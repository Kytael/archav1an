"""BSVD V2 stateful streaming as a VapourSynth filter.

Wraps `BSVDOrtStreamingV2` (ORT-TRT EP on NVIDIA, ORT-MIGraphX EP on AMD) in a
`core.std.ModifyFrame` closure for sequential frame access. Sequential access
is required because the streaming wrapper carries 38 state tensors across
calls; the svtav1-dispatch single-vspipe model guarantees that.

Vendored from ~/gitproj/bsvd/bsvd_ort_streaming_v2.py on 2026-05-21 with one
extension: `ep` parameter selects 'TRT' (NVIDIA, the upstream default) or
'MIGRAPHX' (AMD, new). The TRT and CUDA EP paths are unchanged from upstream.

Output offset: BSVDOrtStreamingV2.step(input[k]) returns denoised(input[k-16])
for shift_num=16. This filter compensates by feeding ahead internally so VS
sees an output clip of the same length as the input.

Edge handling (2026-07-06): both ends are mirror-padded with reflected real
frames. Previously the head ran against zero-initialized state tensors and the
tail was flushed with zero frames, so the first/last shift_num outputs were
denoised against black — visibly under-denoised at the very last frame.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import vapoursynth as vs

FOLD_DIV = 8

VARIANT_CONFIG = {
    'bsvd-64': dict(chns=[64, 128, 256], mid_ch=64, in_ch=4, out_ch=3),
    'bsvd-32': dict(chns=[32, 64, 128], mid_ch=32, in_ch=4, out_ch=3),
}


def _state_input_names_v2():
    names = []
    for prefix in ('t1', 't2'):
        names += [f'{prefix}_s1_pop', f'{prefix}_s2_pop', f'{prefix}_s3_pop']
        for memconv in ('d0', 'd1', 'u2', 'u1'):
            for buf in ('c1l', 'c1c', 'c2l', 'c2c'):
                names.append(f'{prefix}_{memconv}_{buf}')
    return names


def _state_output_names_v2():
    names = []
    for prefix in ('t1', 't2'):
        names += [f'n_{prefix}_s1_push', f'n_{prefix}_s2_push', f'n_{prefix}_s3_push']
        for memconv in ('d0', 'd1', 'u2', 'u1'):
            for buf in ('c1l', 'c1c', 'c2l', 'c2c'):
                names.append(f'n_{prefix}_{memconv}_{buf}')
    return names


SKIP_LENS = [8, 8, 4]
SKIP_SPATIAL_DIV = [1, 1, 2]


class BSVDOrtStreamingV2:
    """V2 stateful streaming wrapper. Step (1, 3, H, W) → (1, 3, H, W), shifted 16 frames.

    `ep`: 'TRT' (TensorRT EP on NVIDIA), 'CUDA' (CUDA EP on NVIDIA), or
    'MIGRAPHX' (MIGraphX EP on AMD). On AMD the device tensor is still allocated
    as 'cuda:N' because the ROCm torch build exposes HIP devices through the
    CUDA API — ORT-MIGX picks up the HIP context via device_id.
    """

    def __init__(self, onnx_path, H, W, B=1, variant='bsvd-64',
                 shift_num=16, fp16=True, device_id=0, ep='TRT',
                 trt_cache_dir=None):
        self.onnx_path = str(onnx_path)
        self.H, self.W, self.B = H, W, B
        self.shift_num = shift_num
        self.dtype = np.float16 if fp16 else np.float32
        self.torch_dtype = torch.float16 if fp16 else torch.float32
        self.device = torch.device(f'cuda:{device_id}')
        self.device_id = device_id
        self.ep = ep.upper()

        cfg = VARIANT_CONFIG[variant]
        self.chns, self.in_ch, self.mid_ch, self.out_ch = (
            cfg['chns'], cfg['in_ch'], cfg['mid_ch'], cfg['out_ch'])
        lyr0, lyr1, lyr2 = self.chns
        self.skip_chs = [3, lyr0, lyr1]
        f1, f2 = lyr1 // FOLD_DIV, lyr2 // FOLD_DIV

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        if self.ep == 'TRT':
            cache = trt_cache_dir or str(Path(self.onnx_path).parent / 'trt_cache_v2')
            Path(cache).mkdir(exist_ok=True, parents=True)
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            provider_options = [
                {'device_id': device_id, 'trt_fp16_enable': fp16,
                 'trt_engine_cache_enable': True, 'trt_engine_cache_path': cache,
                 'trt_max_workspace_size': 4 * 1024 * 1024 * 1024},
                {'device_id': device_id}, {}]
            self._io_device_type = 'cuda'
        elif self.ep == 'MIGRAPHX':
            # Compiled-model cache: the graph compile is ~163 s per session on
            # Strix at 1080p — cache next to the ONNX like the TRT engines.
            # ORT >=1.23 exposes a single EP-managed migraphx_model_cache_dir
            # (the older save/load_compiled_model option pairs are gone).
            cache = trt_cache_dir or str(Path(self.onnx_path).parent / 'migx_cache')
            Path(cache).mkdir(exist_ok=True, parents=True)
            provider_options = [
                {'device_id': device_id, 'migraphx_fp16_enable': fp16,
                 'migraphx_model_cache_dir': cache}, {}]
            providers = ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
            self._io_device_type = 'cuda'
        elif self.ep == 'CUDA':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            provider_options = [{'device_id': device_id}, {}]
            self._io_device_type = 'cuda'
        else:
            raise ValueError(f"Unknown EP {ep!r}; expected TRT, CUDA, or MIGRAPHX")
        self.sess = ort.InferenceSession(
            self.onnx_path, sess_options=sess_opts,
            providers=providers, provider_options=provider_options)
        self.io = self.sess.io_binding()

        self.state_input_names = _state_input_names_v2()
        self.state_output_names = _state_output_names_v2()

        self.fifos = []
        for db in range(2):
            for q in range(3):
                L = SKIP_LENS[q]
                C = self.skip_chs[q]
                div = SKIP_SPATIAL_DIV[q]
                Hd, Wd = H // div, W // div
                self.fifos.append([
                    torch.zeros(B, C, Hd, Wd, device=self.device, dtype=self.torch_dtype)
                    for _ in range(L)
                ])

        bbc_shapes_per_db = [
            (B, f1, H // 2, W // 2), (B, lyr1, H // 2, W // 2),
            (B, f1, H // 2, W // 2), (B, lyr1, H // 2, W // 2),
            (B, f2, H // 4, W // 4), (B, lyr2, H // 4, W // 4),
            (B, f2, H // 4, W // 4), (B, lyr2, H // 4, W // 4),
            (B, f2, H // 4, W // 4), (B, lyr2, H // 4, W // 4),
            (B, f2, H // 4, W // 4), (B, lyr2, H // 4, W // 4),
            (B, f1, H // 2, W // 2), (B, lyr1, H // 2, W // 2),
            (B, f1, H // 2, W // 2), (B, lyr1, H // 2, W // 2),
        ]
        self.bbc_shapes = bbc_shapes_per_db * 2
        self.bbc_a = [torch.zeros(*s, device=self.device, dtype=self.torch_dtype) for s in self.bbc_shapes]
        self.bbc_b = [torch.zeros(*s, device=self.device, dtype=self.torch_dtype) for s in self.bbc_shapes]
        self.bbc_in = self.bbc_a
        self.bbc_out = self.bbc_b

        self.push_buffers = []
        for db in range(2):
            for q in range(3):
                C = self.skip_chs[q]
                div = SKIP_SPATIAL_DIV[q]
                Hd, Wd = H // div, W // div
                self.push_buffers.append(
                    torch.zeros(B, C, Hd, Wd, device=self.device, dtype=self.torch_dtype))

        self.frame_in = torch.zeros(B, self.in_ch, H, W, device=self.device, dtype=self.torch_dtype)
        self.frame_out = torch.zeros(B, self.out_ch, H, W, device=self.device, dtype=self.torch_dtype)
        self.frame_count = 0

    def reset(self):
        for fifo in self.fifos:
            for t in fifo:
                t.zero_()
        for t in self.bbc_a:
            t.zero_()
        for t in self.bbc_b:
            t.zero_()
        self.frame_count = 0

    def _bind(self):
        self.io.clear_binding_inputs()
        self.io.clear_binding_outputs()
        self.io.bind_input(name='frame', device_type=self._io_device_type,
                           device_id=self.device_id,
                           element_type=self.dtype, shape=tuple(self.frame_in.shape),
                           buffer_ptr=self.frame_in.data_ptr())
        in_idx = 0
        for db in range(2):
            for q in range(3):
                fifo = self.fifos[db * 3 + q]
                pop_t = fifo[-1]
                self.io.bind_input(name=self.state_input_names[in_idx],
                                   device_type=self._io_device_type, device_id=self.device_id,
                                   element_type=self.dtype, shape=tuple(pop_t.shape),
                                   buffer_ptr=pop_t.data_ptr())
                in_idx += 1
            for j in range(16):
                t = self.bbc_in[db * 16 + j]
                self.io.bind_input(name=self.state_input_names[in_idx],
                                   device_type=self._io_device_type, device_id=self.device_id,
                                   element_type=self.dtype, shape=tuple(t.shape),
                                   buffer_ptr=t.data_ptr())
                in_idx += 1
        assert in_idx == 38

        self.io.bind_output(name='out_frame', device_type=self._io_device_type,
                            device_id=self.device_id,
                            element_type=self.dtype, shape=tuple(self.frame_out.shape),
                            buffer_ptr=self.frame_out.data_ptr())
        out_idx = 0
        for db in range(2):
            for q in range(3):
                push_t = self.push_buffers[db * 3 + q]
                self.io.bind_output(name=self.state_output_names[out_idx],
                                    device_type=self._io_device_type, device_id=self.device_id,
                                    element_type=self.dtype, shape=tuple(push_t.shape),
                                    buffer_ptr=push_t.data_ptr())
                out_idx += 1
            for j in range(16):
                t = self.bbc_out[db * 16 + j]
                self.io.bind_output(name=self.state_output_names[out_idx],
                                    device_type=self._io_device_type, device_id=self.device_id,
                                    element_type=self.dtype, shape=tuple(t.shape),
                                    buffer_ptr=t.data_ptr())
                out_idx += 1
        assert out_idx == 38

    def step(self, noisy_rgb, sigma):
        if isinstance(sigma, (int, float)):
            sigma_plane = torch.full(
                (self.B, 1, self.H, self.W), float(sigma),
                device=self.device, dtype=self.torch_dtype)
        else:
            sigma_plane = sigma.to(device=self.device, dtype=self.torch_dtype)

        frame_4ch = torch.cat([noisy_rgb.to(self.torch_dtype), sigma_plane], dim=1)
        self.frame_in.copy_(frame_4ch)
        # ORT's TRT/CUDA EPs run on their own non-blocking stream and input
        # sync is the caller's responsibility — flush torch's stream so the
        # engine can't read frame_in before the copy above lands.
        torch.cuda.current_stream(self.device).synchronize()
        self._bind()
        self.sess.run_with_iobinding(self.io)

        for fifo_idx in range(6):
            pushed = self.push_buffers[fifo_idx]
            old_oldest = self.fifos[fifo_idx].pop()
            self.fifos[fifo_idx].insert(0, pushed)
            self.push_buffers[fifo_idx] = old_oldest

        self.bbc_in, self.bbc_out = self.bbc_out, self.bbc_in
        self.frame_count += 1
        return self.frame_out


def _vsframe_rgbs_to_torch(f: vs.VideoFrame, device, dtype) -> torch.Tensor:
    """RGBS VS frame → (1, 3, H, W) torch tensor."""
    arr = np.stack([np.asarray(f[c]) for c in range(3)], axis=0)  # (3, H, W) float32
    return torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=dtype)


def _torch_to_vsframe_rgbs(t: torch.Tensor, fout: vs.VideoFrame) -> vs.VideoFrame:
    """Write (1, 3, H, W) torch tensor into a mutable RGBS VS frame copy."""
    arr = t.float().clamp(0, 1).squeeze(0).cpu().numpy()  # (3, H, W)
    for c in range(3):
        np.asarray(fout[c])[:] = arr[c]
    return fout


def build_bsvd_streaming(source_clip: vs.VideoNode, *,
                         onnx_path: str, sigma: float, ep: str = 'TRT',
                         device_id: int = 0, fp16: bool = True,
                         variant: str = 'bsvd-64',
                         shift_num: int = 16,
                         cache_size: int = 64) -> vs.VideoNode:
    """Wrap BSVDOrtStreamingV2 as a VS clip via ModifyFrame.

    `source_clip` must be RGBS. Output has same format / dimensions / num_frames.

    Sequential access is the fast path. Non-sequential access (e.g., when this
    clip feeds SMDegrain's `prefilter=` and motion search requests frames
    [n-tr, n+tr]) is served from an LRU output cache of size `cache_size`
    frames. Entries are stored in the engine's dtype (float16 when fp16=True,
    lossless): ~11.9 MiB/frame at 1080p, so ~760 MiB at the default 64
    (float32 doubles that; 4K is ~4x). A cache miss
    that would require seeking BACKWARD past the cache window triggers a full
    streamer reset and replay from frame 0 — correct but slow; log to stderr.
    """
    if source_clip.format.id != int(vs.RGBS):
        raise ValueError(f"source_clip must be RGBS, got {source_clip.format.name}")

    H, W = source_clip.height, source_clip.width
    num_frames = source_clip.num_frames

    streamer = BSVDOrtStreamingV2(
        onnx_path=onnx_path, H=H, W=W, B=1, variant=variant,
        shift_num=shift_num, fp16=fp16, device_id=device_id, ep=ep)
    torch_dtype = streamer.torch_dtype
    device = streamer.device

    cache: 'OrderedDict[int, np.ndarray]' = OrderedDict()  # output_idx -> (3,H,W) numpy on CPU
    # next_in_idx counts positions in the mirror-padded feed sequence:
    # positions 0..shift_num-1 are the reflected head (f_shift..f_1), then the
    # N real frames, then shift_num reflected tail frames. Output for real
    # frame i emerges at feed position i + 2*shift_num.
    state = {'next_in_idx': 0}
    lock = threading.Lock()

    def _reflect_idx(i, n):
        """numpy-style 'reflect' (no edge repeat): ...f2 f1 | f0..f(n-1) | f(n-2)..."""
        if n == 1:
            return 0
        period = 2 * n - 2
        i %= period
        return i if i < n else period - i
    # Cache in the engine's dtype — fp16 entries are lossless for an fp16
    # engine and halve the footprint; the float32 upcast happens at plane
    # write time in selector().
    cache_dtype = np.float16 if torch_dtype == torch.float16 else np.float32

    def _store_output(out_tensor, output_idx):
        arr = out_tensor.clamp(0, 1).squeeze(0).cpu().numpy().astype(cache_dtype, copy=False)
        cache[output_idx] = arr
        cache.move_to_end(output_idx)
        while len(cache) > cache_size:
            cache.popitem(last=False)

    def _feed_one():
        j = state['next_in_idx']
        src_idx = _reflect_idx(j - shift_num, num_frames)
        input_frame = source_clip.get_frame(src_idx)
        noisy = _vsframe_rgbs_to_torch(input_frame, device, torch_dtype)
        out = streamer.step(noisy, sigma)
        output_idx = j - 2 * shift_num
        if 0 <= output_idx < num_frames:
            _store_output(out, output_idx)
        state['next_in_idx'] = j + 1

    def _reset_and_replay_to(target_output):
        import sys
        print(f"[bsvd-vs] cache-miss reset, replaying 0..{target_output + 2 * shift_num}",
              file=sys.stderr)
        streamer.reset()
        cache.clear()
        state['next_in_idx'] = 0
        while state['next_in_idx'] <= target_output + 2 * shift_num:
            _feed_one()

    def selector(n, f):
        with lock:
            if n not in cache:
                if n < state['next_in_idx'] - 2 * shift_num:
                    # Backward miss past produced range — need to replay.
                    _reset_and_replay_to(n)
                else:
                    # Forward: feed until output n exists.
                    while state['next_in_idx'] <= n + 2 * shift_num:
                        _feed_one()
            arr = cache[n]
            cache.move_to_end(n)  # LRU bump

        fout = f.copy()
        for c in range(3):
            np.asarray(fout[c])[:] = arr[c]
        return fout

    return vs.core.std.ModifyFrame(source_clip, source_clip, selector)
