"""Prototype: STASUNet ep16 streaming wrapper with a torch CUDA FIFO of 5 frames.

Architectural hypothesis: caching source frames in a Python-side FIFO and
re-invoking ORT-TRT EP directly avoids VS's 5-clip dispatch overhead and the
implicit re-decoding work that the existing vstrt path triggers.

Realistic expectation: the bottleneck is the engine's tiled Swin inference
(~180-225 ms per 1080p output). VS+vstrt already shares source frames across
the 5 offset clips, so the I/O savings are small. This prototype exists to
measure that gap empirically.

Implementation:
- ORT session with TensorRT EP on the fixed-512 ONNX, engine cache shared
  alongside the ONNX (same convention as BSVD V2 wrapper)
- 5-frame deque of torch CUDA tensors (3, H, W) in [-1, 1] for the model
- Per step: feed source frame N+shift, return denoised frame N (shift=2)
- Tiling: 512×512 tiles with 64-pixel overlap, linear-edge blending
- Warmup: frames < shift_num are reflected (first frame repeated)
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


class STASUNetStreaming:
    def __init__(self, onnx_path, H, W, *, tile=512, overlap=64, shift_num=2,
                 device_id=0, fp16=True, trt_cache_dir=None):
        self.onnx_path = str(onnx_path)
        self.H, self.W = H, W
        self.tile = tile
        self.overlap = overlap
        self.shift_num = shift_num
        self.window = 5
        # ONNX I/O is fp32; TRT EP does internal fp16 via trt_fp16_enable.
        self.dtype_np = np.float32
        self.torch_dtype = torch.float32
        self._fp16_internal = fp16
        self.device = torch.device(f'cuda:{device_id}')
        self.device_id = device_id

        # DCN v2 plugin must be initialized before ORT loads the ONNX
        # (same dependency as the vstrt path; see svtav1-dispatch.py:103-107)
        import ctypes
        plug = ctypes.CDLL('/usr/lib/libnvinfer_plugin.so.10', mode=ctypes.RTLD_GLOBAL)
        plug.initLibNvInferPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        plug.initLibNvInferPlugins.restype = ctypes.c_bool
        assert plug.initLibNvInferPlugins(None, b''), 'initLibNvInferPlugins failed'

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        cache = trt_cache_dir or str(Path(self.onnx_path).parent / 'trt_cache_stasunet')
        Path(cache).mkdir(exist_ok=True, parents=True)
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = [
            {'device_id': device_id, 'trt_fp16_enable': fp16,
             'trt_engine_cache_enable': True, 'trt_engine_cache_path': cache,
             'trt_max_workspace_size': 4 * 1024 * 1024 * 1024},
            {'device_id': device_id}, {}]
        self.sess = ort.InferenceSession(
            self.onnx_path, sess_options=sess_opts,
            providers=providers, provider_options=provider_options)

        # Frame FIFO: deque of (3, H, W) torch CUDA tensors in [-1, 1].
        # Index 0 = oldest (= "m2"), index 4 = newest (= "p2").
        self.fifo: deque = deque(maxlen=self.window)
        self.frame_count = 0

        # Pre-compute tile positions; blend weights are computed per-tile
        # because edges that touch the frame boundary must NOT taper.
        self.tile_positions = self._tile_positions(H, W, tile, overlap)
        self.blend_weights = [
            self._make_blend_weight_for_tile(y, x, H, W, tile, overlap)
            for (y, x) in self.tile_positions
        ]

    @staticmethod
    def _tile_positions(H, W, tile, overlap):
        stride = tile - overlap
        ys = list(range(0, max(H - tile, 0) + 1, stride))
        xs = list(range(0, max(W - tile, 0) + 1, stride))
        if ys[-1] + tile < H:
            ys.append(H - tile)
        if xs[-1] + tile < W:
            xs.append(W - tile)
        return [(y, x) for y in ys for x in xs]

    def _make_blend_weight_for_tile(self, y, x, H, W, tile, overlap):
        # Ramp only the edges that overlap with another tile. Edges touching
        # the frame boundary stay at 1 so the outermost pixels aren't multiplied
        # by 0 (which would leave the final image black at the edges after
        # weight-sum normalization).
        def _make_axis(start, axis_len):
            w = np.ones(tile, dtype=np.float32)
            if overlap > 0:
                ramp = np.linspace(0, 1, overlap, endpoint=False, dtype=np.float32)
                if start > 0:  # there's a neighbor on the low side
                    w[:overlap] = ramp
                if start + tile < axis_len:  # there's a neighbor on the high side
                    w[-overlap:] = ramp[::-1]
            return w
        w_y = _make_axis(y, H)
        w_x = _make_axis(x, W)
        w2d = np.outer(w_y, w_x)
        return torch.from_numpy(w2d).to(self.device, dtype=self.torch_dtype)

    def reset(self):
        self.fifo.clear()
        self.frame_count = 0

    def _stack_input(self):
        """Return (1, 15, H, W) tensor by concatenating 5 frames along C."""
        assert len(self.fifo) == self.window
        return torch.cat(list(self.fifo), dim=0).unsqueeze(0)  # (1, 15, H, W)

    def _run_tiled(self, stacked):
        """Tile the (1, 15, H, W) input, run engine per tile, stitch + blend."""
        out = torch.zeros(1, 3, self.H, self.W, device=self.device, dtype=self.torch_dtype)
        weight_sum = torch.zeros(1, 1, self.H, self.W, device=self.device, dtype=self.torch_dtype)
        for (y, x), bw_2d in zip(self.tile_positions, self.blend_weights):
            bw = bw_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, tile, tile)
            tile_in = stacked[:, :, y:y + self.tile, x:x + self.tile].contiguous()
            # ORT IO binding for zero-copy on CUDA
            io = self.sess.io_binding()
            io.bind_input(name='input', device_type='cuda', device_id=self.device_id,
                          element_type=self.dtype_np, shape=tuple(tile_in.shape),
                          buffer_ptr=tile_in.data_ptr())
            tile_out = torch.empty(1, 3, self.tile, self.tile, device=self.device, dtype=self.torch_dtype)
            io.bind_output(name='output', device_type='cuda', device_id=self.device_id,
                           element_type=self.dtype_np, shape=tuple(tile_out.shape),
                           buffer_ptr=tile_out.data_ptr())
            self.sess.run_with_iobinding(io)
            out[:, :, y:y + self.tile, x:x + self.tile] += tile_out * bw
            weight_sum[:, :, y:y + self.tile, x:x + self.tile] += bw

        out = out / weight_sum.clamp(min=1e-6)
        return out

    def step(self, noisy_rgb_3hw):
        """Feed one RGB frame in [0, 1]; advance state; return one denoised frame.

        Returns the denoised version of input[frame_count - shift_num]; the
        first `shift_num` calls return warmup data (denoised version of the
        replayed first frame).
        """
        # Normalize [0, 1] → [-1, 1]
        nrm = (noisy_rgb_3hw.to(self.device, self.torch_dtype) * 2 - 1)
        if len(self.fifo) == 0:
            # Initial warmup: pre-fill FIFO with this frame repeated
            for _ in range(self.window):
                self.fifo.append(nrm.clone())
        else:
            self.fifo.append(nrm)

        stacked = self._stack_input()
        den = self._run_tiled(stacked)
        # Denormalize [-1, 1] → [0, 1]
        den = ((den + 1) * 0.5).clamp(0, 1)
        self.frame_count += 1
        return den

    def flush(self, n_steps):
        """At end of clip: replay the last frame `n_steps` times to drain delay."""
        if not self.fifo:
            return []
        last = self.fifo[-1]
        outs = []
        for _ in range(n_steps):
            self.fifo.append(last.clone())
            outs.append(((self._run_tiled(self._stack_input()) + 1) * 0.5).clamp(0, 1))
            self.frame_count += 1
        return outs


def build_stasunet_streaming(source_clip, *, onnx_path, device_id=0, fp16=True,
                             tile=512, overlap=64, shift_num=2, cache_size=128):
    """VS clip wrapping STASUNetStreaming. Mirrors bsvd_vs_filter.build_bsvd_streaming.

    Sequential access fast path; non-sequential served from LRU output cache.
    """
    import threading
    from collections import OrderedDict
    import vapoursynth as vs

    if source_clip.format.id != int(vs.RGBS):
        raise ValueError(f"source_clip must be RGBS, got {source_clip.format.name}")

    H, W = source_clip.height, source_clip.width
    num_frames = source_clip.num_frames

    streamer = STASUNetStreaming(
        onnx_path=onnx_path, H=H, W=W, tile=tile, overlap=overlap,
        shift_num=shift_num, device_id=device_id, fp16=fp16)
    device = streamer.device
    torch_dtype = streamer.torch_dtype

    cache: 'OrderedDict[int, np.ndarray]' = OrderedDict()
    state = {'next_in_idx': 0}
    lock = threading.Lock()

    def _vsframe_to_torch(f):
        arr = np.stack([np.asarray(f[c]) for c in range(3)], axis=0)
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)

    def _store_output(out_tensor, output_idx):
        arr = out_tensor.float().clamp(0, 1).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        cache[output_idx] = arr
        cache.move_to_end(output_idx)
        while len(cache) > cache_size:
            cache.popitem(last=False)

    def _feed_one():
        k = state['next_in_idx']
        if k < num_frames:
            input_frame = source_clip.get_frame(k)
            t = _vsframe_to_torch(input_frame)
        else:
            t = torch.zeros(3, H, W, device=device, dtype=torch.float32)
        out = streamer.step(t)
        output_idx = k - shift_num
        if 0 <= output_idx < num_frames:
            _store_output(out, output_idx)
        state['next_in_idx'] = k + 1

    def _reset_and_replay_to(target_output):
        import sys
        print(f"[stasunet-vs] cache-miss reset, replaying 0..{target_output + shift_num}",
              file=sys.stderr)
        streamer.reset()
        cache.clear()
        state['next_in_idx'] = 0
        while state['next_in_idx'] <= target_output + shift_num:
            _feed_one()

    def selector(n, f):
        with lock:
            if n not in cache:
                if n < state['next_in_idx'] - shift_num:
                    _reset_and_replay_to(n)
                else:
                    while state['next_in_idx'] <= n + shift_num:
                        _feed_one()
            arr = cache[n]
            cache.move_to_end(n)

        fout = f.copy()
        for c in range(3):
            np.asarray(fout[c])[:] = arr[c]
        return fout

    return vs.core.std.ModifyFrame(source_clip, source_clip, selector)
