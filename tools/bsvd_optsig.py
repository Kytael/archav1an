"""BSVD optimal-σ predictor (V3 1080p calibration).

Pre-pass over the first ~30 frames of a source video; returns a constant σ
locked to the predictor grid {0.04, 0.05, 0.06, 0.07, 0.08, 0.10}. Used by
the BSVD denoise path when --bsvd-sigma=auto.

Vendored from ~/gitproj/bsvd/{optimal_sigma_predictor.py, sigma_estimator.py}
on 2026-05-21 so archav1an is self-contained. If the BSVD repo's coefficients
or σ_estimator architecture change, sync here.

Coefficients (V3, 33-clip grouped LOO MAE 0.0058):
    features:   sigma_raw, std_intensity, mean_motion
    means:      0.00926570, 0.18122130, 0.02040566
    stds:       0.00082168, 0.03697929, 0.00889156
    bias:       0.08030303
    weights:    0.00448410, -0.01410607, -0.00599844
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


V3_MODEL = {
    'features': ['sigma_raw', 'std_intensity', 'mean_motion'],
    'feature_means': [0.00926570, 0.18122130, 0.02040566],
    'feature_stds':  [0.00082168, 0.03697929, 0.00889156],
    'bias': 0.08030303,
    'weights': [0.00448410, -0.01410607, -0.00599844],
    'sigma_grid': [0.04, 0.05, 0.06, 0.07, 0.08, 0.10],
}


class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class _SigmaEstimator(nn.Module):
    """Fully-conv σ-map predictor over T-frame channel stack. Inference only.

    Mirrors ~/gitproj/bsvd/sigma_estimator.py:37-62; trained checkpoint format
    (ckpt['args'] dict + ckpt['state_dict']) is unchanged.
    """

    def __init__(self, ch: int = 32, n_blocks: int = 4, sigma_max: float = 0.20, T: int = 1):
        super().__init__()
        self.sigma_max = float(sigma_max)
        self.T = int(T)
        self.in_conv = nn.Conv2d(3 * self.T, ch, 3, padding=1)
        self.blocks = nn.Sequential(*[_ResBlock(ch) for _ in range(n_blocks)])
        self.out_conv = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x):
        h = F.relu(self.in_conv(x))
        h = self.blocks(h)
        return torch.sigmoid(self.out_conv(h)) * self.sigma_max


class OptimalSigmaPredictor:
    """Streaming pre-pass: observe N frames, then lock a single σ.

    Vendored from ~/gitproj/bsvd/optimal_sigma_predictor.py:51-180.
    """

    def __init__(self, sigma_estimator_pth, device='cuda', model: dict | None = None):
        m = model or V3_MODEL
        self.features = list(m['features'])
        self.mu = np.array(m['feature_means'], dtype=np.float64)
        self.sd = np.array(m['feature_stds'], dtype=np.float64)
        self.bias = float(m['bias'])
        self.weights = np.array(m['weights'], dtype=np.float64)
        self.sigma_grid = np.array(m['sigma_grid'], dtype=np.float64)
        self.device = torch.device(device)

        self._needs_sigma_estimator = 'sigma_raw' in self.features
        self._sig_model = None
        self._sig_T = None
        self._sig_half = None
        if self._needs_sigma_estimator:
            ckpt = torch.load(str(sigma_estimator_pth), map_location=self.device, weights_only=False)
            sa = ckpt['args']
            self._sig_T = int(sa['T'])
            self._sig_half = self._sig_T // 2
            self._sig_model = _SigmaEstimator(
                ch=sa['ch'], n_blocks=sa['n_blocks'],
                sigma_max=float(sa.get('gauss_sigma_max', 0.20)),
                T=self._sig_T,
            ).to(self.device).half().eval()
            self._sig_model.load_state_dict(ckpt['state_dict'])
        self._needs_motion = 'mean_motion' in self.features
        self.reset()

    def reset(self):
        self._intens_sum = 0.0
        self._intens_sq_sum = 0.0
        self._pixel_count = 0
        self._n_frames = 0
        self._sigma_raw_sum = 0.0
        self._sigma_raw_count = 0
        self._motion_sum = 0.0
        self._motion_count = 0
        self._prev_gray = None
        self._sig_buffer: deque = deque(maxlen=self._sig_T) if self._sig_T else deque()
        self._locked_sigma: float | None = None

    @torch.inference_mode()
    def observe(self, frame_b3hw):
        f = frame_b3hw.to(self.device).float()
        gray = f.mean(dim=1)
        self._intens_sum += float(gray.sum())
        self._intens_sq_sum += float((gray * gray).sum())
        self._pixel_count += int(gray.numel())

        if self._needs_motion:
            if self._prev_gray is not None:
                diff = (gray - self._prev_gray).abs()
                self._motion_sum += float(diff.sum())
                self._motion_count += int(diff.numel())
            self._prev_gray = gray

        if self._sig_model is not None:
            f16 = f[0].half() if f.dim() == 4 else f.half()
            self._sig_buffer.append(f16)
            stack = list(self._sig_buffer)
            while len(stack) < self._sig_T:
                stack.insert(0, stack[0])
            x = torch.stack(stack, dim=0).reshape(1, 3 * self._sig_T, *f16.shape[-2:])
            sigma_map = self._sig_model(x)
            self._sigma_raw_sum += float(sigma_map.float().mean())
            self._sigma_raw_count += 1

        self._n_frames += 1

    def _feature_vec(self):
        mean_int = self._intens_sum / max(self._pixel_count, 1)
        var_int = max(self._intens_sq_sum / max(self._pixel_count, 1) - mean_int ** 2, 0)
        std_int = float(var_int ** 0.5)
        feat = {
            'mean_intensity': mean_int,
            'std_intensity': std_int,
            'mean_motion': (self._motion_sum / max(self._motion_count, 1)) if self._motion_count > 0 else 0.0,
            'sigma_raw': (self._sigma_raw_sum / max(self._sigma_raw_count, 1)) if self._sigma_raw_count > 0 else 0.0,
        }
        return np.array([feat[f] for f in self.features]), feat

    def lock(self) -> float:
        if self._n_frames == 0:
            raise RuntimeError("lock() called before observing any frames")
        x, _ = self._feature_vec()
        xn = (x - self.mu) / self.sd
        raw = self.bias + float((self.weights * xn).sum())
        snapped = float(self.sigma_grid[int(np.argmin(np.abs(self.sigma_grid - raw)))])
        self._locked_sigma = snapped
        return snapped

    @property
    def sigma(self) -> float | None:
        return self._locked_sigma

    @property
    def stats(self) -> dict:
        if self._n_frames == 0:
            return {}
        _, feat = self._feature_vec()
        feat = dict(feat)
        feat['n_frames'] = self._n_frames
        feat['locked_sigma'] = self._locked_sigma
        return feat


def compute_sigma_for_video(video_path, *, sigma_estimator_pth,
                            warmup_frames: int = 30, device: str = 'cuda:0') -> float:
    """Decode `warmup_frames` from `video_path` via PyAV and return locked σ.

    Snaps to the V3 predictor's grid {0.04, 0.05, 0.06, 0.07, 0.08, 0.10}.
    Prints feature trace to stderr for diagnostics.
    """
    import av

    pred = OptimalSigmaPredictor(sigma_estimator_pth=sigma_estimator_pth, device=device)
    container = av.open(str(video_path))
    try:
        for n, frame in enumerate(container.decode(video=0)):
            if n >= warmup_frames:
                break
            arr = frame.to_ndarray(format='rgb24')
            t = torch.from_numpy(arr).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            pred.observe(t)
    finally:
        container.close()

    sigma = pred.lock()
    s = pred.stats
    feat_str = ' '.join(
        f'{k}={v:.4f}' for k, v in s.items()
        if k in ('sigma_raw', 'std_intensity', 'mean_intensity', 'mean_motion'))
    print(f"[bsvd-optsig] σ_eff={sigma:.3f} ({feat_str})", file=sys.stderr)
    return sigma


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="BSVD V3 optimal-σ predictor pre-pass.")
    ap.add_argument('video')
    ap.add_argument('--sigma-estimator-pth', required=True)
    ap.add_argument('--warmup-frames', type=int, default=30)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    σ = compute_sigma_for_video(
        args.video, sigma_estimator_pth=args.sigma_estimator_pth,
        warmup_frames=args.warmup_frames, device=args.device)
    print(σ)
