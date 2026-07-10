# tools/optsig_pref/tests/test_features.py
import numpy as np
from tools.optsig_pref.features import brightness, tnoise, spatial_std_flat

def _noisy(T=20, H=64, W=64, mean=120, sigma_dn=3.0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.full((H, W), mean, np.float32)
    return np.clip(base[None] + rng.normal(0, sigma_dn, (T, H, W)), 0, 255).astype(np.float32)

def _motion(T=12, H=64, W=64):
    # smooth ramp translated +3 DN/frame: every block sees a uniform directional
    # shift => genuine content motion, no static blocks.
    xx = np.arange(W, dtype=np.float32)[None, :]
    return np.stack([np.tile(100.0 + (xx + 3.0 * t), (H, 1)) for t in range(T)]).astype(np.float32)

def test_brightness_scales_0_1():
    f = np.full((5, 8, 8), 128, np.float32)
    assert abs(brightness(f) - 128 / 255) < 1e-6

def test_tnoise_recovers_sigma_static_scene():
    v, frac = tnoise(_noisy(sigma_dn=3.0))
    assert 2.0 < v < 4.0 and frac > 0.5

def test_tnoise_nan_when_all_motion():
    # motion everywhere => static_frac below motion_frac_min => NaN
    v, frac = tnoise(_motion())
    assert np.isnan(v) and frac < 0.05

def test_tnoise_static_under_heavy_noise():
    # heavy independent noise is NOT motion (block-mean-of-diff ~0): blocks stay
    # static and tnoise returns a value. Documents the gate semantics.
    v, frac = tnoise(_noisy(sigma_dn=8.0))
    assert not np.isnan(v) and frac > 0.5

def test_tnoise_short_window_returns_nan():
    f = np.full((1, 64, 64), 120.0, np.float32)   # <2 frames => no diffs
    v, frac = tnoise(f)
    assert np.isnan(v) and frac == 0.0
