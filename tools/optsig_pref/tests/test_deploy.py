# tools/optsig_pref/tests/test_deploy.py
import json
import numpy as np
import pytest
from tools import bsvd_optsig as m

def test_module_has_no_torch():
    src = open("tools/bsvd_optsig.py").read()
    assert "import torch" not in src and "from torch" not in src

def test_window_math_and_rule(tmp_path, monkeypatch):
    art = dict(kind="brightness-threshold", features=["brightness"], threshold=0.41,
               sigma_low=0.01, sigma_high=0.05, sigma_grid=[0.01, 0.05],
               window=dict(start_fraction=0.40, length=180))
    mj = tmp_path / "m.json"; mj.write_text(json.dumps(art))
    seen = {}
    def fake_decode(path, start, length):
        seen.update(start=start, length=length)
        return np.full((5, 8, 8), 200.0, np.float32)   # bright -> sigma_low
    monkeypatch.setattr(m, "decode_window", fake_decode)
    s = m.compute_sigma_for_video("x.mov", model_json=str(mj), _nframes=lambda p: 1000, verbose=False)
    assert seen["start"] == 400 and seen["length"] == 180
    assert s == 0.01

def test_dark_clip_gets_high_sigma(tmp_path, monkeypatch):
    art = dict(kind="brightness-threshold", features=["brightness"], threshold=0.41,
               sigma_low=0.01, sigma_high=0.05, sigma_grid=[0.01, 0.05],
               window=dict(start_fraction=0.40, length=180))
    mj = tmp_path / "m.json"; mj.write_text(json.dumps(art))
    monkeypatch.setattr(m, "decode_window", lambda p, s, l: np.full((5, 8, 8), 60.0, np.float32))
    assert m.compute_sigma_for_video("x.mov", model_json=str(mj), _nframes=lambda p: 1000, verbose=False) == 0.05

def test_short_clip_window_clamped(tmp_path, monkeypatch):
    art = dict(kind="brightness-threshold", features=["brightness"], threshold=0.41,
               sigma_low=0.01, sigma_high=0.05, sigma_grid=[0.01, 0.05],
               window=dict(start_fraction=0.40, length=180))
    mj = tmp_path / "m.json"; mj.write_text(json.dumps(art))
    seen = {}
    def fake_decode(path, start, length):
        seen.update(start=start)
        return np.full((5, 8, 8), 60.0, np.float32)
    monkeypatch.setattr(m, "decode_window", fake_decode)
    m.compute_sigma_for_video("x.mov", model_json=str(mj), _nframes=lambda p: 200, verbose=False)
    assert seen["start"] == 20   # min(round(0.4*200)=80, 200-180=20) -> clamped to 20

def test_wrong_kind_rejected(tmp_path):
    mj = tmp_path / "m.json"; mj.write_text(json.dumps(dict(kind="linear")))
    with pytest.raises(ValueError):
        m.compute_sigma_for_video("x.mov", model_json=str(mj), _nframes=lambda p: 1000)
