"""--bsvd-tile carries three forms, and the split-host path re-parses it.

The denoise half of a remote run receives this flag as a string that the same
parser reads back, so a value that does not survive the round trip silently
denoises at the wrong tile size rather than failing.
"""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_dispatch():
    spec = importlib.util.spec_from_file_location(
        "svtav1_dispatch", REPO / "tools" / "svtav1-dispatch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("text,expected", [
    ("auto", "auto"),
    ("1096x976", (1096, 976)),
    ("576", 576),
])
def test_every_tile_form_survives_the_round_trip(text, expected):
    mod = _load_dispatch()
    assert mod.parse_tile_arg(text) == expected
    assert mod.tile_arg(expected) == text
    assert mod.parse_tile_arg(mod.tile_arg(expected)) == expected


def test_a_rectangular_tile_is_not_forwarded_as_a_python_repr():
    """"(1096, 976)" would reach the remote half and fail to parse."""
    mod = _load_dispatch()
    assert mod.tile_arg((1096, 976)) == "1096x976"
    assert "(" not in mod.tile_arg([1096, 976])


def test_the_generated_vpy_quotes_auto_rather_than_naming_it(tmp_path):
    """tile=auto in the script would be a NameError at VPY evaluation."""
    mod = _load_dispatch()
    vpy = tmp_path / "clip.vpy"
    mod.write_denoise_vpy(
        str(vpy), str(tmp_path / "in.MOV"), str(tmp_path / "c.ffindex"),
        "color_real_life", 512, 2, use_bsvd=True,
        bsvd_onnx=str(tmp_path / "m.onnx"), bsvd_sigma=0.05, bsvd_ep="TRT",
        bsvd_device=0, bsvd_tile="auto", bsvd_overlap=16, bsvd_window=750,
        bsvd_margin=32)
    text = vpy.read_text()
    assert "tile='auto'" in text, "auto must be a string literal in the script"
    assert "tile=auto," not in text


def test_the_generated_vpy_emits_a_rectangular_tile_as_a_tuple(tmp_path):
    mod = _load_dispatch()
    vpy = tmp_path / "clip.vpy"
    mod.write_denoise_vpy(
        str(vpy), str(tmp_path / "in.MOV"), str(tmp_path / "c.ffindex"),
        "color_real_life", 512, 2, use_bsvd=True,
        bsvd_onnx=str(tmp_path / "m.onnx"), bsvd_sigma=0.05, bsvd_ep="TRT",
        bsvd_device=0, bsvd_tile=(1096, 976), bsvd_overlap=16,
        bsvd_window=750, bsvd_margin=32)
    assert "tile=(1096, 976)" in vpy.read_text()
