import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_dispatch():
    spec = importlib.util.spec_from_file_location(
        "svtav1_dispatch", REPO / "tools" / "svtav1-dispatch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_remote_source_is_used_verbatim_and_skips_staging():
    d = _load_dispatch()
    src, needs_staging = d.resolve_remote_src(
        remote_source="/mnt/media/dance/SetA/2003/event-c/MVI_0068.MOV",
        remote_root="~/archav1an",
        input_file="/tmp/staged/MVI_0068.MOV")
    assert src == "/mnt/media/dance/SetA/2003/event-c/MVI_0068.MOV"
    assert needs_staging is False


def test_without_remote_source_it_stages_into_temp_remote():
    d = _load_dispatch()
    src, needs_staging = d.resolve_remote_src(
        remote_source=None, remote_root="~/archav1an",
        input_file="/tmp/staged/MVI_0068.MOV")
    assert src == "Temp/_remote/MVI_0068.MOV"
    assert needs_staging is True
