"""The VPY plugin search order decides which build of ffms2/vszip/vship runs."""
import importlib.util
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_dispatch():
    spec = importlib.util.spec_from_file_location(
        "svtav1_dispatch", REPO / "tools" / "svtav1-dispatch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prefix_leads_and_usr_local_is_gone(monkeypatch):
    monkeypatch.delenv("VS_PREFIX", raising=False)
    dirs = _load_dispatch().vs_plugin_dirs()
    assert dirs[0] == "/opt/archav1an/lib/vapoursynth"
    # gpu1's pacman vszip is still 13.0 and lacks Dither, so the prefix must win.
    assert "/usr/lib/vapoursynth" in dirs
    # Unowned March-era copies would shadow current pacman builds.
    assert not any("/usr/local" in d for d in dirs)


def test_vs_prefix_is_honoured(monkeypatch):
    monkeypatch.setenv("VS_PREFIX", "/srv/vs")
    dirs = _load_dispatch().vs_plugin_dirs()
    assert dirs[0] == "/srv/vs/lib/vapoursynth"


def test_generated_denoise_vpy_embeds_the_search_list(tmp_path, monkeypatch):
    monkeypatch.delenv("VS_PREFIX", raising=False)
    mod = _load_dispatch()
    vpy = tmp_path / "x.vpy"
    mod.write_denoise_vpy(str(vpy), source="/t/x.MOV",
                          cachefile=str(tmp_path / "x.ffindex"),
                          model_name="bsvd", tile=0, streams=1)
    text = vpy.read_text()
    assert "/opt/archav1an/lib/vapoursynth" in text
    assert "/usr/local/lib/vapoursynth" not in text
    compile(text, str(vpy), "exec")     # the generated script must be valid Python


def test_generated_ssimu2_scripts_embed_the_search_list(monkeypatch):
    """Both metric backends build their own VPY and must render the same list."""
    import subprocess as _sp

    mod = _load_dispatch()
    monkeypatch.delenv("VS_PREFIX", raising=False)
    captured = []

    def fake_run(argv, **kw):
        captured.append(argv[-1])
        return _sp.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    for tool in ("vs-hip", "vs-zip"):
        mod.measure_ssimu2("/t/a.mkv", "/t/b.mkv", tool)

    assert len(captured) == 2
    for script in captured:
        assert "/opt/archav1an/lib/vapoursynth" in script
        assert "/usr/local/lib/vapoursynth" not in script
        compile(script, "<ssimu2>", "exec")
