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


def test_prefix_bin_goes_to_the_front_of_path(monkeypatch):
    """which() resolved to March /usr/local builds on every host until this."""
    import os
    mod = _load_dispatch()
    monkeypatch.setenv("VS_PREFIX", "/opt/archav1an")
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    if not os.path.isdir("/opt/archav1an/bin"):
        import pytest
        pytest.skip("no prefix bin on this host")
    mod.prefer_prefix_bin()
    assert os.environ["PATH"].split(os.pathsep)[0] == "/opt/archav1an/bin"


def test_prefix_bin_is_not_duplicated_when_already_first(monkeypatch):
    import os
    mod = _load_dispatch()
    monkeypatch.setenv("VS_PREFIX", "/opt/archav1an")
    monkeypatch.setenv("PATH", "/opt/archav1an/bin:/usr/bin")
    if not os.path.isdir("/opt/archav1an/bin"):
        import pytest
        pytest.skip("no prefix bin on this host")
    mod.prefer_prefix_bin()
    assert os.environ["PATH"].count("/opt/archav1an/bin") == 1


def test_a_missing_prefix_bin_leaves_path_alone(monkeypatch):
    import os
    mod = _load_dispatch()
    monkeypatch.setenv("VS_PREFIX", "/nonexistent-prefix-xyz")
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    mod.prefer_prefix_bin()
    assert os.environ["PATH"] == "/usr/local/bin:/usr/bin"


def _vpy_for(tmp_path, **kw):
    mod = _load_dispatch()
    vpy = tmp_path / "x.vpy"
    mod.write_denoise_vpy(str(vpy), source="/t/x.MOV",
                          cachefile=str(tmp_path / "x.ffindex"),
                          model_name="bsvd", tile=0, streams=1,
                          use_bsvd=True, bsvd_onnx="/m/b.onnx", **kw)
    return vpy.read_text()


def test_bsvd_without_tile_uses_the_full_frame_streamer(tmp_path):
    text = _vpy_for(tmp_path)
    assert "build_bsvd_streaming" in text
    assert "build_bsvd_windowed_tiled" not in text
    compile(text, "<vpy>", "exec")


def test_bsvd_tile_switches_to_the_windowed_builder(tmp_path):
    """The 2070S cannot hold the full-frame state, so it must take this path."""
    text = _vpy_for(tmp_path, bsvd_tile=576, bsvd_window=1500, bsvd_margin=32)
    assert "build_bsvd_windowed_tiled" in text
    assert "build_bsvd_streaming" not in text
    assert "tile=576" in text and "window=1500" in text and "margin=32" in text
    compile(text, "<vpy>", "exec")
