# tools/optsig_pref/ladder.vpy
import os as _o, sys as _s
_s.path.insert(0, "/opt/archav1an/venv/lib/python3.14/site-packages")
from vstools import vs, core, initialize_clip
core.max_cache_size = 4096
import glob as _g
for _d in ("/usr/lib/vapoursynth", "/usr/local/lib/vapoursynth"):
    for _p in sorted(_g.glob(_o.path.join(_d, "*.so"))):
        try:
            core.std.LoadPlugin(_p)
        except vs.Error:
            pass
_s.path.insert(0, "/home/user/archav1an/tools")
from bsvd_vs_filter import build_bsvd_streaming
CLIP = _o.environ["CLIP_PATH"]; ST = int(_o.environ["WIN_START"]); LN = int(_o.environ["WIN_LEN"])
SIG = float(_o.environ["BSVD_SIGMA"]); CACHE = _o.environ["FFMS_CACHE"]
ONNX = "/home/user/archav1an/models/bsvd_realpair_ep14_stateful_v2_dyn_fp16.onnx"
LEAD = 32; a = max(ST - LEAD, 0); pre = ST - a
src = core.ffms2.Source(source=CLIP, cachefile=CACHE)[a:ST + LN]
src = initialize_clip(src)
rgb = core.resize.Bicubic(src, format=vs.RGBS, matrix_in_s="709", range_in_s="limited")
den = build_bsvd_streaming(rgb, onnx_path=ONNX, sigma=SIG, ep="TRT", device_id=0, fp16=True)[pre:]
core.resize.Point(den, format=vs.RGB24, dither_type="error_diffusion").set_output()
