from tools.optsig_pref.make_windows import window_for
def test_window_for_basic():
    assert window_for(9109) == (3644, 180)   # round(0.40*9109)=3644
def test_window_for_short_excluded():
    assert window_for(250) is None           # N<300 -> excluded
