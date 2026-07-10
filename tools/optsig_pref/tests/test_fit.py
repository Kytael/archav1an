# tools/optsig_pref/tests/test_fit.py
import numpy as np
from tools.optsig_pref.fit import best_constant, interval_mae, fit_fold, predict, GRID

def test_best_constant_picks_min_mae_rung():
    y = np.array([0.04, 0.05, 0.05, 0.06])
    assert best_constant(y) == 0.05

def test_interval_mae_zero_inside():
    assert interval_mae(0.055, 0.05, 0.06) == 0.0

def test_interval_mae_distance_outside():
    assert abs(interval_mae(0.07, 0.05, 0.06) - 0.01) < 1e-9

def test_fit_fold_signs_enforced():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (30, 2))
    # adversarial target rewards POSITIVE brightness weight; constraint must clip it to 0
    y = 0.03 * X[:, 0] + 0.02 * X[:, 1] + 0.02
    m = fit_fold(X, y)
    assert m["w"][0] <= 1e-12 and m["w"][1] >= -1e-12

def test_predict_clamped_to_grid():
    m = dict(w=np.array([0.0, 0.0, 99.0]), mu=np.zeros(2), sd=np.ones(2))
    assert predict(m, [0.5, 0.5]) == GRID[-1]
