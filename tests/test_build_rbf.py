import numpy as np
import pytest

from blackbox.blackbox import build_rbf


def make_points(n, d, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.random((n, d))
    fs = rng.random(n)
    points = np.column_stack([xs, fs])
    return points


def test_rbf_returns_callable():
    points = make_points(8, 2)
    fit = build_rbf(points)
    assert callable(fit)


def test_rbf_accepts_array():
    points = make_points(8, 2)
    fit = build_rbf(points)
    x = np.array([0.3, 0.7])
    val = fit(x)
    assert np.isscalar(val) or np.ndim(val) == 0


def test_rbf_interpolates_training_points():
    # RBF fit with polynomial term interpolates the training data exactly
    # (up to numerical precision) when the system is nonsingular.
    points = make_points(12, 3, seed=1)
    fit = build_rbf(points)
    for i in range(points.shape[0]):
        pred = fit(points[i, 0:-1])
        assert pred == pytest.approx(points[i, -1], rel=1e-6, abs=1e-9)


def test_rbf_handle_list_input():
    points = make_points(6, 1, seed=2)
    fit = build_rbf(points)
    val = fit([0.5])
    assert np.isscalar(val) or np.ndim(val) == 0


def test_rbf_with_polynomial_only_dim():
    # 1D case should still interpolate
    points = make_points(7, 1, seed=3)
    fit = build_rbf(points)
    for i in range(points.shape[0]):
        assert fit(points[i, 0:-1]) == pytest.approx(points[i, -1], rel=1e-6, abs=1e-9)
