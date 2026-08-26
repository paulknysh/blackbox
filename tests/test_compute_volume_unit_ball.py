import numpy as np
import pytest

from blackbox.blackbox import compute_volume_unit_ball


def test_volume_unit_ball_dim1():
    # volume of 1D unit ball (segment of length 2) is 2
    assert compute_volume_unit_ball(1) == pytest.approx(2.0)


def test_volume_unit_ball_dim2():
    # area of unit disk is pi
    assert compute_volume_unit_ball(2) == pytest.approx(np.pi)


def test_volume_unit_ball_dim3():
    # volume of unit sphere is 4/3 * pi
    assert compute_volume_unit_ball(3) == pytest.approx(4.0 / 3.0 * np.pi)


def test_volume_unit_ball_dim4():
    # volume of 4D unit ball is pi^2 / 2
    assert compute_volume_unit_ball(4) == pytest.approx(np.pi**2 / 2.0)


def test_volume_unit_ball_known_values():
    # spot-check a few more dimensions against the closed form
    expected = {
        5: 8.0 * np.pi**2 / 15.0,
        6: np.pi**3 / 6.0,
        7: 16.0 * np.pi**3 / 105.0,
        8: np.pi**4 / 24.0,
    }
    for d, val in expected.items():
        assert compute_volume_unit_ball(d) == pytest.approx(val)


def test_volume_unit_ball_is_positive_and_decreasing_for_high_d():
    # unit-ball volume peaks around d=5 and then decreases
    low = compute_volume_unit_ball(5)
    high = compute_volume_unit_ball(20)
    assert low > 0
    assert high > 0
    assert high < low
