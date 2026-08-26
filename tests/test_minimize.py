import numpy as np
import pytest

import blackbox as bb


class SerialExecutor:
    """A stand-in for multiprocessing.Pool that runs tasks serially.

    Mirrors the context-manager + map interface used by ``minimize`` so the
    optimization can be tested without spawning worker processes.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, func, iterable):
        return list(map(func, iterable))


def quad(x):
    return (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2


def test_minimize_returns_expected_keys():
    domain = [[-5, 5], [-5, 5]]
    result = bb.minimize(
        f=quad, domain=domain, budget=40, batch=4, executor=SerialExecutor
    )
    assert isinstance(result, dict)
    for key in ("best_x", "best_f", "all_xs", "all_fs"):
        assert key in result


def test_minimize_finds_minimum_of_quadratic():
    domain = [[-5, 5], [-5, 5]]
    result = bb.minimize(
        f=quad, domain=domain, budget=60, batch=4, executor=SerialExecutor
    )
    # the true minimum of quad is 0 at x = (1, 1)
    assert result["best_f"] == pytest.approx(0.0, abs=1.0)
    assert np.allclose(result["best_x"], [1.0, 1.0], atol=1.0)


def test_minimize_number_of_evaluations_equals_budget():
    domain = [[-5, 5], [-5, 5]]
    budget, batch = 40, 4
    result = bb.minimize(
        f=quad, domain=domain, budget=budget, batch=batch, executor=SerialExecutor
    )
    assert result["all_xs"].shape[0] == budget
    assert result["all_fs"].shape[0] == budget


def test_minimize_budget_adjustment():
    # budget not divisible by batch gets rounded up
    domain = [[-5, 5], [-5, 5]]
    budget, batch = 21, 4
    result = bb.minimize(
        f=quad, domain=domain, budget=budget, batch=batch, executor=SerialExecutor
    )
    adjusted = budget - budget % batch + batch
    assert result["all_xs"].shape[0] == adjusted


def test_minimize_insufficient_budget_returns_none():
    domain = [[-5, 5], [-5, 5]]
    # with budget=2, batch=2 the number of global samples n=2 <= d=2
    result = bb.minimize(
        f=quad, domain=domain, budget=2, batch=2, executor=SerialExecutor
    )
    assert result == {}


def test_minimize_constant_function():
    domain = [[0, 1], [0, 1]]

    def const(x):
        return 3.0

    result = bb.minimize(
        f=const, domain=domain, budget=20, batch=2, executor=SerialExecutor
    )
    assert np.allclose(result["all_fs"], 3.0)
    assert result["best_f"] == pytest.approx(3.0)


def test_minimize_1d():
    domain = [[-5, 5]]

    def f1d(x):
        return (x[0] - 2.0) ** 2

    result = bb.minimize(
        f=f1d, domain=domain, budget=30, batch=3, executor=SerialExecutor
    )
    assert result["best_f"] == pytest.approx(0.0, abs=1.0)
    assert abs(result["best_x"][0] - 2.0) < 1.0
