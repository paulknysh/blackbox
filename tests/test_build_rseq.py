import numpy as np

from blackbox.blackbox import build_rseq


def test_rseq_shape():
    n, d = 13, 4
    rseq = build_rseq(n, d)
    assert rseq.shape == (n, d)


def test_rseq_in_unit_cube():
    rseq = build_rseq(200, 5)
    assert np.all(rseq >= 0.0)
    assert np.all(rseq < 1.0)


def test_rseq_deterministic():
    a = build_rseq(50, 3)
    b = build_rseq(50, 3)
    assert np.allclose(a, b)


def test_rseq_single_dim():
    rseq = build_rseq(10, 1)
    assert rseq.shape == (10, 1)


def test_rseq_zero_points():
    # edge case: no points requested should not crash and returns an array
    # (note: current implementation yields shape (0,) rather than (0, d))
    rseq = build_rseq(0, 3)
    assert isinstance(rseq, np.ndarray)
    assert len(rseq) == 0
