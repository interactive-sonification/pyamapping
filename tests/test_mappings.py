import numpy as np
import pytest

from pyamapping.mappings import (
    bilin,
    fold,
    gain,
    linpoly,
    norm_peak,
    norm_rms,
    remove_dc,
)


def test_bilin():
    pytest.approx(
        bilin(np.array([0, 20, 40, 60, 80, 100]), 60, 20, 80, 0, -20, 60),
        np.array([-30.0, -20.0, -10.0, 0.0, 60.0, 120.0]),
    )


def test_fold():
    pytest.approx(
        fold(np.arange(0, 13), 0, 4),
        np.array([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4]),
    )


def test_linpoly():
    pytest.approx(
        linpoly(np.arange(-2, 3), 2.5, 100, 500, curve=1),
        np.array([172.0, 268.0, 300.0, 332.0, 428.0]),
    )


def test_norm_peak():
    pytest.approx(
        np.max(norm_peak(np.random.rand(10), 5)),
        5,
    )


def test_norm_rms():
    pytest.approx(
        norm_rms(np.array([1, 0, 0, -1]), 1),
        np.array([1.41421356, 0.0, 0.0, -1.41421356]),
    )


def test_remove_dc():
    pytest.approx(
        remove_dc(np.array([1, 2, 3, 4])),
        np.array([-1.5, -0.5, 0.5, 1.5]),
    )


def test_gain():
    pytest.approx(
        gain(np.array([1, 2, 3, 4]), amp=2),
        np.array([2, 4, 6, 8]),
    )
