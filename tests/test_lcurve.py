import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyamapping.mappings import lcurve


class TestLcurveDefaultParams:
    """Test lcurve with default parameters (Fermi function: m=0, n=1, tau=1)."""

    def test_zero_input_is_half(self):
        # (1 + 0) / (1 + 1) = 0.5
        assert lcurve(0.0) == pytest.approx(0.5)

    def test_known_values(self):
        assert_allclose(
            lcurve(np.array([-1, -0.5, 0, 0.5, 1])),
            [0.26894142, 0.37754067, 0.5, 0.62245933, 0.73105858],
            rtol=1e-6,
        )

    def test_approaches_zero_for_large_negative_x(self):
        assert lcurve(-500.0) == pytest.approx(0.0, abs=1e-6)

    def test_approaches_one_for_large_positive_x(self):
        assert lcurve(500.0) == pytest.approx(1.0, abs=1e-6)


class TestLcurveParameters:
    """Test effect of m, n, and tau parameters."""

    def test_m_equals_n_is_constant_one(self):
        # (1 + c*exp) / (1 + c*exp) = 1 for any x when m == n
        for x in [-1.0, 0.0, 1.0, 10.0]:
            assert lcurve(x, m=2.0, n=2.0) == pytest.approx(1.0)

    def test_n_zero_is_constant_one(self):
        # (1 + m*exp) / (1 + 0) = 1 + m*exp, but with m=0: always 1
        assert lcurve(0.0, m=0.0, n=0.0) == pytest.approx(1.0)

    def test_tau_scales_x_axis(self):
        # lcurve(x, tau=2) == lcurve(x/2, tau=1)
        assert lcurve(2.0, tau=2.0) == pytest.approx(lcurve(1.0, tau=1.0))

    def test_negative_tau_flips_curve(self):
        # negative tau mirrors the curve: lcurve(x, tau=-1) == lcurve(-x, tau=1)
        assert lcurve(1.0, tau=-1.0) == pytest.approx(lcurve(-1.0, tau=1.0))

    def test_large_tau_flattens_curve(self):
        # very large tau -> exp(-x/tau) -> 1, output approaches (1+m)/(1+n)
        assert lcurve(10.0, tau=1e9) == pytest.approx(lcurve(0.0, tau=1e9), rel=1e-4)


class TestLcurveProperties:
    """Test mathematical properties of the L-curve."""

    def test_monotonically_increasing_default(self):
        x = np.linspace(-5, 5, 200)
        assert np.all(np.diff(lcurve(x)) > 0)

    def test_point_symmetry_around_zero_default(self):
        # Fermi function: lcurve(-x) + lcurve(x) == 1
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert lcurve(-x) + lcurve(x) == pytest.approx(1.0)

    def test_output_bounded_to_unit_interval_default(self):
        x = np.linspace(-100, 100, 500)
        result = lcurve(x)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_scalar_input_returns_scalar(self):
        assert isinstance(float(lcurve(0.5)), float)

    def test_array_input_returns_ndarray(self):
        assert isinstance(lcurve(np.array([0.0, 1.0])), np.ndarray)