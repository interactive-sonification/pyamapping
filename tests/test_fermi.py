import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.mappings import fermi, lcurve


class TestFermiDefaultParams:
    """Test fermi function with default parameters (tau=1, mu=0)."""

    def test_zero_input_is_half(self):
        assert fermi(0.0) == pytest.approx(0.5)

    def test_known_values(self):
        assert_allclose(
            fermi(np.array([-1, -0.5, 0, 0.5, 1])),
            [0.26894142, 0.37754067, 0.5, 0.62245933, 0.73105858],
            rtol=1e-6,
        )

    def test_approaches_zero_for_large_negative_x(self):
        assert fermi(-500.0) == pytest.approx(0.0, abs=1e-6)

    def test_approaches_one_for_large_positive_x(self):
        assert fermi(500.0) == pytest.approx(1.0, abs=1e-6)


class TestFermiParameters:
    """Test effect of tau and mu parameters."""

    def test_mu_shifts_midpoint(self):
        # midpoint (output=0.5) should occur at x=mu
        assert fermi(2.0, mu=2.0) == pytest.approx(0.5)
        assert fermi(-3.0, mu=-3.0) == pytest.approx(0.5)

    def test_tau_scales_x_axis(self):
        # fermi(x, tau=2) == fermi(x/2, tau=1)
        assert fermi(2.0, tau=2.0) == pytest.approx(fermi(1.0, tau=1.0))

    def test_negative_tau_flips_curve(self):
        # negative tau mirrors the curve
        assert fermi(1.0, tau=-1.0) == pytest.approx(fermi(-1.0, tau=1.0))

    def test_large_tau_flattens_curve(self):
        # very large tau -> output approaches 0.5 everywhere
        assert fermi(10.0, tau=1e9) == pytest.approx(0.5, abs=1e-4)

    def test_small_tau_sharpens_curve(self):
        # very small tau -> step function behaviour
        assert fermi(0.1, tau=0.001) == pytest.approx(1.0, abs=1e-6)
        assert fermi(-0.1, tau=0.001) == pytest.approx(0.0, abs=1e-6)


class TestFermiProperties:
    """Test mathematical properties of the Fermi function."""

    def test_point_symmetry_around_mu(self):
        # fermi(mu - x) + fermi(mu + x) == 1
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert fermi(-x) + fermi(x) == pytest.approx(1.0)

    def test_point_symmetry_with_nonzero_mu(self):
        mu = 3.0
        for x in [0.5, 1.0, 2.0]:
            assert fermi(mu - x, mu=mu) + fermi(mu + x, mu=mu) == pytest.approx(1.0)

    def test_monotonically_increasing(self):
        x = np.linspace(-5, 5, 200)
        assert np.all(np.diff(fermi(x)) > 0)

    def test_output_bounded_to_unit_interval(self):
        x = np.linspace(-100, 100, 500)
        result = fermi(x)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_array_input_returns_ndarray(self):
        assert isinstance(fermi(np.array([0.0, 1.0])), np.ndarray)


class TestFermiVsLcurve:
    """Fermi is a special case of lcurve (m=0, n=1) — results should match."""

    def test_default_params_match_lcurve(self):
        x = np.linspace(-3, 3, 50)
        assert_allclose(fermi(x), lcurve(x), rtol=1e-9)

    def test_tau_param_matches_lcurve(self):
        x = np.linspace(-3, 3, 50)
        assert_allclose(fermi(x, tau=2.0), lcurve(x, tau=2.0), rtol=1e-9)
