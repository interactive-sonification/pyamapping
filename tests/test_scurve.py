import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyamapping.mappings import scurve


class TestScurveEndpoints:
    """Test boundary and clamping behaviour."""

    def test_zero_maps_to_zero(self):
        assert scurve(0.0) == pytest.approx(0.0)

    def test_one_maps_to_one(self):
        assert scurve(1.0) == pytest.approx(1.0)

    def test_below_zero_clamps_to_zero(self):
        assert scurve(-1.0) == pytest.approx(0.0)
        assert scurve(-1e9) == pytest.approx(0.0)

    def test_above_one_clamps_to_one(self):
        assert scurve(2.0) == pytest.approx(1.0)
        assert scurve(1e9) == pytest.approx(1.0)


class TestScurveKnownValues:
    """Test known values along the curve."""

    def test_midpoint_is_half(self):
        assert scurve(0.5) == pytest.approx(0.5)

    def test_known_values(self):
        assert_allclose(
            scurve(np.arange(0, 1, 0.25)),
            [0.0, 0.15625, 0.5, 0.84375],
            rtol=1e-6,
        )

    def test_quarter_point(self):
        assert scurve(0.25) == pytest.approx(0.15625)

    def test_three_quarter_point(self):
        assert scurve(0.75) == pytest.approx(0.84375)


class TestScurveProperties:
    """Test mathematical properties of the S-curve."""

    def test_point_symmetry_around_midpoint(self):
        # scurve is symmetric around (0.5, 0.5): scurve(1-x) == 1 - scurve(x)
        for x in [0.1, 0.25, 0.3, 0.75]:
            assert scurve(1 - x) == pytest.approx(1 - scurve(x))

    def test_monotonically_increasing(self):
        x = np.linspace(0, 1, 200)
        assert np.all(np.diff(scurve(x)) >= 0)

    def test_output_bounded_to_unit_interval(self):
        x = np.linspace(-2, 3, 200)
        result = scurve(x)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_slope_at_midpoint_is_steepest(self):
        # derivative at 0.5 should be greater than at 0.25 and 0.75
        eps = 1e-5
        slope_mid = (scurve(0.5 + eps) - scurve(0.5 - eps)) / (2 * eps)
        slope_quarter = (scurve(0.25 + eps) - scurve(0.25 - eps)) / (2 * eps)
        assert slope_mid > slope_quarter

    def test_scalar_input_returns_float(self):
        assert isinstance(scurve(0.5), float)

    def test_array_input_returns_ndarray(self):
        assert isinstance(scurve(np.array([0.25, 0.75])), np.ndarray)