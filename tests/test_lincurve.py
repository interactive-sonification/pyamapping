import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.mappings import lincurve


class TestLincurveEndpoints:
    """Test that x1->y1 and x2->y2 mappings are exact."""

    def test_x1_maps_to_y1_default_params(self):
        assert lincurve(0.0, 0, 1) == pytest.approx(-1.0)

    def test_x2_maps_to_y2_default_params(self):
        assert lincurve(1.0, 0, 1) == pytest.approx(1.0)

    def test_x1_maps_to_y1_custom(self):
        assert lincurve(2.0, 2, 5, y1=10.0, y2=20.0) == pytest.approx(10.0)

    def test_x2_maps_to_y2_custom(self):
        assert lincurve(5.0, 2, 5, y1=10.0, y2=20.0) == pytest.approx(20.0)


class TestLincurveLinearFallback:
    """Test the linear branch triggered when abs(curve) < 0.001."""

    def test_zero_curve_midpoint(self):
        # With curve=0, should be linear: midpoint maps to midpoint
        result = lincurve(0.5, 0, 1, 0, 1, curve=0.0)
        assert result == pytest.approx(0.5)

    def test_tiny_curve_is_linear(self):
        # curve=0.0009 is below threshold, expect linear behaviour
        result = lincurve(0.5, 0, 1, 0, 1, curve=0.0009)
        assert result == pytest.approx(0.5)

    def test_linear_array_input(self):
        x = np.array([0.0, 0.5, 1.0])
        result = lincurve(x, 0, 1, 0, 1, curve=0.0)
        assert_allclose(result, [0.0, 0.5, 1.0])


class TestLincurveCurvedMapping:
    """Test exponential (non-linear) mapping with meaningful curve values."""

    def test_array_input_known_values(self):
        result = lincurve(np.array([0.0, 0.1, 0.4, 0.7, 1.0]), 0, 1, 0, 0.4)
        assert_allclose(
            result,
            [0.0, 0.08385643, 0.25474431, 0.34852956, 0.4],
            rtol=1e-5,
        )

    def test_negative_curve(self):
        # Negative curve bends the other way; endpoints must still hold
        assert lincurve(0.0, 0, 1, 0, 1, curve=-2.0) == pytest.approx(0.0)
        assert lincurve(1.0, 0, 1, 0, 1, curve=-2.0) == pytest.approx(1.0)

    def test_positive_curve(self):
        assert lincurve(0.0, 0, 1, 0, 1, curve=2.0) == pytest.approx(0.0)
        assert lincurve(1.0, 0, 1, 0, 1, curve=2.0) == pytest.approx(1.0)

    def test_positive_vs_negative_curve_symmetry(self):
        # For x in [0,1]->[0,1], curve and -curve should be symmetric around 0.5
        x = 0.3
        pos = lincurve(x, 0, 1, 0, 1, curve=2.0)
        neg = lincurve(1 - x, 0, 1, 0, 1, curve=-2.0)
        assert pos == pytest.approx(1 - neg, rel=1e-5)


class TestLincurvePolarityInversion:
    """Test y2 < y1 (inverted range)."""

    def test_inverted_range_endpoints(self):
        assert lincurve(0.0, 0, 1, y1=1.0, y2=0.0) == pytest.approx(1.0)
        assert lincurve(1.0, 0, 1, y1=1.0, y2=0.0) == pytest.approx(0.0)

    def test_inverted_range_midpoint_is_between_bounds(self):
        result = lincurve(0.5, 0, 1, y1=1.0, y2=0.0)
        assert 0.0 <= result <= 1.0


class TestLincurveExtrapolation:
    """Default (clip=None) should extrapolate beyond [x1, x2]."""

    def test_extrapolates_below_x1(self):
        # x < x1 with linear curve should give z < y1
        result = lincurve(-1.0, 0, 1, 0, 1, curve=0.0)
        assert result == pytest.approx(-1.0)

    def test_extrapolates_above_x2(self):
        result = lincurve(2.0, 0, 1, 0, 1, curve=0.0)
        assert result == pytest.approx(2.0)


class TestLincurveClipping:
    """Test all clip modes."""

    def test_clip_max_caps_at_y2(self):
        # x > x2 would extrapolate above y2=1; clip="max" should cap it
        result = lincurve(2.0, 0, 1, 0, 1, curve=0.0, clip="max")
        assert result == pytest.approx(1.0)

    def test_clip_max_does_not_affect_lower_values(self):
        result = lincurve(0.5, 0, 1, 0, 1, curve=0.0, clip="max")
        assert result == pytest.approx(0.5)

    def test_clip_min_caps_at_y1(self):
        result = lincurve(-1.0, 0, 1, 0, 1, curve=0.0, clip="min")
        assert result == pytest.approx(0.0)

    def test_clip_min_does_not_affect_upper_values(self):
        result = lincurve(0.5, 0, 1, 0, 1, curve=0.0, clip="min")
        assert result == pytest.approx(0.5)

    def test_clip_minmax_caps_both_sides(self):
        assert lincurve(-1.0, 0, 1, 0, 1, curve=0.0, clip="minmax") == pytest.approx(0.0)
        assert lincurve(2.0, 0, 1, 0, 1, curve=0.0, clip="minmax") == pytest.approx(1.0)

    def test_clip_unknown_string_implies_minmax(self):
        # Any string other than "min"/"max" should clip both sides
        assert lincurve(-1.0, 0, 1, 0, 1, curve=0.0, clip="both") == pytest.approx(0.0)
        assert lincurve(2.0, 0, 1, 0, 1, curve=0.0, clip="both") == pytest.approx(1.0)

    def test_clip_minmax_array_input(self):
        x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        result = lincurve(x, 0, 1, 0, 1, curve=0.0, clip="minmax")
        assert_allclose(result, [0.0, 0.0, 0.5, 1.0, 1.0])