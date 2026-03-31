import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyamapping.mappings import curvelin, lincurve

class TestCurvelinEndpoints:
    """Test that x1->y1 and x2->y2 mappings are exact."""

    def test_x1_maps_to_y1(self):
        assert curvelin(0.0, 0, 0.5, 0, 10) == pytest.approx(0.0)

    def test_x2_maps_to_y2(self):
        assert curvelin(0.5, 0, 0.5, 0, 10) == pytest.approx(10.0)

    def test_x1_maps_to_y1_custom(self):
        assert curvelin(2.0, 2, 5, 10, 20) == pytest.approx(10.0)

    def test_x2_maps_to_y2_custom(self):
        assert curvelin(5.0, 2, 5, 10, 20) == pytest.approx(20.0)


class TestCurvelinLinearFallback:
    """Test the linear branch triggered when abs(curve) < 0.001."""

    def test_zero_curve_midpoint(self):
        assert curvelin(0.5, 0, 1, 0, 1, curve=0.0) == pytest.approx(0.5)

    def test_tiny_curve_is_linear(self):
        assert curvelin(0.5, 0, 1, 0, 1, curve=0.0009) == pytest.approx(0.5)

    def test_linear_array_input(self):
        x = np.array([0.0, 0.5, 1.0])
        assert_allclose(curvelin(x, 0, 1, 0, 1, curve=0.0), [0.0, 0.5, 1.0])


class TestCurvelinCurvedMapping:
    """Test exponential (non-linear) mapping with meaningful curve values."""

    def test_known_values(self):
        assert_allclose(
            curvelin(np.array([0, 0.1, 0.3, 0.5]), 0, 0.5, 0, 10),
            [0.0, 0.94934752, 3.65734932, 10.0],
            rtol=1e-5,
        )

    def test_negative_curve_endpoints(self):
        assert curvelin(0.0, 0, 1, 0, 1, curve=-2.0) == pytest.approx(0.0)
        assert curvelin(1.0, 0, 1, 0, 1, curve=-2.0) == pytest.approx(1.0)

    def test_positive_curve_endpoints(self):
        assert curvelin(0.0, 0, 1, 0, 1, curve=2.0) == pytest.approx(0.0)
        assert curvelin(1.0, 0, 1, 0, 1, curve=2.0) == pytest.approx(1.0)


class TestCurvelinPolarityInversion:
    """Test y2 < y1 (inverted range)."""

    def test_inverted_range_endpoints(self):
        assert curvelin(0.0, 0, 1, y1=1.0, y2=0.0) == pytest.approx(1.0)
        assert curvelin(1.0, 0, 1, y1=1.0, y2=0.0) == pytest.approx(0.0)

    def test_inverted_range_midpoint_is_between_bounds(self):
        result = curvelin(0.5, 0, 1, y1=1.0, y2=0.0)
        assert 0.0 <= result <= 1.0


class TestCurvelinExtrapolation:
    """Default (clip=None) should extrapolate beyond [x1, x2]."""

    def test_extrapolates_below_x1(self):
        result = curvelin(-1.0, 0, 1, 0, 1, curve=0.0)
        assert result < 0.0

    def test_extrapolates_above_x2(self):
        result = curvelin(2.0, 0, 1, 0, 1, curve=0.0)
        assert result > 1.0


class TestCurvelinClipping:
    """Test all clip modes."""

    def test_clip_max_caps_at_y2(self):
        assert curvelin(2.0, 0, 1, 0, 1, curve=0.0, clip="max") == pytest.approx(1.0)

    def test_clip_max_does_not_affect_lower_values(self):
        assert curvelin(0.5, 0, 1, 0, 1, curve=0.0, clip="max") == pytest.approx(0.5)

    def test_clip_min_caps_at_y1(self):
        assert curvelin(-1.0, 0, 1, 0, 1, curve=0.0, clip="min") == pytest.approx(0.0)

    def test_clip_min_does_not_affect_upper_values(self):
        assert curvelin(0.5, 0, 1, 0, 1, curve=0.0, clip="min") == pytest.approx(0.5)

    def test_clip_minmax_caps_both_sides(self):
        assert curvelin(-1.0, 0, 1, 0, 1, curve=0.0, clip="minmax") == pytest.approx(0.0)
        assert curvelin(2.0, 0, 1, 0, 1, curve=0.0, clip="minmax") == pytest.approx(1.0)

    def test_clip_unknown_string_implies_minmax(self):
        assert curvelin(-1.0, 0, 1, 0, 1, curve=0.0, clip="both") == pytest.approx(0.0)
        assert curvelin(2.0, 0, 1, 0, 1, curve=0.0, clip="both") == pytest.approx(1.0)

    def test_clip_minmax_array_input(self):
        result = curvelin(np.array([-1.0, 0.5, 2.0]), 0, 1, 0, 1, curve=0.0, clip="minmax")
        assert_allclose(result, [0.0, 0.5, 1.0])


class TestCurvelinVsLincurve:
    """curvelin and lincurve should be inverses of each other."""

    def test_curvelin_is_inverse_of_lincurve(self):
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert curvelin(lincurve(x, 0, 1, 0, 1), 0, 1, 0, 1) == pytest.approx(x, rel=1e-6)

    def test_lincurve_is_inverse_of_curvelin(self):
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert lincurve(curvelin(x, 0, 1, 0, 1), 0, 1, 0, 1) == pytest.approx(x, rel=1e-6)