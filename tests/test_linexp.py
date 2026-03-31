import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyamapping.mappings import linexp

class TestLinexpEndpoints:
    """Test that x1->y1 and x2->y2 mappings are exact."""

    def test_x1_maps_to_y1(self):
        assert linexp(1.0, 1, 8, 2, 256) == pytest.approx(2.0)

    def test_x2_maps_to_y2(self):
        assert linexp(8.0, 1, 8, 2, 256) == pytest.approx(256.0)

    def test_x1_maps_to_y1_custom(self):
        assert linexp(0.0, 0, 4, 100, 300) == pytest.approx(100.0)

    def test_x2_maps_to_y2_custom(self):
        assert linexp(4.0, 0, 4, 100, 300) == pytest.approx(300.0)


class TestLinexpMapping:
    """Test exponential mapping between endpoints."""

    def test_known_value(self):
        assert linexp(5, 1, 8, 2, 256) == pytest.approx(32.0)

    def test_midpoint_is_geometric_mean(self):
        # exponential mapping: midpoint maps to sqrt(y1*y2)
        assert linexp(0.5, 0, 1, 1, 100) == pytest.approx(10.0, rel=1e-6)

    def test_inverted_range(self):
        assert linexp(5, 1, 8, 256, 2) == pytest.approx(16.0, rel=1e-5)

    def test_array_input(self):
        result = linexp(np.array([1.0, 8.0]), 1, 8, 2, 256)
        assert_allclose(result, [2.0, 256.0])


class TestLinexpExtrapolation:
    """Default (clip=None) should extrapolate beyond [x1, x2]."""

    def test_extrapolates_above_x2(self):
        result = linexp(9.0, 1, 8, 2, 256)
        assert result > 256.0

    def test_extrapolates_below_x1(self):
        result = linexp(0.0, 1, 8, 2, 256)
        assert result < 2.0


class TestLinexpClipping:
    """Test all clip modes."""

    def test_clip_max_caps_at_y2(self):
        assert linexp(7, 0, 5, 100, 300, clip="max") == pytest.approx(300.0)

    def test_clip_max_does_not_affect_lower_values(self):
        assert linexp(2.5, 0, 5, 100, 300, clip="max") == pytest.approx(
            linexp(2.5, 0, 5, 100, 300)
        )

    def test_clip_min_caps_at_y1(self):
        assert linexp(1, 2, 5, 100, 300, clip="min") == pytest.approx(100.0)

    def test_clip_min_does_not_affect_upper_values(self):
        assert linexp(3.5, 2, 5, 100, 300, clip="min") == pytest.approx(
            linexp(3.5, 2, 5, 100, 300)
        )

    def test_clip_minmax_caps_both_sides(self):
        assert linexp(1, 2, 5, 100, 300, clip="minmax") == pytest.approx(100.0)
        assert linexp(6, 2, 5, 100, 300, clip="minmax") == pytest.approx(300.0)

    def test_clip_unknown_string_implies_minmax(self):
        assert linexp(1, 2, 5, 100, 300, clip="both") == pytest.approx(100.0)
        assert linexp(6, 2, 5, 100, 300, clip="both") == pytest.approx(300.0)

    def test_clip_inverted_range(self):
        assert linexp(5, 1, 8, 256, 2, clip="minmax") == pytest.approx(16.0, rel=1e-5)

    def test_clip_minmax_array_input(self):
        result = linexp(np.array([0.0, 4.0, 9.0]), 1, 8, 2, 256, clip="minmax")
        assert_allclose(result, [2.0, linexp(4.0, 1, 8, 2, 256), 256.0])


class TestLinexpProperties:
    """Test mathematical properties of linexp."""

    def test_output_is_always_positive_for_positive_bounds(self):
        x = np.linspace(0, 10, 100)
        assert np.all(linexp(x, 1, 8, 2, 256) > 0)

    def test_monotonically_increasing(self):
        x = np.linspace(1, 8, 100)
        assert np.all(np.diff(linexp(x, 1, 8, 2, 256)) > 0)

    def test_ratio_of_outputs_depends_only_on_ratio_of_inputs(self):
        # exponential mapping: equal steps in x -> equal ratios in y
        y1 = linexp(1.0, 0, 3, 1, 8)
        y2 = linexp(2.0, 0, 3, 1, 8)
        y3 = linexp(3.0, 0, 3, 1, 8)
        assert y2 / y1 == pytest.approx(y3 / y2, rel=1e-6)