import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.mappings import linlin

class TestLinlinEndpoints:
    """Test that x1->y1 and x2->y2 mappings are exact."""

    def test_x1_maps_to_y1(self):
        assert linlin(0.0, 0, 1, 10.0, 20.0) == pytest.approx(10.0)

    def test_x2_maps_to_y2(self):
        assert linlin(1.0, 0, 1, 10.0, 20.0) == pytest.approx(20.0)

    def test_array_input(self):
        origin = np.array([-1, 0, 1, 2, 4])
        target = np.array([-50, 0, 50, 100, 200])
        assert_allclose(linlin(origin, 0, 2, 0, 100), target)


class TestLinlinInterpolation:
    """Test linear mapping between endpoints."""

    def test_midpoint(self):
        assert linlin(0.5, 0, 1, 0, 1) == pytest.approx(0.5)

    def test_midpoint_custom_range(self):
        assert linlin(50, 0, 100, 0, 1) == pytest.approx(0.5)

    def test_negative_range(self):
        assert linlin(-0.5, -1, 0, 0, 100) == pytest.approx(50.0)


class TestLinlinExtrapolation:
    """Default (clip=None) should extrapolate beyond [x1, x2]."""

    def test_extrapolates_below_x1(self):
        assert linlin(-1.0, 0, 1, 0, 100) == pytest.approx(-100.0)

    def test_extrapolates_above_x2(self):
        assert linlin(2.0, 0, 1, 0, 100) == pytest.approx(200.0)


class TestLinlinPolarityInversion:
    """Test y2 < y1 (inverted range)."""

    def test_x1_maps_to_y1_inverted(self):
        assert linlin(0, 0, 1, 1, 0) == pytest.approx(1.0)

    def test_x2_maps_to_y2_inverted(self):
        assert linlin(1, 0, 1, 1, 0) == pytest.approx(0.0)

    def test_midpoint_inverted(self):
        assert linlin(0.5, 0, 1, 1, 0) == pytest.approx(0.5)


class TestLinlinClipping:
    """Test all clip modes, including inverted y1/y2 ranges."""

    def test_clip_max_caps_at_y2(self):
        assert linlin(105, 0, 100, 100, 200, clip="max") == pytest.approx(200.0)

    def test_clip_max_does_not_affect_lower_values(self):
        assert linlin(50, 0, 100, 100, 200, clip="max") == pytest.approx(150.0)

    def test_clip_min_caps_at_y1(self):
        assert linlin(-10, 0, 100, 100, 200, clip="min") == pytest.approx(100.0)

    def test_clip_min_does_not_affect_upper_values(self):
        assert linlin(50, 0, 100, 100, 200, clip="min") == pytest.approx(150.0)

    def test_clip_minmax_caps_both_sides(self):
        assert linlin(3, 5, 105, 100, 1000, clip="minmax") == pytest.approx(100.0)
        assert linlin(105, 0, 100, 100, 1000, clip="minmax") == pytest.approx(1000.0)

    def test_clip_unknown_string_implies_minmax(self):
        assert linlin(-10, 0, 100, 100, 200, clip="both") == pytest.approx(100.0)
        assert linlin(110, 0, 100, 100, 200, clip="both") == pytest.approx(200.0)

    def test_clip_max_inverted_range(self):
        # y1=200, y2=100 — "max" should still cap at the higher bound (200)
        assert linlin(-10, 0, 100, 200, 100, clip="max") == pytest.approx(200.0)

    def test_clip_min_inverted_range(self):
        # y1=200, y2=100 — "min" should still floor at the lower bound (100)
        assert linlin(105, 0, 100, 200, 100, clip="min") == pytest.approx(100.0)

    def test_clip_minmax_array_input(self):
        result = linlin(np.array([-10.0, 50.0, 110.0]), 0, 100, 0, 1, clip="minmax")
        assert_allclose(result, [0.0, 0.5, 1.0])