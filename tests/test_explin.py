import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.mappings import explin, linexp


class TestExplinEndpoints:
    """Test that x1->y1 and x2->y2 mappings are exact."""

    def test_x1_maps_to_y1(self):
        assert explin(220.0, 220, 440, 0, 12) == pytest.approx(0.0)

    def test_x2_maps_to_y2(self):
        assert explin(440.0, 220, 440, 0, 12) == pytest.approx(12.0)

    def test_x1_maps_to_y1_custom(self):
        assert explin(0.001, 0.001, 1.0, -30, 0) == pytest.approx(-30.0)

    def test_x2_maps_to_y2_custom(self):
        assert explin(1.0, 0.001, 1.0, -30, 0) == pytest.approx(0.0)


class TestExplinMapping:
    """Test logarithmic mapping between endpoints."""

    def test_known_value(self):
        f = 220 * 2 ** (-5 / 12)
        assert explin(f, 220, 440, 0, 12) == pytest.approx(-5.0)

    def test_geometric_midpoint_maps_to_linear_midpoint(self):
        # log mapping: geometric mean of x1,x2 maps to arithmetic mean of y1,y2
        geo_mid = np.sqrt(1.0 * 100.0)
        assert explin(geo_mid, 1.0, 100.0, 0.0, 1.0) == pytest.approx(0.5, rel=1e-6)

    def test_inverted_y_range(self):
        assert explin(0.01, 0.001, 1.0, 0, -30) == pytest.approx(
            -explin(0.01, 0.001, 1.0, 0, 30), rel=1e-6
        )

    def test_array_input(self):
        result = explin(np.array([220.0, 440.0]), 220, 440, 0, 12)
        assert_allclose(result, [0.0, 12.0])


class TestExplinExtrapolation:
    """Default (clip=None) should extrapolate beyond [x1, x2]."""

    def test_extrapolates_above_x2(self):
        result = explin(2.0, 0.001, 1.0, 0, 1)
        assert result > 1.0

    def test_extrapolates_below_x1(self):
        result = explin(0.0001, 0.001, 1.0, 0, 1)
        assert result < 0.0


class TestExplinClipping:
    """Test all clip modes."""

    def test_clip_max_caps_at_y2(self):
        assert explin(2, 0.001, 1.0, 0, 1, clip="max") == pytest.approx(1.0)

    def test_clip_max_does_not_affect_lower_values(self):
        assert explin(0.01, 0.001, 1.0, -30, 0, clip="max") == pytest.approx(
            explin(0.01, 0.001, 1.0, -30, 0), rel=1e-6
        )

    def test_clip_min_caps_at_y1(self):
        assert explin(0.0001, 0.001, 1.0, 0, 1, clip="min") == pytest.approx(0.0)

    def test_clip_min_does_not_affect_upper_values(self):
        assert explin(0.1, 0.001, 1.0, 0, 1, clip="min") == pytest.approx(
            explin(0.1, 0.001, 1.0, 0, 1), rel=1e-6
        )

    def test_clip_minmax_caps_both_sides(self):
        assert explin(0.0001, 0.001, 1.0, 0, 1, clip="minmax") == pytest.approx(0.0)
        assert explin(2.0, 0.001, 1.0, 0, 1, clip="minmax") == pytest.approx(1.0)

    def test_clip_unknown_string_implies_minmax(self):
        assert explin(0.0001, 0.001, 1.0, 0, 1, clip="both") == pytest.approx(0.0)
        assert explin(2.0, 0.001, 1.0, 0, 1, clip="both") == pytest.approx(1.0)

    def test_clip_inverted_y_range(self):
        assert explin(0.01, 0.001, 1.0, 0, -30, clip="minmax") == pytest.approx(
            -10.0, rel=1e-5
        )

    def test_clip_minmax_array_input(self):
        result = explin(np.array([0.0001, 0.01, 2.0]), 0.001, 1.0, 0, 1, clip="minmax")
        assert_allclose(result, [0.0, explin(0.01, 0.001, 1.0, 0, 1), 1.0])


class TestExplinProperties:
    """Test mathematical properties of explin."""

    def test_monotonically_increasing(self):
        x = np.geomspace(0.001, 1.0, 100)
        assert np.all(np.diff(explin(x, 0.001, 1.0, 0, 1)) > 0)

    def test_equal_ratios_in_x_give_equal_steps_in_y(self):
        # log mapping: equal ratios in x -> equal differences in y
        y1 = explin(1.0, 1, 100, 0, 1)
        y2 = explin(10.0, 1, 100, 0, 1)
        y3 = explin(100.0, 1, 100, 0, 1)
        assert y2 - y1 == pytest.approx(y3 - y2, rel=1e-6)

    def test_explin_is_inverse_of_linexp(self):
        # explin and linexp should be inverses of each other
        for x in [220.0, 311.0, 440.0]:
            assert explin(
                linexp(x, 220, 440, 220, 440), 220, 440, 220, 440
            ) == pytest.approx(x, rel=1e-9)
