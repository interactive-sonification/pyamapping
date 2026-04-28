import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.chainable_array import ChainableArray
from pyamapping.mappings import bilin, interp_spline


class TestInterpSplineLinear:
    """Test linear interpolation (k=1, default) between control points."""

    def test_exact_at_control_points(self):
        assert_allclose(interp_spline([-1, 0, 1], [-1, 0, 1]), [-1, 0, 1])

    def test_midpoint_between_control_points(self):
        assert interp_spline(0.5, [-1, 0, 1], [-1, 0, 1]) == pytest.approx(0.5)

    def test_custom_control_points(self):
        assert interp_spline(50, [0, 100], [0, 200]) == pytest.approx(100.0)

    def test_three_point_piecewise(self):
        # two segments with different slopes
        result = interp_spline(
            np.array([0, 25, 50, 75, 100]),
            [0, 50, 100],
            [0, 10, 30],
        )
        assert_allclose(result, [0, 5, 10, 20, 30])


class TestInterpSplineExtrapolation:
    """Test that values outside xc range are extrapolated."""

    def test_extrapolates_below(self):
        result = interp_spline(-1, [0, 1], [0, 10])
        assert result == pytest.approx(-10.0)

    def test_extrapolates_above(self):
        result = interp_spline(2, [0, 1], [0, 10])
        assert result == pytest.approx(20.0)


class TestInterpSplineReturnType:
    """Test that return type depends on input type."""

    def test_chainable_array_input_returns_plain_ndarray(self):
        x = ChainableArray(np.array([0.0, 0.5, 1.0]))
        result = interp_spline(x, [-1, 0, 1], [-1, 0, 1])
        assert isinstance(result, np.ndarray)

    def test_plain_array_input_returns_chainable_array(self):
        x = np.array([0.0, 0.5, 1.0])
        result = interp_spline(x, [-1, 0, 1], [-1, 0, 1])
        assert isinstance(result, ChainableArray)


class TestBilin:
    """Test bilin as a two-segment piecewise linear mapping via interp_spline."""

    def test_known_values(self):
        assert_allclose(
            bilin(np.array([0, 20, 40, 60, 80, 100]), 60, 20, 80, 0, -20, 60),
            [-30.0, -20.0, -10.0, 0.0, 60.0, 120.0],
        )

    def test_xcenter_maps_to_ycenter(self):
        assert bilin(60, 60, 20, 80, 0, -20, 60) == pytest.approx(0.0)

    def test_xmin_maps_to_ymin(self):
        assert bilin(20, 60, 20, 80, 0, -20, 60) == pytest.approx(-20.0)

    def test_xmax_maps_to_ymax(self):
        assert bilin(80, 60, 20, 80, 0, -20, 60) == pytest.approx(60.0)

    def test_extrapolates_below_xmin(self):
        result = bilin(0, 60, 20, 80, 0, -20, 60)
        assert result < -20.0

    def test_extrapolates_above_xmax(self):
        result = bilin(100, 60, 20, 80, 0, -20, 60)
        assert result > 60.0

    def test_symmetric_mapping(self):
        # with symmetric x and y ranges, xcenter should map to ycenter=0
        assert bilin(0.5, 0.5, 0.0, 1.0, 0.0, -1.0, 1.0) == pytest.approx(0.0)

    def test_two_slopes_differ(self):
        # slope left of xcenter: (-20-0)/(20-60) = 0.5
        # slope right of xcenter: (60-0)/(80-60) = 3.0
        slope_left = (
            bilin(40, 60, 20, 80, 0, -20, 60) - bilin(20, 60, 20, 80, 0, -20, 60)
        ) / 20
        slope_right = (
            bilin(80, 60, 20, 80, 0, -20, 60) - bilin(60, 60, 20, 80, 0, -20, 60)
        ) / 20
        assert slope_left != pytest.approx(slope_right)
