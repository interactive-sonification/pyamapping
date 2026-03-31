import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.mappings import linpoly


class TestLinpolyEndpoints:
    """Test that -xmax->y1 and xmax->y2 mappings are exact."""

    def test_xmax_maps_to_y2(self):
        assert linpoly(1.0, xmax=1.0, y1=-1.0, y2=1.0) == pytest.approx(1.0)

    def test_negative_xmax_maps_to_y1(self):
        assert linpoly(-1.0, xmax=1.0, y1=-1.0, y2=1.0) == pytest.approx(-1.0)

    def test_zero_maps_to_midpoint(self):
        # (1 + 0) / 2 = 0.5 -> y1 + 0.5*(y2-y1) = midpoint
        assert linpoly(0.0, xmax=1.0, y1=0.0, y2=1.0) == pytest.approx(0.5)

    def test_custom_range_endpoints(self):
        assert linpoly(2.5, xmax=2.5, y1=100, y2=500) == pytest.approx(500.0)
        assert linpoly(-2.5, xmax=2.5, y1=100, y2=500) == pytest.approx(100.0)


class TestLinpolyCurve:
    """Test effect of curve parameter on polynomial order."""

    def test_known_values_curve_1(self):
        assert_allclose(
            linpoly(np.arange(-2, 3), 2.5, 100, 500, curve=1),
            [172.0, 268.0, 300.0, 332.0, 428.0],
            rtol=1e-6,
        )

    def test_curve_zero_is_linear(self):
        # curve=0 -> order=1, pure linear mapping
        x = np.array([-1.0, 0.0, 1.0])
        assert_allclose(linpoly(x, xmax=1.0, y1=0.0, y2=1.0, curve=0), [0.0, 0.5, 1.0])

    def test_positive_curve_increases_order(self):
        # curve=1 -> order=2 (quadratic), curve=2 -> order=3 (cubic)
        # higher order -> more compression near zero
        mid_curve1 = linpoly(0.5, xmax=1.0, y1=0.0, y2=1.0, curve=1)
        mid_curve2 = linpoly(0.5, xmax=1.0, y1=0.0, y2=1.0, curve=2)
        assert mid_curve2 < mid_curve1  # more curved -> closer to midpoint

    def test_negative_curve_order(self):
        # curve=-1 -> order = 1/(1-(-1)) = 0.5 (square root)
        order = 1 / (1 - (-1))
        expected = 0 + 1 * (1 + (0.5 / 1.0) ** order) / 2
        assert linpoly(0.5, xmax=1.0, y1=0.0, y2=1.0, curve=-1) == pytest.approx(expected)


class TestLinpolyProperties:
    """Test mathematical properties of linpoly."""

    def test_odd_symmetry_around_midpoint(self):
        # linpoly(-x) and linpoly(x) should be symmetric around the output midpoint
        for x in [0.25, 0.5, 0.75, 1.0]:
            assert linpoly(x, y1=0.0, y2=1.0) + linpoly(-x, y1=0.0, y2=1.0) == pytest.approx(1.0)

    def test_monotonically_increasing(self):
        x = np.linspace(-1, 1, 200)
        assert np.all(np.diff(linpoly(x)) > 0)

    def test_output_bounded_within_range_for_input_within_xmax(self):
        x = np.linspace(-1, 1, 100)
        result = linpoly(x, xmax=1.0, y1=0.0, y2=1.0)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestLinpolyClipping:
    """Test all clip modes."""

    def test_clip_max_caps_at_y2(self):
        assert linpoly(2.0, xmax=1.0, y1=0.0, y2=1.0, clip="max") == pytest.approx(1.0)

    def test_clip_min_caps_at_y1(self):
        assert linpoly(-2.0, xmax=1.0, y1=0.0, y2=1.0, clip="min") == pytest.approx(0.0)

    def test_clip_minmax_caps_both_sides(self):
        assert linpoly(2.0, xmax=1.0, y1=0.0, y2=1.0, clip="minmax") == pytest.approx(1.0)
        assert linpoly(-2.0, xmax=1.0, y1=0.0, y2=1.0, clip="minmax") == pytest.approx(0.0)

    def test_clip_unknown_string_implies_minmax(self):
        assert linpoly(2.0, xmax=1.0, y1=0.0, y2=1.0, clip="both") == pytest.approx(1.0)
        assert linpoly(-2.0, xmax=1.0, y1=0.0, y2=1.0, clip="both") == pytest.approx(0.0)

    def test_clip_does_not_affect_values_within_range(self):
        assert linpoly(0.5, xmax=1.0, y1=0.0, y2=1.0, clip="minmax") == pytest.approx(
            linpoly(0.5, xmax=1.0, y1=0.0, y2=1.0)
        )