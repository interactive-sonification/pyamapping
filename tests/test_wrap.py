import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.mappings import wrap


class TestWrapDefaultRange:
    """Test wrapping with default range [-1, 1]."""

    def test_value_within_range_unchanged(self):
        assert wrap(0.0) == pytest.approx(0.0)
        assert wrap(-0.5) == pytest.approx(-0.5)

    def test_upper_bound_wraps_to_lower(self):
        assert wrap(1.0) == pytest.approx(-1.0)

    def test_wraps_above_range(self):
        assert wrap(1.5) == pytest.approx(-0.5)

    def test_wraps_below_range(self):
        assert wrap(-1.5) == pytest.approx(0.5)


class TestWrapCustomRange:
    """Test wrapping with custom ranges."""

    def test_known_values(self):
        assert_allclose(
            wrap(np.arange(-3, 5), 0, 3),
            [0, 1, 2, 0, 1, 2, 0, 1],
        )

    def test_value_at_y1_is_y1(self):
        assert wrap(0.0, 0, 3) == pytest.approx(0.0)

    def test_value_at_y2_wraps_to_y1(self):
        assert wrap(3.0, 0, 3) == pytest.approx(0.0)

    def test_multiple_periods_above(self):
        assert wrap(7.0, 0, 3) == pytest.approx(1.0)

    def test_multiple_periods_below(self):
        assert wrap(-5.0, 0, 3) == pytest.approx(1.0)

    def test_float_range(self):
        assert wrap(1.75, 0.0, 0.5) == pytest.approx(0.25)


class TestWrapProperties:
    """Test mathematical properties of wrap."""

    def test_output_always_in_range(self):
        x = np.linspace(-10, 10, 200)
        result = wrap(x, 0, 3)
        assert np.all(result >= 0.0)
        assert np.all(result < 3.0)

    def test_periodic_with_period_equal_to_range(self):
        # wrap(x) == wrap(x + (y2-y1))
        period = 3.0
        x = np.linspace(-5, 5, 50)
        assert_allclose(wrap(x, 0, period), wrap(x + period, 0, period))

    def test_array_input_returns_ndarray(self):
        assert isinstance(wrap(np.array([0.0, 1.0])), np.ndarray)

    def test_scalar_input_returns_scalar(self):
        assert isinstance(float(wrap(0.5)), float)
