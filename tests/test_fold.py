import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.mappings import fold


class TestFoldDefaultRange:
    """Test folding with default range [-1, 1]."""

    def test_value_within_range_unchanged(self):
        assert fold(0.0) == pytest.approx(0.0)
        assert fold(-0.5) == pytest.approx(-0.5)
        assert fold(0.5) == pytest.approx(0.5)

    def test_upper_bound_stays_at_upper(self):
        assert fold(1.0) == pytest.approx(1.0)

    def test_lower_bound_stays_at_lower(self):
        assert fold(-1.0) == pytest.approx(-1.0)

    def test_folds_back_above_upper(self):
        assert fold(1.5) == pytest.approx(0.5)

    def test_folds_back_below_lower(self):
        assert fold(-1.5) == pytest.approx(-0.5)


class TestFoldCustomRange:
    """Test folding with custom ranges."""

    def test_known_values(self):
        assert_allclose(
            fold(np.arange(0, 13), 0, 4),
            [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4],
        )

    def test_value_at_y1_stays_at_y1(self):
        assert fold(0.0, 0, 4) == pytest.approx(0.0)

    def test_value_at_y2_stays_at_y2(self):
        assert fold(4.0, 0, 4) == pytest.approx(4.0)

    def test_folds_above_y2(self):
        assert fold(5.0, 0, 4) == pytest.approx(3.0)

    def test_folds_below_y1(self):
        assert fold(-1.0, 0, 4) == pytest.approx(1.0)

    def test_multiple_folds(self):
        # after two full periods the pattern repeats
        assert fold(8.0, 0, 4) == pytest.approx(fold(0.0, 0, 4))
        assert fold(9.0, 0, 4) == pytest.approx(fold(1.0, 0, 4))


class TestFoldBoundsOrder:
    """Test that y1/y2 order is irrelevant due to internal sorting."""

    def test_swapped_bounds_give_same_result(self):
        x = np.linspace(-5, 5, 50)
        assert_allclose(fold(x, 0, 4), fold(x, 4, 0))

    def test_negative_range(self):
        assert fold(-3.0, -4, 0) == pytest.approx(-3.0)
        assert fold(1.0, -4, 0) == pytest.approx(-1.0)


class TestFoldProperties:
    """Test mathematical properties of fold."""

    def test_output_always_in_range(self):
        x = np.linspace(-20, 20, 500)
        result = fold(x, 0, 4)
        assert np.all(result >= 0.0)
        assert np.all(result <= 4.0)

    def test_periodic_with_period_twice_the_range(self):
        # fold has period 2*L where L = y2 - y1
        period = 2 * 4
        x = np.linspace(-5, 5, 50)
        assert_allclose(fold(x, 0, 4), fold(x + period, 0, 4))

    def test_symmetric_around_bounds(self):
        # fold(y2 + d) == fold(y2 - d): mirrors at upper bound
        for d in [0.5, 1.0, 1.5]:
            assert fold(4 + d, 0, 4) == pytest.approx(fold(4 - d, 0, 4))

    def test_array_input_returns_ndarray(self):
        assert isinstance(fold(np.array([0.0, 1.0])), np.ndarray)

    def test_scalar_input_returns_scalar(self):
        assert isinstance(float(fold(0.5)), float)