import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.mappings import clip

class TestClipScalar:
    """Test clipping of scalar float/int values."""

    def test_value_below_minimum_is_clipped(self):
        assert clip(2, 3, 5) == 3

    def test_value_above_maximum_is_clipped(self):
        assert clip(6, 3, 5) == 5

    def test_value_within_range_is_unchanged(self):
        assert clip(4, 3, 5) == 4

    def test_value_equal_to_minimum(self):
        assert clip(3, 3, 5) == 3

    def test_value_equal_to_maximum(self):
        assert clip(5, 3, 5) == 5

    def test_negative_range(self):
        assert clip(-10, -5, -1) == -5
        assert clip(-3, -5, -1) == -3
        assert clip(0, -5, -1) == -1

    def test_float_values(self):
        assert clip(1.5, 2.0, 4.0) == pytest.approx(2.0)
        assert clip(3.5, 2.0, 4.0) == pytest.approx(3.5)
        assert clip(5.0, 2.0, 4.0) == pytest.approx(4.0)


class TestClipScalarDefaultBounds:
    """Test that default bounds (-inf, +inf) leave values unchanged."""

    def test_no_minimum_large_negative(self):
        assert clip(-1e9) == -1e9

    def test_no_maximum_large_positive(self):
        assert clip(1e9) == 1e9

    def test_only_minimum_clips_below(self):
        assert clip(-5, minimum=0) == 0
        assert clip(5, minimum=0) == 5

    def test_only_maximum_clips_above(self):
        assert clip(5, maximum=3) == 3
        assert clip(1, maximum=3) == 1


class TestClipArray:
    """Test clipping of numpy array inputs."""

    def test_array_all_clipped(self):
        assert_allclose(clip(np.array([0, 1, 6, 7]), 2, 4), [2, 2, 4, 4])

    def test_array_none_clipped(self):
        a = np.array([2.0, 3.0, 4.0])
        assert_allclose(clip(a, 0, 10), a)

    def test_array_mixed(self):
        a1 = np.arange(6)
        assert_allclose(clip(a1, 2, 4), [2, 2, 2, 3, 4, 4])

    def test_array_default_bounds(self):
        a = np.array([-1e9, 0.0, 1e9])
        assert_allclose(clip(a), a)

    def test_array_only_minimum(self):
        assert_allclose(clip(np.array([0, 2, 5]), minimum=3), [3, 3, 5])

    def test_array_only_maximum(self):
        assert_allclose(clip(np.array([0, 2, 5]), maximum=3), [0, 2, 3])


class TestClipScalarArrayConsistency:
    """Scalar and array code paths should return identical results."""

    def test_same_result_for_value_below(self):
        assert clip(1, 3, 5) == clip(np.array([1]), 3, 5)[0]

    def test_same_result_for_value_within(self):
        assert clip(4, 3, 5) == clip(np.array([4]), 3, 5)[0]

    def test_same_result_for_value_above(self):
        assert clip(7, 3, 5) == clip(np.array([7]), 3, 5)[0]