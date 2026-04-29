import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.mappings import norm_peak


class TestNormPeakDefaultPeak:
    """Test peak normalization with default peak=1.0."""

    def test_max_abs_is_one(self):
        x = np.array([1.0, -2.0, 3.0, -0.5])
        result = norm_peak(x)
        assert np.max(np.abs(result)) == pytest.approx(1.0)

    def test_positive_array_max_is_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.max(norm_peak(x)) == pytest.approx(1.0)

    def test_negative_array_min_is_minus_one(self):
        x = np.array([-4.0, -3.0, -2.0, -1.0])
        assert np.min(norm_peak(x)) == pytest.approx(-1.0)

    def test_shape_preserved(self):
        x = np.random.rand(20)
        assert norm_peak(x).shape == x.shape


class TestNormPeakCustomPeak:
    """Test peak normalization with custom peak values."""

    def test_custom_peak_random_array(self):
        result = norm_peak(np.random.rand(10), peak=5.0)
        assert np.max(np.abs(result)) == pytest.approx(5.0)

    def test_custom_peak_known_array(self):
        x = np.array([1.0, -2.0, 3.0, -0.5])
        result = norm_peak(x, peak=6.0)
        assert np.max(np.abs(result)) == pytest.approx(6.0)

    def test_scaling_is_linear(self):
        x = np.array([1.0, 2.0, 3.0])
        result = norm_peak(x, peak=2.0)
        assert_allclose(result, [2 / 3, 4 / 3, 2.0])

    def test_ratios_between_elements_preserved(self):
        x = np.array([1.0, 2.0, 4.0])
        result = norm_peak(x, peak=5.0)
        assert result[1] / result[0] == pytest.approx(2.0)
        assert result[2] / result[1] == pytest.approx(2.0)


class TestNormPeakEdgeCases:
    """Test edge cases including zero arrays and sign preservation."""

    def test_zero_array_returned_unchanged(self):
        x = np.zeros(5)
        assert_allclose(norm_peak(x), x)

    def test_zero_array_with_custom_peak_returned_unchanged(self):
        x = np.zeros(5)
        assert_allclose(norm_peak(x, peak=3.0), x)

    def test_already_normalized_array_unchanged(self):
        x = np.array([-1.0, 0.0, 0.5, 1.0])
        assert_allclose(norm_peak(x), x)

    def test_sign_preserved(self):
        x = np.array([-3.0, 1.0, 2.0])
        result = norm_peak(x)
        assert result[0] < 0
        assert result[1] > 0

    def test_single_element_array(self):
        assert norm_peak(np.array([5.0])) == pytest.approx(1.0)

    def test_single_negative_element(self):
        assert norm_peak(np.array([-5.0])) == pytest.approx(-1.0)
