import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.mappings import norm_rms


class TestNormRmsDefaultRms:
    """Test RMS normalization with default rms=1.0."""

    def test_rms_is_one(self):
        x = np.array([1.0, -2.0, 3.0, -0.5])
        result = norm_rms(x)
        assert np.sqrt(np.mean(result**2)) == pytest.approx(1.0)

    def test_known_values(self):
        assert_allclose(
            norm_rms(np.array([1, 0, 0, -1]), 1),
            [1.41421356, 0.0, 0.0, -1.41421356],
            rtol=1e-6,
        )

    def test_shape_preserved(self):
        x = np.random.rand(20)
        assert norm_rms(x).shape == x.shape


class TestNormRmsCustomRms:
    """Test RMS normalization with custom target RMS values."""

    def test_custom_rms_achieved(self):
        x = np.array([1.0, -2.0, 3.0, -0.5])
        result = norm_rms(x, rms=2.0)
        assert np.sqrt(np.mean(result**2)) == pytest.approx(2.0)

    def test_random_array_custom_rms(self):
        x = np.random.rand(100)
        result = norm_rms(x, rms=0.5)
        assert np.sqrt(np.mean(result**2)) == pytest.approx(0.5)

    def test_scaling_is_linear(self):
        # doubling target rms doubles all values
        x = np.array([1.0, 2.0, 3.0])
        result1 = norm_rms(x, rms=1.0)
        result2 = norm_rms(x, rms=2.0)
        assert_allclose(result2, result1 * 2)

    def test_ratios_between_elements_preserved(self):
        x = np.array([1.0, 2.0, 4.0])
        result = norm_rms(x, rms=3.0)
        assert result[1] / result[0] == pytest.approx(2.0)
        assert result[2] / result[1] == pytest.approx(2.0)


class TestNormRmsEdgeCases:
    """Test edge cases including zero arrays and sign preservation."""

    def test_zero_array_returned_unchanged(self):
        x = np.zeros(5)
        assert_allclose(norm_rms(x), x)

    def test_zero_array_with_custom_rms_returned_unchanged(self):
        x = np.zeros(5)
        assert_allclose(norm_rms(x, rms=3.0), x)

    def test_already_unit_rms_unchanged(self):
        # array with rms=1: single element [1.0]
        x = np.array([1.0])
        assert_allclose(norm_rms(x), x)

    def test_sign_preserved(self):
        x = np.array([-3.0, 1.0, 2.0])
        result = norm_rms(x)
        assert result[0] < 0
        assert result[1] > 0

    def test_single_element_array(self):
        result = norm_rms(np.array([5.0]))
        assert np.sqrt(np.mean(result**2)) == pytest.approx(1.0)
