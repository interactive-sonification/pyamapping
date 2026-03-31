import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.mappings import remove_dc


class TestRemoveDcBasic:
    """Test basic DC removal behaviour."""

    def test_known_values(self):
        assert_allclose(
            remove_dc(np.array([1, 2, 3, 4])),
            [-1.5, -0.5, 0.5, 1.5],
        )

    def test_mean_of_result_is_zero(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.mean(remove_dc(x)) == pytest.approx(0.0)

    def test_mean_of_random_array_is_zero(self):
        x = np.random.rand(100) + 5.0  # large DC offset
        assert np.mean(remove_dc(x)) == pytest.approx(0.0, abs=1e-12)

    def test_shape_preserved(self):
        x = np.random.rand(20)
        assert remove_dc(x).shape == x.shape


class TestRemoveDcEdgeCases:
    """Test edge cases for DC removal."""

    def test_zero_mean_array_unchanged(self):
        x = np.array([-1.0, 0.0, 1.0])
        assert_allclose(remove_dc(x), x)

    def test_constant_array_becomes_zero(self):
        x = np.full(5, 3.0)
        assert_allclose(remove_dc(x), np.zeros(5))

    def test_single_element_becomes_zero(self):
        assert remove_dc(np.array([7.0])) == pytest.approx(0.0)

    def test_negative_dc_offset(self):
        x = np.array([-5.0, -4.0, -3.0])
        assert np.mean(remove_dc(x)) == pytest.approx(0.0)


class TestRemoveDcProperties:
    """Test mathematical properties of DC removal."""

    def test_values_sum_to_zero(self):
        x = np.array([1.0, 3.0, 5.0, 7.0])
        assert np.sum(remove_dc(x)) == pytest.approx(0.0)

    def test_relative_differences_preserved(self):
        # DC removal is a shift — differences between elements are unchanged
        x = np.array([1.0, 3.0, 6.0, 10.0])
        result = remove_dc(x)
        assert_allclose(np.diff(result), np.diff(x))

    def test_idempotent(self):
        # applying twice gives same result as once
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert_allclose(remove_dc(remove_dc(x)), remove_dc(x))

    def test_variance_preserved(self):
        # DC removal does not change variance
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.var(remove_dc(x)) == pytest.approx(np.var(x))