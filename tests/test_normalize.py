import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyamapping.mappings import normalize


class TestNormalizeDefaultRange:
    """Test normalization to default range [-1, 1]."""

    def test_min_maps_to_minus_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize(x)
        assert result.min() == pytest.approx(-1.0)

    def test_max_maps_to_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize(x)
        assert result.max() == pytest.approx(1.0)

    def test_random_array_bounds(self):
        result = normalize(np.random.rand(100))
        assert result.min() == pytest.approx(-1.0)
        assert result.max() == pytest.approx(1.0)


class TestNormalizeCustomRange:
    """Test normalization to custom target ranges."""

    def test_custom_range(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = normalize(x, y1=0.0, y2=100.0)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(100.0)

    def test_midpoint_maps_correctly(self):
        x = np.array([0.0, 0.5, 1.0])
        result = normalize(x, y1=0.0, y2=1.0)
        assert result[1] == pytest.approx(0.5)

    def test_known_values(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert_allclose(normalize(x, y1=0.0, y2=1.0), [0.0, 0.25, 0.5, 0.75, 1.0])


class TestNormalizePolarityInversion:
    """Test normalization with y1 > y2 (inverted range)."""

    def test_inverted_range_min_maps_to_y1(self):
        x = np.array([1.0, 2.0, 3.0])
        result = normalize(x, y1=1.0, y2=0.0)
        assert result.max() == pytest.approx(1.0)

    def test_inverted_range_max_maps_to_y2(self):
        x = np.array([1.0, 2.0, 3.0])
        result = normalize(x, y1=1.0, y2=0.0)
        assert result.min() == pytest.approx(0.0)

    def test_inverted_range_is_mirror_of_normal(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normal = normalize(x, y1=0.0, y2=1.0)
        inverted = normalize(x, y1=1.0, y2=0.0)
        assert_allclose(normal + inverted, np.ones_like(x))


class TestNormalizeProperties:
    """Test mathematical properties of normalize."""

    def test_output_shape_matches_input(self):
        x = np.random.rand(20)
        assert normalize(x).shape == x.shape

    def test_constant_array_with_zero_range(self):
        # all same values -> x2 - x1 = 0 -> nan (numpy does not raise)
        x = np.array([3.0, 3.0, 3.0])
        result = normalize(x)
        assert np.all(np.isnan(result))

    def test_linear_mapping_preserves_order(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize(x)
        assert np.all(np.diff(result) > 0)

    def test_already_normalized_input(self):
        x = np.array([-1.0, 0.0, 1.0])
        assert_allclose(normalize(x), x)