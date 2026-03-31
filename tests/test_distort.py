import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyamapping.mappings import distort


class TestDistortScalar:
    """Test distortion of scalar values."""

    def test_zero_input_is_zero(self):
        assert distort(0.0) == pytest.approx(0.0)

    def test_positive_input_is_below_threshold(self):
        # x / (1 + |x|) is always < 1 for finite x
        assert distort(1.0) == pytest.approx(0.5)
        assert distort(2.0) == pytest.approx(2 / 3)
        assert distort(3.0) == pytest.approx(0.75)

    def test_negative_input_mirrors_positive(self):
        # function is odd: distort(-x) == -distort(x)
        assert distort(-1.0) == pytest.approx(-0.5)
        assert distort(-2.0) == pytest.approx(-2 / 3)

    def test_output_bounded_below_one(self):
        assert abs(distort(1e9)) < 1.0

    def test_output_bounded_above_minus_one(self):
        assert distort(-1e9) > -1.0

    def test_custom_threshold(self):
        assert distort(1.0, threshold=2.0) == pytest.approx(1 / 3)

    def test_threshold_scales_output(self):
        # doubling threshold halves the compression at x=threshold
        assert distort(1.0, threshold=2.0) == pytest.approx(distort(0.5, threshold=1.0))


class TestDistortArray:
    """Test distortion of array inputs."""

    def test_known_values(self):
        assert_allclose(
            distort([0, 1, 2, 3], threshold=1),
            [0.0, 0.5, 2/3, 0.75],
            rtol=1e-6,
        )

    def test_negative_array_mirrors_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        assert_allclose(distort(-x), -distort(x))

    def test_array_output_bounded(self):
        x = np.array([-1e9, -1.0, 0.0, 1.0, 1e9])
        result = distort(x)
        assert np.all(result > -1.0)
        assert np.all(result < 1.0)

    def test_custom_threshold_array(self):
        assert_allclose(
            distort(np.array([0, 1, 2]), threshold=2.0),
            [0.0, 1/3, 0.5],
            rtol=1e-6,
        )


class TestDistortProperties:
    """Test mathematical properties of the distortion function."""

    def test_odd_symmetry(self):
        # distort(-x) == -distort(x) for all x
        for x in [0.1, 1.0, 5.0, 100.0]:
            assert distort(-x) == pytest.approx(-distort(x))

    def test_output_range_is_open_minus_one_to_one(self):
        # output strictly between -1 and 1 for all finite inputs
        for x in [-1e9, -100, -1, 0, 1, 100, 1e9]:
            assert -1.0 < distort(x) < 1.0 or distort(x) == 0.0

    def test_monotonically_increasing(self):
        x = np.linspace(-10, 10, 100)
        result = distort(x)