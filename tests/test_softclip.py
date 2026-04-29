import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.mappings import softclip


class TestSoftclipLinearRegion:
    """Test the perfectly linear region within [-0.5, 0.5]."""

    def test_zero_is_zero(self):
        assert softclip(0.0) == pytest.approx(0.0)

    def test_positive_boundary(self):
        assert softclip(0.5) == pytest.approx(0.5)

    def test_negative_boundary(self):
        assert softclip(-0.5) == pytest.approx(-0.5)

    def test_midpoint_linear(self):
        assert softclip(0.25) == pytest.approx(0.25)

    def test_linear_region_is_identity(self):
        x = np.linspace(-0.5, 0.5, 50)
        assert_allclose(softclip(x), x)


class TestSoftclipNonlinearRegion:
    """Test the compressed region outside [-0.5, 0.5]."""

    def test_known_positive_values(self):
        assert_allclose(
            softclip(np.arange(1, 5)),
            [0.75, 0.875, 0.91666667, 0.9375],
            rtol=1e-6,
        )

    def test_just_outside_boundary_positive(self):
        # (|0.6| - 0.25) / 0.6
        assert softclip(0.6) == pytest.approx(0.35 / 0.6, rel=1e-6)

    def test_just_outside_boundary_negative(self):
        assert softclip(-0.6) == pytest.approx(-0.35 / 0.6, rel=1e-6)

    def test_output_approaches_one_for_large_x(self):
        # (x - 0.25) / x -> 1 as x -> inf
        assert softclip(1e9) == pytest.approx(1.0, rel=1e-6)

    def test_output_approaches_minus_one_for_large_negative_x(self):
        assert softclip(-1e9) == pytest.approx(-1.0, rel=1e-6)


class TestSoftclipProperties:
    """Test mathematical properties of softclip."""

    def test_odd_symmetry(self):
        for x in [0.3, 0.5, 1.0, 5.0, 100.0]:
            assert softclip(-x) == pytest.approx(-softclip(x))

    def test_output_bounded(self):
        x = np.array([-1e9, -10.0, -1.0, 0.0, 1.0, 10.0, 1e9])
        result = softclip(x)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_monotonically_increasing(self):
        x = np.linspace(-10, 10, 200)
        assert np.all(np.diff(softclip(x)) > 0)

    def test_continuous_at_boundary(self):
        # softclip should be continuous at x=±0.5
        assert softclip(0.5 - 1e-9) == pytest.approx(softclip(0.5 + 1e-9), rel=1e-4)
        assert softclip(-0.5 - 1e-9) == pytest.approx(softclip(-0.5 + 1e-9), rel=1e-4)

    def test_scalar_input_returns_float(self):
        assert isinstance(softclip(0.3), float)
        assert isinstance(softclip(1.0), float)

    def test_array_input_returns_ndarray(self):
        assert isinstance(softclip(np.array([0.3, 1.0])), np.ndarray)
