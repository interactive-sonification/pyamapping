import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyamapping.mappings import interp, ecdf, ecdf_to_lin, lin_to_ecdf


class TestEcdf:
    """Test empirical cumulative distribution function."""

    def test_output_is_tuple_of_two_arrays(self):
        xs, ys = ecdf(np.array([3.0, 1.0, 2.0]))
        assert isinstance(xs, np.ndarray)
        assert isinstance(ys, np.ndarray)

    def test_xs_are_sorted(self):
        xs, ys = ecdf(np.array([3.0, 1.0, 2.0]))
        assert np.all(np.diff(xs) >= 0)

    def test_ys_range_from_step_to_one(self):
        xs, ys = ecdf(np.array([1.0, 2.0, 3.0, 4.0]))
        assert ys[0] == pytest.approx(0.25)
        assert ys[-1] == pytest.approx(1.0)

    def test_ys_are_uniformly_spaced(self):
        xs, ys = ecdf(np.array([1.0, 2.0, 3.0, 4.0]))
        assert_allclose(np.diff(ys), np.full(3, 0.25))

    def test_lengths_match(self):
        x = np.random.rand(20)
        xs, ys = ecdf(x)
        assert len(xs) == len(ys) == 20

    def test_selection_slice(self):
        x = np.arange(1, 11, dtype=float)
        xs, ys = ecdf(x, selection=np.s_[::2])
        assert len(xs) == 5
        assert len(ys) == 5

    def test_already_sorted_input(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        xs, ys = ecdf(x)
        assert_allclose(xs, x)


class TestInterp:
    """Test piecewise linear interpolation with clipping."""

    def test_exact_at_control_points(self):
        assert interp(0, [-1, 0, 1], [-1, 0, 1]) == pytest.approx(0.0)
        assert interp(-1, [-1, 0, 1], [-1, 0, 1]) == pytest.approx(-1.0)
        assert interp(1, [-1, 0, 1], [-1, 0, 1]) == pytest.approx(1.0)

    def test_midpoint_interpolation(self):
        assert interp(0.5, [0, 1], [0, 10]) == pytest.approx(5.0)

    def test_clips_below(self):
        # numpy.interp clips by default, unlike interp_spline
        assert interp(-2, [-1, 0, 1], [-1, 0, 1]) == pytest.approx(-1.0)

    def test_clips_above(self):
        assert interp(2, [-1, 0, 1], [-1, 0, 1]) == pytest.approx(1.0)

    def test_array_input(self):
        result = interp(np.array([-1.0, 0.0, 1.0]), [-1, 0, 1], [-1, 0, 1])
        assert_allclose(result, [-1.0, 0.0, 1.0])

    def test_default_yc(self):
        # default yc=[-1, 0, 1] maps xc=[-1, 0, 1] linearly
        assert interp(0.0, [-1, 0, 1]) == pytest.approx(0.0)


class TestLinToEcdf:
    """Test mapping via empirical cumulative distribution function."""

    def test_output_range_is_zero_to_one(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lin_to_ecdf(ref, ref)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_min_maps_to_lowest_quantile(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert lin_to_ecdf(1.0, ref) == pytest.approx(0.2)

    def test_max_maps_to_one(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert lin_to_ecdf(5.0, ref) == pytest.approx(1.0)

    def test_below_min_clips_to_zero(self):
        ref = np.array([1.0, 2.0, 3.0])
        assert lin_to_ecdf(0.0, ref) == pytest.approx(1/3)

    def test_above_max_clips_to_one(self):
        ref = np.array([1.0, 2.0, 3.0])
        assert lin_to_ecdf(10.0, ref) == pytest.approx(1.0)

    def test_sorted_and_unsorted_give_same_result(self):
        ref = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 2.0])
        x = np.array([1.0, 2.0, 3.0])
        assert_allclose(
            lin_to_ecdf(x, ref, sorted=False),
            lin_to_ecdf(x, np.sort(ref), sorted=True),
        )

    def test_monotonically_increasing(self):
        ref = np.random.rand(50)
        x = np.linspace(0, 1, 20)
        result = lin_to_ecdf(x, ref)
        assert np.all(np.diff(result) >= 0)



class TestEcdfToLin:
    """Test inverse ECDF mapping (quantiles back to feature values)."""

    def test_max_quantile_maps_to_max_value(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert ecdf_to_lin(1.0, ref) == pytest.approx(5.0)

    def test_min_quantile_maps_to_min_value(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert ecdf_to_lin(0.2, ref) == pytest.approx(1.0)

    def test_midpoint_quantile(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert ecdf_to_lin(0.6, ref) == pytest.approx(3.0)

    def test_below_min_quantile_clips(self):
        ref = np.array([1.0, 2.0, 3.0])
        assert ecdf_to_lin(0.0, ref) == pytest.approx(1.0)

    def test_above_max_quantile_clips(self):
        ref = np.array([1.0, 2.0, 3.0])
        assert ecdf_to_lin(1.5, ref) == pytest.approx(3.0)

    def test_sorted_and_unsorted_give_same_result(self):
        ref = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 2.0])
        x = np.array([0.2, 0.5, 0.8])
        assert_allclose(
            ecdf_to_lin(x, ref, sorted=False),
            ecdf_to_lin(x, np.sort(ref), sorted=True),
        )

    def test_monotonically_increasing(self):
        ref = np.random.rand(50)
        x = np.linspace(0.1, 0.9, 20)
        result = ecdf_to_lin(x, ref)
        assert np.all(np.diff(result) >= 0)

    def test_array_input(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ecdf_to_lin(np.array([0.2, 0.6, 1.0]), ref)
        assert_allclose(result, [1.0, 3.0, 5.0])


class TestLinEcdfRoundtrip:
    """Test that lin_to_ecdf and ecdf_to_lin are inverses of each other."""

    def test_lin_to_ecdf_to_lin(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # map ref values to quantiles and back — should recover original values
        quantiles = lin_to_ecdf(ref, ref)
        assert_allclose(ecdf_to_lin(quantiles, ref), ref, rtol=1e-6)

    def test_ecdf_to_lin_to_ecdf(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        quantiles = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        recovered = lin_to_ecdf(ecdf_to_lin(quantiles, ref), ref)
        assert_allclose(recovered, quantiles, rtol=1e-6)

    def test_roundtrip_unsorted_ref(self):
        ref = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
        quantiles = lin_to_ecdf(np.sort(ref), ref)
        assert_allclose(ecdf_to_lin(quantiles, ref), np.sort(ref), rtol=1e-6)