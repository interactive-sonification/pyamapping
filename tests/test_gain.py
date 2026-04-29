import numpy as np
from numpy.testing import assert_allclose

from pyamapping.mappings import db_to_amp, gain


class TestGainAmp:
    """Test gain applied as a scalar amplitude factor."""

    def test_known_values(self):
        assert_allclose(gain(np.array([1, 2, 3, 4]), amp=2), [2, 4, 6, 8])

    def test_amp_one_unchanged(self):
        x = np.array([1.0, 2.0, 3.0])
        assert_allclose(gain(x, amp=1.0), x)

    def test_amp_half_attenuates(self):
        x = np.array([2.0, 4.0, 6.0])
        assert_allclose(gain(x, amp=0.5), [1.0, 2.0, 3.0])

    def test_amp_zero_silences(self):
        x = np.array([1.0, 2.0, 3.0])
        assert_allclose(gain(x, amp=0.0), np.zeros(3))


class TestGainDb:
    """Test gain applied as decibels."""

    def test_0db_unchanged(self):
        x = np.array([1.0, 2.0, 3.0])
        assert_allclose(gain(x, db=0), x, rtol=1e-6)

    def test_minus_6db_approximately_halves(self):
        x = np.array([1.0, 2.0, 3.0])
        assert_allclose(gain(x, db=-6), x * db_to_amp(-6), rtol=1e-6)

    def test_20db_multiplies_by_ten(self):
        x = np.array([1.0, 2.0, 3.0])
        assert_allclose(gain(x, db=20), x * 10.0, rtol=1e-6)

    def test_minus_20db_divides_by_ten(self):
        x = np.array([1.0, 2.0, 3.0])
        assert_allclose(gain(x, db=-20), x * 0.1, rtol=1e-6)


class TestGainCombined:
    """Test gain applied as both db and amp simultaneously."""

    def test_db_and_amp_both_applied(self):
        x = np.array([1.0, 2.0, 3.0])
        assert_allclose(gain(x, db=20, amp=2.0), x * 10.0 * 2.0, rtol=1e-6)

    def test_db_and_amp_order_independent(self):
        # db*amp == amp*db mathematically
        x = np.array([1.0, 2.0, 3.0])
        assert_allclose(
            gain(x, db=6, amp=0.5),
            x * db_to_amp(6) * 0.5,
            rtol=1e-6,
        )


class TestGainEdgeCases:
    """Test edge cases and no-op behaviour."""

    def test_no_args_returns_copy(self):
        x = np.array([1.0, 2.0, 3.0])
        result = gain(x)
        assert_allclose(result, x)
        assert result is not x  # should be a copy, not the same object

    def test_shape_preserved(self):
        x = np.random.rand(20)
        assert gain(x, amp=2.0).shape == x.shape

    def test_negative_amp_inverts_polarity(self):
        x = np.array([1.0, -2.0, 3.0])
        assert_allclose(gain(x, amp=-1.0), [-1.0, 2.0, -3.0])
