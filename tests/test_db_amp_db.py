import pytest

from pyamapping.mappings import amp_to_db, db_to_amp


class TestDbToAmp:
    """Test decibel to linear amplitude conversion."""

    def test_0db_is_unity(self):
        assert db_to_amp(0) == pytest.approx(1.0)

    def test_20db_is_ten(self):
        assert db_to_amp(20) == pytest.approx(10.0)

    def test_minus_20db_is_tenth(self):
        assert db_to_amp(-20) == pytest.approx(0.1)

    def test_6db_approximately_doubles(self):
        assert db_to_amp(6) == pytest.approx(2.0, rel=1e-2)

    def test_negative_inf_db_is_silence(self):
        assert db_to_amp(-float("inf")) == pytest.approx(0.0)

    def test_positive_db_is_above_unity(self):
        assert db_to_amp(10) > 1.0

    def test_negative_db_is_below_unity(self):
        assert db_to_amp(-10) < 1.0


class TestAmpToDb:
    """Test linear amplitude to decibel conversion."""

    def test_unity_is_0db(self):
        assert amp_to_db(1.0) == pytest.approx(0.0)

    def test_ten_is_20db(self):
        assert amp_to_db(10.0) == pytest.approx(20.0)

    def test_tenth_is_minus_20db(self):
        assert amp_to_db(0.1) == pytest.approx(-20.0)

    def test_double_amplitude_is_approx_6db(self):
        assert amp_to_db(2.0) == pytest.approx(6.0, rel=1e-2)

    def test_half_amplitude_is_approx_minus_6db(self):
        assert amp_to_db(0.5) == pytest.approx(-6.0, rel=1e-2)

    def test_zero_amplitude_is_minus_inf(self):
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            result = amp_to_db(0.0)
        assert result == -float("inf")


class TestDbAmpRoundtrip:
    """Test that the two functions are exact inverses of each other."""

    def test_db_to_amp_to_db(self):
        for db in range(128):
            assert amp_to_db(db_to_amp(db)) == pytest.approx(db, rel=1e-9)

    def test_amp_to_db_to_amp(self):
        for amp in [0.001, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]:
            assert db_to_amp(amp_to_db(amp)) == pytest.approx(amp, rel=1e-9)

    def test_negative_db_roundtrip(self):
        for db in [-60, -40, -20, -6]:
            assert amp_to_db(db_to_amp(db)) == pytest.approx(db, rel=1e-9)
