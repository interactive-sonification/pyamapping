import pytest
import numpy as np

from pyamapping.mappings import hz_to_mel, mel_to_hz

class TestHzToMel:
    """Test Hz to mel conversion for both Slaney and HTK formulas."""

    def test_zero_hz_is_zero_mel_slaney(self):
        assert hz_to_mel(0) == pytest.approx(0.0)

    def test_zero_hz_is_zero_mel_htk(self):
        assert hz_to_mel(0, htk=True) == pytest.approx(0.0)

    def test_known_value_slaney(self):
        assert hz_to_mel(440) == pytest.approx(6.6, rel=1e-6)

    def test_known_value_htk(self):
        # 440 Hz -> mel via O'Shaughnessy formula
        assert hz_to_mel(440, htk=True) == pytest.approx(2595 * np.log10(1 + 440 / 700), rel=1e-6)

    def test_linear_regime_below_1000hz(self):
        # Slaney: linear law applies below 1000 Hz
        assert hz_to_mel(200) == pytest.approx(3.0 * 200 / 200, rel=1e-6)

    def test_log_regime_above_1000hz(self):
        # Slaney: log law applies at and above 1000 Hz — boundary value
        assert hz_to_mel(1000) == pytest.approx(15.0, rel=1e-6)

    def test_array_input_slaney(self):
        result = hz_to_mel(np.array([0, 200, 1000, 4000]))
        assert isinstance(result, np.ndarray)
        assert result[2] == pytest.approx(15.0, rel=1e-6)

    def test_array_input_htk(self):
        result = hz_to_mel(np.array([0, 440, 1000]), htk=True)
        assert isinstance(result, np.ndarray)


class TestMelToHz:
    """Test mel to Hz conversion for both Slaney and HTK formulas."""

    def test_zero_mel_is_zero_hz_slaney(self):
        assert mel_to_hz(0) == pytest.approx(0.0)

    def test_zero_mel_is_zero_hz_htk(self):
        assert mel_to_hz(0, htk=True) == pytest.approx(0.0)

    def test_known_value_slaney(self):
        assert mel_to_hz(6.6) == pytest.approx(440.0, rel=1e-6)


    def test_known_value_htk(self):
        assert mel_to_hz(2595 * np.log10(1 + 440 / 700), htk=True) == pytest.approx(440, rel=1e-6)

    def test_linear_regime_below_mel_15(self):
        # Slaney: linear law applies below mel=15
        assert mel_to_hz(9.0) == pytest.approx(200.0 / 3 * 9, rel=1e-6)

    def test_boundary_mel_15_is_1000hz(self):
        assert mel_to_hz(15) == pytest.approx(1000.0, rel=1e-6)

    def test_array_input_slaney(self):
        result = mel_to_hz(np.array([0, 9, 15, 30]))
        assert isinstance(result, np.ndarray)
        assert result[2] == pytest.approx(1000.0, rel=1e-6)


class TestHzMelRoundtrip:
    """Test that hz_to_mel and mel_to_hz are exact inverses of each other."""

    def test_hz_to_mel_to_hz_slaney(self):
        for hz in np.arange(1, 128):
            assert mel_to_hz(hz_to_mel(hz)) == pytest.approx(hz, rel=1e-6)

    def test_mel_to_hz_to_mel_slaney(self):
        for mel in np.arange(1, 128):
            assert hz_to_mel(mel_to_hz(mel)) == pytest.approx(mel, rel=1e-6)

    def test_hz_to_mel_to_hz_htk(self):
        for hz in [100, 440, 1000, 4000, 8000]:
            assert mel_to_hz(hz_to_mel(hz, htk=True), htk=True) == pytest.approx(hz, rel=1e-9)

    def test_mel_to_hz_to_mel_htk(self):
        for mel in [100, 500, 1000, 2000]:
            assert hz_to_mel(mel_to_hz(mel, htk=True), htk=True) == pytest.approx(mel, rel=1e-9)

    def test_htk_and_slaney_differ(self):
        # The two formulas are different — their results should not be equal
        assert hz_to_mel(440, htk=True) != pytest.approx(hz_to_mel(440, htk=False))