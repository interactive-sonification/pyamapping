import pytest

from pyamapping.mappings import midi_to_ratio, ratio_to_midi


class TestMidiToRatio:
    """Test MIDI interval to frequency ratio conversion."""

    def test_unison_is_ratio_one(self):
        assert midi_to_ratio(0) == pytest.approx(1.0)

    def test_octave_is_ratio_two(self):
        assert midi_to_ratio(12) == pytest.approx(2.0)

    def test_two_octaves_is_ratio_four(self):
        assert midi_to_ratio(24) == pytest.approx(4.0)

    def test_perfect_fifth(self):
        assert midi_to_ratio(7) == pytest.approx(1.4983070768766815)

    def test_negative_interval_inverts_ratio(self):
        assert midi_to_ratio(-12) == pytest.approx(0.5)

    def test_fractional_interval(self):
        assert midi_to_ratio(6.0) == pytest.approx(2**0.5, rel=1e-9)


class TestRatioToMidi:
    """Test frequency ratio to MIDI interval conversion."""

    def test_ratio_one_is_unison(self):
        assert ratio_to_midi(1.0) == pytest.approx(0.0)

    def test_ratio_two_is_octave(self):
        assert ratio_to_midi(2.0) == pytest.approx(12.0)

    def test_ratio_four_is_two_octaves(self):
        assert ratio_to_midi(4.0) == pytest.approx(24.0)

    def test_ratio_half_is_negative_octave(self):
        assert ratio_to_midi(0.5) == pytest.approx(-12.0)

    def test_perfect_fifth_ratio(self):
        assert ratio_to_midi(1.4983070768766815) == pytest.approx(7.0, rel=1e-6)


class TestMidiRatioRoundtrip:
    """Test that the two functions are exact inverses of each other."""

    def test_midi_to_ratio_to_midi(self):
        for midi in [-12, -7, 0, 5, 7, 12, 24]:
            assert ratio_to_midi(midi_to_ratio(midi)) == pytest.approx(midi, rel=1e-9)

    def test_ratio_to_midi_to_ratio(self):
        for ratio in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            assert midi_to_ratio(ratio_to_midi(ratio)) == pytest.approx(ratio, rel=1e-9)

    def test_fractional_roundtrip(self):
        assert ratio_to_midi(midi_to_ratio(3.5)) == pytest.approx(3.5, rel=1e-9)
