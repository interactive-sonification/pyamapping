import numpy as np
import pytest

from pyamapping.mappings import midi_to_cps, cps_to_midi


class TestMidiToCps:
    """Test MIDI note to frequency conversion."""

    def test_a4_is_440hz(self):
        assert midi_to_cps(69) == pytest.approx(440.0)

    def test_a5_is_880hz(self):
        assert midi_to_cps(81) == pytest.approx(880.0)

    def test_a3_is_220hz(self):
        assert midi_to_cps(57) == pytest.approx(220.0)

    def test_middle_c(self):
        assert midi_to_cps(60) == pytest.approx(261.6255, rel=1e-4)

    def test_octave_up_doubles_frequency(self):
        assert midi_to_cps(69 + 12) == pytest.approx(midi_to_cps(69) * 2)

    def test_octave_down_halves_frequency(self):
        assert midi_to_cps(69 - 12) == pytest.approx(midi_to_cps(69) / 2)

    def test_fractional_midi_note(self):
        # halfway between two semitones should be between their frequencies
        f_low = midi_to_cps(60)
        f_high = midi_to_cps(61)
        assert f_low < midi_to_cps(60.5) < f_high


class TestCpsToMidi:
    """Test frequency to MIDI note conversion."""

    def test_440hz_is_a4(self):
        assert cps_to_midi(440.0) == pytest.approx(69.0)

    def test_880hz_is_a5(self):
        assert cps_to_midi(880.0) == pytest.approx(81.0)

    def test_220hz_is_a3(self):
        assert cps_to_midi(220.0) == pytest.approx(57.0)

    def test_middle_c_frequency(self):
        assert cps_to_midi(261.6255) == pytest.approx(60.0, rel=1e-4)

    def test_octave_up_adds_12(self):
        assert cps_to_midi(880.0) == pytest.approx(cps_to_midi(440.0) + 12)

    def test_octave_down_subtracts_12(self):
        assert cps_to_midi(220.0) == pytest.approx(cps_to_midi(440.0) - 12)


class TestMidiCpsRoundtrip:
    """Test that the two functions are exact inverses of each other."""

    def test_midi_to_cps_to_midi(self):
        for midi in [0, 21, 60, 69, 81, 108, 127]:
            assert cps_to_midi(midi_to_cps(midi)) == pytest.approx(midi, rel=1e-9)

    def test_cps_to_midi_to_cps(self):
        for cps in [27.5, 110.0, 220.0, 440.0, 880.0, 1760.0]:
            assert midi_to_cps(cps_to_midi(cps)) == pytest.approx(cps, rel=1e-9)

    def test_fractional_roundtrip(self):
        assert cps_to_midi(midi_to_cps(60.5)) == pytest.approx(60.5, rel=1e-9)