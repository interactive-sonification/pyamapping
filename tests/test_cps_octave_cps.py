import pytest

from pyamapping.mappings import cps_to_octave, octave_to_cps


class TestCpsToOctave:
    """Test frequency to decimal octave conversion."""

    def test_440hz_is_4_75(self):
        # reference point: 440 Hz -> 4.75
        assert cps_to_octave(440) == pytest.approx(4.75)

    def test_middle_c_is_4(self):
        # MIDI 60 / C4 / 261.626 Hz -> 4.0
        assert cps_to_octave(261.626) == pytest.approx(4.0, rel=1e-4)

    def test_880hz_is_5_75(self):
        assert cps_to_octave(880) == pytest.approx(5.75)

    def test_220hz_is_3_75(self):
        assert cps_to_octave(220) == pytest.approx(3.75)

    def test_octave_up_increments_by_one(self):
        assert cps_to_octave(880) == pytest.approx(cps_to_octave(440) + 1)

    def test_octave_down_decrements_by_one(self):
        assert cps_to_octave(220) == pytest.approx(cps_to_octave(440) - 1)


class TestOctaveToCps:
    """Test decimal octave to frequency conversion."""

    def test_4_75_is_440hz(self):
        assert octave_to_cps(4.75) == pytest.approx(440.0)

    def test_4_is_middle_c(self):
        assert octave_to_cps(4.0) == pytest.approx(261.626, rel=1e-4)

    def test_5_75_is_880hz(self):
        assert octave_to_cps(5.75) == pytest.approx(880.0)

    def test_3_75_is_220hz(self):
        assert octave_to_cps(3.75) == pytest.approx(220.0)

    def test_increment_by_one_doubles_frequency(self):
        assert octave_to_cps(5.75) == pytest.approx(octave_to_cps(4.75) * 2)

    def test_decrement_by_one_halves_frequency(self):
        assert octave_to_cps(3.75) == pytest.approx(octave_to_cps(4.75) / 2)


class TestCpsOctaveRoundtrip:
    """Test that the two functions are exact inverses of each other."""

    def test_cps_to_octave_to_cps(self):
        for cps in [27.5, 110.0, 220.0, 261.626, 440.0, 880.0, 1760.0]:
            assert octave_to_cps(cps_to_octave(cps)) == pytest.approx(cps, rel=1e-9)

    def test_octave_to_cps_to_octave(self):
        for octave in [1.0, 2.75, 3.75, 4.0, 4.75, 5.75, 7.0]:
            assert cps_to_octave(octave_to_cps(octave)) == pytest.approx(
                octave, rel=1e-9
            )
