import numpy as np
import pytest

from pyamapping import (
    amp_to_db,
    clip,
    cps_to_midi,
    db_to_amp,
    hz_to_mel,
    linlin,
    mel_to_hz,
    midi_to_cps,
)
from pyamapping.mappings import (
    bilin,
    cps_to_octave,
    curvelin,
    distort,
    explin,
    lcurve,
    lincurve,
    linexp,
    midi_to_ratio,
    octave_to_cps,
    ratio_to_midi,
    scurve,
    softclip,
)


def test_linlin():
    origin = np.array([-1, 0, 1, 2, 4])
    target = np.array([-50, 0, 50, 100, 200])
    assert np.array_equal(target, linlin(origin, 0, 2, 0, 100))

    # test clipping
    assert 100 == linlin(-10, 0, 100, 100, 200, "min")
    assert 200 == linlin(105, 0, 100, 100, 200, "max")

    assert 200 == linlin(-10, 0, 100, 200, 100, "max")
    assert 100 == linlin(105, 0, 100, 200, 100, "min")

    assert 100 == linlin(3, 5, 105, 100, 1000, "minmax")
    assert 1000 == linlin(105, 0, 100, 100, 1000, "minmax")

    # test y1 > y2
    assert 0.0 == linlin(1, 0, 1, 1, 0)
    assert 0.5 == linlin(0.5, 0, 1, 1, 0)
    assert 1.0 == linlin(0, 0, 1, 1, 0)


def test_clip():
    for x, y in zip([2, 3, 4, 5, 6], [3, 3, 4, 5, 5]):
        assert y == clip(x, 3, 5)

    a1 = np.arange(6)
    a2 = np.array([2, 2, 2, 3, 4, 4])
    assert np.array_equal(clip(a1, 2, 4), a2)


def test_midi_to_cps_to_midi():
    assert midi_to_cps(69) == 440
    assert cps_to_midi(440) == 69
    for x in range(128):
        assert x == cps_to_midi(midi_to_cps(x))


def test_midi_ratio_midi():
    pytest.approx(midi_to_ratio(7), 1.4983070768766815)
    pytest.approx(ratio_to_midi(2), 12.0)


def test_cps_octave_cps():
    pytest.approx(octave_to_cps(5.75), 880)
    pytest.approx(cps_to_octave(220), 3.75)


def test_db_amp_db():
    for x in range(128):
        assert x == pytest.approx(amp_to_db(db_to_amp(x)))


def test_hz_mel_hz():
    pytest.approx(hz_to_mel(440), 549.64)
    pytest.approx(mel_to_hz(549.64), 440)
    for x in np.arange(1, 128):
        assert x == pytest.approx(hz_to_mel(mel_to_hz(x)))


def test_linexp():
    pytest.approx(linexp(5, 1, 8, 2, 256), 32.0)
    pytest.approx(linexp(7, 0, 5, 100, 300, "max"), 300)


def test_explin():
    f = 220 * 2 ** (-5 / 12)
    pytest.approx(explin(f, 220, 440, 0, 12), -5.0)
    pytest.approx(explin(0.01, 0.001, 1.0, -30, 0, "max"), -20.0)


def test_lincurve():
    pytest.approx(
        lincurve(np.array([0.0, 0.1, 0.4, 0.7, 1.0]), 0, 1, 0, 0.4),
        np.array([0.0, 0.08385643, 0.25474431, 0.34852956, 0.4]),
    )


def test_curvelin():
    pytest.approx(
        curvelin(np.array([0, 0.1, 0.3, 0.5]), 0, 0.5, 0, 10),
        np.array([0.0, 0.94934752, 3.65734932, 10.0]),
    )


def test_bilin():
    pytest.approx(
        bilin(np.array([0, 20, 40, 60, 80, 100]), 60, 20, 80, 0, -20, 60),
        np.array([-30.0, -20.0, -10.0, 0.0, 60.0, 120.0]),
    )


def test_distort():
    pytest.approx(
        distort([0, 1, 2, 3], 1),
        np.array([0.0, 0.5, 0.66666667, 0.75]),
    )


def test_softclip():
    pytest.approx(
        softclip(np.arange(1, 5)),
        np.array([0.75, 0.875, 0.91666667, 0.9375]),
    )


def test_scurve():
    pytest.approx(
        scurve(np.arange(0, 1, 0.25)),
        np.array([0.0, 0.15625, 0.5, 0.84375]),
    )


def test_lcurve():
    pytest.approx(
        lcurve(np.array([-1, -0.5, 0, 0.5, 1])),
        np.array([0.26894142, 0.37754067, 0.5, 0.62245933, 0.73105858]),
    )
