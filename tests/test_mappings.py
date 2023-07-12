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


def test_midi_cps():
    assert midi_to_cps(69) == 440
    assert cps_to_midi(440) == 69
    for x in range(128):
        assert x == cps_to_midi(midi_to_cps(x))


def test_db_amp():
    for x in range(128):
        assert x == pytest.approx(amp_to_db(db_to_amp(x)))


def test_hz_mel():
    pytest.approx(hz_to_mel(440), 549.64)
    pytest.approx(mel_to_hz(549.64), 440)
    for x in range(128):
        assert x == pytest.approx(hz_to_mel(mel_to_hz(x)))
