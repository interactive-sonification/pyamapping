"""Collection of audio related mapping functions"""
from typing import Optional, Union

import numpy as np


def linlin(
    value: Union[float, np.ndarray],
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    clip: Optional[str] = None,
) -> Union[float, np.ndarray]:
    """Map value linearly so that [x1, x2] is mapped to [y1, y2]

    linlin is implemented in analogy to the SC3 linlin, yet this
    function extrapolates by default.
    A frequently used invocation is with x1 < x2, i.e. thinking
    of them as a range [x1,x2]

    Parameters
    ----------
    value : float or np.ndarray
        value(s) to be mapped
    x1 : float
        source value 1
    x2 : float
        source value 2
    y1 : float
        destination value to be reached for value == x1
    y2 : float
        destination value to be reached for value == x2
    clip: None or string
        None extrapolates, "min" or "max" clip at floor resp. ceiling
        of the destination range, any other value defaults to "minmax",
        i.e. it clips on both sides.

    Returns
    -------
    float or np.ndarray
        the mapping result
    """
    z = (value - x1) / (x2 - x1) * (y2 - y1) + y1
    if clip is None:
        return z
    if y1 > y2:
        x1, x2, y1, y2 = x2, x1, y2, y1
    if clip == "max":
        return np.minimum(z, y2)
    elif clip == "min":
        return np.maximum(z, y1)
    else:  # imply clip to be "minmax"
        return np.minimum(np.maximum(z, y1), y2)


def clip(
    value: Union[float, np.ndarray],
    minimum: float = -float("inf"),
    maximum: float = float("inf"),
) -> Union[float, np.ndarray]:
    """Clips a value to a certain range

    Parameters
    ----------
    value : float or np.ndarray
        Value(s) to clip
    minimum : float, optional
        Minimum output value, by default -float("inf")
    maximum : float, optional
        Maximum output value, by default float("inf")

    Returns
    -------
    float
        clipped value
    """
    if type(value) == np.ndarray:
        return np.maximum(np.minimum(value, maximum), minimum)
    else:  # ToDo: check if better performance than above numpy code - if not: delete
        if value < minimum:
            return minimum
        if value > maximum:
            return maximum
        return value


def midi_to_cps(midi_note: float) -> float:
    """Convert MIDI note to cycles per second

    Parameters
    ----------
    m : float
        midi note

    Returns
    -------
    float
        corresponding cycles per seconds
    """
    return 440.0 * 2 ** ((midi_note - 69) / 12.0)


midicps = midi_to_cps


def cps_to_midi(cps: float) -> float:
    """Convert cycles per second to MIDI note

    Parameters
    ----------
    cps : float
        cycles per second

    Returns
    -------
    float
        corresponding MIDI note
    """
    return 69 + 12 * np.log2(cps / 440.0)


cpsmidi = cps_to_midi


def hz_to_mel(hz):
    """Convert a value in Hertz to Mels

    Parameters
    ----------
    hz : number of array
        value in Hz, can be an array

    Returns:
    --------
    _ : number of array
        value in Mels, same type as the input.
    """
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    """Convert a value in Hertz to Mels

    Parameters
    ----------
    hz : number of array
        value in Hz, can be an array

    Returns:
    --------
    _ : number of array
        value in Mels, same type as the input.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def db_to_amp(decibels: float) -> float:
    """Convert a decibels to a linear amplitude.

    Parameters
    ----------
    decibels : float
        Decibel value to convert

    Returns
    -------
    float
        Corresponding linear amplitude
    """
    return 10 ** (decibels / 20.0)


dbamp = db_to_amp


def amp_to_db(amp: float) -> float:
    """Convert a linear amplitude to decibels.

    Parameters
    ----------
    amp : float
        Linear amplitude to convert

    Returns
    -------
    float
        Corresponding decibels
    """
    return 20 * np.log10(amp)


ampdb = amp_to_db
