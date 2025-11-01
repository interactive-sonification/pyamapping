"""Collection of audio related mapping functions."""

from typing import Any, Callable, List, Optional, TypeVar, Union

import numpy as np
from numpy.typing import ArrayLike

NDArrayType = TypeVar("NDArrayType", bound=np.ndarray)


def linlin(
    value: Union[float, ArrayLike],
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    clip: Optional[str] = None,
) -> Union[float, np.ndarray]:
    """Map value linearly so that [x1, x2] is mapped to [y1, y2].

    linlin is implemented in analogy to the SC3 linlin, yet this
    function extrapolates by default.
    A frequently used invocation is with x1 < x2, i.e. thinking
    of them as a range [x1,x2]

    Parameters
    ----------
    value : float or np.ndarray (ArrayLike)
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


def linexp(
    value: Union[float, ArrayLike],
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    clip: Optional[str] = None,
) -> Union[float, np.ndarray]:
    """Map value exponentially so that [x1, x2] is mapped to [y1, y2].

    linexp is implemented in analogy to the SC3 linexp, yet this
    function extrapolates by default.
    A frequently used invocation is with x1 < x2, i.e. thinking
    of them as a range [x1,x2]

    Parameters
    ----------
    value : float or np.ndarray (ArrayLike)
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
    z = np.exp((value - x1) / (x2 - x1) * (np.log(y2) - np.log(y1)) + np.log(y1))
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


def explin(
    value: Union[float, ArrayLike],
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    clip: Optional[str] = None,
) -> Union[float, np.ndarray]:
    """Map value logarithmically so that [x1, x2] is mapped to [y1, y2].

    explin is implemented in analogy to the SC3 explin, yet this
    function extrapolates by default.
    A frequently used invocation is with x1 < x2, i.e. thinking
    of them as a range [x1,x2]

    Parameters
    ----------
    value : float or np.ndarray (ArrayLike)
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
    z = np.log(value / x1) / np.log(x2 / x1) * (y2 - y1) + y1

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


def lincurve(
    x: Union[float, ArrayLike],
    x1: float,
    x2: float,
    y1: float = -1.0,
    y2: float = 1.0,
    curve: float = -2.0,
    clip: Optional[str] = None,
) -> Union[float, np.ndarray]:
    """Map value exponentially so that [x1, x2] is mapped to [y1, y2].

    lincurve is implemented in analogy to the SC3 lincurve, yet this
    function extrapolates by default.
    A frequently used invocation is with x1 < x2,
    i.e. thinking of them as a range [x1, x2]
    x1 is mapped to y1. Use y2 < y1 for polarity inversion, i.e. curve = -curve
    returns y1 + (y2 - y1) /
                (1.0 - exp(curve)) * (1 - exp(curve) ** ((x - x1) / (x2 - x1)))
            yoffset +   yrange  * (this goes from 0 =(1-grow**0) to (1-grow**1)

    Parameters
    ----------
    value : float or np.ndarray (ArrayLike)
        value(s) to be mapped
    x1 : float
        source value 1
    x2 : float
        source value 2
    y1 : float
        destination value to be reached for value == x1
    y2 : float
        destination value to be reached for value == x2
    curve : float
        specification of the curvature. TBA
    clip: None or string
        None extrapolates, "min" or "max" clip at floor resp. ceiling
        of the destination range, any other value defaults to "minmax",
        i.e. it clips on both sides.

    Returns
    -------
    float or np.ndarray
        the mapping result
    """
    if abs(curve) < 0.001:
        z = (x - x1) / (x2 - x1) * (y2 - y1) + y1
    else:
        z = y1 + (y2 - y1) / (1.0 - np.exp(curve)) * (
            1 - np.exp((curve * (x - x1) / (x2 - x1)))
        )

    if y1 > y2:
        y1, y2 = y2, y1
    if clip:
        if clip == "max":
            z = np.minimum(z, y2)
        elif clip == "min":
            z = np.maximum(z, y1)
        else:  # imply clip to be "minmax"
            z = np.minimum(np.maximum(z, y1), y2)
    return z


def curvelin(
    x: Union[float, ArrayLike],
    x1: float,
    x2: float,
    y1: float = -1.0,
    y2: float = 1.0,
    curve: float = -2.0,
    clip: Optional[str] = None,
) -> Union[float, np.ndarray]:
    """Map (assumedly exponentially curved) x from [x1, x2] linearly to [y1, y2].

    This is done by applying a curve parameter as in sc3. the input range can include 0,
    different from explin a clipping is performed according to the clip argument.

    curvelin is implemented in analogy to the SC3 curvelin, yet extrapolates by default.
    A frequently used invocation is with x1<x2, i.e. a range [x1, x2]
    Use x1 is mapped to y1, use y2 < y1 for polarity inversion.

    Parameters
    ----------
    value : float or np.ndarray (ArrayLike)
        value(s) to be mapped
    x1 : float
        source value 1
    x2 : float
        source value 2
    y1 : float
        destination value to be reached for value == x1
    y2 : float
        destination value to be reached for value == x2
    curve : float
        specification of the curvature. TBA
    clip: None or string
        None extrapolates, "min" or "max" clip at floor resp. ceiling
        of the destination range, any other value defaults to "minmax",
        i.e. it clips on both sides.

    Returns
    -------
    float or np.ndarray
        the mapping result
    """
    if abs(curve) < 0.001:
        z = (x - x1) / (x2 - x1) * (y2 - y1) + y1
    else:
        a = (x2 - x1) / (1.0 - np.exp(curve))
        z = np.log((x1 + a - x) / a) * (y2 - y1) / curve + y1

    if y1 > y2:
        y1, y2 = y2, y1
    if clip:
        if clip == "max":
            z = np.minimum(z, y2)
        elif clip == "min":
            z = np.maximum(z, y1)
        else:  # imply clip to be "minmax"
            z = np.minimum(np.maximum(z, y1), y2)
    return z


def linpoly(
    x: Union[float, ArrayLike],
    xmax: float = 1.0,
    y1: float = -1.0,
    y2: float = 1.0,
    curve: float = 2.0,
    clip: Optional[str] = None,
) -> Union[float, np.ndarray]:
    """Map x between [-xmax, xmax] to [y1, y2] using a polynomial mapping.

    The mapping is y1 + (y2 - y1) * (1 + (x/xmax)**order) / 2
    where order = 1 + curve if curve>0 else (1 - 1 / (1 + curve))


    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): values to be mapped
        xmax (float): source scale, defaults to 1.0
        y1 (float, optional): target range low value, Defaults to -1.0.
        y2 (float, optional): target ragne 2nd value. Defaults to 1.0.
        curve (float, optional):  defaults to 2
            if curve > 0: polynomial order is curve + 1
            if curve < 0: polynomial order is 1 - 1/curve
        clip (Optional[str], optional): clip flags (min / max / minmax).
            Defaults to None.

    Returns
    -------
        Union[float, np.ndarray]: mapping result for x
    """
    order = 1 + curve if curve >= 0 else (1 / (1 - curve))
    z = y1 + (y2 - y1) * (1 + np.sign(x) * (np.abs(x) / xmax) ** order) / 2
    if clip is None:
        return z
    if y1 > y2:
        y1, y2 = y2, y1
    if clip == "max":
        return np.minimum(z, y2)
    elif clip == "min":
        return np.maximum(z, y1)
    else:  # imply clip to be "minmax"
        return np.minimum(np.maximum(z, y1), y2)


def interp_spline(
    x: Union[float, ArrayLike],
    xc: Union[List[float], np.ndarray],
    yc: Union[List[float], np.ndarray] = [-1, 0, 1],
    k=1,
    **kwarg,
) -> Union[float, np.ndarray]:
    """Apply scipy.interpolate.interp_spline interpolation.

    applicable for piecewise linear mappings, with extrapolation.
    interp_spline is slower than numpy.interp() for smaller data sets (e.g. <5000)
    however, it allows extrapolation.

    Parameters
    ----------
    x : float or np.ndarray (ArrayLike)
        value(s) to be mapped
    xc : Union[List[float], np.ndarray]
        x coordinates of line segment function, must be sorted
    yc : Union[List[float], np.ndarray]
        y coordinates of line segment function, requires len(yc)==len(xc)
    k : interpolation order (see interp_spline, defaults to 1 = linear)

    Returns
    -------
    float or np.ndarray
        the mapping result, extrapolating beyond bounds

    """
    from scipy.interpolate import make_interp_spline

    spl = make_interp_spline(xc, yc, k=k)  # k=1: linear
    return spl(x) if isinstance(x, ChainableArray) else chain(spl(x))


def interp(
    x: Union[float, ArrayLike],
    xc: Union[List[float], np.ndarray],
    yc: Union[List[float], np.ndarray] = [-1, 0, 1],
    **kwarg,
) -> Union[float, np.ndarray]:
    """Apply numpy.interp interpolation.

    applicable for piecewise linear mappings, with extrapolation
    interp is faster than interp_spline() for small x (e.g. <5000),
    but it clips by default.

    Parameters
    ----------
    x : float or np.ndarray (ArrayLike)
        value(s) to be mapped
    xc : Union[List[float], np.ndarray]
        x coordinates of line segment function, must be sorted
    yc : Union[List[float], np.ndarray]
        y coordinates of line segment function, requires len(yc)==len(xc)

    Returns
    -------
    float or np.ndarray
        the mapping result, clipping beyond bounds

    """
    return np.interp(x, xc, yc, **kwarg)


def bilin(
    x: Union[float, ArrayLike],
    xcenter: float,
    xmin: float,
    xmax: float,
    ycenter: float = 0,
    ymin: float = -1,
    ymax: float = 1,
    **kwargs,
) -> Union[float, np.ndarray]:
    """Bilin compatibility function. implements sc3 bilin function.

    This maps x in 2 linear segments as given by coordinates.

    Parameter:
    ---------
        x (Union[float, np.typing.ArrayLike]): _description_
        xcenter (float): _description_
        xmin (float): _description_
        xmax (float): _description_
        ycenter (float): _description_
        ymin (float): _description_
        ymax (float): _description_

    Returns
    -------
    Union[float, np.ndarray],
        the mapping result
    """
    return interp_spline(x, [xmin, xcenter, xmax], [ymin, ycenter, ymax], **kwargs)


def clip(
    value: Union[float, ArrayLike],
    minimum: float = -float("inf"),
    maximum: float = float("inf"),
) -> Union[float, np.ndarray]:
    """Clips a value to a certain range.

    Parameters
    ----------
    value : float or np.ndarray (ArrayLike)
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
    if isinstance(value, np.ndarray):
        return np.maximum(np.minimum(value, maximum), minimum)
    else:  # ToDo: check if better performance than above numpy code - if not: delete
        if value < minimum:
            return minimum
        if value > maximum:
            return maximum
        return value


def midi_to_cps(midi_note: float) -> float:
    """Convert MIDI note to cycles per second.

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
    """Convert cycles per second to MIDI note.

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


def midi_to_ratio(midi_note: float) -> float:
    """Convert MIDI difference to ratio.

    Parameters
    ----------
    m : float
        MIDI note

    Returns
    -------
    float
        corresponding ratio
    """
    return 2 ** (midi_note / 12.0)


midiratio = midi_to_ratio


def ratio_to_midi(ratio: float) -> float:
    """Convert ratio to MIDI difference.

    Parameters
    ----------
    ratio : float
        ratio (e.g. of frequencies)

    Returns
    -------
    float
        corresponding MIDI difference
    """
    return 12 * np.log2(ratio)


ratiomidi = ratio_to_midi


def cps_to_octave(cps: float) -> float:
    """Convert cycles per second into decimal octaves.

    reference Middle C (i.e. MIDI 60, C_4, 261.626 Hz) yields 4 (octaves).

    Parameters
    ----------
    cps : float
        cycles per second

    Returns
    -------
    float
        octaves relative to Middle C (MIDI 60 C_4, 261.626 HZ)
    """
    return np.log2(cps / 440) + 4.75


cpsoct = cps_to_octave


def octave_to_cps(octave: float) -> float:
    """Convert octaves to cps.

    reference 4.75 yields 440 Hz, i.e. 4 -> freq of Middle C (C4)

    Parameters
    ----------
    octave : float
        octave

    Returns
    -------
    float
        cycles per second
    """
    return 440 * 2 ** (octave - 4.75)


octcps = octave_to_cps


def hz_to_mel(hz):
    """Convert a value in Hertz to Mels.

    Parameters
    ----------
    hz : number of array
        frequencies in Hz, can be an array

    Returns
    -------
    _ : number of array
        mel scale value, same type as the input.
    """
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    """Convert a frequency in Hz to mel using .

    Parameters
    ----------
    mel : number of array
        melody value

    Returns
    -------
    _ : number of array
        cps in Hz, same type as the input.
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


def distort(
    x: Union[float, ArrayLike], threshold: float = 1.0
) -> Union[float, np.ndarray]:
    """Apply value distortion x/(threshold + |x|).

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): input value or array
        threshold (float, optional): defaults to 1.0.

    Returns
    -------
        Union[float, np.ndarray]: distorted value / array
    """
    return x / (threshold + np.abs(x))


def softclip(
    x: Union[float, ArrayLike], threshold: float = 1.0
) -> Union[float, np.ndarray]:
    """Apply softclip distortion to x.

    This yields a perfectly linear region within [-0.5, 0.5],
    outside values computed by (|x| - 0.25) / x

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): input value or array
        threshold (float, optional): defaults to 1.0.

    Returns
    -------
        Union[float, np.ndarray]: softclip distorted value / array
    """
    condition = np.abs(x) > 0.5
    return (np.abs(x) - 0.25) / x * condition + (1 - condition) * x
    # return (np.abs(x) - 0.25)/x if np.abs(x)<0.5 else x


def scurve(
    x: Union[float, ArrayLike], threshold: float = 1.0
) -> Union[float, np.ndarray]:
    """Map value onto an S-curve bound to [0,1].

    Implements v * v * (3-(2*v)) mit v = x.clip(0,1)

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): input value or array
        threshold (float, optional): defaults to 1.0.

    Returns
    -------
        Union[float, np.ndarray]: scurve distorted value / array
    """
    v = clip(x, 0, 1)
    return v**2 * (3 - 2 * v)


def lcurve(
    x: Union[float, ArrayLike], m: float = 0.0, n: float = 1.0, tau: float = 1.0
) -> Union[float, np.ndarray]:
    """Map value or array onto an L-curve.

    Implements (1 + m * exp(-x/tau) + 1) / (1 + n * exp(-x/tau))
    - equal to fermi function with default parameters
    - note that different to the sc3 implementation, tau is inside
    the exp function (unclear tau placement in sc3...)

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): input value or array
        m (float, optional): numerator factor defaults to 0.0.
        n (float, optional): denumerator factor defaults to 1.0.
        tau (float, optional): scale constant, defaults to 1.0.

    Returns
    -------
        Union[float, np.ndarray]: lcurve distorted value / array
    """
    return (1 + m * np.exp(-x / tau)) / (1 + n * np.exp(-x / tau))


def fermi(
    x: Union[float, ArrayLike], tau: float = 1.0, mu: float = 0.0
) -> Union[float, np.ndarray]:
    """Apply fermi function to value or array.

    Implements 1 / (1 + exp(-x/tau))

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): input value or array
        tau (float, optional): scale constant, defaults to 1.0.

    Returns
    -------
        Union[float, np.ndarray]: fermi distorted value / array
    """
    return 1.0 / (1 + np.exp(-(x - mu) / tau))


def normalize(
    x: Union[float, ArrayLike], y1: float = -1.0, y2: float = 1.0
) -> Union[float, np.ndarray]:
    """Normalize array to target range [y1, y2].

    Linear mapping [min(x), max(x)] to [y1, y2]. Use y1 > y2 to change polarity.

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): input value or array
        y1 (float, optional): mapping target for min(x). Defaults to -1.0.
        y2 (float, optional): mapping target for max(x). Defaults to  1.0.

    Returns
    -------
        Union[float, np.ndarray]: normalized / scaled array
    """
    x1, x2 = np.amin(x), np.amax(x)
    return (x - x1) / (x2 - x1) * (y2 - y1) + y1


def wrap(
    x: Union[float, ArrayLike], y1: float = -1.0, y2: float = 1.0
) -> Union[float, np.ndarray]:
    """Wrap array around target range [y1, y2].

    This implements the mapping y1 + x % (y2 - y1).
    The order of y1, y2 is irrelevant.

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): input value or array
        y1 (float, optional): 1st wrap bound. Defaults to -1.0.
        y2 (float, optional): 2nd wrap bound. Defaults to  1.0.

    Returns
    -------
        Union[float, np.ndarray]: wraped array
    """
    return y1 + x % (y2 - y1)


def fold(
    x: Union[float, ArrayLike], y1: float = -1.0, y2: float = 1.0
) -> Union[float, np.ndarray]:
    """Fold array around target range [y1, y2].

    This implements (np.abs((x - y2) % (2 * L) - L) + y1),
    ordering bounds so that y1 < y2.

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): input value or array
        y1 (float, optional): 1st fold bound. Defaults to -1.0.
        y2 (float, optional): 2nd fold bound. Defaults to  1.0.

    Returns
    -------
        Union[float, np.ndarray]: folded array
    """
    if y2 < y1:
        y1, y2 = y2, y1
    L = y2 - y1
    return np.abs((x - y2) % (2 * L) - L) + y1


def remove_dc(
    x: Union[float, ArrayLike],
) -> Union[float, np.ndarray]:
    """Remove DC bias.

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): input value or array

    Returns
    -------
        Union[float, np.ndarray]: normalized / scaled array
    """
    return x - np.mean(x)


def norm_peak(x: Union[float, ArrayLike], peak=1.0):
    """Normalize array so that max(abs(x)) = peak.

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): input value or array
        peak (float): target peak

    Returns
    -------
        Union[float, np.ndarray]: normalized (scaled) array
    """
    peak_of_x = np.max(np.abs(x))
    return (x / peak_of_x) * peak if peak_of_x != 0 else x


def norm_rms(x: Union[float, ArrayLike], rms=1.0):
    """Normalize array so that its RMS value equals `rms`.

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): input value or array
        rms (float): target rms of array

    Returns
    -------
        Union[float, np.ndarray]: rms normalized (scaled) array
    """
    rms_of_x = np.sqrt(np.mean(x**2))
    return (x / rms_of_x) * rms if rms_of_x != 0 else x


def gain(
    x: Union[float, ArrayLike], db: Optional[float] = None, amp: Optional[float] = None
):
    """Apply gain, either as dB (SPL) or scalar factor amp.

    No operation done if neither argument is given, it applies both if both are given.

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): input value or array
        db (None or float): dB SPL = gain 10**(db/20), e.g. -6 dB ~ factor 0.5
        amp (None or float): gain factor

    Returns
    -------
        Union[float, np.ndarray]: scaled (amplified / attenuated) array
    """
    if db:
        sig = x * dbamp(db)
    else:
        sig = x.copy()
    if amp:
        sig *= amp
    return sig


def linspace(
    x: Union[float, int, ArrayLike], x1: float, x2: float, endpoint: bool = True
) -> np.ndarray:
    """Create np.linspace from x1 to x2 in int(x) resp len(x) steps.

    Parameters
    ----------
        x (Union[float, int, ArrayLike]): length or array of which only shape is used
        x1 (float): target interval one side
        x2 (float): target interval other side
        endpoint (bool): forwarded to np.linspace

    Returns
    -------
        Union[float, np.ndarray]: array of length len(x)
            (resp. int(x) if x is float) of numbers between x1 and x2
    """
    if isinstance(x, np.ndarray):
        return np.linspace(x1, x2, x.shape[0], endpoint=endpoint)
    else:
        return np.linspace(x1, x2, int(abs(x)), endpoint=endpoint)


def lin_to_ecdf(
    x: Union[float, ArrayLike], ref_data: np.ndarray, sorted: bool = False
) -> Union[float, np.ndarray]:
    """Map data using empiric cumulative distribution function as mapping.

    This meann feature values are mapped to quantiles.
    if sorted==True: ref_data is regarded as sorted, speeding repeated invocations.

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): value or array to map
        ref_data (np.ndarray): reference data used to create ecdf.
        sorted (bool): whether ref_data is sorted.
            Defaults to False, i.e. by default data will be sorted.

    Returns
    -------
        np.ndarray: resulting mapped data
    """
    if sorted:
        return interp(
            x, ref_data, np.arange(1, len(ref_data) + 1) / float(len(ref_data))
        )
    else:
        return interp(x, *ecdf(ref_data))


def ecdf_to_lin(
    x: Union[float, ArrayLike], ref_data: np.ndarray, sorted: bool = False
) -> Union[float, np.ndarray]:
    """Map data using inverse empiric cumulative distribution function.

    This means that quantiles are mapped back to estimated feature values.
    - if ref_data is omitted, x is used instead
    - if sorted==True: ref_data is regarded as sorted, speeding repeated invocations

    Parameters
    ----------
        x (Union[float, np.typing.ArrayLike]): value or array to map
        ref_data (np.ndarray): reference data used to create ecdf.
        sorted (bool): whether ref_data is sorted.
            Defaults to False, i.e. data will be sorted.

    Returns
    -------
        np.ndarray: resulting mapped data
    """
    if sorted:
        return interp(
            x, np.arange(1, len(ref_data) + 1) / float(len(ref_data)), ref_data
        )
    else:
        xc, yc = ecdf(ref_data)
        return interp(x, yc, xc)


# create subclass of numpy.ndarray


class ChainableArray(np.ndarray):
    """subclass for simpler numpy mapping by chaining syntax."""

    def __new__(cls, input_array, *args, **kwargs):
        """Create new instance."""
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        """Finalize array."""
        if obj is None:
            return

    def to_array(self):
        """Convert self to np.ndarray."""
        return np.array(self)

    def to_asig(self, sr=44100):
        """Convert self to pya.Asig."""
        from pya import Asig

        return Asig(self, sr=sr)

    def plot(self, *args, **kwargs):
        """Plot self via matplotlib."""
        import matplotlib.pyplot as plt

        sr = kwargs.pop("sr", None)
        if sr:
            xs = np.arange(0, self.shape[0]) / sr
            plt.plot(xs, self, *args, **kwargs)
            plt.xlabel("time [s]")
        else:
            xs = kwargs.pop("xs", None)
            if xs is not None:
                plt.plot(xs, self, *args, **kwargs)
            else:
                plt.plot(self, *args, **kwargs)

        return self

    def mapvec(
        self: NDArrayType, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> NDArrayType:
        """Map fn on self by using np.vectorize().

        Parameters
        ----------
            self (NDArrayType): array to map
            fn (Callable[..., Any]): function to call on each element

        Returns
        -------
            NDArrayType: mapping result as ChainableArray
        """
        return np.vectorize(fn)(self, *args, **kwargs)

    def map(
        self: NDArrayType, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> NDArrayType:
        """Apply function fn directly to self; on fail suggest to use mapvec().

        Parameters
        ----------
            self (np.ndarray): array used as input of fn
            fn (Callable[..., Any]): mapping function
            args and kwargs are passed on to fn

        Raises
        ------
            TypeError: if fn fails to operate on np.ndarray as first argument.
            mapvec() is then proposed as alternative.

        Returns
        -------
            ChainableArray: the mapping result as ChainableArray
        """
        try:
            return chain(fn(self, *args, **kwargs))
        except (TypeError, ValueError, AttributeError) as e:
            raise TypeError(
                f"Function {fn.__name__} does not support NumPy arrays directly. "
                "Use .mapvec() instead for np.vectorize elementwise mapping."
            ) from e

    def __getattr__(self, name: str) -> Callable:
        """Dynamically handle method calls."""
        if name.startswith("dynamic_"):
            return lambda *args: f"Called {name} with {args}"
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


def ecdf(
    x: np.ndarray, selection: slice = slice(None, None, None)
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical cumulative distribution function.

    Usable for handcrafted mapping functions such as with using ChainableArray.interp()

    Example 1: (compute once - use many)
    >>> myecdf = ecdf(data); chain(otherdata).interp(*ecdf)

    Example 2: (compute and map in one go)
    >>> chain(otherdata).interp(*ecdf(data))

    Example 3: (use a sparser (more smooth) ecdf mapping)
    >>> chain(otherdata).interp(*ecdf(data, np.s_[::5]))

    Parameters
    ----------
        x (np.ndarray): array
        selection (slice): slice applied to x and y coordinates of the resulting tuple

    Returns
    -------
        tuple[np.ndarray, np.ndarray]: sorted array of x and y coordinates of the ecdf
    """
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs[selection], ys[selection]


def register_numpy_ufunc(fn: np.ufunc, name: Union[None, str] = None) -> None:
    """Register numpy ufunc with one or two ndarray arguments."""
    nin = fn.nin
    if nin == 1:

        def method1(self, *args, **kwargs):
            return ChainableArray(fn(self, *args, **kwargs))

        method = method1

    elif nin == 2:

        def method2(self, other, *args, **kwargs):
            return ChainableArray(fn(self, other, *args, **kwargs))

        method = method2

    else:
        print("warning: np.ufunc fn has nin not in [1,2]")

        def default_method(x):
            return None

        method = default_method

    method.__name__ = fn.__name__ if not name else name
    method.__doc__ = (
        f"{method.__name__} implements numpy.{method.__name__}"
        + f"function for ChainableArray. See help(np.{fn.__name__})"
    )

    setattr(ChainableArray, method.__name__, method)


def register_chain_fn(fn: Callable, name: Union[None, str] = None) -> None:
    """Register function fn for chaining, optionally under given name."""

    def method(self, *args, **kwargs):
        return ChainableArray(fn(self, *args, **kwargs))

    method.__name__ = fn.__name__ if not name else name
    method.__doc__ = (
        f"{method.__name__} implements the {method.__name__}"
        + "operation for ChainableArray. Argument: np.ndarray"
    )

    setattr(ChainableArray, method.__name__, method)


def _list_numpy_ufuncs():
    """Return all numpy ufuncs with 1 or 2 ndarray arguments."""
    ufunc_list = []
    for attr_name in dir(np):  # all attributes in numpy
        attr = getattr(np, attr_name)
        if isinstance(attr, np.ufunc):
            if attr.nin <= 2:
                ufunc_list.append(attr)
            else:
                print(attr, attr.nin)
    return ufunc_list


# create class methods for numpy functions and pyamapping functions
for fn in _list_numpy_ufuncs():  # numpy_mapping_functions:
    name = "abs" if fn.__name__ == "absolute" else None
    register_numpy_ufunc(fn, name)

# register some non-ufunc which nontheless should workd
register_chain_fn(np.angle, "angle")
ChainableArray.magnitude = ChainableArray.abs

pyamapping_functions = [
    amp_to_db,
    bilin,
    clip,
    cps_to_midi,
    cps_to_octave,
    curvelin,
    db_to_amp,
    distort,
    ecdf_to_lin,
    ecdf,
    explin,
    fermi,
    fold,
    gain,
    hz_to_mel,
    interp_spline,
    interp,
    lcurve,
    lin_to_ecdf,
    lincurve,
    linexp,
    linlin,
    linpoly,
    linspace,
    mel_to_hz,
    midi_to_cps,
    midi_to_ratio,
    norm_peak,
    norm_rms,
    normalize,
    octave_to_cps,
    ratio_to_midi,
    remove_dc,
    scurve,
    softclip,
    wrap,
]

for fn in pyamapping_functions:
    register_chain_fn(fn, None)

register_chain_fn(cpsmidi, "cpsmidi")
register_chain_fn(midicps, "midicps")
register_chain_fn(ratiomidi, "ratiomidi")
register_chain_fn(midiratio, "midiratio")
register_chain_fn(cpsoct, "cpsoct")
register_chain_fn(octcps, "octcps")
register_chain_fn(ampdb, "ampdb")
register_chain_fn(dbamp, "dbamp")


def chain(input_array: ArrayLike) -> ChainableArray:
    """Turn np.ndarray into ChainableArray."""
    # ToDo: check difference to input_array.view(ChainableArray)
    return ChainableArray(input_array)
