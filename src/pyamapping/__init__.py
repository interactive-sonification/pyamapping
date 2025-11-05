import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from typing import Callable, Union

import numpy as np

from pyamapping.chainable_array import ChainableArray, chain
from pyamapping.mappings import (  # some synonyms; the class and helper functions
    amp_to_db,
    ampdb,
    bilin,
    clip,
    cps_to_midi,
    cps_to_octave,
    cpsmidi,
    cpsoct,
    curvelin,
    db_to_amp,
    dbamp,
    distort,
    ecdf,
    ecdf_to_lin,
    explin,
    fermi,
    fold,
    gain,
    hz_to_mel,
    interp,
    interp_spline,
    lcurve,
    lin_to_ecdf,
    lincurve,
    linexp,
    linlin,
    linpoly,
    mel_to_hz,
    midi_to_cps,
    midi_to_ratio,
    midicps,
    midiratio,
    norm_peak,
    norm_rms,
    normalize,
    octave_to_cps,
    octcps,
    ratio_to_midi,
    ratiomidi,
    remove_dc,
    scurve,
    softclip,
    wrap,
)

# defined here: register_chain_fn,


__all__ = [
    "amp_to_db",
    "bilin",
    "clip",
    "cps_to_midi",
    "cps_to_octave",
    "curvelin",
    "db_to_amp",
    "distort",
    "ecdf_to_lin",
    "ecdf",
    "explin",
    "fermi",
    "fold",
    "gain",
    "interp_spline",
    "interp",
    "lcurve",
    "lin_to_ecdf",
    "lincurve",
    "linexp",
    "linlin",
    "linpoly",
    "midi_to_cps",
    "midi_to_ratio",
    "norm_peak",
    "norm_rms",
    "normalize",
    "octave_to_cps",
    "ratio_to_midi",
    "remove_dc",
    "scurve",
    "softclip",
    "wrap",
    "mel_to_hz",
    "hz_to_mel",
    # some synonyms
    "ampdb",
    "cpsmidi",
    "dbamp",
    "midicps",
    # class and helper functions
    "ChainableArray",
    "chain",
    "register_chain_fn",
]  # type: ignore


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
