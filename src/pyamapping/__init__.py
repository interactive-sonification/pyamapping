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


from pyamapping.mappings import (  # some synonyms; the class and helper functions
    ChainableArray,
    amp_to_db,
    ampdb,
    bilin,
    chain,
    clip,
    cps_to_midi,
    cps_to_octave,
    cpsmidi,
    curvelin,
    db_to_amp,
    dbamp,
    distort,
    ecdf,
    ecdf_to_lin,
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
    linlog,
    linpoly,
    linspace,
    mel_to_hz,
    midi_to_cps,
    midi_to_ratio,
    midicps,
    norm_peak,
    norm_rms,
    normalize,
    octave_to_cps,
    ratio_to_midi,
    register_chain_fn,
    remove_dc,
    scurve,
    softclip,
    wrap,
)

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
    "linlog",
    "linpoly",
    "linspace",
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
