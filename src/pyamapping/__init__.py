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


from pyamapping.mappings import (
    amp_to_db,
    ampdb,
    clip,
    cps_to_midi,
    cpsmidi,
    db_to_amp,
    dbamp,
    hz_to_mel,
    linlin,
    mel_to_hz,
    midi_to_cps,
    midicps,
)

__all__ = [
    "amp_to_db",
    "ampdb",
    "clip",
    "cps_to_midi",
    "cpsmidi",
    "db_to_amp",
    "dbamp",
    "hz_to_mel",
    "linlin",
    "mel_to_hz",
    "midi_to_cps",
    "midicps",
]
