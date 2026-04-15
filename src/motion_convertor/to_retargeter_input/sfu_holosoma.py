"""
SFU → holosoma retargeter input.

holosoma retargeter expects unified .npz format as input — same as to_unified_input/sfu.py.
Delegates directly to that module.
"""
from pathlib import Path

from ..to_unified_input.sfu import convert as _sfu_convert


def convert(npz_path: Path | str, out_path: Path | str) -> None:
    """
    Convert a SFU .npz to holosoma retargeter input (unified format).

    Parameters
    ----------
    npz_path : source SFU .npz file
    out_path : destination path (should end with _input_raw.npz)
    """
    _sfu_convert(npz_path, out_path)
