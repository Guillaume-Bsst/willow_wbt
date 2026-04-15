"""
holosoma retargeter output → holosoma trainer input (form B).

holosoma raw retargeter output is already form B — it contains body_pos_w,
joint_pos, velocities, etc. at the retargeter's output FPS.

This module is a no-op: it copies (or symlinks) the raw output to the
trainer_input path so train.py can find it without searching the run folder.
"""
import shutil
from pathlib import Path


def convert(output_raw_path: Path | str, out_path: Path | str) -> None:
    """
    Copy holosoma retargeter output to trainer input location.

    Parameters
    ----------
    output_raw_path : holosoma retargeter output .npz
    out_path        : destination path (trainer_input.npz)
    """
    output_raw_path = Path(output_raw_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_raw_path, out_path)
