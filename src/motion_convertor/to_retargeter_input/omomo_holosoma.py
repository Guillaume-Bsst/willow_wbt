"""
OMOMO → holosoma retargeter input.

For robot_only tasks: holosoma expects unified .npz format (same as to_unified_input/omomo.py).
For object_interaction tasks: holosoma expects .pt tensors from InterAct 2-step pipeline.

This module handles both task types:
  - robot_only    → delegates to to_unified_input/omomo.py
  - object_interaction → runs InterAct subprocess chain (process_omomo + interact2mimic)
                         via cfg/processing/interact.yaml
"""
from pathlib import Path

from ..to_unified_input.omomo import convert as _omomo_unified


def convert_robot_only(seq_data: dict, out_path: Path | str) -> None:
    """
    Convert an OMOMO robot_only sequence to holosoma retargeter input.

    Produces unified .npz (same as to_unified_input/omomo output).

    Parameters
    ----------
    seq_data : one entry from the joblib-loaded OMOMO pickle dict
    out_path : destination .npz path
    """
    _omomo_unified(seq_data, out_path)


def convert_object_interaction(out_dir: Path | str) -> None:
    """
    Run the full InterAct 2-step pipeline to produce .pt tensors for
    holosoma object_interaction retargeting.

    This function orchestrates two subprocess calls (both in env: interact):
      Step 1: process_omomo.py — raw .p → sequences_canonical/
      Step 2: interact2mimic.py — sequences_canonical/ → .pt tensors

    ⚠️ HARDCODED PATHS: Both InterAct scripts use hardcoded paths relative
    to their working directories. See cfg/processing/interact.yaml and
    src/motion_convertor/TODO.md (Gap 2) for required directory structure.

    Before calling this, ensure the InterAct working directories have the
    expected symlink/file layout:
      InterAct/data/omomo/raw/  → OMOMO raw .p files
      InterAct/models/smplh/    → SMPL-H body models
      InterAct/models/smplx/    → SMPL-X body models

    Parameters
    ----------
    out_dir : directory where the final .pt files should be copied from
              InterAct/simulation/intermimic/InterAct/omomo/
    """
    from .._subprocess import run_entry_point
    from .._config import repo_root
    import shutil

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — process_omomo.py (no CLI args, hardcoded paths)
    run_entry_point("processing", "interact", "process_omomo", args={})

    # Step 2 — interact2mimic.py (--dataset_name omomo)
    run_entry_point("processing", "interact", "interact2mimic", args={"dataset": "omomo"})

    # Copy output .pt files to out_dir
    pt_output_dir = (
        repo_root()
        / "src/motion_convertor/third_party/InterAct/simulation/intermimic/InterAct/omomo"
    )
    if pt_output_dir.exists():
        for pt_file in pt_output_dir.glob("*.pt"):
            shutil.copy2(pt_file, out_dir / pt_file.name)
