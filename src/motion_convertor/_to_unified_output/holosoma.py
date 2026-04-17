"""
holosoma retargeter output → unified format.

Input : holosoma .npz with keys body_pos_w (T,B,3), body_names list, etc.
Output: unified .npz with global_joint_positions (T,22,3) Z-up metres.

holosoma body_names are MuJoCo link names. We map them to SMPL-X joint indices
using the smplx→G1 mapping from holosoma's config_types/data_type.py.

For the 7 SMPL-X joints without a G1 body counterpart
(Spine1=3, Spine2=6, Spine3=9, Neck=12, L_Collar=13, R_Collar=14, Head=15),
we use their nearest tracked parent body as a proxy:
  Spine1  → Pelvis (pelvis_contour_link)
  Spine2  → Pelvis
  Spine3  → Pelvis
  Neck    → Pelvis
  Head    → Pelvis
  L_Collar→ L_Shoulder (left_shoulder_roll_link)
  R_Collar→ R_Shoulder (right_shoulder_roll_link)
"""
import numpy as np
from pathlib import Path

from ..unified import save_unified

# SMPL-X joint index → G1 body name that represents it
# 7 joints without G1 equivalent use proxy (parent body)
_SMPLX_TO_G1 = {
    0:  "pelvis_contour_link",            # Pelvis
    1:  "left_hip_pitch_link",            # L_Hip
    2:  "right_hip_pitch_link",           # R_Hip
    3:  "pelvis_contour_link",            # Spine1 → proxy: Pelvis
    4:  "left_knee_link",                 # L_Knee
    5:  "right_knee_link",                # R_Knee
    6:  "pelvis_contour_link",            # Spine2 → proxy: Pelvis
    7:  "left_ankle_intermediate_1_link", # L_Ankle
    8:  "right_ankle_intermediate_1_link",# R_Ankle
    9:  "pelvis_contour_link",            # Spine3 → proxy: Pelvis
    10: "left_ankle_roll_sphere_5_link",  # L_Foot
    11: "right_ankle_roll_sphere_5_link", # R_Foot
    12: "pelvis_contour_link",            # Neck → proxy: Pelvis
    13: "left_shoulder_roll_link",        # L_Collar → proxy: L_Shoulder
    14: "right_shoulder_roll_link",       # R_Collar → proxy: R_Shoulder
    15: "pelvis_contour_link",            # Head → proxy: Pelvis
    16: "left_shoulder_roll_link",        # L_Shoulder
    17: "right_shoulder_roll_link",       # R_Shoulder
    18: "left_elbow_link",               # L_Elbow
    19: "right_elbow_link",              # R_Elbow
    20: "left_rubber_hand_link",         # L_Wrist
    21: "right_rubber_hand_link",        # R_Wrist
}


def convert(npz_path: Path | str, out_path: Path | str, height: float) -> None:
    """
    Convert holosoma retargeter output to unified format.

    Parameters
    ----------
    npz_path : path to holosoma output .npz
    out_path : destination unified .npz path
    height   : subject height in metres (carry over from input_unified.npz)
    """
    npz_path = Path(npz_path)
    out_path = Path(out_path)

    data = np.load(npz_path, allow_pickle=True)
    body_pos_w = data["body_pos_w"]       # (T, B, 3)
    body_names = list(data["body_names"]) # list of str

    # Build name → index lookup
    name_to_idx = {name: i for i, name in enumerate(body_names)}

    T = body_pos_w.shape[0]
    positions = np.zeros((T, 22, 3), dtype=np.float32)

    for smplx_idx, g1_body in _SMPLX_TO_G1.items():
        if g1_body not in name_to_idx:
            raise ValueError(
                f"Body '{g1_body}' not found in holosoma output. "
                f"Available: {body_names}"
            )
        b_idx = name_to_idx[g1_body]
        positions[:, smplx_idx, :] = body_pos_w[:, b_idx, :]

    # Object poses (object_interaction only)
    object_poses = None
    if "object_pos_w" in data and data["object_pos_w"] is not None:
        object_pos = data["object_pos_w"]    # (T, 3)
        object_quat = data["object_quat_w"]  # (T, 4) wxyz
        object_poses = np.hstack([object_quat, object_pos]).astype(np.float32)  # (T, 7)

    save_unified(out_path, positions, height, object_poses)
