"""
LAFAN (BVH) → unified format.

Input : .bvh file, 22 joints, Y-up, centimetres, 30 Hz.
Output: unified .npz at 30 Hz, global_joint_positions (T,22,3) Z-up metres.

LAFAN 22-joint BVH order (depth-first):
  0 Hips, 1 LeftUpLeg, 2 LeftLeg, 3 LeftFoot, 4 LeftToe,
  5 RightUpLeg, 6 RightLeg, 7 RightFoot, 8 RightToe,
  9 Spine, 10 Spine1, 11 Spine2, 12 Neck, 13 Head,
  14 LeftShoulder, 15 LeftArm, 16 LeftForeArm, 17 LeftHand,
  18 RightShoulder, 19 RightArm, 20 RightForeArm, 21 RightHand

Reordered to SMPL-X 22-joint convention:
  SMPL-X idx ← LAFAN idx
  0  Pelvis    ← 0  Hips
  1  L_Hip     ← 1  LeftUpLeg
  2  R_Hip     ← 5  RightUpLeg
  3  Spine1    ← 9  Spine
  4  L_Knee    ← 2  LeftLeg
  5  R_Knee    ← 6  RightLeg
  6  Spine2    ← 10 Spine1
  7  L_Ankle   ← 3  LeftFoot
  8  R_Ankle   ← 7  RightFoot
  9  Spine3    ← 11 Spine2
  10 L_Foot    ← 4  LeftToe
  11 R_Foot    ← 8  RightToe
  12 Neck      ← 12 Neck
  13 L_Collar  ← 14 LeftShoulder
  14 R_Collar  ← 18 RightShoulder
  15 Head      ← 13 Head
  16 L_Shoulder← 15 LeftArm
  17 R_Shoulder← 19 RightArm
  18 L_Elbow   ← 16 LeftForeArm
  19 R_Elbow   ← 20 RightForeArm
  20 L_Wrist   ← 17 LeftHand
  21 R_Wrist   ← 21 RightHand

Requires: bvhio
"""
import numpy as np
from pathlib import Path

from ..unified import save_unified

# Y-up → Z-up rotation matrix
_R_Y2Z = np.array([[1, 0, 0],
                   [0, 0, -1],
                   [0, 1, 0]], dtype=np.float32)

# Permutation: SMPL-X index i ← LAFAN index _LAFAN_TO_SMPLX[i]
_LAFAN_TO_SMPLX = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 18, 13, 15, 19, 16, 20, 17, 21]


def _collect_joints(root) -> list:
    """Depth-first traversal of the bvhio joint tree — matches BVH joint order."""
    result = [root]
    for child in root.Children:
        result.extend(_collect_joints(child))
    return result


def convert(bvh_path: Path | str, out_path: Path | str) -> None:
    """
    Convert a LAFAN .bvh file to unified format.

    Parameters
    ----------
    bvh_path : source .bvh file
    out_path : destination unified .npz path
    """
    import bvhio

    bvh_path = Path(bvh_path)
    out_path = Path(out_path)

    root = bvhio.readAsHierarchy(str(bvh_path))
    joints = _collect_joints(root)
    num_joints = len(joints)
    num_frames = len(root.Keyframes)

    # Build (T, num_joints, 3) in LAFAN BVH order, Y-up, centimetres
    all_pos = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    for frame_idx in range(num_frames):
        root.loadPose(frame_idx)
        for j_idx, joint in enumerate(joints):
            pos = joint.PositionWorld  # glm.vec3, Y-up cm
            all_pos[frame_idx, j_idx, 0] = pos.x
            all_pos[frame_idx, j_idx, 1] = pos.y
            all_pos[frame_idx, j_idx, 2] = pos.z

    # cm → metres
    all_pos /= 100.0

    # Y-up → Z-up: apply rotation to each position vector
    # R @ pos  (broadcast over T and J)
    all_pos = all_pos @ _R_Y2Z.T  # (T, J, 3)

    # Reorder from LAFAN order → SMPL-X order
    positions = all_pos[:, _LAFAN_TO_SMPLX, :]  # (T, 22, 3)

    save_unified(out_path, positions, height=1.75)
