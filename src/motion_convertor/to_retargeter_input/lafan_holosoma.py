"""
LAFAN → holosoma retargeter input.

holosoma expects .npy of shape (T, 22, 3), Y-up, metres, in LAFAN_DEMO_JOINTS order.
holosoma applies Y→Z-up transform internally.

LAFAN_DEMO_JOINTS order (holosoma's expected order):
  0 Hips, 1 RightUpLeg, 2 RightLeg, 3 RightFoot, 4 RightToeBase,
  5 LeftUpLeg, 6 LeftLeg, 7 LeftFoot, 8 LeftToeBase,
  9 Spine, 10 Spine1, 11 Spine2, 12 Neck, 13 Head,
  14 RightShoulder, 15 RightArm, 16 RightForeArm, 17 RightHand,
  18 LeftShoulder, 19 LeftArm, 20 LeftForeArm, 21 LeftHand

bvhio joint order (depth-first):
  0 Hips, 1 LeftUpLeg, 2 LeftLeg, 3 LeftFoot, 4 LeftToe(=LeftToeBase),
  5 RightUpLeg, 6 RightLeg, 7 RightFoot, 8 RightToe(=RightToeBase),
  9 Spine, 10 Spine1, 11 Spine2, 12 Neck, 13 Head,
  14 LeftShoulder, 15 LeftArm, 16 LeftForeArm, 17 LeftHand,
  18 RightShoulder, 19 RightArm, 20 RightForeArm, 21 RightHand

Requires: bvhio
"""
import numpy as np
from pathlib import Path

# Permutation: holosoma index i ← bvhio index _BVHIO_TO_HOLOSOMA[i]
_BVHIO_TO_HOLOSOMA = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 18, 19, 20, 21, 14, 15, 16, 17]


def _collect_joints(root) -> list:
    """Depth-first traversal of the bvhio joint tree — matches BVH joint order."""
    result = [root]
    for child in root.Children:
        result.extend(_collect_joints(child))
    return result


def convert(bvh_path: Path | str, out_path: Path | str) -> None:
    """
    Convert a LAFAN .bvh file to holosoma retargeter input.

    Produces a .npy file of shape (T, 22, 3), Y-up, metres, joint order
    matching holosoma's LAFAN_DEMO_JOINTS convention.

    Parameters
    ----------
    bvh_path : source .bvh file
    out_path : destination .npy path
    """
    import bvhio

    bvh_path = Path(bvh_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root = bvhio.readAsHierarchy(str(bvh_path))
    joints = _collect_joints(root)
    num_joints = len(joints)
    num_frames = len(root.Keyframes)

    # Build (T, num_joints, 3) in bvhio order, Y-up, centimetres
    all_pos = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    for frame_idx in range(num_frames):
        root.loadPose(frame_idx)
        for j_idx, joint in enumerate(joints):
            pos = joint.PositionWorld
            all_pos[frame_idx, j_idx, 0] = pos.x
            all_pos[frame_idx, j_idx, 1] = pos.y
            all_pos[frame_idx, j_idx, 2] = pos.z

    # cm → metres (holosoma expects metres)
    all_pos /= 100.0

    # Reorder from bvhio order → holosoma LAFAN_DEMO_JOINTS order
    # Keep Y-up — holosoma applies Y→Z-up transform internally
    positions = all_pos[:, _BVHIO_TO_HOLOSOMA, :]  # (T, 22, 3)

    np.save(str(out_path), positions)
