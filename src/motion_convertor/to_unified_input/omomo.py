"""
OMOMO → unified format.

Input : per-sequence dict from .p pickle file (loaded via joblib).
Output: unified .npz at 30 Hz, global_joint_positions (T,22,3) Z-up metres,
        plus object_poses (T,7) [qw,qx,qy,qz,x,y,z] if object present.

OMOMO raw pickle keys (per-sequence, keyed by sequence index):
    seq_name      : str
    root_orient   : (T, 3)     root axis-angle
    pose_body     : (T, 63)    body pose axis-angle (21 joints × 3)
    trans         : (T, 3)     root translation, metres
    betas         : (1, 16)    SMPL-H shape params — use betas[0]
    gender        : str
    obj_rot       : (T, 3, 3)  object rotation matrix
    obj_trans     : (T, 3, 1)  object translation — use obj_trans[:, :, 0]

SMPL-H forward kinematics via human_body_prior.BodyModel returns Jtr with
52 joints total (22 body + 15 per hand = 52). The joints24 subset is:
  joints 0-21 (body) + joint 37 (L_Hand proxy) + joint 52... Actually:
  joints24 = first 22 body joints + left index finger tip (37) + right equiv
  → In practice we take .Jtr[:, :22, :] for the 22 SMPL-X-compatible joints.
  Joints 22 (L_Hand) and 23 (R_Hand) are end sites not in .Jtr directly;
  SMPL-H body model gives exactly joints 0-21 in SMPL order for body joints.

Requires: human_body_prior (from InterAct), torch, scipy
"""
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

from ..unified import save_unified
from .._config import body_model_path


def _run_smplh_fk(root_orient, pose_body, trans, betas, gender, model_dir: Path):
    """
    Run SMPL-H forward kinematics. Returns joint positions (T, 52, 3).
    """
    from human_body_prior.body_model.body_model import BodyModel

    gender = str(gender).lower()
    model_path = model_dir / gender / "model.npz"
    T = root_orient.shape[0]

    model = BodyModel(
        bm_fname=str(model_path),
        num_betas=16,
    )

    # Pad pose_body to full SMPL-H pose (156 total: root(3) + body(63) + hands(90))
    pose_hand = np.zeros((T, 90), dtype=np.float32)
    poses_full = np.concatenate([root_orient, pose_body, pose_hand], axis=1)  # (T, 156)

    batch_size = 512
    all_joints = []
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        output = model(
            root_orient=torch.from_numpy(root_orient[start:end]).float(),
            pose_body=torch.from_numpy(pose_body[start:end]).float(),
            pose_hand=torch.from_numpy(pose_hand[start:end]).float(),
            betas=torch.from_numpy(betas).float().unsqueeze(0).expand(end - start, -1),
            trans=torch.from_numpy(trans[start:end]).float(),
        )
        all_joints.append(output.Jtr.detach().numpy())

    return np.concatenate(all_joints, axis=0)  # (T, 52, 3)


def convert(seq_data: dict, out_path: Path | str) -> None:
    """
    Convert a single OMOMO sequence dict to unified format.

    Parameters
    ----------
    seq_data : one entry from the joblib-loaded OMOMO pickle dict.
               Expected keys: root_orient, pose_body, trans, betas, gender,
               obj_rot, obj_trans (all optional for robot_only tasks).
    out_path : destination unified .npz path
    """
    out_path = Path(out_path)

    root_orient = seq_data["root_orient"].astype(np.float32)   # (T, 3)
    pose_body = seq_data["pose_body"].astype(np.float32)        # (T, 63)
    trans = seq_data["trans"].astype(np.float32)                # (T, 3)
    betas = seq_data["betas"][0].astype(np.float32)             # (1,16) → (16,)
    gender = str(seq_data["gender"])

    model_dir = body_model_path("OMOMO")
    joints52 = _run_smplh_fk(root_orient, pose_body, trans, betas, gender, model_dir)

    # First 22 joints are the SMPL-X-compatible body joints (Z-up already)
    # SMPL-H joints 0-21 match SMPL-X convention exactly
    joints22 = joints52[:, :22, :].astype(np.float32)  # (T, 22, 3)

    # Height: T-pose with betas only
    from human_body_prior.body_model.body_model import BodyModel
    tpose_model_path = model_dir / gender.lower() / "model.npz"
    tpose_model = BodyModel(bm_fname=str(tpose_model_path), num_betas=16)
    tpose_output = tpose_model(
        betas=torch.from_numpy(betas).float().unsqueeze(0),
    )
    tpose_joints = tpose_output.Jtr[0, :22, :].detach().numpy()
    height = float(tpose_joints[:, 2].max() - tpose_joints[:, 2].min())

    # Object poses (only present in object_interaction sequences)
    object_poses = None
    if "obj_rot" in seq_data and seq_data["obj_rot"] is not None:
        obj_rot = seq_data["obj_rot"].astype(np.float32)         # (T, 3, 3)
        obj_trans = seq_data["obj_trans"][:, :, 0].astype(np.float32)  # (T, 3, 1) → (T, 3)

        quat_xyzw = Rotation.from_matrix(obj_rot).as_quat()      # (T, 4) xyzw
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)  # (T, 4) wxyz
        object_poses = np.hstack([quat_wxyz, obj_trans])          # (T, 7)

    save_unified(out_path, joints22, height, object_poses)
