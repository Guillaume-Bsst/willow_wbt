# holosoma_retargeting — Format Specification

## Overview

| Property | Value |
|----------|-------|
| Repo | [amazon-far/holosoma](https://github.com/amazon-far/holosoma) — `src/holosoma_retargeting/` |
| Supported datasets | LAFAN, SFU, OMOMO robot_only + object_interaction |
| Object interaction | ✅ Supported |
| Input formats | `.npy` (LAFAN), `.npz` unified (SFU, OMOMO) |
| Output format | `.npz` with full simulation state (`qpos`, body positions, velocities) |
| Supported robots | Unitree G1 (29-DOF, 27-DOF), Booster T1 (23-DOF), Unitree H1 (partial) |

---

## Input Formats

### `lafan` format — `.npy`

- Shape: `(T, 22, 3)` — global joint positions, Y-up
- Units: metres
- **Coordinate transform applied internally**: Y→Z-up swap
- **Spine correction**: `Spine1` z-coordinate adjusted by `-0.06 m`
- **Scale factor**: `1.27 / 1.7 = 0.747` (fixed, based on LAFAN height distribution)
- 22 joints in `LAFAN_DEMO_JOINTS` order (Right-first): Hips, RightUpLeg, RightLeg, RightFoot, RightToeBase, LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase, Spine, Spine1, Spine2, Neck, Head, RightShoulder, RightArm, RightForeArm, RightHand, LeftShoulder, LeftArm, LeftForeArm, LeftHand
- Note: bvhio returns joints in Left-first order — must reorder to match `LAFAN_DEMO_JOINTS`

### `smplx` format — `.npz` (unified format)

This is the **primary input format** for holosoma_retargeting. It matches the pipeline unified format exactly.

| Key | Shape | Description |
|-----|-------|-------------|
| `global_joint_positions` | `(T, 22, 3)` | Global joint positions, Z-up, metres |
| `height` | scalar | Subject height in metres |
| `object_poses` *(optional)* | `(T, 7)` | `[qw, qx, qy, qz, x, y, z]` — object interaction only |

- Coordinate system: **Z-up**
- 22 SMPL-X joints — see [datasets/SFU.md](../raw_datasets/SFU.md#smpl-x-joint-convention-22-body-joints)
- **Scale factor**: `ROBOT_HEIGHT / height` (dynamic, per-sequence)
- FPS: expects 30 Hz (SFU raw is 120 Hz → downsample ×4 before passing)

### `mocap` format — `.npy`

- Shape: `(T, 51, 3)` — global joint positions
- Default human height: **1.78 m**
- Downsampled by factor 4 before retargeting
- Used for climbing sequences only

### `smplh` format — `.pt` (PyTorch)

- InterMimic/InterAct format — used for OMOMO with object interaction
- Contains 45 SMPL-H body joints + object poses
- Scale factor: per-task based on subject height

---

## Output Format

`.npz` file with full robot simulation state:

| Key | Shape | Description |
|-----|-------|-------------|
| `fps` | `[float]` | Output FPS |
| `joint_pos` | `(T, N_dof)` | Robot joint positions (radians) |
| `joint_vel` | `(T, N_dof)` | Robot joint velocities (rad/s) |
| `body_pos_w` | `(T, N_bodies, 3)` | Body CoM positions, world frame, metres |
| `body_quat_w` | `(T, N_bodies, 4)` | Body quaternions, world frame, **wxyz** |
| `body_lin_vel_w` | `(T, N_bodies, 3)` | Body linear velocities (m/s) |
| `body_ang_vel_w` | `(T, N_bodies, 3)` | Body angular velocities (rad/s) |
| `joint_names` | list | Joint names, same order as `joint_pos` |
| `body_names` | list | Body names, same order as `body_*` arrays |

For **object_interaction** tasks, additionally:

| Key | Shape | Description |
|-----|-------|-------------|
| `object_pos_w` | `(T, 3)` | Object position, world frame |
| `object_quat_w` | `(T, 4)` | Object quaternion, **wxyz** |
| `object_lin_vel_w` | `(T, 3)` | Object linear velocity |
| `object_ang_vel_w` | `(T, 3)` | Object angular velocity |

### Training input (form A — native bridge)

This output is directly consumable as **form A** by the holosoma trainer. The native bridge (`convert_data_format_mj.py`) reads `qpos` from this file and converts it to the 50 Hz training format internally.

| Key | Shape | Description |
|-----|-------|-------------|
| `qpos` | `(T, 7 + N_dof)` | `[x, y, z, qw, qx, qy, qz, dof_0, ..., dof_N]` |

- Base 7: position (3) + quaternion wxyz (4)
- DOF indices 7:36 for G1 29-DOF

→ Full trainer spec: [training/holosoma.md](../training/holosoma.md)

---

## Quaternion Convention

**All quaternions in holosoma_retargeting output use wxyz format.**

```python
# wxyz (holosoma) → xyzw (some other libs)
q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
```

---

## Supported Robots

| Robot | DOF | URDF | Height | Datasets |
|-------|-----|------|--------|---------|
| `unitree_g1` 29-DOF (default) | 29 | `g1_29dof_retargeting.urdf` | 1.32 m | lafan, smplx, smplh, mocap |
| `unitree_g1` 27-DOF | 27 | `g1_27dof_retargeting.urdf` | 1.32 m | lafan, smplx, smplh, mocap |
| `booster_t1` | 23 | `t1_23dof.urdf` | 1.20 m | lafan, smplh, mocap |
| `unitree_h1` | 19 | `h1_19dof.urdf` | — | ⚠️ model exists but not fully registered |

Select DOF mode with `--robot-config.robot-dof 27` or `29`.

→ Detailed spec for G1: [robots/G1.md](../robots/G1.md)

---

## Assets Required for object_interaction

- Object URDF: `01_retargeted_motions/assets/OMOMO/objects/{object_name}/{object_name}.urdf`
- Object mesh: `01_retargeted_motions/assets/OMOMO/objects/{object_name}/{object_name}.obj`
- Generated from `captured_objects/` by `prep_omomo_for_rt.py`

---

## Converter responsibilities (`motion_convertor`)

| Direction | Operation |
|-----------|-----------|
| LAFAN → holosoma input | BVH → global joint positions `.npy` (T,23,3), Y-up, metres |
| SFU → holosoma input | SMPL-X FK → unified `.npz` (T,22,3) + height, downsample 120→30 Hz |
| OMOMO → holosoma input | SMPL-H FK → unified `.npz` (T,22,3) + height + object_poses (T,7) |
| holosoma output → unified | Extract `body_pos_w[pelvis_idx]` + joint positions → reformat as `global_joint_positions (T,22,3)` |
