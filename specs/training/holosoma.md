# holosoma — Trainer Specification

## Overview

| Property | Value |
|----------|-------|
| Repo | `modules/third_party/holosoma` |
| Algorithm | PPO (Proximal Policy Optimization) |
| Task | Whole-Body Tracking (WBT) |
| Supported robots | G1 (29 DOF), T1 (23 DOF) |
| Supported simulators | IsaacGym, IsaacSim, MuJoCo (classic + MJWarp) |
| Control frequency | 50 Hz (physics 200 Hz, decimation 4) |
| Max episode length | 10.0 s |

---

## Training Input Format

holosoma accepts **two input forms**:

| Form | Format | Who produces it |
|------|--------|-----------------|
| **A — retargeter-native** | `qpos (T, 7+N)` at 30 Hz | holosoma_retargeting output — holosoma runs its own bridge internally |
| **B — pre-converted** | `body_pos_w`, `joint_pos`, etc. at 50 Hz | output of `convert_data_format_mj.py` run externally |

> Our `src/motion_convertor/to_trainer_input/` does **not** wrap the native bridge. It simply passes the unified format through — holosoma handles the conversion internally when given form A.

### Form A — retargeter-native keys

| Key | Shape | Description |
|-----|-------|-------------|
| `qpos` | `(T, 7 + N_dof)` | `[base_pos(3) \| base_quat(4) \| dof_pos(N)]` — 30 Hz, wxyz |

### Form B — pre-converted keys

| Key | Shape | Description |
|-----|-------|-------------|
| `fps` | scalar | Playback FPS — 50 Hz |
| `joint_pos` | `(T, J)` | Robot joint positions |
| `joint_vel` | `(T, J)` | Robot joint velocities (finite differences) |
| `body_pos_w` | `(T, B, 3)` | World-frame body positions |
| `body_quat_w` | `(T, B, 4)` | World-frame body quaternions — **xyzw** |
| `body_lin_vel_w` | `(T, B, 3)` | World-frame body linear velocities |
| `body_ang_vel_w` | `(T, B, 3)` | World-frame body angular velocities |
| `joint_names` | list[str] | Joint names in order |
| `body_names` | list[str] | All MuJoCo body names in order |

Optional object keys (both forms): `object_pos_w (T,3)`, `object_quat_w (T,4)`, `object_lin_vel_w (T,3)`, `object_ang_vel_w (T,3)`

### ⚠️ Quaternion convention
Form B uses **xyzw** (MuJoCo output convention). Form A uses **wxyz** (holosoma_retargeting convention).

---

## Native bridge (`convert_data_format_mj.py`)

Tool provided natively by holosoma to convert form A → form B.  
Source: `modules/third_party/holosoma/src/holosoma_retargeting/data_conversion/convert_data_format_mj.py`

This bridge is **internal to holosoma** — our pipeline does not call it directly.

### Processing

1. Reads `qpos` from holosoma_retargeting output
2. Runs forward kinematics in MuJoCo to compute `body_pos_w`, `body_quat_w`
3. Computes velocities via finite differences
4. Interpolates to 50 Hz (SLERP for quaternions, LERP for positions)

### Output

Form B: training-ready `.npz` at 50 Hz with all keys listed in [Form B keys](#form-b--pre-converted-keys).

---

## Observation Space (G1 WBT)

### Actor observation — 97 dimensions

| Component | Dim | Description |
|-----------|-----|-------------|
| `motion_command` | 1 | Scalar motion command |
| `motion_ref_ori_b` | 6 | Reference orientation in base frame (2×3 rotation matrix) |
| `base_ang_vel` | 3 | Base angular velocity in base frame |
| `dof_pos` | 29 | Joint positions relative to defaults |
| `dof_vel` | 29 | Joint velocities |
| `actions` | 29 | Previous actions |
| **Total** | **97** | |

### Critic observation — 229 dimensions (no object)

| Component | Dim | Description |
|-----------|-----|-------------|
| `motion_command` | 1 | Scalar motion command |
| `motion_ref_pos_b` | 3 | Reference root position in base frame |
| `motion_ref_ori_b` | 6 | Reference orientation in base frame |
| `robot_body_pos_b` | 42 | 14 bodies × 3D position |
| `robot_body_ori_b` | 84 | 14 bodies × 6D orientation (rotation matrix) |
| `base_lin_vel` | 3 | Base linear velocity |
| `base_ang_vel` | 3 | Base angular velocity |
| `dof_pos` | 29 | Joint positions |
| `dof_vel` | 29 | Joint velocities |
| `actions` | 29 | Previous actions |
| **Total** | **229** | |

### Critic observation — 241 dimensions (with object)

Adds 12 dims to the 229-dim critic: `obj_pos_b (3)`, `obj_ori_b (6)`, `obj_lin_vel_b (3)`.

---

## Action Space (G1 WBT)

| Property | Value |
|----------|-------|
| Dimensions | 29 |
| Type | Joint position targets |
| Control | PD controllers with per-joint Kp/Kd gains |
| Action scale | 0.25 (default, configurable) |

---

## Tracked Bodies — G1 (14 bodies)

| Index | Body name |
|-------|-----------|
| 0 | `pelvis` |
| 1 | `left_hip_roll_link` |
| 2 | `left_knee_link` |
| 3 | `left_ankle_roll_link` |
| 4 | `right_hip_roll_link` |
| 5 | `right_knee_link` |
| 6 | `right_ankle_roll_link` |
| 7 | `torso_link` (reference body) |
| 8 | `left_shoulder_roll_link` |
| 9 | `left_elbow_link` |
| 10 | `left_wrist_yaw_link` |
| 11 | `right_shoulder_roll_link` |
| 12 | `right_elbow_link` |
| 13 | `right_wrist_yaw_link` |

---

## G1 Joint Order (29 DOF)

| Index | Joint name |
|-------|-----------|
| 0 | `left_hip_pitch` |
| 1 | `left_hip_roll` |
| 2 | `left_hip_yaw` |
| 3 | `left_knee` |
| 4 | `left_ankle_pitch` |
| 5 | `left_ankle_roll` |
| 6 | `right_hip_pitch` |
| 7 | `right_hip_roll` |
| 8 | `right_hip_yaw` |
| 9 | `right_knee` |
| 10 | `right_ankle_pitch` |
| 11 | `right_ankle_roll` |
| 12 | `waist_yaw` |
| 13 | `waist_roll` |
| 14 | `waist_pitch` |
| 15 | `left_shoulder_pitch` |
| 16 | `left_shoulder_roll` |
| 17 | `left_shoulder_yaw` |
| 18 | `left_elbow` |
| 19 | `left_wrist_roll` |
| 20 | `left_wrist_pitch` |
| 21 | `left_wrist_yaw` |
| 22 | `right_shoulder_pitch` |
| 23 | `right_shoulder_roll` |
| 24 | `right_shoulder_yaw` |
| 25 | `right_elbow` |
| 26 | `right_wrist_roll` |
| 27 | `right_wrist_pitch` |
| 28 | `right_wrist_yaw` |

---

## Training Output

### PyTorch checkpoint (`.pt`)

| Key | Description |
|-----|-------------|
| `actor_model_state_dict` | Actor network weights |
| `critic_model_state_dict` | Critic network weights |
| `actor_optimizer_state_dict` | Adam optimizer state |
| `critic_optimizer_state_dict` | Adam optimizer state |
| `actor_obs_normalizer_state_dict` | Running mean/var for actor obs |
| `critic_obs_normalizer_state_dict` | Running mean/var for critic obs |
| `iter` | Training iteration at save time |
| `infos` | Training metrics |
| `experiment_config` | Full `ExperimentConfig` serialized |
| `wandb_run_path` | W&B run path for logging |

### ONNX export

**Policy-only** (`opset=13`):

| Tensor | Shape | Description |
|--------|-------|-------------|
| Input: `actor_obs` | `(batch, 97)` | Actor observation |
| Output: `action` | `(batch, 29)` | Joint position targets |

Metadata embedded: `dof_names`, `kp`, `kd`, `action_scale`, `command_ranges`, `robot_urdf`

**Motion + Policy** (for inference with reference replay):

| Tensor | Shape | Description |
|--------|-------|-------------|
| Input: `obs` | `(batch, 97)` | Actor observation |
| Input: `time_step` | `(batch, 1)` | Current timestep |
| Output: `actions` | `(batch, 29)` | Joint position targets |
| Output: `joint_pos` | `(batch, T, 29)` | Reference joint positions |
| Output: `joint_vel` | `(batch, T, 29)` | Reference joint velocities |
| Output: `ref_pos_xyz` | `(batch, 3)` | Reference base position |
| Output: `ref_quat_xyzw` | `(batch, 4)` | Reference base quaternion (xyzw) |

---

## Training Hyperparameters (G1 WBT PPO)

| Parameter | Value |
|-----------|-------|
| `num_envs` | 4096 |
| `num_learning_iterations` | 30 000 |
| `num_learning_epochs` | 5 |
| `save_interval` | 4 000 |
| `entropy_coef` | 0.005 |
| `init_noise_std` | 1.0 |
| `actor_learning_rate` | 1e-3 |
| `critic_learning_rate` | 1e-3 |
| `empirical_normalization` | True |
| `weight_decay` | 0.0 |

---

## Simulator Details

| Simulator | Backend | Notes |
|-----------|---------|-------|
| IsaacGym | GPU | Classic Isaac RL |
| IsaacSim | GPU | Newer Nvidia Isaac |
| MuJoCo classic | CPU | Single env, debug |
| MuJoCo MJWarp | GPU | Batched, fastest iteration |

Physics FPS: **200 Hz** — control decimation: **4** → control at **50 Hz**

---

## Key Source Files

| File | Role |
|------|------|
| `src/holosoma/config_values/wbt/g1/observation.py` | Actor/critic obs dimensions |
| `src/holosoma/config_values/wbt/g1/command.py` | Tracked body list |
| `src/holosoma/config_values/wbt/g1/experiment.py` | PPO hyperparameters |
| `src/holosoma/agents/ppo/ppo.py` | Training loop, checkpoint save |
| `src/holosoma/utils/inference_helpers.py` | ONNX export |
| `src/holosoma_retargeting/data_conversion/convert_data_format_mj.py` | Policy convertor |
