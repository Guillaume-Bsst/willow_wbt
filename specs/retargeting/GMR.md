# GMR — Format Specification

## Overview

| Property | Value |
|----------|-------|
| Repo | [YanjieZe/GMR](https://github.com/YanjieZe/GMR) |
| Supported datasets | LAFAN, SFU, OMOMO robot_only |
| Object interaction | ❌ Not supported |
| Input format | `.bvh` (LAFAN) or `.npz` SMPL-X (SFU, OMOMO) |
| Output format | Pickle dict with robot joint angles |
| Supported robots | 17 robots — see table below |

---

## Input Formats

### BVH (for LAFAN)

- File: `.bvh` as described in [datasets/LAFAN.md](../raw_datasets/LAFAN.md)
- GMR reads BVH directly — no pre-conversion needed
- **Coordinate system correction applied internally**: rotation matrix `[[1,0,0],[0,0,-1],[0,1,0]]` (Y-up → Z-up)
- **Units**: centimeters divided by 100 → meters internally
- **Height assumption**: hardcoded **1.75 m**

### SMPL-X NPZ (for SFU, OMOMO)

- File: `.npz` with SMPL-X parameters
- **Required keys**:

| Key | Shape | Description |
|-----|-------|-------------|
| `pose_body` | `(T, 63)` | 21 body joints, axis-angle |
| `root_orient` | `(T, 3)` | Root rotation, axis-angle |
| `trans` | `(T, 3)` | Root translation [x, y, z], metres |
| `betas` | `(16,)` | Shape parameters |
| `gender` | string | `"male"` / `"female"` / `"neutral"` |
| `mocap_frame_rate` | float | Source FPS (e.g. 120.0) |

- Optional: `pose_hand`, `pose_jaw`, `pose_eye` (set to zeros if absent)
- **Height**: computed from betas as `1.66 + 0.1 * betas[0]` metres
- **Coordinate system**: Z-up (SMPL-X standard) — no conversion needed

---

## Output Format

Pickle file (`.pkl`) with the following keys:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `fps` | scalar | float | Output FPS |
| `root_pos` | `(T, 3)` | float32 | Base translation [x, y, z] in metres |
| `root_rot` | `(T, 4)` | float32 | Base quaternion **[x, y, z, w]** (xyzw!) |
| `dof_pos` | `(T, N_dof)` | float32 | Joint positions for N robot DOFs |
| `local_body_pos` | None | — | Not computed |
| `link_body_list` | None | — | Not stored |

### ⚠️ Quaternion convention

GMR saves quaternions in **xyzw** format.  
The unified format and MuJoCo use **wxyz**.  
Convert with: `q_wxyz = [q[3], q[0], q[1], q[2]]`

---

## Supported Robots

GMR uses IK JSON config files (`ik_configs/`) to define human → robot joint mappings. 17 robots are supported.

| Robot | DOF | IK config (SMPL-X) | BVH LAFAN support |
|-------|-----|--------------------|--------------------|
| `unitree_g1` | 29 | `smplx_to_g1.json` | ✅ `bvh_lafan1_to_g1.json` |
| `unitree_g1_with_hands` | 29+hands | `smplx_to_g1.json` | ✅ |
| `unitree_h1` | 19 | `smplx_to_h1.json` | ❌ |
| `unitree_h1_2` | 19 | `smplx_to_h1_2.json` | ✅ `bvh_xsens_to_h1_2.json` |
| `booster_t1` | 23 | `smplx_to_t1.json` | ❌ |
| `booster_t1_29dof` | 29 | `smplx_to_t1_29dof.json` | ✅ `bvh_lafan1_to_t1_29dof.json` |
| `stanford_toddy` | 30 | `smplx_to_toddy.json` | ✅ `bvh_lafan1_to_toddy.json` |
| `fourier_n1` | 26 | `smplx_to_n1.json` | ✅ `bvh_lafan1_to_n1.json` |
| `engineai_pm01` | 28 | `smplx_to_pm01.json` | ✅ `bvh_lafan1_to_pm01.json` |
| `kuavo_s45` | 28 | `smplx_to_kuavo.json` | ❌ |
| `hightorque_hi` | 25 | `smplx_to_hi.json` | ❌ |
| `galaxea_r1pro` | 24 | `smplx_to_r1pro.json` | ❌ (wheeled) |
| `berkeley_humanoid_lite` | 27 | `smplx_to_bhl.json` | ❌ |
| `booster_k1` | 22 | `smplx_to_k1.json` | ❌ |
| `pnd_adam_lite` | 26 | `smplx_to_adam.json` | ❌ |
| `tienkung` | 20 | `smplx_to_tienkung.json` | ❌ |
| `fourier_gr3` | 36 | `smplx_to_gr3.json` | ❌ |

All robots support SMPL-X input. BVH LAFAN support requires a dedicated `bvh_lafan1_to_{robot}.json` config.

→ Detailed spec for G1: [robots/G1.md](../robots/G1.md)

---

## Preprocessing Expected by GMR

- For LAFAN: raw BVH file, no pre-processing needed
- For SFU/OMOMO: raw SMPL-X `.npz`, no pre-processing needed
- GMR handles FPS internally (reads `mocap_frame_rate`)
- GMR does **not** accept the unified `global_joint_positions` format directly — it needs raw SMPL-X params or BVH

---

## Converter responsibilities (`motion_convertor`)

| Direction | Operation |
|-----------|-----------|
| LAFAN → GMR input | Pass `.bvh` directly (no conversion needed) |
| SFU → GMR input | Pass raw `.npz` directly (no conversion needed) |
| OMOMO → GMR input | Extract SMPL-H params from pickle → reformat as SMPL-X `.npz` |
| GMR output → unified | Load pickle, convert `root_rot` xyzw→wxyz, reformat as `global_joint_positions (T,22,3)` via FK |
| Any input → unified | Run SMPL-X FK on input params to get `global_joint_positions` |
