# SFU (AMASS) — Format Specification

## Overview

| Property | Value |
|----------|-------|
| File format | `.npz` (NumPy compressed, AMASS SMPL-X stage-ii fits) |
| Frame rate | 120 Hz |
| Coordinate system | Z-up, right-handed |
| Units | meters |
| Body model | SMPL-X (21 body joints + root) |
| Joint count | 22 (body only, no hands/face) |

---

## NPZ Keys

### Pose parameters (axis-angle)

| Key | Shape | Description |
|-----|-------|-------------|
| `root_orient` | `(T, 3)` | Root rotation, axis-angle |
| `pose_body` | `(T, 63)` | 21 body joints × 3, axis-angle |
| `pose_hand` | `(T, 90)` | 15 joints × 2 hands × 3, axis-angle |
| `pose_jaw` | `(T, 3)` | Jaw rotation, axis-angle |
| `pose_eye` | `(T, 6)` | 2 eyes × 3, axis-angle |
| `poses` | `(T, 165)` | Concatenated: `[root_orient(3) \| pose_body(63) \| pose_hand(90) \| pose_jaw(3) \| pose_eye(6)]` |

### Translation and shape

| Key | Shape | Description |
|-----|-------|-------------|
| `trans` | `(T, 3)` | Root translation [x, y, z] in metres |
| `betas` | `(16,)` | SMPL-X shape parameters (16 components) |
| `num_betas` | scalar | = 16 |

### Metadata

| Key | Type | Description |
|-----|------|-------------|
| `gender` | string | `"male"` / `"female"` / `"neutral"` |
| `surface_model_type` | string | `"smplx"` |
| `mocap_frame_rate` | float | Typically `120.0` Hz |
| `mocap_time_length` | float | Total duration in seconds |

### MoCap markers (optional)

| Key | Shape | Description |
|-----|-------|-------------|
| `markers_latent` | `(53, 3)` | 53 marker latent positions |
| `latent_labels` | array | Marker names |
| `markers_latent_vids` | array | Marker video IDs |

---

## SMPL-X Joint Convention (22 body joints)

| Index | Name |
|-------|------|
| 0 | Pelvis |
| 1 | L_Hip |
| 2 | R_Hip |
| 3 | Spine1 |
| 4 | L_Knee |
| 5 | R_Knee |
| 6 | Spine2 |
| 7 | L_Ankle |
| 8 | R_Ankle |
| 9 | Spine3 |
| 10 | L_Foot |
| 11 | R_Foot |
| 12 | Neck |
| 13 | L_Collar |
| 14 | R_Collar |
| 15 | Head |
| 16 | L_Shoulder |
| 17 | R_Shoulder |
| 18 | L_Elbow |
| 19 | R_Elbow |
| 20 | L_Wrist |
| 21 | R_Wrist |

---

## Processing Notes

- **Forward kinematics** required to get global joint positions from axis-angle params → use `human_body_prior` or `smplx` python package
- **Downsampling**: 120 Hz → 30 Hz (factor of 4) before feeding to retargeters
- **Height**: computed from `betas` via SMPL-X forward kinematics (max joint height)
- **Axis-angle**: use `scipy.spatial.transform.Rotation.from_rotvec()` to convert

---

## Directory Structure

```
SFU/
├── 0005/
│   ├── neutral_stagei.npz       # rest pose
│   ├── 0005_Walking001_stageii.npz
│   └── ...
├── 0008/
├── 0012/
├── 0015/
├── 0017/
├── 0018/
└── LICENSE.txt
```

### Naming convention

```
{subject_id}_{action}{index}_stageii.npz
```

`neutral_stagei.npz` contains the T-pose / rest pose for the subject (no motion).

---

## Typical Sequence Stats

| Property | Value |
|----------|-------|
| Typical frame count | 1 000–5 000 frames at 120 Hz |
| Duration | ~10–40 seconds |
