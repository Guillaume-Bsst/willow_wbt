# Specs

Technical format specifications for all datasets, robots, retargeters, and trainers. These are the reference documents for implementing adapters in `src/motion_convertor/`.

## Robots

| Robot | Spec | DOF modes | Retargeters | Trainers |
|-------|------|-----------|-------------|---------|
| Unitree G1 | [robots/G1.md](robots/G1.md) | 29-DOF, 27-DOF | GMR, holosoma_retargeting | holosoma |

## Raw datasets

| Dataset | Spec | Format | FPS | Coord |
|---------|------|--------|-----|-------|
| LAFAN | [raw_datasets/LAFAN.md](raw_datasets/LAFAN.md) | `.bvh` | 30 Hz | Y-up |
| SFU (AMASS) | [raw_datasets/SFU.md](raw_datasets/SFU.md) | `.npz` SMPL-X | 120 Hz | Z-up |
| OMOMO | [raw_datasets/OMOMO.md](raw_datasets/OMOMO.md) | `.p` pickle | 30 Hz | Z-up |

## Retargeting

| Retargeter | Spec | Input | Output | Quaternion |
|------------|------|-------|--------|-----------|
| GMR | [retargeting/GMR.md](retargeting/GMR.md) | `.bvh` / SMPL-X `.npz` | `.pkl` | xyzw |
| holosoma_retargeting | [retargeting/holosoma_retargeting.md](retargeting/holosoma_retargeting.md) | unified `.npz` / `.npy` | `.npz` full sim state | wxyz |

## Training

| Trainer | Spec | Algorithm | Input | Output | Obs dim | Action dim |
|---------|------|-----------|-------|--------|---------|------------|
| holosoma | [training/holosoma.md](training/holosoma.md) | PPO | `.npz` motion (50 Hz) | `.pt` + `.onnx` | 97 (actor) | 29 |

## Unified format (reminder)

All `*_unified.npz` in `01_retargeted_motions/` follow this schema:

| Key | Shape | Description |
|-----|-------|-------------|
| `global_joint_positions` | `(T, 22, 3)` | Z-up, world frame, metres |
| `height` | float | Subject height, metres |
| `object_poses` *(optional)* | `(T, 7)` | `[qw, qx, qy, qz, x, y, z]` |

## Key cross-cutting concerns

| Topic | Detail |
|-------|--------|
| Coordinate system | All unified data is **Z-up**. LAFAN is Y-up → needs rotation |
| Quaternion convention | Unified = **wxyz**. GMR output = xyzw → needs swap. Training `.npz` = xyzw (MuJoCo convention) |
| FPS | SFU is 120 Hz → downsample ×4 to 30 Hz before retargeting. Trainer expects 50 Hz → upsample in policy convertor |
| Height | LAFAN has no height → assume 1.75 m. OMOMO/SFU computed from body model |
| Joint count | Unified = 22 joints (SMPL-X). OMOMO has 24 (drop L_Hand, R_Hand) |
