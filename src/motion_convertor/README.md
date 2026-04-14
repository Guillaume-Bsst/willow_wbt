# Motion Convertor

The motion convertor is the **adapter layer** between all the heterogeneous solutions in the pipeline. It translates motion data between each solution's native format and the unified format, enabling any retargeter to consume any dataset and any trainer to consume any retargeter's output — without modifying the solutions themselves.

It is a passive tool. It does not call retargeters or trainers. That is the responsibility of `scripts/retarget.py`.

---

## Role in the pipeline

```
00_raw_datasets/{dataset}/          (BVH, pickle, AMASS npz, ...)
        │
        │  prep_{dataset}()         dataset → retargeter native input
        ▼
{retargeter} native input
        │
        │  [retargeter — called by scripts/retarget.py]
        ▼
{retargeter} native output
        │
        │  unify_input_{dataset}()  native input → unified
        │  unify_output_{retargeter}() native output → unified
        ▼
01_retargeted_motions/{dataset}/{retargeter}/run_{timestamp}/
    ├── {seq}_input_raw.{ext}
    ├── {seq}_input_unified.npz
    ├── {seq}_output_raw.{ext}
    └── {seq}_output_unified.npz
```

→ Full format and run structure: [data/01_retargeted_motions/README.md](../../data/01_retargeted_motions/README.md)

---

## Unified format

The unified format is the **contract** between all pipeline stages. Any solution that speaks this format can be plugged in anywhere.

| Key | Shape | Description |
|-----|-------|-------------|
| `global_joint_positions` | `(T, 22, 3)` | Joint positions in world frame (metres) |
| `height` | `float` | Subject height in metres |
| `object_poses` *(optional)* | `(T, 7)` | `[qw, qx, qy, qz, x, y, z]` — object interaction only |

- 22 joints, SMPL-X convention, world frame, metres
- Both human (input) and robot (output) motions use this format — enabling direct source/retargeted comparison

---

## Dataset adapters

Each dataset requires a dedicated adapter because joint conventions, body models, fps, and file formats differ across datasets. Adding a new dataset means writing one new adapter here, without touching anything else.

| Dataset | Raw format | Body model | Notes |
|---------|-----------|-----------|-------|
| **LAFAN1** | `.bvh` | — (BVH skeleton) | No SMPL model needed |
| **OMOMO** | `.p` (pickle) | SMPL-H (`smplh/`) | Also generates object assets (`.urdf`, `.xml`) for object_interaction |
| **SFU** | `.npz` (AMASS) | SMPL-X (`models_smplx_v1_1/`) | Via AMASS SMPL-X forward kinematics |

---

## Retargeter adapters

Each retargeter also has its own native I/O. Adding a new retargeter means writing one new adapter here.

| Retargeter | Native input | Native output |
|------------|-------------|--------------|
| **GMR** | `.npy` `(T, 22, 3)` | `.pt` / `.npy` robot joint angles |
| **holosoma_retargeter** | `.npz` `(T, 22, 3)` + optional `object_poses` | `.npz` with `qpos` |

---

## Adding a new solution

**New dataset:**
1. Add download instructions to `data/00_raw_datasets/README.md`
2. Write `prep_{dataset}()` — converts raw data to the retargeter's expected input
3. Write `unify_input_{dataset}()` — converts the native input to unified format

**New retargeter:**
1. Add submodule to `modules/01_retargeting/`
2. Write `unify_output_{retargeter}()` — converts its native output to unified format

---

## third_party/

| Submodule | Role |
|-----------|------|
| **InterAct** ([wzyabcas/InterAct](https://github.com/wzyabcas/InterAct)) | Human-object interaction processing — SMPL conversion, object pose estimation, physics-based HOI correction |
| **InterMimic** ([Sirui-Xu/InterMimic](https://github.com/Sirui-Xu/InterMimic)) | Imitation learning for interaction — converts OMOMO sequences for physics simulation (IsaacGym / IsaacLab) |
