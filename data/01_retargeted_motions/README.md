# Retargeted Motions

Outputs of the retargeting pipeline. Organized as `{dataset}/{retargeter}/`.

Each retargeter run is stored in a timestamped subfolder with its config:

```
{dataset}/{retargeter}/
├── run_YYYYMMDD_HHMMSS/
│   ├── config.yaml                  # exact parameters used for this run
│   ├── {sequence}_input_raw.{ext}   # native input format for the retargeter
│   ├── {sequence}_input_unified.npz # input converted to unified format
│   ├── {sequence}_output_raw.{ext}  # native output format of the retargeter
│   └── {sequence}_output_unified.npz# output converted to unified format
└── latest -> run_YYYYMMDD_HHMMSS/   # symlink to the most recent run
```

The `_unified` files are produced by `src/motion_convertor/` from the `_raw` files.

---

## Unified format

All `_unified.npz` files follow the same schema:

| Key                    | Shape      | Description                                      |
|------------------------|------------|--------------------------------------------------|
| `global_joint_positions` | `(T, 22, 3)` | Joint positions in world frame (metres)        |
| `height`               | `float`    | Subject height in metres                         |
| `object_poses` *(optional)* | `(T, 7)` | `[qw, qx, qy, qz, x, y, z]` — object interaction sequences only |

- `T` — number of frames
- `J = 22` joints (SMPL-X joint convention)
- Coordinate system: world frame, metres

---

## Structure

```
01_retargeted_motions/
├── assets/                          # scene assets used during retargeting (may be tweaked per retargeter)
│   ├── LAFAN/
│   │   └── terrains/
│   ├── OMOMO/
│   │   ├── objects/                 # object meshes/URDFs (largebox, smallbox, ...)
│   │   └── terrains/
│   └── SFU/
│       └── terrains/
├── LAFAN/
│   ├── GMR/
│   └── holosoma_retargeter/
├── SFU/
│   ├── GMR/
│   └── holosoma_retargeter/
├── OMOMO_robot_only/
│   ├── GMR/
│   └── holosoma_retargeter/
└── OMOMO_object_interaction/
    └── holosoma_retargeter/
```

Assets in `assets/` are shared across sequences of the same dataset. Object assets (`.obj`, `.urdf`, `.xml`) are stored under `assets/OMOMO/objects/{object_name}/`. Terrains are stored under `assets/{dataset}/terrains/`.
