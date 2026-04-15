# LAFAN — Format Specification

## Overview

| Property | Value |
|----------|-------|
| File format | `.bvh` (Biovision Hierarchy) |
| Frame rate | 30 Hz (frame time = 0.033333s) |
| Coordinate system | Y-up, right-handed |
| Units | centimeters (positions), degrees (rotations) |
| Joint count | 23 |

---

## BVH File Structure

```
HIERARCHY
  ROOT Hips
    CHANNELS 6  Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT LeftUpLeg
      CHANNELS 3  Zrotation Yrotation Xrotation
      ...
MOTION
Frames: 7341
Frame Time: 0.033333
-0.12 91.32 2.45 0.01 -0.03 0.12 ...
```

- **HIERARCHY** — defines skeleton tree with joint names and local offsets (`OFFSET x y z`)
- **MOTION** — one row per frame, values in channel order defined in HIERARCHY
- Root has 6 channels: 3 translations + 3 rotations
- All other joints: 3 rotation channels only (Zrotation, Yrotation, Xrotation)
- Rotation order: **ZYX Euler** (intrinsic), values in **degrees**
- Positions in **centimeters** → divide by 100 for meters

---

## Joint List (23 joints, in BVH order)

| Index | Name |
|-------|------|
| 0 | Hips (root) |
| 1 | LeftUpLeg |
| 2 | LeftLeg |
| 3 | LeftFoot |
| 4 | LeftToe |
| 5 | RightUpLeg |
| 6 | RightLeg |
| 7 | RightFoot |
| 8 | RightToe |
| 9 | Spine |
| 10 | Spine1 |
| 11 | Spine2 |
| 12 | Neck |
| 13 | Head |
| 14 | LeftShoulder |
| 15 | LeftArm |
| 16 | LeftForeArm |
| 17 | LeftHand |
| 18 | RightShoulder |
| 19 | RightArm |
| 20 | RightForeArm |
| 21 | RightHand |

> Note: 22 joints listed above + root = 23 total.

---

## Coordinate System Conversion

LAFAN is **Y-up**. The pipeline unified format and retargeters use **Z-up**.

Conversion rotation matrix (Y-up → Z-up):
```
R = [[1,  0,  0],
     [0,  0, -1],
     [0,  1,  0]]
```

Apply to all joint positions after computing global positions from BVH.

---

## Height

Human height is **not stored** in LAFAN files. Assumed: **1.75 m** (hardcoded in GMR), **1.70 m** (used in holosoma_retargeting with scale factor `1.27 / 1.7`).

---

## Typical Sequence Stats

| Property | Value |
|----------|-------|
| File size | 5–6.5 MB |
| Typical duration | ~4–9 minutes |
| Typical frame count | 7 000–16 000 frames |
| Subjects | subject1 – subject5 |

---

## Naming Convention

```
{action}{index}_subject{id}.bvh
```
Examples: `walk1_subject1.bvh`, `dance2_subject4.bvh`, `aiming1_subject1.bvh`
