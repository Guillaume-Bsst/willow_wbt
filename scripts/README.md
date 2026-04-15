# Scripts

Pipeline entry points. Each script orchestrates one stage of the pipeline — it selects which solution to use, calls the appropriate module, and uses `src/motion_convertor/` to handle all data I/O. The modules themselves are never modified.

---

## retarget.py

Runs a retargeting job for a given dataset/retargeter/robot combination.

**What it does:**
1. Reads raw data from `data/00_raw_datasets/{dataset}/`
2. Calls `motion_convertor.to_retargeter_input()` → `{seq}_input_raw.{ext}`
3. Calls `motion_convertor.to_unified_input()` → `{seq}_input_unified.npz`
4. Calls `modules/01_retargeting/{retargeter}` to run retargeting → `{seq}_output_raw.{ext}`
5. Calls `motion_convertor.to_unified_output()` → `{seq}_output_unified.npz`
6. Writes `config.yaml` and updates the `latest →` symlink

**Output** — `data/01_retargeted_motions/{dataset}_{robot}/{retargeter}/run_{timestamp}/`:
```
{seq}_input_raw.{ext}
{seq}_input_unified.npz
{seq}_output_raw.{ext}
{seq}_output_unified.npz
config.yaml
```

**Supported combinations:**

| Dataset | GMR | holosoma_retargeting |
|---------|-----|---------------------|
| LAFAN | ✅ | ✅ |
| SFU | ✅ | ✅ |
| OMOMO robot_only | ✅ | ✅ |
| OMOMO object_interaction | ❌ | ✅ |

```bash
python scripts/retarget.py --dataset LAFAN --robot G1 --retargeter GMR
python scripts/retarget.py --dataset OMOMO_object_interaction --robot G1 --retargeter holosoma_retargeting
```

---

## train.py

Runs a WBT training job from retargeted motions.

**What it does:**
1. Locates the retargeting run in `data/01_retargeted_motions/{dataset}_{robot}/{retargeter}/latest/` (or a specific `run_{timestamp}`)
2. Calls `motion_convertor.to_trainer_input()` → `{seq}_trainer_input.npz` (written into the existing run folder)
3. Calls `modules/02_training/{trainer}` to run training
4. Saves to `data/02_policies/{dataset}_{robot}/{retargeter}_{trainer}/run_{timestamp}/`:
   - `checkpoint.pt` — trained weights
   - `policy.onnx` — exported policy
   - `config.yaml`
5. Updates the `latest →` symlink

```bash
python scripts/train.py --dataset LAFAN --robot G1 --retargeter GMR --trainer holosoma
python scripts/train.py --dataset LAFAN --robot G1 --retargeter GMR --trainer holosoma --run run_20240301_120000
```

---

## infer.py

Runs inference from a trained policy in simulation or on a real robot.

**What it does:**
1. Reads the ONNX policy from `data/02_policies/{dataset}_{robot}/{retargeter}_{trainer}/latest/`
2. Calls `modules/03_inference/{engine}`

```bash
# MuJoCo sim-to-sim evaluation
python scripts/infer.py --dataset LAFAN --robot G1 --retargeter GMR --trainer holosoma --mode sim

# Real robot deployment
python scripts/infer.py --dataset LAFAN --robot G1 --retargeter GMR --trainer holosoma --mode real
```
