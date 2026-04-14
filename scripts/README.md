# Scripts

Pipeline entry points. Each script orchestrates one stage of the pipeline — it selects which solution to use, calls the appropriate module, and uses the converters to handle all data I/O. The modules themselves are never modified.

---

## retarget.py

Runs a retargeting job for a given dataset/retargeter combination.

**What it does:**
1. Reads raw data from `data/00_raw_datasets/{dataset}/`
2. Calls `src/motion_convertor/` to convert to the retargeter's native input format
3. Calls `modules/01_retargeting/{retargeter}` to run retargeting
4. Calls `src/motion_convertor/` to convert input and output to unified format
5. Saves to `data/01_retargeted_motions/{dataset}/{retargeter}/run_{timestamp}/`:
   - `{seq}_input_raw.{ext}` — native input
   - `{seq}_input_unified.npz` — unified input
   - `{seq}_output_raw.{ext}` — native output
   - `{seq}_output_unified.npz` — unified output
   - `config.yaml` — exact parameters used
6. Updates the `latest →` symlink

**Supported combinations:**

| Dataset | GMR | holosoma_retargeter |
|---------|-----|---------------------|
| LAFAN1 | ✅ | ✅ |
| SFU | ✅ | ✅ |
| OMOMO robot_only | ✅ | ✅ |
| OMOMO object_interaction | ❌ | ✅ |

```bash
python scripts/retarget.py --dataset LAFAN --retargeter GMR --config configs/retarget_gmr.yaml
python scripts/retarget.py --dataset OMOMO_object_interaction --retargeter holosoma_retargeter
```

---

## train.py

Runs a WBT training job from retargeted motions.

**What it does:**
1. Reads unified motions from `data/01_retargeted_motions/{dataset}/{retargeter}/latest/`
2. Calls `src/policy_convertor/` to convert to the trainer's expected input format
3. Calls `modules/02_training/{trainer}` to run training
4. Saves to `data/02_policies/{dataset}/{retargeter}/{trainer}/run_{timestamp}/`:
   - trained checkpoint (`.pt`)
   - exported policy (`.onnx`)
   - `config.yaml`
5. Updates the `latest →` symlink

```bash
python scripts/train.py --dataset LAFAN --retargeter GMR --trainer holosoma --config configs/train_wbt.yaml
```

---

## infer.py

Runs inference from a trained policy in simulation or on a real robot.

**What it does:**
1. Reads the ONNX policy from `data/02_policies/{dataset}/{retargeter}/{trainer}/latest/`
2. Calls `modules/03_inference/{engine}`

**Modes:**
```bash
# MuJoCo sim-to-sim evaluation
python scripts/infer.py --dataset LAFAN --retargeter GMR --trainer holosoma --mode sim

# Real robot deployment
python scripts/infer.py --dataset LAFAN --retargeter GMR --trainer holosoma --mode real
```
