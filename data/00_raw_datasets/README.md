# Raw Motion Data

Raw datasets and body models to download before running the pipeline. No processing here — just acquisition.

Three dataset folders sit alongside this README: `LAFAN/`, `OMOMO/`, `SFU/`.

---

## LAFAN/

```
LAFAN/
└── lafan1/
    ├── walk1_subject1.bvh
    ├── dance2_subject4.bvh
    └── ...
```

1. Download [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) (click "View Raw")
2. Extract `.bvh` files into `LAFAN/lafan1/`

---

## OMOMO/

```
OMOMO/
├── data/
│   ├── train_diffusion_manip_window_120_cano_joints24.p
│   ├── test_diffusion_manip_seq_joints24.p
│   ├── object_bps_npy_files_joints24/
│   ├── captured_objects/
│   └── ...
└── smplh/
    ├── male/model.npz
    ├── female/model.npz
    └── neutral/model.npz
```

1. Download the [OMOMO dataset](https://drive.google.com/file/d/1tZVqLB7II0whI-Qjz-z-AU3ponSEyAmm/view) (`data/` folder) into `OMOMO/data/`
2. Download the [Extended SMPL+H model for AMASS](https://mano.is.tue.mpg.de/download.php) and extract `smplh.tar.xz` into `OMOMO/smplh/`

---

## SFU/

```
SFU/
├── SFU/
│   ├── 0005/
│   │   ├── neutral_stagei.npz
│   │   ├── 0005_Walking001_stageii.npz
│   │   └── ...
│   ├── 0008/
│   └── ...
└── models_smplx_v1_1/
    └── models/
        └── smplx/
            ├── SMPLX_NEUTRAL.npz
            ├── SMPLX_MALE.npz
            └── SMPLX_FEMALE.npz
```

1. Follow [AMASS download instructions](https://amass.is.tue.mpg.de/) and select the **SFU** subset (SMPL-H format) into `SFU/SFU/`
2. Download [SMPL-X models](https://smpl-x.is.tue.mpg.de/) (SMPL-X N neutral format) and extract `models_smplx_v1_1.zip` into `SFU/models_smplx_v1_1/`
