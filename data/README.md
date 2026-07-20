# Dataset Layout

Place evaluation data under `data/` at the repository root. This directory is gitignored; only this README is tracked.

## EVIMO / MVSEC (recommended layout)

```
data/
├── hdf5/          # Event sequences in HDF5 format (one file per scene)
├── gt_flow/       # Ground-truth optical flow (.npz or per-frame .npy)
└── scenes/        # EVIMO scene exports used by eval/run_eval.py
    └── scene13_0/
        └── 13_0/
            └── optical_flow/
                ├── 000001.npy
                └── ...
```

## MVSEC (original format)

If using raw MVSEC HDF5 files directly:

```
data/
└── mvsec/
    ├── <sequence>_data.hdf5
    └── <sequence>_gt.hdf5
```

## Symlink alternative

You may symlink an existing dataset root instead of copying files:

```bash
ln -s /path/to/your/dataset data
```

Point the method configs to the corresponding subdirectories under `data/`.
