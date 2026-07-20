# VecKM Flow Eval

Standardized evaluation pipelines for event-based optical flow baselines on **EVIMO** and **MVSEC**.

This repository vendors and extends three upstream optical-flow methods and provides consistent data handling, inference configs, and metric scripts for benchmarking against **VecKM_flow** ([GitHub](https://github.com/dhyuan99/VecKM_flow), [real-time C++/CUDA](https://github.com/dhyuan99/VecKM_flow_cpp)).

| Method | Directory | Upstream |
|--------|-----------|----------|
| E-RAFT | [`E-RAFT/`](E-RAFT/) | [uzh-rpg/E-RAFT](https://github.com/uzh-rpg/E-RAFT) |
| TCM (Taming Event Flow) | [`taming_event_flow/`](taming_event_flow/) | [tudelft/taming_event_flow](https://github.com/tudelft/taming_event_flow) |
| Secrets (Contrast Maximization) | [`event_based_optical_flow/`](event_based_optical_flow/) | [tub-rip/event_based_optical_flow](https://github.com/tub-rip/event_based_optical_flow) |

---

## Related projects

### VecKM_flow (normal flow)

Event-based **normal flow** estimator from the Perception and Robotics Group at UMD:

| Resource | Link |
|----------|------|
| Python API (ICCV 2025) | [dhyuan99/VecKM_flow](https://github.com/dhyuan99/VecKM_flow) |
| Real-time C++/CUDA | [dhyuan99/VecKM_flow_cpp](https://github.com/dhyuan99/VecKM_flow_cpp) |
| Paper (ICCV 2025) | [Learning Normal Flow Directly From Event Neighborhoods](https://arxiv.org/abs/2412.11284) |
| Real-time paper | [A Real-Time Event-Based Normal Flow Estimator](https://arxiv.org/abs/2504.19417) |

### EVIMO dataset

Evaluation data follows the [EVIMO / EVIMO2](https://better-flow.github.io/evimo/) format. Download sequences and ground truth from the official site:

| Resource | Link |
|----------|------|
| Dataset homepage | [better-flow.github.io/evimo](https://better-flow.github.io/evimo/) |
| EVIMO2 paper | [EVIMO2: An Event Camera Dataset for Motion Segmentation, Optical Flow, ...](https://arxiv.org/abs/2205.03467) |

---

## Repository layout

```
VecKM_Flow_Eval/
├── data/                          # Local datasets (gitignored, see data/README.md)
├── eval/                          # Unified normal-flow / optical-flow evaluation
├── outputs/                       # Inference & metric outputs (gitignored)
├── E-RAFT/                        # E-RAFT baseline + EVIMO/MVSEC extensions
├── taming_event_flow/             # TCM baseline + EVIMO/MVSEC extensions
└── event_based_optical_flow/      # Secrets CM baseline + EVIMO extensions
```

---

## Setup

### 1. Clone and prepare data

```bash
git clone https://github.com/prgumd/VecKM_Flow_Eval.git
cd VecKM_Flow_Eval
```

Create the local data directory and follow [`data/README.md`](data/README.md) for the expected layout:

```bash
mkdir -p data/hdf5 data/gt_flow outputs
```

### 2. Install dependencies

Each baseline has its own environment. Install only what you need:

**E-RAFT**

```bash
cd E-RAFT
conda env create -f environment.yml
conda activate e-raft   # name from environment.yml
```

**TCM**

```bash
cd taming_event_flow
pip install -r requirements.txt
```

**Secrets**

```bash
cd event_based_optical_flow
pip install -e .
```

Download pretrained checkpoints from the respective upstream repositories into each method's `checkpoints/` or `mlruns/` directory.

---

## Running evaluations

All paths below are **relative to the method subdirectory**. Replace placeholders with your local paths under `data/` and `outputs/`.

### 1. E-RAFT

```bash
cd E-RAFT
python main.py --path ../data --dataset mvsec --frequency 20
```

**Config:** [`E-RAFT/config/mvsec_20.json`](E-RAFT/config/mvsec_20.json)

Key fields to edit:

| Field | Description |
|-------|-------------|
| `save_dir` | Output directory (default: `./outputs/e-raft`) |
| `data_loader.test.args.datasets` | Scene / sequence IDs |
| `data_loader.test.args.filter` | Frame index range |
| `test.checkpoint` | Path to pretrained weights |

### 2. TCM (Taming Event Flow)

```bash
cd taming_event_flow
python eval_flow.py <model_name> --config configs/eval_dsec.yml
```

**Config:** [`taming_event_flow/configs/eval_dsec.yml`](taming_event_flow/configs/eval_dsec.yml)

Set the data root:

```yaml
data:
  path: ../data/hdf5
```

For MVSEC evaluation, use `configs/eval_mvsec.yml` instead.

### 3. Secrets (Contrast Maximization)

```bash
cd event_based_optical_flow
python main.py --config_file ./configs/evimo_no_timeaware.yaml
```

**Config:** [`event_based_optical_flow/configs/evimo_no_timeaware.yaml`](event_based_optical_flow/configs/evimo_no_timeaware.yaml)

Key fields:

```yaml
data:
  root: "../data/hdf5"
  gt: "../data/gt_flow"
  sequence: "<scene_id>"    # e.g. 15_05

output:
  output_dir: "../outputs/secrets/<scene_id>"
```

Ensure `is_dnn: false` for the contrast-maximization (non-DNN) pipeline.

---

## Evaluation metrics

This repo supports **two score families** for comparing predictions against EVIMO ground truth:

| Score type | CLI `--score-type` | Metrics | Implementation |
|------------|-------------------|---------|----------------|
| **Normal flow** | `normal-flow` | mean error, sharp-angle rate | `eval/metrics.py` |
| **Optical flow (projection)** | `optical-flow-projection` | mean error, sharp-angle rate | `eval/metrics.py` |
| **Optical flow (AEE)** | `optical-flow-aee` | AEE, % out | `eval/metrics.py` |

Use the unified evaluator at [`eval/run_eval.py`](eval/run_eval.py):

```bash
# Normal-flow scores
python3 -m eval.run_eval --method eraft --score-type normal-flow --scene 13_0 --version dsec

# Optical-flow projection scores
python3 -m eval.run_eval --method tcm --score-type optical-flow-projection --all-scenes

# Optical-flow AEE scores
python3 -m eval.run_eval --method secrets --score-type optical-flow-aee --all-scenes
```

Common options:

| Flag | Description |
|------|-------------|
| `--pred-root` | Override prediction root (defaults under `outputs/<method>/EVIMO/`) |
| `--gt-root` | Ground-truth root (default: `data/scenes`) |
| `--output-csv` | Summary CSV path (default: `outputs/metrics/<method>_<score>_summary.csv`) |
| `--verbose-frames` | Print per-frame metrics |

Expected layout:

```
data/scenes/scene13_0/13_0/optical_flow/000001.npy
outputs/e-raft/EVIMO/Eraft_saved/13_0_dsec/flow/1.npy
outputs/tcm/EVIMO/scene13_0/TCM_DSEC_60Hz/flow_npy/1.npy
outputs/secrets/EVIMO/scene13_0/pred_masked_npy/pred_masked0.npy
```

All three score types are implemented in [`eval/metrics.py`](eval/metrics.py) and invoked via `python3 -m eval.run_eval`.

Training / inference metrics reported by each baseline (FWL, RSAT, etc.) are separate from this post-processing evaluation.

---

## Output convention

```
outputs/
├── e-raft/       # E-RAFT predictions
├── tcm/          # TCM predictions
└── secrets/      # Secrets CM predictions
```

---

## Data preparation

To convert raw EVIMO scene exports into MVSEC-compatible layout for E-RAFT, use [`E-RAFT/EVIMO2MVSEC.py`](E-RAFT/EVIMO2MVSEC.py). Configure paths at the top of the script before running.

If you use preprocessed data under `data/scenes/` (see [`data/README.md`](data/README.md)), this step is optional.

---

## Reproducibility

When benchmarking across methods, keep these settings consistent:

- GPU device ID
- Voxel bin count / event window size
- Crop resolution (e.g. 480×640 for EVIMO, 260×346 for MVSEC outdoor)
- Train/eval split and frame index ranges

---

## Citation

If you use this evaluation suite, please cite the baseline method papers, the EVIMO dataset, and VecKM_flow as appropriate:

```bibtex
@InProceedings{Gehrig3dv2021,
  author    = {Mathias Gehrig and Mario Millh{\"a}usler and Daniel Gehrig and Davide Scaramuzza},
  title     = {{E-RAFT}: Dense Optical Flow from Event Cameras},
  booktitle = {International Conference on 3D Vision (3DV)},
  year      = {2021}
}

@InProceedings{Paredes-Valles_2023_ICCV,
  author    = {Paredes-Vall{\'e}s, Federico and Scheper, Kirk Y. W. and De Wagter, Christophe and de Croon, Guido C. H. E.},
  title     = {Taming Contrast Maximization for Learning Sequential, Low-latency, Event-based Optical Flow},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023}
}

@Article{Shiba24pami,
  author  = {Shintaro Shiba and Yannick Klose and Yoshimitsu Aoki and Guillermo Gallego},
  title   = {Secrets of Event-based Optical Flow, Depth, and Ego-Motion by Contrast Maximization},
  journal = {IEEE Trans. Pattern Anal. Mach. Intell. (T-PAMI)},
  year    = {2024}
}

@article{Burner2022evimo2,
  author  = {Levi Burner and Anton Mitrokhin and Cornelia Ferm{\"u}ller and Yiannis Aloimonos},
  title   = {{EVIMO2}: An Event Camera Dataset for Motion Segmentation, Optical Flow, Structure from Motion, and Visual Inertial Odometry in Indoor Scenes with Monocular or Stereo Algorithms},
  journal = {arXiv preprint arXiv:2205.03467},
  year    = {2022}
}

@article{yuan2024learning,
  title   = {Learning Normal Flow Directly From Event Neighborhoods},
  author  = {Yuan, Dehao and Burner, Levi and Wu, Jiayi and Liu, Minghui and Chen, Jingxi and Aloimonos, Yiannis and Ferm{\"u}ller, Cornelia},
  journal = {arXiv preprint arXiv:2412.11284},
  year    = {2024}
}

@article{yuan2025real,
  title   = {A Real-Time Event-Based Normal Flow Estimator},
  author  = {Yuan, Dehao and Ferm{\"u}ller, Cornelia},
  journal = {arXiv preprint arXiv:2504.19417},
  year    = {2025}
}
```
