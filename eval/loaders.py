"""Prediction / ground-truth path discovery and loading for each baseline method."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

TCM_VERSION_DIRS = {
    "dsec": "TCM_DSEC_60Hz",
    "mvsec": "TCM_MVSEC_60Hz",
}


@dataclass(frozen=True)
class FramePair:
    frame_index: int
    pred_path: Path
    gt_path: Path


@dataclass(frozen=True)
class SceneJob:
    scene: str
    version: Optional[str]
    frames: List[FramePair]


def gt_scene_dir(gt_root: Path, scene: str) -> Path:
    return gt_root / f"scene{scene}" / scene / "optical_flow"


def discover_eraft_jobs(pred_root: Path, gt_root: Path) -> List[SceneJob]:
    jobs = []
    for scene_dir in sorted(pred_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        parts = scene_dir.name.split("_")
        if len(parts) < 3 or parts[2] not in {"dsec", "mvsec"}:
            continue
        scene = f"{parts[0]}_{parts[1]}"
        version = parts[2]
        pred_dir = scene_dir / "flow"
        gt_dir = gt_scene_dir(gt_root, scene)
        if not pred_dir.is_dir() or not gt_dir.is_dir():
            continue
        frames = _frames_from_numeric_stems(pred_dir, gt_dir)
        if frames:
            jobs.append(SceneJob(scene=scene, version=version, frames=frames))
    return jobs


def discover_tcm_jobs(pred_root: Path, gt_root: Path) -> List[SceneJob]:
    jobs = []
    for scene_dir in sorted(pred_root.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene"):
            continue
        scene = scene_dir.name.replace("scene", "", 1)
        gt_dir = gt_scene_dir(gt_root, scene)
        if not gt_dir.is_dir():
            continue
        for version, method_dir in TCM_VERSION_DIRS.items():
            pred_dir = scene_dir / method_dir / "flow_npy"
            if not pred_dir.is_dir():
                continue
            frames = _frames_from_numeric_stems(pred_dir, gt_dir)
            if frames:
                jobs.append(SceneJob(scene=scene, version=version, frames=frames))
    return jobs


def discover_secrets_jobs(pred_root: Path, gt_root: Path) -> List[SceneJob]:
    jobs = []
    for scene_dir in sorted(pred_root.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene"):
            continue
        scene = scene_dir.name.replace("scene", "", 1)
        pred_dir = scene_dir / "pred_masked_npy"
        gt_dir = scene_dir / "gt_flow_npy"
        if not pred_dir.is_dir() or not gt_dir.is_dir():
            continue
        frames = []
        for pred_path in sorted(
            pred_dir.glob("pred_masked*.npy"),
            key=lambda path: int(path.stem.replace("pred_masked", "")),
        ):
            frame_index = int(pred_path.stem.replace("pred_masked", ""))
            gt_path = gt_dir / f"gt_flow{frame_index}.npy"
            if gt_path.exists():
                frames.append(FramePair(frame_index, pred_path, gt_path))
        if frames:
            jobs.append(SceneJob(scene=scene, version=None, frames=frames))
    return jobs


def _frames_from_numeric_stems(pred_dir: Path, gt_dir: Path) -> List[FramePair]:
    frames = []
    for pred_path in sorted(pred_dir.glob("*.npy"), key=lambda path: int(path.stem)):
        frame_index = int(pred_path.stem)
        gt_path = gt_dir / f"{frame_index:06d}.npy"
        if gt_path.exists():
            frames.append(FramePair(frame_index, pred_path, gt_path))
    return frames


def load_arrays(frame: FramePair):
    return np.load(frame.pred_path), np.load(frame.gt_path)
