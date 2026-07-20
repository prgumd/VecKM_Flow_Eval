#!/usr/bin/env python3
"""Unified EVIMO evaluation CLI for baseline methods."""

import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import numpy as np

from eval.loaders import (
    SceneJob,
    discover_eraft_jobs,
    discover_secrets_jobs,
    discover_tcm_jobs,
    load_arrays,
)
from eval.metrics import (
    compute_normal_flow_scores,
    compute_optical_flow_aee_scores,
    compute_optical_flow_projection_scores,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

METHOD_DEFAULTS = {
    "eraft": {
        "pred_root": REPO_ROOT / "outputs" / "e-raft" / "EVIMO" / "Eraft_saved",
        "discover": discover_eraft_jobs,
    },
    "tcm": {
        "pred_root": REPO_ROOT / "outputs" / "tcm" / "EVIMO",
        "discover": discover_tcm_jobs,
    },
    "secrets": {
        "pred_root": REPO_ROOT / "outputs" / "secrets" / "EVIMO",
        "discover": discover_secrets_jobs,
    },
}

SCORE_TYPES = {
    "normal-flow": {
        "compute": compute_normal_flow_scores,
        "frame_fields": ("mean_error", "sharp_angle_rate"),
        "summary_fields": (
            "mean_error_mean",
            "mean_error_median",
            "sharp_angle_rate_mean",
            "sharp_angle_rate_median",
        ),
    },
    "optical-flow-projection": {
        "compute": compute_optical_flow_projection_scores,
        "frame_fields": ("mean_error", "sharp_angle_rate"),
        "summary_fields": (
            "mean_error_mean",
            "mean_error_median",
            "sharp_angle_rate_mean",
            "sharp_angle_rate_median",
        ),
    },
    "optical-flow-aee": {
        "compute": compute_optical_flow_aee_scores,
        "frame_fields": ("aee", "percent_out", "evaluated_pixels", "outlier_pixels", "percent_out_global"),
        "summary_fields": (
            "evaluated_pixels",
            "outlier_pixels",
            "aee_mean",
            "aee_median",
            "percent_out_mean",
            "percent_out_median",
            "percent_out_global",
        ),
    },
}


def parse_args():
    parser = ArgumentParser(
        description=(
            "Evaluate baseline predictions on EVIMO. "
            "Supports normal-flow scores and optical-flow scores (projection or AEE)."
        )
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=tuple(METHOD_DEFAULTS),
        help="Baseline method to evaluate.",
    )
    parser.add_argument(
        "--score-type",
        required=True,
        choices=tuple(SCORE_TYPES),
        help=(
            "normal-flow: projection-based normal-flow scores; "
            "optical-flow-projection: projection scores for optical-flow baselines; "
            "optical-flow-aee: AEE / percent-out for optical-flow baselines."
        ),
    )
    parser.add_argument("--scene", default="13_0", help="Scene id, for example 13_0.")
    parser.add_argument(
        "--version",
        default="dsec",
        choices=("dsec", "mvsec"),
        help="Prediction variant for methods that expose dsec/mvsec outputs.",
    )
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Evaluate all discovered scene folders under the prediction root.",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=None,
        help="Root directory containing saved predictions.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=REPO_ROOT / "data" / "scenes",
        help="Root directory containing scene*/<scene>/optical_flow ground truth.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV path for aggregated results.",
    )
    parser.add_argument(
        "--verbose-frames",
        action="store_true",
        help="Print per-frame metrics.",
    )
    return parser.parse_args()


def default_output_csv(method: str, score_type: str) -> Path:
    return REPO_ROOT / "outputs" / "metrics" / f"{method}_{score_type.replace('-', '_')}_summary.csv"


def select_jobs(all_jobs: List[SceneJob], scene: str, version: str, evaluate_all: bool) -> List[SceneJob]:
    if evaluate_all:
        return all_jobs
    selected = [job for job in all_jobs if job.scene == scene and (job.version in (None, version))]
    if not selected:
        raise RuntimeError(f"No predictions found for scene={scene} version={version}.")
    return selected


def summarize_frames(frame_metrics: List[dict], score_type: str) -> dict:
    if score_type in {"normal-flow", "optical-flow-projection"}:
        mean_errors = [item["mean_error"] for item in frame_metrics]
        sharp_rates = [item["sharp_angle_rate"] for item in frame_metrics]
        return {
            "mean_error_mean": float(np.mean(mean_errors)),
            "mean_error_median": float(np.median(mean_errors)),
            "sharp_angle_rate_mean": float(np.mean(sharp_rates)),
            "sharp_angle_rate_median": float(np.median(sharp_rates)),
        }

    aees = [item["aee"] for item in frame_metrics]
    out_rates = [item["percent_out"] for item in frame_metrics]
    evaluated_pixels = sum(item["evaluated_pixels"] for item in frame_metrics)
    outlier_pixels = sum(item["outlier_pixels"] for item in frame_metrics)
    return {
        "evaluated_pixels": evaluated_pixels,
        "outlier_pixels": outlier_pixels,
        "aee_mean": float(np.mean(aees)),
        "aee_median": float(np.median(aees)),
        "percent_out_mean": float(np.mean(out_rates)),
        "percent_out_median": float(np.median(out_rates)),
        "percent_out_global": float(100.0 * outlier_pixels / evaluated_pixels) if evaluated_pixels else 0.0,
    }


def evaluate_job(job: SceneJob, score_type: str, verbose_frames: bool) -> dict:
    compute = SCORE_TYPES[score_type]["compute"]
    frame_metrics = []
    skipped_missing_gt = 0
    skipped_no_overlap = 0

    for frame in job.frames:
        try:
            pred, gt = load_arrays(frame)
            metrics = compute(pred, gt)
            frame_metrics.append(metrics)
            if verbose_frames:
                rendered = ", ".join(f"{key}={value:.6f}" for key, value in metrics.items())
                print(f"scene={job.scene} version={job.version} frame={frame.frame_index}: {rendered}")
        except FileNotFoundError:
            skipped_missing_gt += 1
        except ValueError as exc:
            skipped_no_overlap += 1
            if verbose_frames:
                print(f"scene={job.scene} frame={frame.frame_index}: skipped ({exc})")

    if not frame_metrics:
        raise RuntimeError("No valid frames were evaluated.")

    result = {
        "scene": job.scene,
        "predicted_frames": len(job.frames),
        "evaluated_frames": len(frame_metrics),
        "skipped_missing_gt": skipped_missing_gt,
        "skipped_no_overlap": skipped_no_overlap,
    }
    if job.version is not None:
        result["version"] = job.version
    result.update(summarize_frames(frame_metrics, score_type))
    print(
        f"scene={job.scene} version={job.version} evaluated={result['evaluated_frames']} "
        + ", ".join(
            f"{key}={result[key]:.6f}"
            for key in SCORE_TYPES[score_type]["summary_fields"]
            if key in result and isinstance(result[key], float)
        )
    )
    return result


def write_results_csv(results: List[dict], output_csv: Path, score_type: str):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene"]
    if any("version" in row for row in results):
        fieldnames.append("version")
    fieldnames.extend(
        [
            "predicted_frames",
            "evaluated_frames",
            "skipped_missing_gt",
            "skipped_no_overlap",
            *SCORE_TYPES[score_type]["summary_fields"],
        ]
    )
    with output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main():
    args = parse_args()
    method_cfg = METHOD_DEFAULTS[args.method]
    pred_root = args.pred_root or method_cfg["pred_root"]
    output_csv = args.output_csv or default_output_csv(args.method, args.score_type)

    jobs = method_cfg["discover"](pred_root, args.gt_root)
    selected_jobs = select_jobs(jobs, args.scene, args.version, args.all_scenes)

    results = []
    for job in selected_jobs:
        try:
            results.append(evaluate_job(job, args.score_type, args.verbose_frames))
        except Exception as exc:
            label = f"scene={job.scene} version={job.version}"
            print(f"Skipping {label}: {exc}")

    if not results:
        raise RuntimeError("No scenes were successfully evaluated.")

    write_results_csv(results, output_csv, args.score_type)
    print(f"Saved CSV summary to {output_csv}")


if __name__ == "__main__":
    main()
