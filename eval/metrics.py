"""Metric computation for normal-flow and optical-flow evaluation."""

import numpy as np


def _gt_to_hwc(gt_flow: np.ndarray) -> np.ndarray:
    if gt_flow.ndim != 3:
        raise ValueError(f"Expected GT flow with 3 dims, got shape {gt_flow.shape}")
    if gt_flow.shape[0] == 2:
        gt = np.transpose(gt_flow, (1, 2, 0))
    else:
        gt = gt_flow
    gt = gt.astype(np.float64, copy=False)
    gt[np.linalg.norm(gt, axis=-1) < 1e-5] = np.nan
    return gt


def _pred_to_hwc(pred_flow: np.ndarray, height: int, width: int) -> np.ndarray:
    if pred_flow.ndim == 3 and pred_flow.shape[0] == 2:
        pred = np.transpose(pred_flow, (1, 2, 0))
    elif pred_flow.ndim == 2:
        if pred_flow.shape[0] == 5:
            u, v = pred_flow[1], pred_flow[2]
            xs = pred_flow[3].astype(int)
            ys = pred_flow[4].astype(int)
        elif pred_flow.shape[0] == 4:
            u, v = pred_flow[0], pred_flow[1]
            xs = pred_flow[2].astype(int)
            ys = pred_flow[3].astype(int)
        else:
            raise ValueError(f"Unsupported sparse prediction shape: {pred_flow.shape}")
        pred = np.full((height, width, 2), np.nan, dtype=np.float64)
        pred[ys, xs, 0] = u
        pred[ys, xs, 1] = v
    else:
        raise ValueError(f"Unsupported prediction shape: {pred_flow.shape}")
    return pred


def _pred_to_chw(pred_flow: np.ndarray, height: int, width: int) -> np.ndarray:
    hwc = _pred_to_hwc(pred_flow, height, width)
    return np.transpose(hwc, (2, 0, 1))


def compute_normal_flow_scores(pred_flow, gt_flow):
    """Projection-based scores used for normal-flow evaluation (mean error, sharp-angle rate)."""
    gt = _gt_to_hwc(gt_flow)
    pred = _pred_to_hwc(pred_flow, gt.shape[0], gt.shape[1])

    pred_mask = np.logical_not(np.isnan(pred))
    gt_mask = np.logical_not(np.isnan(gt))
    intersection = np.logical_and(pred_mask, gt_mask)
    valid = intersection[:, :, 0]
    if not np.any(valid):
        raise ValueError("No overlapping valid pixels for normal-flow evaluation.")

    masked_pred = pred.copy()
    masked_gt = gt.copy()
    masked_pred[~intersection] = np.nan
    masked_gt[~intersection] = np.nan

    dot_product = np.sum(masked_gt * masked_pred, axis=-1)
    magnitude_pred = np.sqrt(np.sum(masked_pred**2, axis=-1))
    projection = dot_product / magnitude_pred
    error = magnitude_pred - projection

    sharp_angle_mask = dot_product > 0
    valid_sharp = np.logical_and(valid, sharp_angle_mask)

    return {
        "mean_error": float(np.nanmean(np.abs(error))),
        "sharp_angle_rate": float(np.count_nonzero(valid_sharp) / np.count_nonzero(valid)),
    }


def compute_optical_flow_projection_scores(pred_flow, gt_flow):
    """Projection-based scores for optical-flow baseline outputs (`*_latest.py` scripts)."""
    return compute_normal_flow_scores(pred_flow, gt_flow)


def compute_optical_flow_aee_scores(
    pred_flow,
    gt_flow,
    outlier_threshold=3.0,
):
    """Endpoint-error scores for optical-flow evaluation (`*_AEE_latest.py` scripts)."""
    if gt_flow.ndim == 3 and gt_flow.shape[0] != 2:
        gt = np.transpose(_gt_to_hwc(gt_flow), (2, 0, 1))
    elif gt_flow.ndim == 3:
        gt = gt_flow.astype(np.float64, copy=False)
        gt[:, np.linalg.norm(gt, axis=0) < 1e-5] = np.nan
    else:
        raise ValueError(f"Unsupported GT shape: {gt_flow.shape}")

    pred = _pred_to_chw(pred_flow, gt.shape[1], gt.shape[2])

    event_mask = np.logical_not(np.logical_and(np.equal(pred[0], 0), np.equal(pred[1], 0)))
    gt_valid_mask = np.logical_not(np.isnan(gt[0]))
    eval_mask = np.logical_and(event_mask, gt_valid_mask)
    if not np.any(eval_mask):
        raise ValueError("No valid event-supported pixels for AEE evaluation.")

    diff = pred - gt
    epe_map = np.sqrt(np.sum(diff**2, axis=0))
    valid_epe = epe_map[eval_mask]
    valid_epe = valid_epe[np.isfinite(valid_epe)]
    if valid_epe.size == 0:
        raise ValueError("No finite endpoint errors after masking.")

    outlier_pixels = int(np.count_nonzero(valid_epe > outlier_threshold))
    return {
        "aee": float(np.mean(valid_epe)),
        "percent_out": float(100.0 * np.mean(valid_epe > outlier_threshold)),
        "evaluated_pixels": int(valid_epe.size),
        "outlier_pixels": outlier_pixels,
        "percent_out_global": float(100.0 * outlier_pixels / valid_epe.size),
    }
