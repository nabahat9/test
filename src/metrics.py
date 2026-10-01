"""Metrics: classification, threshold selection, symbolic-rule metrics and multi-seed aggregation.

Why precision / recall / F1 and not accuracy alone
--------------------------------------------------
Suspicious nodes are a minority (~20 % here).  A model that predicts "normal" for everyone already reaches
~80 % accuracy while detecting nothing.  *Recall* = share of suspicious nodes that are found, *precision* =
share of raised alarms that are correct, *F1* = their harmonic mean.  ROC-AUC measures ranking quality
independent of any threshold (undefined -- returned as NaN -- if only one class is present).

Zero-division convention (same as ``sklearn`` with ``zero_division=0``): precision, recall and F1 are 0.0
when their denominator is 0.

Symbolic-rule metrics (all evaluated on a node mask, by default the *test* nodes)
--------------------------------------------------------------------------------
Let ``pred_v`` in {0,1} be a model's binary decision, ``V_k`` the set of nodes violating rule Rk on the
*observed* graph, ``V = union_k V_k`` and ``s_v = 1[v in V]`` the symbolic verdict.

``R{k}_detection_rate``            |{v in V_k : pred_v = 1}| / |V_k|        (NaN if V_k is empty in the mask)
``rule_violation_detection_rate``  |{v in V   : pred_v = 1}| / |V|          (NaN if V is empty in the mask)
``rule_consistency``               mean over nodes of 1[pred_v == s_v]      (agreement with the symbolic verdict)

Caveat: with rule-derived labels on the *clean* graph, ``s_v == y_v`` so ``rule_consistency`` equals accuracy
and ``rule_violation_detection_rate`` equals recall.  They only differ when the observed graph is perturbed
(then the labels stay clean while the rules see the noisy graph).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .rules import RULE_IDS


# --------------------------------------------------------------------------- #
# Classification metrics
# --------------------------------------------------------------------------- #
def _check_binary_inputs(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true {y_true.shape} and y_score {y_score.shape} must have the same length.")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty set of nodes.")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must be binary (0/1).")
    return y_true.astype(int), y_score


def roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC, or NaN when it is mathematically undefined (only one class present)."""
    y_true, y_score = _check_binary_inputs(y_true, y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Accuracy, precision, recall, F1 (at ``threshold``) and ROC-AUC (threshold-free) + confusion counts."""
    y_true, y_score = _check_binary_inputs(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_safe(y_true, y_score),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": int(len(y_true)), "n_pos": int(y_true.sum()),
    }


# --------------------------------------------------------------------------- #
# Threshold selection (validation data only!)
# --------------------------------------------------------------------------- #
def default_threshold_grid(grid_size: int = 99) -> np.ndarray:
    """Candidate thresholds ``linspace(0.01, 0.99, grid_size)`` (rounded for stable tie handling)."""
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    return np.round(np.linspace(0.01, 0.99, grid_size), 4)


def f1_curve(y_true: np.ndarray, y_score: np.ndarray, grid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """F1 at every threshold of ``grid``."""
    y_true, y_score = _check_binary_inputs(y_true, y_score)
    grid = default_threshold_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    f1s = np.empty(len(grid))
    for i, thr in enumerate(grid):
        tp, fp, fn, _ = confusion_counts(y_true, (y_score >= thr).astype(int))
        f1s[i] = _prf(tp, fp, fn)[2]
    return grid, f1s


def select_threshold(y_val: np.ndarray, score_val: np.ndarray, grid: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Threshold maximising F1 on the **validation** nodes; returns ``(threshold, val_f1)``.

    Ties are broken towards the threshold closest to 0.5 (deterministic, least extreme).  This function only
    ever sees validation data -- the caller applies the returned threshold to the test set unchanged.
    """
    grid, f1s = f1_curve(y_val, score_val, grid)
    best = f1s.max()
    candidates = grid[np.isclose(f1s, best)]
    thr = float(candidates[np.argmin(np.abs(candidates - 0.5))])
    return thr, float(best)


# --------------------------------------------------------------------------- #
# Symbolic-rule metrics
# --------------------------------------------------------------------------- #
def _mask_or_all(mask: Optional[np.ndarray], n: int) -> np.ndarray:
    return np.ones(n, dtype=bool) if mask is None else np.asarray(mask).astype(bool)


def rule_detection_rates(pred: np.ndarray, flag_matrix: np.ndarray, rule_ids: Sequence[str] = RULE_IDS,
                         mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Per-rule and overall detection rates (see module docstring). NaN where no node violates in ``mask``."""
    pred = np.asarray(pred).astype(bool).reshape(-1)
    flags = np.asarray(flag_matrix) > 0
    if flags.shape != (len(pred), len(rule_ids)):
        raise ValueError(f"flag_matrix must have shape {(len(pred), len(rule_ids))}, got {flags.shape}")
    m = _mask_or_all(mask, len(pred))
    out: Dict[str, float] = {}
    any_v = flags.any(axis=1) & m
    out["rule_violation_detection_rate"] = float(pred[any_v].mean()) if any_v.any() else float("nan")
    out["n_violating_any"] = int(any_v.sum())
    for k, rid in enumerate(rule_ids):
        vk = flags[:, k] & m
        out[f"{rid}_detection_rate"] = float(pred[vk].mean()) if vk.any() else float("nan")
        out[f"{rid}_n_violating"] = int(vk.sum())
    return out


def rule_consistency(pred: np.ndarray, symbolic_verdict: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Share of nodes where the model's decision equals the symbolic verdict ``s_v`` (see module docstring)."""
    pred = np.asarray(pred).astype(int).reshape(-1)
    verdict = (np.asarray(symbolic_verdict) > 0).astype(int).reshape(-1)
    m = _mask_or_all(mask, len(pred))
    if not m.any():
        return float("nan")
    return float((pred[m] == verdict[m]).mean())


# --------------------------------------------------------------------------- #
# Multi-seed aggregation
# --------------------------------------------------------------------------- #
CLASSIFICATION_KEYS: Tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "roc_auc")


def aggregate_runs(df: pd.DataFrame, group_cols: Sequence[str],
                   metric_cols: Sequence[str] = CLASSIFICATION_KEYS) -> pd.DataFrame:
    """Mean and *sample* standard deviation (ddof=1) per group; NaN values are skipped.

    Output columns: ``group_cols`` + ``n_runs`` + ``mean_<m>`` / ``std_<m>`` for every metric ``m``.
    The std of a single run is NaN (undefined), never silently 0.
    """
    missing = [c for c in list(group_cols) + list(metric_cols) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from results table: {missing}")
    rows = []
    for keys, sub in df.groupby(list(group_cols), sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(sub))
        for m in metric_cols:
            values = sub[m].astype(float).to_numpy()
            valid = values[~np.isnan(values)]
            row[f"mean_{m}"] = float(valid.mean()) if len(valid) else float("nan")
            row[f"std_{m}"] = float(valid.std(ddof=1)) if len(valid) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def pool_rule_detection(df: pd.DataFrame, group_cols: Sequence[str],
                        rule_ids: Sequence[str] = RULE_IDS) -> pd.DataFrame:
    """Detection rates pooled over runs: (sum of detected violators) / (sum of violators).

    Per-run rule rates are often undefined (a small test split may contain no violator of a rare rule), so the
    plain mean over runs is based on very few runs.  Pooling weights every violating node equally instead.
    Needs the ``<rule>_detection_rate`` and ``<rule>_n_violating`` (+ ``n_violating_any``) columns written by
    ``ModelRun.rows``.  Output: group columns + ``<rule>_pooled_rate`` / ``<rule>_pooled_n`` (rate NaN if n = 0).
    """
    keys = [f"{r}" for r in rule_ids] + ["overall"]
    rate_col = {**{r: f"{r}_detection_rate" for r in rule_ids}, "overall": "rule_violation_detection_rate"}
    n_col = {**{r: f"{r}_n_violating" for r in rule_ids}, "overall": "n_violating_any"}
    missing = [c for c in list(group_cols) + list(rate_col.values()) + list(n_col.values()) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from results table: {missing}")
    rows = []
    for gkeys, sub in df.groupby(list(group_cols), sort=False):
        gkeys = gkeys if isinstance(gkeys, tuple) else (gkeys,)
        row = dict(zip(group_cols, gkeys))
        for k in keys:
            n = sub[n_col[k]].fillna(0).astype(float).to_numpy()
            rate = sub[rate_col[k]].astype(float).to_numpy()
            total = float(n.sum())
            detected = float(np.nansum(np.where(n > 0, rate * n, 0.0)))
            row[f"{k}_pooled_rate"] = detected / total if total > 0 else float("nan")
            row[f"{k}_pooled_n"] = int(total)
        rows.append(row)
    return pd.DataFrame(rows)


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "n/a"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def report_table(agg: pd.DataFrame, group_cols: Sequence[str],
                 metric_cols: Sequence[str] = CLASSIFICATION_KEYS, digits: int = 3) -> pd.DataFrame:
    """Compact "mean ± std" table suitable for a report."""
    out = agg[list(group_cols) + ["n_runs"]].copy()
    for m in metric_cols:
        out[m] = [format_mean_std(a, b, digits) for a, b in zip(agg[f"mean_{m}"], agg[f"std_{m}"])]
    return out
