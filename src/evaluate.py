"""Evaluation orchestration shared by ``run_demo.py`` and all ``experiments/*.py`` scripts.

For every model the protocol is identical:

1. train on ``train_mask`` (early stopping on ``val_mask``),
2. score all nodes,
3. choose a decision threshold on the **validation** nodes only (maximising F1),
4. report the TEST metrics twice, clearly labelled:
       * ``threshold=0.5``            (default threshold)
       * ``val-selected threshold``   (threshold from step 3, applied to test unchanged)

The test set is never used to pick thresholds, hyper-parameters or checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from .metrics import (classification_metrics, default_threshold_grid, roc_auc_safe, rule_consistency,
                      rule_detection_rates, select_threshold)
from .rules import (RuleResult, RuleThresholds, evaluate_all_rules, rule_flag_matrix, symbolic_score)
from .train import TrainSettings, load_trained_model, train_model
from .utils import load_json, message_passing_edges

DISPLAY_NAMES: Dict[str, str] = {
    "mlp": "MLP",
    "gcn": "GCN",
    "gat": "GAT",
    "rulegat": "RuleGAT",
    "rulegat_neural": "RuleGAT (neural branch only)",
    "symbolic": "Symbolic rules only (reference)",
}
MODEL_ORDER: Sequence[str] = ("mlp", "gcn", "gat", "rulegat", "rulegat_neural", "symbolic")
SETTING_DEFAULT = "threshold=0.5"
SETTING_VAL = "val-selected threshold"


@dataclass
class ModelRun:
    """Everything produced by one (model, seed) run."""

    key: str
    seed: int
    scores: np.ndarray                       # decision score per node (in [0, 1])
    neural_prob: Optional[np.ndarray]        # neural probability per node (None for the symbolic reference)
    threshold_val: float
    val_f1: float
    test_default: Dict[str, float]
    test_val_thr: Dict[str, float]
    rule_default: Dict[str, float]
    rule_val_thr: Dict[str, float]
    consistency_default: float
    consistency_val_thr: float
    history: Optional[List[Dict[str, float]]] = None
    best_epoch: Optional[int] = None
    train_time_s: float = 0.0
    settings: Optional[TrainSettings] = None
    model: Optional[torch.nn.Module] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def display(self) -> str:
        return DISPLAY_NAMES.get(self.key, self.key)

    def rows(self, extra: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        """Two flat result rows (one per threshold setting) -- ready for a CSV table."""
        out = []
        for setting, metrics, rule, cons in (
                (SETTING_DEFAULT, self.test_default, self.rule_default, self.consistency_default),
                (SETTING_VAL, self.test_val_thr, self.rule_val_thr, self.consistency_val_thr)):
            row: Dict[str, Any] = {"model": self.display, "model_key": self.key, "seed": self.seed,
                                   "setting": setting, "threshold": metrics["threshold"]}
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                row[k] = metrics[k]
            row["rule_violation_detection_rate"] = rule["rule_violation_detection_rate"]
            row["rule_consistency"] = cons
            row["n_violating_any"] = rule["n_violating_any"]
            for rid in ("R1", "R2", "R3", "R4", "R5"):
                row[f"{rid}_detection_rate"] = rule[f"{rid}_detection_rate"]
                row[f"{rid}_n_violating"] = rule[f"{rid}_n_violating"]
            row["n_test"] = metrics["n"]
            row["n_test_pos"] = metrics["n_pos"]
            row["val_f1_at_threshold"] = self.val_f1
            row["best_epoch"] = self.best_epoch
            row["train_time_s"] = round(self.train_time_s, 3)
            if self.settings is not None:
                row["lambda_constraint"] = self.settings.lambda_constraint
                row["constraint_nodes"] = self.settings.constraint_nodes
                row["symbolic_weight"] = self.settings.symbolic_weight
            if extra:
                row.update(extra)
            out.append(row)
        return out


def evaluate_scores(key: str, seed: int, data: Data, scores: np.ndarray, neural_prob: Optional[np.ndarray],
                    flag_matrix: np.ndarray, sym: np.ndarray, grid: np.ndarray,
                    **run_kwargs: Any) -> ModelRun:
    """Threshold selection on validation + test metrics for one score vector."""
    y = data.y.cpu().numpy()
    val = data.val_mask.cpu().numpy()
    test = data.test_mask.cpu().numpy()
    thr, val_f1 = select_threshold(y[val], scores[val], grid)           # <-- validation data only
    test_default = classification_metrics(y[test], scores[test], 0.5)
    test_val = classification_metrics(y[test], scores[test], thr)
    pred_default = scores >= 0.5
    pred_val = scores >= thr
    return ModelRun(
        key=key, seed=seed, scores=scores, neural_prob=neural_prob, threshold_val=thr, val_f1=val_f1,
        test_default=test_default, test_val_thr=test_val,
        rule_default=rule_detection_rates(pred_default, flag_matrix, mask=test),
        rule_val_thr=rule_detection_rates(pred_val, flag_matrix, mask=test),
        consistency_default=rule_consistency(pred_default, sym, mask=test),
        consistency_val_thr=rule_consistency(pred_val, sym, mask=test),
        **run_kwargs)


def _predict(model: torch.nn.Module, data: Data, settings: TrainSettings,
             sym_t: torch.Tensor) -> Dict[str, np.ndarray]:
    mp = message_passing_edges(data.edge_index, int(data.num_nodes), settings.message_passing)
    model.eval()
    with torch.no_grad():
        if settings.model_name == "rulegat":
            p, final = model.hybrid_scores(data.x, mp, sym_t)
            return {"neural": p.cpu().numpy(), "final": final.cpu().numpy()}
        p = torch.sigmoid(model(data.x, mp)).cpu().numpy()
        return {"neural": p, "final": p}


def run_models(data: Data, cfg: Mapping[str, Any], seed: int,
               models: Sequence[str] = ("mlp", "gcn", "gat", "rulegat"), include_symbolic: bool = True,
               rulegat_overrides: Optional[Mapping[str, Any]] = None, save_dir: Optional[Path] = None,
               load_dir: Optional[Path] = None, tag_prefix: str = "", verbose: bool = False,
               rule_results: Optional[Mapping[str, RuleResult]] = None) -> Dict[str, ModelRun]:
    """Train (or load), score and evaluate ``models`` on ``data``.

    The symbolic branch runs on ``data`` itself (the *observed* graph).  ``data.y`` is the ground truth.
    ``rulegat`` additionally yields a ``rulegat_neural`` entry: the same trained network scored WITHOUT the
    symbolic branch, which separates "learned with the constraint loss" from "rules used at inference".
    """
    thresholds = RuleThresholds.from_config(cfg.get("rules"))
    results = rule_results if rule_results is not None else evaluate_all_rules(data, thresholds)
    flags = rule_flag_matrix(results)
    sym = symbolic_score(results)
    sym_t = torch.as_tensor(sym, dtype=torch.float32)
    grid = default_threshold_grid(int(cfg.get("threshold", {}).get("grid_size", 99)))
    runs: Dict[str, ModelRun] = {}

    for name in models:
        overrides = dict(rulegat_overrides or {}) if name == "rulegat" else {}
        settings = TrainSettings.from_config(cfg, name, seed, **overrides)
        stem = f"{tag_prefix}{name}_seed{seed}"
        trained = None
        if load_dir is not None and (Path(load_dir) / f"{stem}.pt").exists():
            try:
                model = load_trained_model(Path(load_dir) / f"{stem}.pt", settings, data.x.shape[1])
                hist_path = Path(load_dir) / f"{stem}_history.json"
                history = load_json(hist_path) if hist_path.exists() else None
                trained = (model, history, None, 0.0)
            except (RuntimeError, ValueError):
                trained = None                                           # incompatible checkpoint -> retrain
        if trained is None:
            res = train_model(data, settings, symbolic_score=sym, save_dir=save_dir, tag=stem, verbose=verbose)
            trained = (res.model, res.history, res.best_epoch, res.train_time_s)
        model, history, best_epoch, seconds = trained
        pred = _predict(model, data, settings, sym_t)
        common = dict(history=history, best_epoch=best_epoch, train_time_s=seconds, settings=settings, model=model)
        runs[name] = evaluate_scores(name, seed, data, pred["final"], pred["neural"], flags, sym, grid, **common)
        if name == "rulegat":
            runs["rulegat_neural"] = evaluate_scores("rulegat_neural", seed, data, pred["neural"], pred["neural"],
                                                     flags, sym, grid, **common)

    if include_symbolic:
        runs["symbolic"] = evaluate_scores("symbolic", seed, data, sym.astype(np.float64), None, flags, sym, grid)

    return {k: runs[k] for k in MODEL_ORDER if k in runs}


def runs_to_rows(runs: Mapping[str, ModelRun], extra: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Flatten a ``run_models`` result into CSV rows."""
    rows: List[Dict[str, Any]] = []
    for run in runs.values():
        rows.extend(run.rows(extra))
    return rows
