"""Training loop (Adam, weighted BCE, optional constraint loss, early stopping on validation F1).

Model selection uses ONLY validation data.  Test nodes are never touched during training, early stopping or
threshold selection.

Command line (no source editing needed)::

    python -m src.train --model rulegat --seed 42 --epochs 200 --lr 0.01 --hidden-dim 32 \
        --dropout 0.3 --lambda-constraint 0.5
    python -m src.train --model gat --set data.n_nodes=200 --set train.patience=30
"""
from __future__ import annotations

import argparse
import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import Tensor
from torch_geometric.data import Data

from .dataset import assert_no_leaky_features
from .losses import positive_class_weight, total_loss, weighted_bce
from .metrics import roc_auc_safe
from .models import MODEL_NAMES, build_model
from .utils import (apply_overrides, ensure_dir, get_logger, load_config, message_passing_edges, save_json,
                    set_seed)

CONSTRAINT_NODE_CHOICES = ("train", "all")


@dataclass
class TrainSettings:
    """Everything that defines one training run (saved next to the weights)."""

    model_name: str
    seed: int = 42
    hidden_dim: int = 32
    heads: int = 4
    dropout: float = 0.3
    attn_dropout: float = 0.0
    message_passing: str = "undirected"
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 300
    patience: int = 50
    use_pos_weight: bool = True
    lambda_constraint: float = 0.0
    constraint_nodes: str = "train"
    symbolic_weight: float = 0.5

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], model_name: str, seed: int, **overrides: Any) -> "TrainSettings":
        model, train, rg = cfg.get("model", {}), cfg.get("train", {}), cfg.get("rulegat", {})
        s = cls(model_name=model_name.lower(), seed=int(seed),
                hidden_dim=int(model.get("hidden_dim", 32)), heads=int(model.get("heads", 4)),
                dropout=float(model.get("dropout", 0.3)), attn_dropout=float(model.get("attn_dropout", 0.0)),
                message_passing=str(model.get("message_passing", "undirected")),
                lr=float(train.get("lr", 0.01)), weight_decay=float(train.get("weight_decay", 5e-4)),
                epochs=int(train.get("epochs", 300)), patience=int(train.get("patience", 50)),
                use_pos_weight=bool(train.get("use_pos_weight", True)),
                lambda_constraint=float(rg.get("lambda_constraint", 0.5)) if model_name.lower() == "rulegat" else 0.0,
                constraint_nodes=str(rg.get("constraint_nodes", "train")),
                symbolic_weight=float(rg.get("symbolic_weight", 0.5)))
        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(s, key):
                raise ValueError(f"Unknown training setting '{key}'.")
            setattr(s, key, value)
        s.validate()
        return s

    def validate(self) -> None:
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model '{self.model_name}'. Valid models: {list(MODEL_NAMES)}")
        if self.constraint_nodes not in CONSTRAINT_NODE_CHOICES:
            raise ValueError(f"constraint_nodes must be one of {CONSTRAINT_NODE_CHOICES}, got '{self.constraint_nodes}'")
        if self.lambda_constraint < 0 or self.symbolic_weight < 0:
            raise ValueError("lambda_constraint and symbolic_weight must be >= 0")
        if self.lambda_constraint > 0 and self.model_name != "rulegat":
            raise ValueError("Only the 'rulegat' model supports a constraint loss (lambda_constraint > 0).")
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be >= 1")

    def model_cfg(self) -> Dict[str, Any]:
        return {"hidden_dim": self.hidden_dim, "heads": self.heads, "dropout": self.dropout,
                "attn_dropout": self.attn_dropout}


@dataclass
class TrainResult:
    model: torch.nn.Module
    history: List[Dict[str, float]]
    best_epoch: int
    best_val_f1: float
    settings: TrainSettings
    train_time_s: float
    paths: Dict[str, str] = field(default_factory=dict)


def _val_f1_and_loss(logits: Tensor, data: Data, pos_weight: Optional[Tensor]) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = data.y.cpu().numpy()
    vm = data.val_mask.cpu().numpy()
    pred = (probs[vm] >= 0.5).astype(int)
    return {
        "val_f1": float(f1_score(y[vm], pred, zero_division=0)),
        "val_auc": roc_auc_safe(y[vm], probs[vm]),
        "val_loss": float(weighted_bce(logits, data.y, data.val_mask, pos_weight).item()),
    }


def train_model(data: Data, settings: TrainSettings, symbolic_score: Optional[np.ndarray] = None,
                save_dir: Optional[Path] = None, tag: Optional[str] = None, verbose: bool = False) -> TrainResult:
    """Train one model and return it with its best-validation weights restored.

    ``symbolic_score``: ``[N]`` 0/1 rule verdicts of the *observed* graph; required when
    ``settings.lambda_constraint > 0``.
    """
    settings.validate()
    assert_no_leaky_features(data.feature_names)
    for name in ("x", "edge_index", "y", "train_mask", "val_mask", "test_mask"):
        if getattr(data, name, None) is None:
            raise ValueError(f"Data object is missing '{name}'.")
    if settings.lambda_constraint > 0 and symbolic_score is None:
        raise ValueError("lambda_constraint > 0 needs symbolic_score (rule verdicts per node).")

    set_seed(settings.seed)
    n = int(data.num_nodes)
    x = data.x
    mp_edges = message_passing_edges(data.edge_index, n, settings.message_passing)
    sym = None if symbolic_score is None else torch.as_tensor(symbolic_score, dtype=torch.float32)
    constraint_mask = data.train_mask if settings.constraint_nodes == "train" else torch.ones(n, dtype=torch.bool)
    pos_weight = positive_class_weight(data.y, data.train_mask) if settings.use_pos_weight else None

    model = build_model(settings.model_name, x.shape[1], settings.model_cfg(), settings.symbolic_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)

    history: List[Dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_f1, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    start = time.time()

    for epoch in range(1, settings.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, mp_edges)
        loss, cls, cons = total_loss(logits, data.y, data.train_mask, sym, constraint_mask,
                                     settings.lambda_constraint, pos_weight)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val = _val_f1_and_loss(model(x, mp_edges), data, pos_weight)
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "train_cls_loss": float(cls.item()),
                        "train_constraint_loss": float(cons.item()), **val})

        improved = (val["val_f1"] > best_f1 + 1e-9) or (abs(val["val_f1"] - best_f1) <= 1e-9 and val["val_loss"] < best_loss - 1e-9)
        if improved:
            best_f1, best_loss, best_epoch, stale = val["val_f1"], val["val_loss"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if verbose and (epoch % 25 == 0 or epoch == 1):
            print(f"  [{settings.model_name}] epoch {epoch:3d} loss {loss.item():.4f} "
                  f"val_f1 {val['val_f1']:.3f} val_loss {val['val_loss']:.4f}")
        if stale >= settings.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    result = TrainResult(model=model, history=history, best_epoch=best_epoch, best_val_f1=float(best_f1),
                         settings=settings, train_time_s=time.time() - start)

    if save_dir is not None:
        directory = ensure_dir(save_dir)
        stem = tag or f"{settings.model_name}_seed{settings.seed}"
        weights = directory / f"{stem}.pt"
        torch.save(model.state_dict(), weights)
        save_json(history, directory / f"{stem}_history.json")
        save_json({**asdict(settings), "best_epoch": best_epoch, "best_val_f1": best_f1,
                   "feature_names": list(data.feature_names)}, directory / f"{stem}_config.json")
        result.paths = {"weights": str(weights), "history": str(directory / f"{stem}_history.json"),
                        "config": str(directory / f"{stem}_config.json")}
    return result


def load_trained_model(weights_path: Path, settings: TrainSettings, in_dim: int) -> torch.nn.Module:
    """Rebuild a model from ``settings`` and load saved weights."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"No saved weights at {weights_path}.")
    model = build_model(settings.model_name, in_dim, settings.model_cfg(), settings.symbolic_weight)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and evaluate one model on the synthetic transaction graph.")
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--config", default=None, help="path to config.yaml (default: project config.yaml)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--lambda-constraint", type=float, default=None, help="RuleGAT constraint-loss weight")
    p.add_argument("--symbolic-weight", type=float, default=None, help="RuleGAT inference-time symbolic weight w")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="config override section.key=value (repeatable)")
    p.add_argument("--save", action="store_true", help="save weights/history/config under outputs/models")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    from .dataset import build_dataset
    from .evaluate import run_models
    from .utils import output_dir

    args = _build_parser().parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.overrides)
    seed = int(cfg["seed"]) if args.seed is None else args.seed
    for section, key, value in (("train", "epochs", args.epochs), ("train", "lr", args.lr),
                                ("model", "hidden_dim", args.hidden_dim), ("model", "dropout", args.dropout),
                                ("rulegat", "lambda_constraint", args.lambda_constraint),
                                ("rulegat", "symbolic_weight", args.symbolic_weight)):
        if value is not None:
            cfg.setdefault(section, {})[key] = value
    log = get_logger("rulegat.train")
    data = build_dataset(cfg, seed=seed)
    log.info("graph: %d nodes, %d edges, %d suspicious (rule-derived labels)", data.num_nodes,
             data.edge_index.shape[1], int(data.y.sum()))
    save_dir = output_dir(cfg, "models") if args.save else None
    runs = run_models(data, cfg, seed, models=(args.model,), include_symbolic=False, save_dir=save_dir, verbose=True)
    for run in runs.values():
        log.info("%-32s test@0.5: F1=%.3f  test@val-thr(%.2f): F1=%.3f  ROC-AUC=%.3f", run.display,
                 run.test_default["f1"], run.threshold_val, run.test_val_thr["f1"], run.test_val_thr["roc_auc"])


if __name__ == "__main__":
    main()
