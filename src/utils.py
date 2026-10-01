"""Shared helpers: seeding, config IO, filesystem, logging and edge utilities."""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch
import yaml
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

PathLike = Union[str, os.PathLike]


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch and (optionally) request deterministic kernels."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # warn_only: some scatter kernels have no deterministic variant; we warn instead of crashing.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_config(path: Optional[PathLike] = None) -> Dict[str, Any]:
    """Load the YAML configuration (default: ``<project>/config.yaml``)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}. Pass --config <file> or create config.yaml."
        )
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {cfg_path} must contain a YAML mapping at the top level.")
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Return a deep copy of ``cfg`` with ``section.key=value`` overrides applied.

    Values are parsed with ``yaml.safe_load`` so ``0.1``, ``true`` and ``[1,2]`` get proper types.
    """
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must have the form section.key=value")
        dotted, raw = item.split("=", 1)
        node = out
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override '{dotted}': '{part}' is not a config section.")
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def resolve_path(path: PathLike) -> Path:
    """Absolute paths are kept; relative ones are interpreted relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(path: PathLike) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_dir(cfg: Dict[str, Any], key: str) -> Path:
    """Directory for one of the ``paths:`` entries of the config (created if missing)."""
    defaults = {
        "generated": "data/generated",
        "models": "outputs/models",
        "figures": "outputs/figures",
        "tables": "outputs/tables",
        "logs": "outputs/logs",
    }
    if key not in defaults:
        raise KeyError(f"Unknown output directory '{key}'. Valid keys: {sorted(defaults)}")
    return ensure_dir(cfg.get("paths", {}).get(key, defaults[key]))


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def to_serializable(obj: Any) -> Any:
    """Recursively convert numpy/torch/Path objects to JSON-friendly types (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Tensor):
        return to_serializable(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return to_serializable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return to_serializable({k: getattr(obj, k) for k in obj.__dataclass_fields__})
    return obj


def save_json(obj: Any, path: PathLike) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as handle:
        json.dump(to_serializable(obj), handle, indent=2)
    return p


def load_json(path: PathLike) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, log_file: Optional[PathLike] = None, level: int = logging.INFO) -> logging.Logger:
    """Console (+ optional file) logger; safe to call repeatedly."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)
    if log_file is not None:
        target = resolve_path(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == target
                   for h in logger.handlers):
            file_handler = logging.FileHandler(target, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
    return logger


# --------------------------------------------------------------------------- #
# Graph helpers
# --------------------------------------------------------------------------- #
def message_passing_edges(edge_index: Tensor, num_nodes: int, mode: str = "undirected") -> Tensor:
    """Edge index used for GNN message passing.

    ``mode='directed'``   -> the transaction graph as is (messages flow source -> destination).
    ``mode='undirected'`` -> both directions, so a sender also "sees" its receivers. The *rules* and the
                             stored ``data.edge_index`` always stay directed; only the GNN's message
                             passing graph is symmetrised.
    """
    if mode == "directed":
        return edge_index
    if mode == "undirected":
        from torch_geometric.utils import to_undirected

        return to_undirected(edge_index, num_nodes=num_nodes)
    raise ValueError(f"Unknown message_passing mode '{mode}'. Use 'directed' or 'undirected'.")
