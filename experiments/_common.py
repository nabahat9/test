"""Shared plumbing for the experiment scripts (argument parsing, config, table saving)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.utils import (apply_overrides, get_logger, load_config, output_dir, save_json)  # noqa: E402


def make_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to config.yaml (default: project config.yaml)")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="seeds to run (default: config 'seeds')")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="config override section.key=value (repeatable)")
    p.add_argument("--fixed-graph", action="store_true",
                   help="generate the graph once with the master seed; seeds then only change split + initialisation")
    p.add_argument("--no-plots", action="store_true")
    return p


def setup(args: argparse.Namespace, name: str, default_single_seed: bool = False) -> Tuple[Dict[str, Any], List[int], Any]:
    cfg = apply_overrides(load_config(args.config), args.overrides)
    if args.seeds is not None:
        seeds = list(args.seeds)
    elif default_single_seed:
        seeds = [int(cfg["seed"])]
    else:
        seeds = [int(s) for s in cfg["seeds"]]
    log = get_logger(f"rulegat.{name}", output_dir(cfg, "logs") / f"{name}.log")
    log.info("config: seeds=%s fixed_graph=%s overrides=%s", seeds, args.fixed_graph, args.overrides)
    return cfg, seeds, log


def graph_seed_for(args: argparse.Namespace, cfg: Dict[str, Any], seed: int) -> int:
    """Graph seed: the run seed (independent replicates) or the master seed with ``--fixed-graph``."""
    return int(cfg["seed"]) if args.fixed_graph else int(seed)


def save_table(df: pd.DataFrame, name: str, cfg: Dict[str, Any], log=None) -> Path:
    """Write ``<name>.csv`` and ``<name>.json`` into outputs/tables."""
    directory = output_dir(cfg, "tables")
    csv_path = directory / f"{name}.csv"
    df.to_csv(csv_path, index=False, float_format="%.6g")
    save_json(df.to_dict(orient="records"), directory / f"{name}.json")
    if log is not None:
        log.info("wrote %s (%d rows)", csv_path, len(df))
    return csv_path


def save_run_config(cfg: Dict[str, Any], seeds: Sequence[int], name: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Persist the exact configuration of an experiment for reproducibility."""
    save_json({"experiment": name, "seeds": list(seeds), "config": cfg, **(extra or {})},
              output_dir(cfg, "logs") / f"{name}_config.json")
