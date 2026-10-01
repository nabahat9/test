"""PyG dataset construction: features, rule-derived labels, splits, statistics, graph noise.

Terminology (kept explicit on purpose -- these four things are NOT the same):

* **injection targets**   nodes the generator deliberately made part of a pattern
                          (``injected_anchor_mask`` = one designated target per pattern,
                          ``injected_participant_mask`` = anchors + ring accomplices)
* **rule violations**     (node, rule) pairs found by the symbolic rule engine on the finished graph
* **supervised labels**   ``data.y[v] = 1`` iff node v violates at least one rule on the *clean* graph
* **symbolic predictions** what the rule engine outputs on the graph a model is *given* (the observed graph)
                          -- identical to the labels on the clean graph, different under graph noise

Leakage policy: the neural feature matrix ``x`` contains only ``age_days/2500``, ``device_changes/8`` and
the sine/cosine of the login hour (plus optional degree features).  ``blocked``, risk scores, labels and any
rule output are never placed in ``x``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch_geometric.data import Data

from .data_generation import GenerationConfig, RawGraph, generate_raw_graph
from .rules import (RULE_IDS, RuleResult, RuleThresholds, evaluate_all_rules, flagged_nodes,
                    rule_flag_matrix, violated_rules)
from .utils import PathLike, resolve_path

AGE_SCALE = 2500.0
DEVICE_SCALE = 8.0
DEGREE_LOG_SCALE = float(np.log1p(15.0))

BASE_FEATURE_NAMES: List[str] = ["age_norm", "device_changes_norm", "login_hour_sin", "login_hour_cos"]
STRUCTURAL_FEATURE_NAMES: List[str] = ["out_degree_log", "in_degree_log"]
# Anything that would trivially encode the target must never be a neural input feature.
FORBIDDEN_FEATURE_KEYWORDS: Tuple[str, ...] = ("blocked", "risk", "label", "target", "rule", "symbolic", "violation")


def assert_no_leaky_features(feature_names: Sequence[str]) -> None:
    """Raise if any feature name suggests a target-leaking variable."""
    bad = [f for f in feature_names if any(k in f.lower() for k in FORBIDDEN_FEATURE_KEYWORDS)]
    if bad:
        raise ValueError(f"Target-leaking feature(s) not allowed as neural input: {bad}")


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def degree_features(edge_index: Tensor, num_nodes: int) -> Tensor:
    """log(1+out-degree), log(1+in-degree), scaled -- computed from the *observed* edges only."""
    out_deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
    in_deg = torch.bincount(edge_index[1], minlength=num_nodes).float()
    return torch.stack([torch.log1p(out_deg), torch.log1p(in_deg)], dim=1) / DEGREE_LOG_SCALE


def build_node_features(age_days: np.ndarray, device_changes: np.ndarray, login_hour: np.ndarray,
                        edge_index: Optional[Tensor] = None, structural: bool = False) -> Tuple[Tensor, List[str]]:
    """Neural feature matrix ``x`` and its column names (no target-leaking columns)."""
    hour = np.asarray(login_hour, dtype=np.float64)
    cols = [
        np.asarray(age_days, dtype=np.float64) / AGE_SCALE,
        np.asarray(device_changes, dtype=np.float64) / DEVICE_SCALE,
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
    ]
    x = torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)
    names = list(BASE_FEATURE_NAMES)
    if structural:
        if edge_index is None:
            raise ValueError("structural=True requires edge_index.")
        x = torch.cat([x, degree_features(edge_index, x.shape[0])], dim=1)
        names += STRUCTURAL_FEATURE_NAMES
    assert_no_leaky_features(names)
    return x, names


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def make_split_masks(y: np.ndarray, train: float, val: float, test: float,
                     seed: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Reproducible stratified train/val/test node masks (disjoint, covering all nodes)."""
    total = train + val + test
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1, got {total:.4f}")
    n = len(y)
    idx = np.arange(n)
    y = np.asarray(y).astype(int)
    stratify_ok = min(np.bincount(y, minlength=2)) >= 3

    def split(indices: np.ndarray, size: float, labels: np.ndarray):
        return train_test_split(indices, test_size=size, random_state=seed,
                                stratify=labels if stratify_ok else None)

    train_val_idx, test_idx = split(idx, test, y)
    val_rel = val / (train + val)
    train_idx, val_idx = split(train_val_idx, val_rel, y[train_val_idx])
    masks = []
    for chosen in (train_idx, val_idx, test_idx):
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.as_tensor(np.sort(chosen), dtype=torch.long)] = True
        masks.append(m)
    return masks[0], masks[1], masks[2]


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
def dataset_from_raw(raw: RawGraph, thresholds: RuleThresholds, split: Mapping[str, float],
                     split_seed: int, structural: bool = False) -> Data:
    """Assemble the PyG object; labels ``y`` come from rule violations on the (clean) graph."""
    edge_index = torch.tensor(np.stack([raw.src, raw.dst]), dtype=torch.long)
    edge_amount = torch.tensor(raw.amount, dtype=torch.float32)
    x, names = build_node_features(raw.age_days, raw.device_changes, raw.login_hour, edge_index, structural)
    n = raw.n_nodes

    data = Data(x=x, edge_index=edge_index, edge_amount=edge_amount, num_nodes=n)
    data.age_days = torch.tensor(raw.age_days, dtype=torch.long)
    data.blocked = torch.tensor(raw.blocked, dtype=torch.bool)          # symbolic metadata only (needed by R1)
    data.feature_names = names
    data.structural_features = bool(structural)
    data.device_changes = torch.tensor(raw.device_changes, dtype=torch.long)
    data.login_hour = torch.tensor(raw.login_hour, dtype=torch.long)
    data.injected_anchor_mask = torch.tensor(raw.anchor_mask, dtype=torch.bool)
    data.injected_participant_mask = torch.tensor(raw.participant_mask, dtype=torch.bool)
    data.decoy_mask = torch.tensor(raw.decoy_mask, dtype=torch.bool)
    data.injected_pattern = torch.tensor(raw.pattern_rule, dtype=torch.long)
    data.rings = [tuple(r) for r in raw.rings]
    data.decoy_kind = dict(raw.decoy_kind)

    results = evaluate_all_rules(data, thresholds)
    y_np = np.zeros(n, dtype=np.int64)
    y_np[flagged_nodes(results)] = 1
    data.y = torch.tensor(y_np, dtype=torch.long)                        # rule-derived supervised label

    tr, va, te = make_split_masks(y_np, split.get("train", 0.6), split.get("val", 0.2),
                                  split.get("test", 0.2), split_seed)
    data.train_mask, data.val_mask, data.test_mask = tr, va, te
    return data


def build_dataset(cfg: Mapping[str, Any], seed: Optional[int] = None,
                  graph_seed: Optional[int] = None) -> Data:
    """Generate the synthetic graph described by ``cfg`` and return the PyG ``Data`` object.

    ``seed``        controls the train/val/test split (and is the default graph seed)
    ``graph_seed``  if given, the graph itself is generated with this seed instead
    """
    seed = int(cfg.get("seed", 42)) if seed is None else int(seed)
    gen_seed = seed if graph_seed is None else int(graph_seed)
    data_cfg = cfg.get("data", {})
    gen_cfg = GenerationConfig.from_config(data_cfg, cfg.get("rules"), seed=gen_seed)
    raw = generate_raw_graph(gen_cfg)
    return dataset_from_raw(raw, gen_cfg.thresholds, cfg.get("split", {}), split_seed=seed,
                            structural=bool(data_cfg.get("structural_features", False)))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def graph_statistics(data: Data, thresholds: Optional[RuleThresholds] = None,
                     results: Optional[Mapping[str, RuleResult]] = None) -> Dict[str, Any]:
    """Graph-level statistics including injected-vs-flagged bookkeeping (nothing is assumed equal)."""
    results = results if results is not None else evaluate_all_rules(data, thresholds)
    n = int(data.num_nodes)
    flagged = np.zeros(n, dtype=bool)
    flagged[flagged_nodes(results)] = True
    anchors = data.injected_anchor_mask.numpy()
    parts = data.injected_participant_mask.numpy()
    decoys = data.decoy_mask.numpy()
    flags = rule_flag_matrix(results)
    per_rule = {rid: {"nodes_violating": int(len(results[rid].nodes)),
                      "violations": int(results[rid].n_violations)} for rid in RULE_IDS}
    overlap = {f"{a}&{b}": int(((flags[:, i] > 0) & (flags[:, j] > 0)).sum())
               for i, a in enumerate(RULE_IDS) for j, b in enumerate(RULE_IDS) if i < j}
    y = data.y.numpy()
    return {
        "n_nodes": n,
        "n_edges": int(data.edge_index.shape[1]),
        "n_blocked_accounts": int(data.blocked.sum()),
        "n_injected_anchor_nodes": int(anchors.sum()),
        "n_injected_participant_nodes": int(parts.sum()),
        "n_decoy_nodes": int(decoys.sum()),
        "n_symbolic_flagged_nodes": int(flagged.sum()),
        "n_node_rule_violations": int(sum(v["nodes_violating"] for v in per_rule.values())),
        "n_r3_cycles": int(results["R3"].n_violations),
        "n_nodes_violating_multiple_rules": int((flags.sum(axis=1) > 1).sum()),
        "per_rule": per_rule,
        "rule_pair_overlap": overlap,
        "injected_vs_flagged": {
            "anchors_flagged": int((anchors & flagged).sum()),
            "anchors_not_flagged": int((anchors & ~flagged).sum()),
            "participants_flagged": int((parts & flagged).sum()),
            "participants_not_flagged": int((parts & ~flagged).sum()),
            "flagged_but_not_participant": int((flagged & ~parts).sum()),
            "decoys_flagged": int((decoys & flagged).sum()),
        },
        "n_label_positive": int(y.sum()),
        "label_positive_rate": float(y.mean()),
        "n_train": int(data.train_mask.sum()), "n_val": int(data.val_mask.sum()), "n_test": int(data.test_mask.sum()),
        "positives_train": int(y[data.train_mask.numpy()].sum()),
        "positives_val": int(y[data.val_mask.numpy()].sum()),
        "positives_test": int(y[data.test_mask.numpy()].sum()),
        "n_features": int(data.x.shape[1]),
        "feature_names": list(data.feature_names),
    }


# --------------------------------------------------------------------------- #
# Graph noise (robustness experiment)
# --------------------------------------------------------------------------- #
def perturb_graph(data: Data, add_frac: float = 0.0, rewire_frac: float = 0.0, seed: int = 0,
                  amount_median: float = 400.0, amount_sigma: float = 0.9,
                  amount_max: float = 5000.0) -> Data:
    """Return a copy of ``data`` whose *observed* transaction graph is noisy.

    ``add_frac``     add ``round(add_frac * E)`` irrelevant edges (random endpoints, small amounts)
    ``rewire_frac``  redirect the destination of ``round(rewire_frac * E)`` random edges (edge perturbation)

    Labels ``y``, masks and node metadata are untouched: y stays the *clean-graph* ground truth, so noise
    measures how well a model copes with an imperfectly observed graph.  Edge arrays are re-sorted jointly
    so ``edge_index`` and ``edge_amount`` stay aligned.
    """
    if add_frac < 0 or rewire_frac < 0 or rewire_frac > 1:
        raise ValueError("add_frac must be >= 0 and rewire_frac must be within [0, 1].")
    rng = np.random.default_rng(seed)
    n = int(data.num_nodes)
    src = data.edge_index[0].numpy().copy()
    dst = data.edge_index[1].numpy().copy()
    amount = data.edge_amount.numpy().astype(np.float64).copy()
    e = len(src)
    existing = set(zip(src.tolist(), dst.tolist()))

    n_rewire = int(round(rewire_frac * e))
    if n_rewire > 0:
        for i in rng.choice(e, size=n_rewire, replace=False):
            for _ in range(20):                                   # avoid self loops / duplicates
                v = int(rng.integers(0, n))
                if v != src[i] and (int(src[i]), v) not in existing:
                    existing.discard((int(src[i]), int(dst[i])))
                    existing.add((int(src[i]), v))
                    dst[i] = v
                    break

    n_add = int(round(add_frac * e))
    new_src, new_dst, new_amt = [], [], []
    attempts = 0
    while len(new_src) < n_add and attempts < 50 * max(n_add, 1):
        attempts += 1
        u, v = int(rng.integers(0, n)), int(rng.integers(0, n))
        if u == v or (u, v) in existing:
            continue
        existing.add((u, v))
        new_src.append(u)
        new_dst.append(v)
        new_amt.append(round(float(min(rng.lognormal(np.log(amount_median), amount_sigma), amount_max)), 2))
    if new_src:
        src = np.concatenate([src, new_src])
        dst = np.concatenate([dst, new_dst])
        amount = np.concatenate([amount, new_amt])

    order = np.lexsort((dst, src))
    noisy = data.clone()
    noisy.edge_index = torch.tensor(np.stack([src[order], dst[order]]), dtype=torch.long)
    noisy.edge_amount = torch.tensor(amount[order], dtype=torch.float32)
    if getattr(data, "structural_features", False):
        base = len(BASE_FEATURE_NAMES)
        noisy.x = torch.cat([noisy.x[:, :base], degree_features(noisy.edge_index, n)], dim=1)
    return noisy


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_dataset(data: Data, path: PathLike) -> str:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, p)
    return str(p)


def load_dataset(path: PathLike) -> Data:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"No saved dataset at {p}.")
    return torch.load(p, weights_only=False)
