"""Check one or several nodes: is it suspicious, and why?

    python check_node.py 25                 # one node (seed 42, models trained on the fly ~ a few seconds)
    python check_node.py 0 25 87            # several nodes
    python check_node.py --top 10           # the 10 highest-scoring nodes
    python check_node.py 25 --seed 43 --reuse-models

Output per node: neural score p, symbolic verdict s (which rules, with the actual values),
RuleGAT score (p + w*s)/(1 + w), decision at the validation-selected threshold, and whether
neural and symbolic reasoning agree.  SYNTHETIC data only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.dataset import build_dataset
from src.evaluate import run_models
from src.explain import format_node_report, node_report
from src.rules import RuleThresholds, evaluate_all_rules
from src.utils import apply_overrides, load_config, output_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nodes", nargs="*", type=int, help="node ids to check")
    ap.add_argument("--top", type=int, default=0, help="check the N nodes with the highest RuleGAT score")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--reuse-models", action="store_true", help="load weights from outputs/models if present")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    seed = int(cfg["seed"]) if args.seed is None else args.seed
    data = build_dataset(cfg, seed=seed)
    results = evaluate_all_rules(data, RuleThresholds.from_config(cfg.get("rules")))
    load_dir = output_dir(cfg, "models") if args.reuse_models else None
    runs = run_models(data, cfg, seed, models=("rulegat",), include_symbolic=False, load_dir=load_dir,
                      tag_prefix="demo_" if args.reuse_models else "")
    rg = runs["rulegat"]
    thr = rg.threshold_val                                  # chosen on the validation set only
    p, final = rg.neural_prob, rg.scores

    ids = list(args.nodes)
    if args.top:
        ids += [int(i) for i in np.argsort(-final, kind="stable")[:args.top]]
    if not ids:
        ap.error("give node ids or --top N")
    split = np.full(data.num_nodes, "train", dtype=object)
    split[data.val_mask.numpy()], split[data.test_mask.numpy()] = "val", "test"

    print(f"seed {seed} | {data.num_nodes} nodes | decision threshold (val-selected) = {thr:.2f}\n")
    for v in ids:
        if not 0 <= v < data.num_nodes:
            print(f"Node {v}: out of range (0..{data.num_nodes - 1})\n")
            continue
        rec = node_report(v, results, p, final, thr)
        d05, dval = final[v] >= 0.5, final[v] >= thr
        lab = lambda b: "SUSPECT" if b else "normal"
        print(format_node_report(rec))
        print(f"  => decision @0.50 (as in the README/slides): {lab(d05)}   |   @{thr:.2f} (val-selected): {lab(dval)}"
              f"   split: {split[v]}   true label: {int(data.y[v])}")
        if rec["rules_violated"] and not dval:
            print("  !  a rule is violated but the fused score stays under the val-selected threshold: "
                  "the rule verdict itself is the reliable alert here.")
        why = {"both": "neural and rules agree",
               "neural-only": "neural pattern, no rule violated (not explained by the rules)",
               "symbolic-only": "network missed it, a rule catches it (inspect the values above)",
               "neither": "no signal from either branch"}[rec["agreement"]]
        print(f"  why: {why}\n")


if __name__ == "__main__":
    main()