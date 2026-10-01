"""RuleGAT end-to-end demo (synthetic data only).

    python run_demo.py                     # full demo, saves figures to outputs/figures
    python run_demo.py --no-plots          # console only
    python run_demo.py --reuse-models      # load saved weights from outputs/models instead of retraining
    python run_demo.py --seed 43 --set train.epochs=150

Steps: generate graph -> statistics -> symbolic rules -> train/load MLP, GCN, GAT, RuleGAT -> validation-selected
threshold -> test metrics -> rule statistics -> automatic explanations -> neural/symbolic disagreements -> plots.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.dataset import build_dataset, graph_statistics, save_dataset
from src.evaluate import SETTING_DEFAULT, SETTING_VAL, run_models, runs_to_rows
from src.explain import (agreement_counts, disagreement_report, explain_flagged_nodes, format_node_report,
                         rule_statistics)
from src.rules import (RULE_DESCRIPTIONS, RULE_IDS, RuleThresholds, evaluate_all_rules, format_node_explanation,
                       violated_nodes)
from src.utils import apply_overrides, load_config, output_dir, save_json
from src.visualization import (plot_f1_vs_threshold, plot_graph, plot_roc_curves, plot_rule_detection_rates,
                               plot_rule_overlap, plot_training_curves)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def banner(title: str) -> None:
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="section.key=value")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--reuse-models", action="store_true", help="load saved weights when available")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    seed = int(cfg["seed"]) if args.seed is None else args.seed
    thresholds = RuleThresholds.from_config(cfg.get("rules"))
    models_dir, figs_dir, tables_dir = output_dir(cfg, "models"), output_dir(cfg, "figures"), output_dir(cfg, "tables")

    banner("RuleGAT demo -- SYNTHETIC banking transaction graph (not real banking data)")

    # 1-2. dataset + statistics ---------------------------------------------------------------------
    banner("1. Dataset and graph statistics")
    data = build_dataset(cfg, seed=seed)
    save_dataset(data, Path(cfg.get("paths", {}).get("generated", "data/generated")) / f"graph_seed{seed}.pt")
    results = evaluate_all_rules(data, thresholds)
    stats = graph_statistics(data, thresholds, results)
    ivf = stats["injected_vs_flagged"]
    print(f"nodes: {stats['n_nodes']}   directed transactions: {stats['n_edges']}   seed: {seed}")
    print(f"neural features ({stats['n_features']}): {stats['feature_names']}   "
          f"(blocked / risk score / labels / rule outputs are NOT neural inputs)")
    print(f"split (train/val/test): {stats['n_train']}/{stats['n_val']}/{stats['n_test']}   "
          f"positives: {stats['positives_train']}/{stats['positives_val']}/{stats['positives_test']}")
    print("\nInjected targets vs. symbolic detections (these sets are NOT assumed equal):")
    print(f"  injected anchor nodes (1 designated target per pattern) : {stats['n_injected_anchor_nodes']}")
    print(f"  injected participant nodes (anchors + ring accomplices) : {stats['n_injected_participant_nodes']}")
    print(f"  decoy nodes (near misses, intentionally NOT violating)  : {stats['n_decoy_nodes']}")
    print(f"  symbolically flagged nodes (>= 1 rule violated)         : {stats['n_symbolic_flagged_nodes']}")
    print(f"  (node, rule) violations: {stats['n_node_rule_violations']}   distinct large 3-cycles: {stats['n_r3_cycles']}")
    print(f"  anchors flagged: {ivf['anchors_flagged']}/{stats['n_injected_anchor_nodes']}   "
          f"participants flagged: {ivf['participants_flagged']}/{stats['n_injected_participant_nodes']}   "
          f"flagged but not injected: {ivf['flagged_but_not_participant']}   decoys flagged: {ivf['decoys_flagged']}")
    print(f"  supervised label y = 1 iff >= 1 rule violated on the clean graph -> "
          f"{stats['n_label_positive']} positives ({100 * stats['label_positive_rate']:.1f}%)")

    # 3. rules -------------------------------------------------------------------------------------------
    banner("2. Symbolic rules (evaluated on raw graph metadata)")
    rs = rule_statistics(results)
    for rid in RULE_IDS:
        print(f"  {rid}: {rs['nodes_per_rule'][rid]:3d} nodes, {rs['violations_per_rule'][rid]:3d} violations | "
              f"{RULE_DESCRIPTIONS[rid]}")
    print(f"  nodes violating more than one rule: {rs['nodes_violating_multiple_rules']}")
    print("  rule overlap matrix (rows/cols R1..R5, diagonal = nodes per rule):")
    print("   ", np.array2string(rs["overlap_matrix"], prefix="    "))

    # 4-8. models ------------------------------------------------------------------------------------------
    banner("3. Training / loading models (MLP, GCN, GAT, RuleGAT) + validation-selected thresholds")
    runs = run_models(data, cfg, seed, tag_prefix="demo_", save_dir=models_dir,
                      load_dir=models_dir if args.reuse_models else None, verbose=False)
    for run in runs.values():
        if run.best_epoch:
            print(f"  {run.display:<32s} best epoch {run.best_epoch:3d}   val F1 (thr 0.5) checkpoint; "
                  f"val-selected threshold = {run.threshold_val:.2f} (val F1 {run.val_f1:.3f})")

    # 9-11. metrics ------------------------------------------------------------------------------------------
    banner("4. TEST metrics")
    df = pd.DataFrame(runs_to_rows(runs))
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    for setting, label in ((SETTING_DEFAULT, "Setting A -- default threshold 0.5"),
                           (SETTING_VAL, "Setting B -- threshold selected on VALIDATION nodes (max val F1), applied unchanged to test")):
        sub = df[df["setting"] == setting]
        print(f"\n{label}")
        table = sub[cols].round(3).copy()
        if setting == SETTING_VAL:
            table.insert(1, "threshold", sub["threshold"].round(2).to_numpy())
        print(table.to_string(index=False))
    print("\nTest set is small (n = %d, %d positives): single-run numbers are noisy -> see experiments/run_multiseed.py."
          % (stats["n_test"], stats["positives_test"]))
    print("CAUTION: labels are defined BY the rules, so 'RuleGAT' (rules used at inference) and the symbolic-only\n"
          "reference are aligned with the labels by construction. Compare the neural branch rows for a fair\n"
          "neural-vs-neural view, and see experiments/run_robustness.py for a less circular test.")

    banner("5. Symbolic-rule metrics on TEST nodes (threshold 0.5)")
    rule_cols = ["model", "rule_violation_detection_rate", "rule_consistency"] + [f"{r}_detection_rate" for r in RULE_IDS]
    print(df[df["setting"] == SETTING_DEFAULT][rule_cols].round(3).to_string(index=False))
    print("(NaN = no test node violates that rule)")

    # 12-13. rule statistics, agreement, explanations -------------------------------------------------------------------
    rg = runs["rulegat"]
    neural_prob, final_score = rg.neural_prob, rg.scores
    banner("6. Neural vs symbolic agreement (neural branch of RuleGAT, threshold 0.5)")
    for name, mask in (("all nodes", None), ("test nodes", data.test_mask.numpy())):
        c = agreement_counts(neural_prob, results, 0.5, mask)
        print(f"  {name:<10s}: flagged by both = {c['both']:3d} | neural only = {c['neural_only']:3d} | "
              f"symbolic only = {c['symbolic_only']:3d} | neither = {c['neither']:3d}")

    banner("7. Automatic explanations")
    print("Q: What rules does each rule's first violating node violate?")
    for rid in RULE_IDS:
        nodes = violated_nodes(results)[rid]
        if nodes:
            print(format_node_explanation(nodes[0], results))
    print("\nRuleGAT reports for the highest-scoring flagged nodes:")
    for rec in explain_flagged_nodes(results, neural_prob, final_score, 0.5, max_nodes=6):
        print(format_node_report(rec))
        print()

    banner("8. Where neural and symbolic reasoning disagree (test nodes)")
    dis = disagreement_report(results, neural_prob, 0.5, data.test_mask.numpy(), max_nodes=3)
    for key, title in (("symbolic_only", "Rule violated but neural branch says 'normal' (neural miss):"),
                       ("neural_only", "Neural branch says 'suspicious' but no rule violated (learned alarm / false positive):")):
        print(title)
        if not dis[key]:
            print("  (none)")
        for rec in dis[key]:
            print("  " + format_node_report(rec).replace("\n", "\n  "))

    # 14. plots ---------------------------------------------------------------------------------------------------
    df.to_csv(tables_dir / "demo_results.csv", index=False)
    save_json({"graph_statistics": stats, "config": cfg, "seed": seed}, tables_dir / "demo_summary.json")
    if not args.no_plots:
        banner("9. Figures")
        y, test, val = data.y.numpy(), data.test_mask.numpy(), data.val_mask.numpy()
        learned = {r.display: r for r in runs.values() if r.key != "symbolic"}
        paths = [
            plot_training_curves({r.display: r.history for r in runs.values() if r.history and r.key != "rulegat_neural"},
                                 figs_dir / "demo_training_curves.png"),
            plot_roc_curves(y[test], {n: r.scores[test] for n, r in learned.items()}, figs_dir / "demo_roc.png"),
            plot_f1_vs_threshold(y[val], {n: r.scores[val] for n, r in learned.items()}, y[test],
                                 {n: r.scores[test] for n, r in learned.items()},
                                 {n: r.threshold_val for n, r in learned.items()}, figs_dir / "demo_f1_vs_threshold.png"),
            plot_rule_detection_rates({r.display: r.rule_default for r in runs.values()}, figs_dir / "demo_rule_detection.png"),
            plot_rule_overlap(rs["overlap_matrix"], figs_dir / "demo_rule_overlap.png"),
            plot_graph(data, results, figs_dir / "demo_graph.png", neural_prob=neural_prob),
        ]
        for p in paths:
            print("  saved", p)
    print("\nDone. Tables in", tables_dir, "| models in", models_dir)


if __name__ == "__main__":
    main()
