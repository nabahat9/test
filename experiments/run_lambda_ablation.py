"""Experiment B: RuleGAT with different constraint-loss weights lambda in {0, 0.1, 0.5, 1.0}.

For every seed, lambda and constraint-node scope the same trained network is scored twice:
    RuleGAT                        hybrid inference (neural + symbolic)
    RuleGAT (neural branch only)   the network alone -> isolates the effect of the *constraint loss*

    python experiments/run_lambda_ablation.py                              # both constraint-node scopes
    python experiments/run_lambda_ablation.py --constraint-nodes train     # train nodes only (no transductive use)

Outputs: rulegat_lambda_raw.csv (per run), rulegat_lambda_results.csv (mean/std over seeds), figure.
"""
from __future__ import annotations

from _common import graph_seed_for, make_parser, save_run_config, save_table, setup

import pandas as pd

from src.dataset import build_dataset
from src.evaluate import run_models, runs_to_rows
from src.metrics import aggregate_runs
from src.utils import output_dir
from src.visualization import plot_lambda_performance


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--constraint-nodes", nargs="+", default=["train", "all"], choices=["train", "all"],
                        help="nodes the constraint loss is applied to (default: both variants)")
    parser.add_argument("--lambdas", type=float, nargs="+", default=None)
    args = parser.parse_args()
    cfg, seeds, log = setup(args, "lambda_ablation")
    lambdas = args.lambdas if args.lambdas is not None else [float(v) for v in cfg["rulegat"]["lambda_grid"]]
    save_run_config(cfg, seeds, "lambda_ablation", {"lambdas": lambdas, "constraint_nodes": args.constraint_nodes})

    rows = []
    for seed in seeds:
        data = build_dataset(cfg, seed=seed, graph_seed=graph_seed_for(args, cfg, seed))
        for scope in args.constraint_nodes:
            for lam in lambdas:
                runs = run_models(data, cfg, seed, models=("rulegat",), include_symbolic=False,
                                  rulegat_overrides={"lambda_constraint": lam, "constraint_nodes": scope})
                rows.extend(runs_to_rows(runs, {"variant_scope": f"constraint_nodes={scope}"}))
                log.info("seed %d scope=%-5s lambda=%.1f | hybrid F1 %.3f | neural-only F1 %.3f AUC %.3f", seed, scope,
                         lam, runs["rulegat"].test_default["f1"], runs["rulegat_neural"].test_default["f1"],
                         runs["rulegat_neural"].test_default["roc_auc"])
    raw = pd.DataFrame(rows)
    save_table(raw, "rulegat_lambda_raw", cfg, log)
    agg = aggregate_runs(raw, ["model", "variant_scope", "setting", "lambda_constraint"])
    save_table(agg, "rulegat_lambda_results", cfg, log)

    if not args.no_plots:
        sub = agg[agg["setting"] == "threshold=0.5"].copy()
        sub["variant"] = sub["model"] + " | " + sub["variant_scope"]
        plot_lambda_performance(sub, output_dir(cfg, "figures") / "lambda_performance.png")


if __name__ == "__main__":
    main()
