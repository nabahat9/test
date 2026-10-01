"""Experiments A and C: MLP vs GCN vs GAT vs RuleGAT, threshold 0.5 vs validation-selected threshold.

    python experiments/run_baselines.py                    # master seed only
    python experiments/run_baselines.py --seeds 42 43 44   # several seeds (one row set per seed)

Outputs (outputs/tables): baseline_results.csv / .json; (outputs/figures): baseline_*.png
"""
from __future__ import annotations

from _common import graph_seed_for, make_parser, save_run_config, save_table, setup

import pandas as pd

from src.dataset import build_dataset
from src.evaluate import runs_to_rows, run_models
from src.utils import output_dir
from src.visualization import plot_f1_vs_threshold, plot_roc_curves, plot_training_curves


def main() -> None:
    args = make_parser(__doc__).parse_args()
    cfg, seeds, log = setup(args, "baselines", default_single_seed=True)
    save_run_config(cfg, seeds, "baselines")
    rows, first = [], None
    for seed in seeds:
        data = build_dataset(cfg, seed=seed, graph_seed=graph_seed_for(args, cfg, seed))
        runs = run_models(data, cfg, seed)
        rows.extend(runs_to_rows(runs))
        first = first or (data, runs)
        for run in runs.values():
            log.info("seed %d | %-32s thr=0.5: F1 %.3f AUC %.3f | val-thr %.2f: F1 %.3f", seed, run.display,
                     run.test_default["f1"], run.test_default["roc_auc"], run.threshold_val, run.test_val_thr["f1"])
    df = pd.DataFrame(rows)
    save_table(df, "baseline_results", cfg, log)

    if not args.no_plots and first is not None:
        data, runs = first
        figs = output_dir(cfg, "figures")
        test, val = data.test_mask.numpy(), data.val_mask.numpy()
        y = data.y.numpy()
        learned = {r.display: r for r in runs.values() if r.key != "symbolic"}
        plot_training_curves({r.display: r.history for r in runs.values() if r.history and r.key != "rulegat_neural"},
                             figs / "baseline_training_curves.png")
        plot_roc_curves(y[test], {n: r.scores[test] for n, r in learned.items()}, figs / "baseline_roc.png")
        plot_f1_vs_threshold(y[val], {n: r.scores[val] for n, r in learned.items()}, y[test],
                             {n: r.scores[test] for n, r in learned.items()},
                             {n: r.threshold_val for n, r in learned.items()}, figs / "baseline_f1_vs_threshold.png")


if __name__ == "__main__":
    main()
