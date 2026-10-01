"""Robustness experiment: GAT vs RuleGAT when the *observed* transaction graph is noisy.

Noise settings come from config.yaml (``robustness.settings``): irrelevant added edges and edge rewiring.
Labels stay the clean-graph, rule-derived ground truth; the models AND the rule engine only see the noisy graph.
The symbolic-only reference is included so the effect of noise on the rules themselves is visible.
No conclusion is pre-supposed -- read the resulting table.

    python experiments/run_robustness.py
    python experiments/run_robustness.py --seeds 42 43 44 45 46 --set robustness.settings=...

Outputs: robustness_raw.csv, robustness_results.csv, robustness_report.csv, figure robustness_f1.png
"""
from __future__ import annotations

from _common import graph_seed_for, make_parser, save_run_config, save_table, setup

import pandas as pd

from src.dataset import build_dataset, perturb_graph
from src.evaluate import run_models, runs_to_rows
from src.metrics import aggregate_runs, report_table
from src.utils import output_dir
from src.visualization import plot_robustness


def main() -> None:
    args = make_parser(__doc__).parse_args()
    cfg, seeds, log = setup(args, "robustness")
    settings = cfg["robustness"]["settings"]
    save_run_config(cfg, seeds, "robustness", {"fixed_graph": args.fixed_graph})
    data_cfg = cfg.get("data", {})
    rows = []
    for seed in seeds:
        clean = build_dataset(cfg, seed=seed, graph_seed=graph_seed_for(args, cfg, seed))
        for idx, s in enumerate(settings):
            noisy = perturb_graph(clean, add_frac=float(s["add_frac"]), rewire_frac=float(s["rewire_frac"]),
                                  seed=seed * 1000 + idx, amount_median=float(data_cfg.get("amount_median", 400.0)),
                                  amount_sigma=float(data_cfg.get("amount_sigma", 0.9)),
                                  amount_max=float(data_cfg.get("amount_max", 5000.0)))
            runs = run_models(noisy, cfg, seed, models=("gat", "rulegat"), include_symbolic=True)
            rows.extend(runs_to_rows(runs, {"noise": s["name"], "add_frac": s["add_frac"],
                                            "rewire_frac": s["rewire_frac"], "n_edges_observed": int(noisy.edge_index.shape[1])}))
            log.info("seed %d | %-13s | GAT F1 %.3f | RuleGAT F1 %.3f | symbolic-only F1 %.3f", seed, s["name"],
                     runs["gat"].test_default["f1"], runs["rulegat"].test_default["f1"], runs["symbolic"].test_default["f1"])
    raw = pd.DataFrame(rows)
    save_table(raw, "robustness_raw", cfg, log)
    groups = ["noise", "model", "setting"]
    agg = aggregate_runs(raw, groups)
    save_table(agg, "robustness_results", cfg, log)
    save_table(report_table(agg, groups), "robustness_report", cfg, log)
    if not args.no_plots:
        sub = agg[agg["setting"] == "threshold=0.5"]
        plot_robustness(sub, output_dir(cfg, "figures") / "robustness_f1.png", "f1", [s["name"] for s in settings])
    print(report_table(agg[agg["setting"] == "threshold=0.5"], groups).to_string(index=False))


if __name__ == "__main__":
    main()
