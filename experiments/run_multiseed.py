"""Experiment D: repeat the model comparison over several seeds and report mean ± std.

Each seed changes the graph (unless --fixed-graph), the train/val/test split and the network initialisation.

    python experiments/run_multiseed.py                      # seeds 42..46 from config.yaml
    python experiments/run_multiseed.py --seeds 42 43 44 45 46 47 48
    python experiments/run_multiseed.py --fixed-graph        # same graph, different split/init

Outputs: multiseed_raw.csv, multiseed_results.csv (mean_/std_ per metric), multiseed_report.csv
(formatted "mean ± std"), multiseed_rule_metrics.csv (symbolic-rule metrics, mean over seeds where defined),
multiseed_rule_pooled.csv (rule detection pooled over all seeds: detected / violating) and its figure.
"""
from __future__ import annotations

from _common import graph_seed_for, make_parser, save_run_config, save_table, setup

import pandas as pd

from src.dataset import build_dataset
from src.evaluate import run_models, runs_to_rows
from src.metrics import aggregate_runs, pool_rule_detection, report_table
from src.utils import output_dir
from src.visualization import plot_rule_detection_rates

RULE_METRICS = ["rule_violation_detection_rate", "rule_consistency", "R1_detection_rate", "R2_detection_rate",
                "R3_detection_rate", "R4_detection_rate", "R5_detection_rate"]


def main() -> None:
    args = make_parser(__doc__).parse_args()
    cfg, seeds, log = setup(args, "multiseed")
    save_run_config(cfg, seeds, "multiseed", {"fixed_graph": args.fixed_graph})
    rows = []
    for seed in seeds:
        data = build_dataset(cfg, seed=seed, graph_seed=graph_seed_for(args, cfg, seed))
        runs = run_models(data, cfg, seed)
        rows.extend(runs_to_rows(runs))
        log.info("seed %d done | test positives: %d / %d", seed, int(data.y[data.test_mask].sum()), int(data.test_mask.sum()))
    raw = pd.DataFrame(rows)
    save_table(raw, "multiseed_raw", cfg, log)
    groups = ["model", "setting"]
    agg = aggregate_runs(raw, groups)
    save_table(agg, "multiseed_results", cfg, log)
    save_table(report_table(agg, groups), "multiseed_report", cfg, log)
    save_table(aggregate_runs(raw, groups, RULE_METRICS), "multiseed_rule_metrics", cfg, log)
    pooled = pool_rule_detection(raw, groups)
    save_table(pooled, "multiseed_rule_pooled", cfg, log)
    if not args.no_plots:
        sub = pooled[pooled["setting"] == "threshold=0.5"].set_index("model")
        rates = {m: {f"{k}_detection_rate" if k != "overall" else "rule_violation_detection_rate":
                     sub.loc[m, f"{k}_pooled_rate"] for k in ["R1", "R2", "R3", "R4", "R5", "overall"]}
                 for m in sub.index}
        n_any = {k: int(sub.iloc[0][f"{k}_pooled_n"]) for k in ["R1", "R2", "R3", "R4", "R5", "overall"]}
        plot_rule_detection_rates(rates, output_dir(cfg, "figures") / "multiseed_rule_detection.png", n_any,
                                  f"Rule detection rate pooled over {len(seeds)} seeds' test nodes (threshold 0.5); "
                                  "n = violating test nodes")
    print(report_table(agg, groups).to_string(index=False))


if __name__ == "__main__":
    main()
