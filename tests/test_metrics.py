"""Metric, threshold-selection, rule-metric and aggregation tests (known values)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from src.metrics import (aggregate_runs, classification_metrics, default_threshold_grid, f1_curve, format_mean_std,
                         report_table, roc_auc_safe, rule_consistency, rule_detection_rates, select_threshold)


def test_classification_metrics_known_confusion_matrix():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.4, 0.6, 0.1, 0.2])
    m = classification_metrics(y, s, 0.5)                    # pred = [1, 0, 1, 0, 0]
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5) and m["f1"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(5 / 6)               # 5 of 6 (pos, neg) pairs ranked correctly
    assert m["n"] == 5 and m["n_pos"] == 2 and m["threshold"] == 0.5


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    s = rng.random(60)
    pred = (s >= 0.4).astype(int)
    m = classification_metrics(y, s, 0.4)
    assert m["precision"] == pytest.approx(precision_score(y, pred, zero_division=0))
    assert m["recall"] == pytest.approx(recall_score(y, pred, zero_division=0))
    assert m["f1"] == pytest.approx(f1_score(y, pred, zero_division=0))


def test_zero_division_and_undefined_auc():
    y = np.array([1, 0, 1, 0])
    m = classification_metrics(y, np.array([0.1, 0.2, 0.3, 0.4]), 0.5)        # nothing predicted positive
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0 and not np.isnan(m["roc_auc"])
    single = classification_metrics(np.zeros(4), np.array([0.1, 0.9, 0.5, 0.2]), 0.5)   # pred = [0, 1, 1, 0] (>= 0.5)
    assert np.isnan(single["roc_auc"]) and single["accuracy"] == pytest.approx(0.5)
    assert np.isnan(roc_auc_safe(np.ones(3), np.array([0.2, 0.5, 0.9])))
    assert roc_auc_safe(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_metric_input_validation():
    with pytest.raises(ValueError, match="same length"):
        classification_metrics(np.array([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError, match="binary"):
        classification_metrics(np.array([0, 2]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="empty"):
        classification_metrics(np.array([]), np.array([]))


def test_threshold_grid_and_selection():
    grid = default_threshold_grid()
    assert grid.min() == pytest.approx(0.01) and grid.max() == pytest.approx(0.99) and 0.5 in grid
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.85, 0.9, 0.95])
    thr, f1 = select_threshold(y, s)
    assert f1 == pytest.approx(1.0) and 0.85 < thr <= 0.9                      # best plateau is (0.85, 0.9]
    g, curve = f1_curve(y, s)
    assert len(g) == len(curve) and curve.max() == pytest.approx(1.0)
    # tie-break: the plateau (0.21..0.7) contains 0.5 -> 0.5 is returned
    assert select_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.9]))[0] == pytest.approx(0.5)
    # no positives on validation -> deterministic fallback to the neutral threshold
    assert select_threshold(np.zeros(4), np.array([0.1, 0.2, 0.7, 0.9]))[0] == pytest.approx(0.5)


def test_threshold_selection_never_looks_at_test_data(default_data):
    """Flipping every TEST label must not change the validation-selected threshold."""
    import torch
    from src.evaluate import evaluate_scores
    from src.rules import evaluate_all_rules, rule_flag_matrix, symbolic_score
    d = default_data
    res = evaluate_all_rules(d)
    flags, sym = rule_flag_matrix(res), symbolic_score(res)
    rng = np.random.default_rng(3)
    scores = rng.random(d.num_nodes)
    grid = default_threshold_grid()
    a = evaluate_scores("gat", 0, d, scores, scores, flags, sym, grid)
    flipped = d.clone()
    flipped.y = d.y.clone()
    flipped.y[d.test_mask] = 1 - flipped.y[d.test_mask]
    b = evaluate_scores("gat", 0, flipped, scores, scores, flags, sym, grid)
    assert a.threshold_val == b.threshold_val and a.val_f1 == b.val_f1
    assert a.test_val_thr["f1"] != b.test_val_thr["f1"] or a.test_val_thr["accuracy"] != b.test_val_thr["accuracy"]


def test_rule_detection_rates_known_values():
    flags = np.zeros((6, 5))
    flags[0, 0] = flags[1, 0] = 1            # R1 violators: nodes 0, 1
    flags[2, 2] = 1                          # R3 violator: node 2
    flags[3, 2] = flags[3, 3] = 1            # node 3 violates R3 and R4
    pred = np.array([1, 0, 1, 1, 1, 0])
    r = rule_detection_rates(pred, flags)
    assert r["R1_detection_rate"] == pytest.approx(0.5)
    assert r["R3_detection_rate"] == pytest.approx(1.0)
    assert r["R4_detection_rate"] == pytest.approx(1.0)
    assert np.isnan(r["R2_detection_rate"]) and np.isnan(r["R5_detection_rate"])
    assert r["rule_violation_detection_rate"] == pytest.approx(3 / 4)          # violators {0,1,2,3}, detected {0,2,3}
    assert r["n_violating_any"] == 4 and r["R3_n_violating"] == 2
    masked = rule_detection_rates(pred, flags, mask=np.array([0, 1, 0, 0, 1, 1], dtype=bool))   # nodes 1, 4, 5 only
    assert masked["R1_detection_rate"] == pytest.approx(0.0) and masked["R1_n_violating"] == 1   # node 1 missed
    assert np.isnan(masked["R3_detection_rate"])                                                  # no R3 node in mask
    with pytest.raises(ValueError, match="flag_matrix"):
        rule_detection_rates(pred, flags[:, :3])


def test_rule_consistency_known_values():
    verdict = np.array([1, 1, 0, 0])
    assert rule_consistency(np.array([1, 0, 0, 1]), verdict) == pytest.approx(0.5)
    assert rule_consistency(verdict, verdict) == pytest.approx(1.0)
    assert rule_consistency(np.array([1, 0, 0, 1]), verdict, mask=np.array([1, 0, 1, 0], dtype=bool)) == pytest.approx(1.0)
    assert np.isnan(rule_consistency(verdict, verdict, mask=np.zeros(4, dtype=bool)))


def test_on_clean_graph_consistency_equals_accuracy_and_detection_equals_recall(default_data):
    from src.rules import evaluate_all_rules, rule_flag_matrix, symbolic_score
    d = default_data
    res = evaluate_all_rules(d)
    rng = np.random.default_rng(1)
    scores = rng.random(d.num_nodes)
    test = d.test_mask.numpy()
    y = d.y.numpy()
    m = classification_metrics(y[test], scores[test], 0.5)
    pred = scores >= 0.5
    assert rule_consistency(pred, symbolic_score(res), mask=test) == pytest.approx(m["accuracy"])
    rates = rule_detection_rates(pred, rule_flag_matrix(res), mask=test)
    assert rates["rule_violation_detection_rate"] == pytest.approx(m["recall"])


def test_multiseed_aggregation_is_mean_and_sample_std():
    df = pd.DataFrame({"model": ["A"] * 3 + ["B"] * 2, "accuracy": [0.6, 0.7, 0.8, 1.0, 0.5],
                       "precision": [0.5] * 5, "recall": [0.5] * 5, "f1": [0.1, 0.2, 0.3, 0.4, 0.6],
                       "roc_auc": [0.7, np.nan, 0.9, 0.8, 0.8]})
    agg = aggregate_runs(df, ["model"]).set_index("model")
    assert agg.loc["A", "n_runs"] == 3
    assert agg.loc["A", "mean_accuracy"] == pytest.approx(0.7) and agg.loc["A", "std_accuracy"] == pytest.approx(np.std([0.6, 0.7, 0.8], ddof=1))
    assert agg.loc["A", "mean_roc_auc"] == pytest.approx(0.8)                        # NaN skipped
    assert agg.loc["A", "std_roc_auc"] == pytest.approx(np.std([0.7, 0.9], ddof=1))
    assert agg.loc["B", "std_f1"] == pytest.approx(np.std([0.4, 0.6], ddof=1))
    single = aggregate_runs(df.iloc[:1], ["model"])
    assert np.isnan(single.loc[0, "std_accuracy"])                                  # undefined, not 0
    with pytest.raises(KeyError):
        aggregate_runs(df, ["nope"])
    assert format_mean_std(0.8123, 0.0456) == "0.812 ± 0.046" and format_mean_std(float("nan"), 0.1) == "n/a"
    assert report_table(aggregate_runs(df, ["model"]), ["model"]).loc[0, "accuracy"] == "0.700 ± 0.100"


def test_pooled_rule_detection_weights_by_violators():
    from src.metrics import pool_rule_detection
    rows = []
    for model, r1, n1 in (("A", 1.0, 1), ("A", 0.0, 3), ("A", float("nan"), 0), ("B", 0.5, 2)):
        rows.append({"model": model, "R1_detection_rate": r1, "R1_n_violating": n1,
                     "R2_detection_rate": float("nan"), "R2_n_violating": 0,
                     "R3_detection_rate": 1.0, "R3_n_violating": 2,
                     "R4_detection_rate": float("nan"), "R4_n_violating": 0,
                     "R5_detection_rate": float("nan"), "R5_n_violating": 0,
                     "rule_violation_detection_rate": 0.5, "n_violating_any": 2})
    pooled = pool_rule_detection(pd.DataFrame(rows), ["model"]).set_index("model")
    assert pooled.loc["A", "R1_pooled_rate"] == pytest.approx(1 / 4)          # (1*1 + 0*3) / (1 + 3), not mean(1, 0)
    assert pooled.loc["A", "R1_pooled_n"] == 4 and pooled.loc["B", "R1_pooled_rate"] == pytest.approx(0.5)
    assert np.isnan(pooled.loc["A", "R2_pooled_rate"]) and pooled.loc["A", "R2_pooled_n"] == 0
    assert pooled.loc["A", "R3_pooled_rate"] == pytest.approx(1.0) and pooled.loc["A", "overall_pooled_n"] == 6
    with pytest.raises(KeyError):
        pool_rule_detection(pd.DataFrame({"model": ["A"]}), ["model"])
