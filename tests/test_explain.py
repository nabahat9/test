"""Explainability tests: reports, agreement categories, statistics."""
import numpy as np
import pytest

from src.explain import (agreement_category, agreement_counts, disagreement_report, explain_flagged_nodes,
                         format_node_report, node_report, rule_statistics)
from src.rules import RULE_IDS, evaluate_all_rules


def _graph(graph_factory):
    # node 0: age 4 days, 8 receivers x 4800 -> R2 and R4;  node 20: blocked sender -> R1;  ring 30-31-32 -> R3
    edges = [(0, 1 + i, 4800) for i in range(8)] + [(20, 21, 10)]
    edges += [(30, 31, 13000), (31, 32, 13000), (32, 30, 13000)]
    return graph_factory(40, edges, ages={0: 4}, blocked=[20])


def test_agreement_categories():
    assert agreement_category(True, True) == "both" and agreement_category(True, False) == "neural-only"
    assert agreement_category(False, True) == "symbolic-only" and agreement_category(False, False) == "neither"


def test_node_report_contains_generated_rule_values(graph_factory):
    res = evaluate_all_rules(_graph(graph_factory))
    p = np.full(40, 0.1)
    p[0] = 0.81
    final = (p + 0.5 * (np.isin(np.arange(40), [0, 20, 30, 31, 32]))) / 1.5
    rec = node_report(0, res, p, final, 0.5)
    assert rec["rules_violated"] == ["R2", "R4"] and rec["agreement"] == "both"
    assert rec["details"]["R2"] == {"age_days": 4.0, "outgoing_count": 8}
    assert rec["details"]["R4"]["unique_receivers"] == 8 and rec["details"]["R4"]["outgoing_amount"] == pytest.approx(38400)
    text = format_node_report(rec)
    assert text.startswith("Node 0") and "Neural score: 0.81" in text and "R2 violated:" in text and "R4 violated:" in text
    assert "38,400" in text and "age = 4 days" in text
    quiet = format_node_report(node_report(5, res, p, final, 0.5))
    assert "no symbolic rule violated" in quiet


def test_agreement_counts_partition_the_nodes(graph_factory):
    res = evaluate_all_rules(_graph(graph_factory))
    p = np.full(40, 0.1)
    p[[0, 20, 5]] = 0.9                                   # 0, 20: both; 5: neural-only ; ring nodes: symbolic-only
    c = agreement_counts(p, res, 0.5)
    assert c == {"both": 2, "neural_only": 1, "symbolic_only": 3, "neither": 34}
    assert sum(c.values()) == 40
    mask = np.zeros(40, dtype=bool)
    mask[[0, 5, 30]] = True
    assert agreement_counts(p, res, 0.5, mask) == {"both": 1, "neural_only": 1, "symbolic_only": 1, "neither": 0}


def test_rule_statistics_overlap_and_counts(graph_factory):
    res = evaluate_all_rules(_graph(graph_factory))
    st = rule_statistics(res)
    assert st["nodes_per_rule"] == {"R1": 1, "R2": 1, "R3": 3, "R4": 1, "R5": 0}
    assert st["violations_per_rule"]["R3"] == 1
    ov = st["overlap_matrix"]
    assert ov.shape == (5, 5) and ov[1, 3] == 1 == ov[3, 1] and ov[0, 1] == 0
    assert ov[2, 2] == 3 and st["nodes_violating_multiple_rules"] == 1 and st["nodes_violating_any_rule"] == 5


def test_explain_flagged_nodes_and_disagreements(graph_factory):
    res = evaluate_all_rules(_graph(graph_factory))
    p = np.full(40, 0.05)
    p[[0, 5]] = [0.9, 0.7]
    final = p.copy()
    final[[0, 20, 30, 31, 32]] += 0.4
    recs = explain_flagged_nodes(res, p, final, 0.5)
    assert [r["node"] for r in recs][:1] == [0]                                       # highest final score first
    assert {r["node"] for r in recs} == {0, 5, 20, 30, 31, 32}
    assert len(explain_flagged_nodes(res, p, final, 0.5, max_nodes=2)) == 2
    dis = disagreement_report(res, p, 0.5, max_nodes=2)
    assert [r["node"] for r in dis["neural_only"]] == [5]
    assert len(dis["symbolic_only"]) == 2 and all(r["agreement"] == "symbolic-only" for r in dis["symbolic_only"])
    assert all(r["rules_violated"] for r in dis["symbolic_only"])
