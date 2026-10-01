"""Rule-engine tests on tiny hand-built graphs with known answers (positive AND negative cases)."""
import numpy as np
import pytest

from src.rules import (RULE_IDS, RuleThresholds, evaluate_all_rules, evaluate_r1, evaluate_r2, evaluate_r3,
                       evaluate_r4, evaluate_r5, evaluate_rule, explanations_for_node, find_large_cycles,
                       flagged_nodes, format_node_explanation, graph_arrays, rule_flag_matrix, symbolic_score,
                       violated_nodes, violated_rules)


# ------------------------------------------------------------------ R1
def test_r1_blocked_account_that_sends_is_flagged(graph_factory):
    g = graph_factory(3, [(0, 1, 50)], blocked=[0])
    assert evaluate_r1(g).to_list() == [0]


def test_r1_blocked_account_that_only_receives_is_not_flagged(graph_factory):
    g = graph_factory(3, [(1, 0, 50)], blocked=[0])
    assert evaluate_r1(g).to_list() == []


def test_r1_unblocked_sender_is_not_flagged(graph_factory):
    g = graph_factory(3, [(0, 1, 50)])
    assert evaluate_r1(g).to_list() == []


# ------------------------------------------------------------------ R2
def _fan_out(u, k, amount, start=1):
    return [(u, start + i, amount) for i in range(k)]


def test_r2_five_day_old_account_with_six_outgoing_edges(graph_factory):
    g = graph_factory(10, _fan_out(0, 6, 10), ages={0: 5})
    assert evaluate_r2(g).to_list() == [0]


def test_r2_boundaries(graph_factory):
    g = graph_factory(10, _fan_out(0, 5, 10), ages={0: 7})            # exactly age 7 and 5 edges -> violates
    assert evaluate_r2(g).to_list() == [0]
    g = graph_factory(10, _fan_out(0, 4, 10), ages={0: 5})            # too few outgoing
    assert evaluate_r2(g).to_list() == []
    g = graph_factory(10, _fan_out(0, 6, 10), ages={0: 8})            # too old
    assert evaluate_r2(g).to_list() == []


# ------------------------------------------------------------------ R3
def test_r3_large_three_cycle(graph_factory):
    g = graph_factory(5, [(0, 1, 12000), (1, 2, 15000), (2, 0, 20000)])
    res = evaluate_r3(g)
    assert res.to_list() == [0, 1, 2]
    assert res.n_violations == 1
    assert res.details[1]["cycles"][0]["nodes"] == (0, 1, 2)


def test_r3_negative_cases(graph_factory):
    small_edge = graph_factory(4, [(0, 1, 12000), (1, 2, 11999), (2, 0, 20000)])
    assert evaluate_r3(small_edge).to_list() == []
    not_a_cycle = graph_factory(4, [(0, 1, 20000), (1, 2, 20000), (0, 2, 20000)])
    assert evaluate_r3(not_a_cycle).to_list() == []
    two_cycle = graph_factory(4, [(0, 1, 20000), (1, 0, 20000)])
    assert evaluate_r3(two_cycle).to_list() == []
    four_cycle = graph_factory(4, [(0, 1, 20000), (1, 2, 20000), (2, 3, 20000), (3, 0, 20000)])
    assert evaluate_r3(four_cycle).to_list() == []


def test_r3_two_cycles_sharing_a_node_and_parallel_edges(graph_factory):
    edges = [(0, 1, 13000), (1, 2, 13000), (2, 0, 13000), (0, 3, 14000), (3, 4, 14000), (4, 0, 14000)]
    res = evaluate_r3(graph_factory(5, edges))
    assert res.to_list() == [0, 1, 2, 3, 4] and res.n_violations == 2
    assert len(res.details[0]["cycles"]) == 2
    parallel = [(0, 1, 500), (0, 1, 13000), (1, 2, 13000), (2, 0, 13000)]      # largest parallel edge counts
    assert evaluate_r3(graph_factory(3, parallel)).to_list() == [0, 1, 2]


def test_find_large_cycles_reports_each_cycle_once(graph_factory):
    g = graph_factory(3, [(0, 1, 20000), (1, 2, 20000), (2, 0, 20000)])
    cycles = find_large_cycles(graph_arrays(g), 12000)
    assert len(cycles) == 1 and cycles[0][0] == (0, 1, 2)


# ------------------------------------------------------------------ R4 / R5
def test_r4_positive_boundary_and_negatives(graph_factory):
    ok = graph_factory(12, _fan_out(0, 8, 4000))                                  # 8 receivers, 32,000
    assert evaluate_r4(ok).to_list() == [0]
    boundary = graph_factory(12, _fan_out(0, 8, 3750))                            # exactly 30,000
    assert evaluate_r4(boundary).to_list() == [0]
    too_few = graph_factory(12, _fan_out(0, 7, 9000))                             # 7 receivers
    assert evaluate_r4(too_few).to_list() == []
    too_small = graph_factory(12, _fan_out(0, 8, 3700))                           # 29,600
    assert evaluate_r4(too_small).to_list() == []
    repeated = graph_factory(12, [(0, 1 + i % 4, 5000) for i in range(10)])       # 10 tx, only 4 unique receivers
    assert evaluate_r4(repeated).to_list() == []


def test_r5_positive_and_negatives(graph_factory):
    ok = graph_factory(12, [(i, 0, 4000) for i in range(1, 9)])
    assert evaluate_r5(ok).to_list() == [0]
    few = graph_factory(12, [(i, 0, 9000) for i in range(1, 8)])
    assert evaluate_r5(few).to_list() == []
    small = graph_factory(12, [(i, 0, 1000) for i in range(1, 10)])
    assert evaluate_r5(small).to_list() == []


# ------------------------------------------------------------------ all rules / aggregation
def test_evaluate_all_rules_structure_and_aggregation(graph_factory):
    edges = _fan_out(0, 8, 5000) + [(9, 10, 100)]                                   # node 0: R2 + R4
    g = graph_factory(12, edges, ages={0: 4}, blocked=[9])                         # node 9: R1
    res = evaluate_all_rules(g)
    assert list(res) == list(RULE_IDS)
    assert violated_nodes(res) == {"R1": [9], "R2": [0], "R3": [], "R4": [0], "R5": []}
    assert flagged_nodes(res).tolist() == [0, 9]
    flags = rule_flag_matrix(res)
    assert flags.shape == (12, 5) and flags[0].tolist() == [0, 1, 0, 1, 0] and flags[9].tolist() == [1, 0, 0, 0, 0]
    assert symbolic_score(res).tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    assert violated_rules(0, res) == ["R2", "R4"]
    assert evaluate_rule("R4", g).to_list() == [0]
    with pytest.raises(ValueError, match="Unknown rule"):
        evaluate_rule("R9", g)


def test_custom_thresholds_are_respected(graph_factory):
    g = graph_factory(12, _fan_out(0, 8, 2000))                                     # 16,000 total
    assert evaluate_all_rules(g)["R4"].to_list() == []
    lenient = RuleThresholds(min_outgoing_amount=10000.0)
    assert evaluate_all_rules(g, lenient)["R4"].to_list() == [0]
    with pytest.raises(ValueError, match="Unknown rule threshold"):
        RuleThresholds.from_config({"bogus": 1})


def test_input_validation_gives_clear_errors(graph_factory):
    g = graph_factory(3, [(0, 1, 5)])
    g.edge_amount = g.edge_amount[:0]
    with pytest.raises(ValueError, match="aligned"):
        evaluate_all_rules(g)
    g2 = graph_factory(3, [(0, 1, 5)])
    del g2.blocked
    with pytest.raises(ValueError, match="blocked"):
        evaluate_all_rules(g2)


# ------------------------------------------------------------------ explanations
def test_explanations_are_generated_from_graph_values(graph_factory):
    g = graph_factory(12, _fan_out(0, 8, 4800), ages={0: 4})                       # 8 receivers, 38,400
    res = evaluate_all_rules(g)
    lines = explanations_for_node(0, res)
    assert len(lines) == 2
    assert "R2" in lines[0] and "age = 4 days" in lines[0] and "outgoing transactions = 8" in lines[0]
    assert "R4" in lines[1] and "8 unique receivers" in lines[1] and "38,400" in lines[1]
    text = format_node_explanation(0, res)
    assert text.startswith("Node 0 violates:") and "  * R2:" in text and "  * R4:" in text


def test_explanation_text_changes_with_the_data(graph_factory):
    a = evaluate_all_rules(graph_factory(12, _fan_out(0, 5, 10), ages={0: 2}))
    b = evaluate_all_rules(graph_factory(12, _fan_out(0, 7, 10), ages={0: 6}))
    assert explanations_for_node(0, a) != explanations_for_node(0, b)
    assert "age = 2 days" in explanations_for_node(0, a)[0] and "age = 6 days" in explanations_for_node(0, b)[0]


def test_explanation_for_clean_node_and_r3_text(graph_factory):
    g = graph_factory(4, [(0, 1, 12000), (1, 2, 15000), (2, 0, 20000)])
    res = evaluate_all_rules(g)
    assert explanations_for_node(3, res) == []
    assert "violates no symbolic rule" in format_node_explanation(3, res)
    assert "0->1->2->0" in explanations_for_node(1, res)[0] and "12,000" in explanations_for_node(1, res)[0]
    with pytest.raises(KeyError):
        res["R1"].explain(3)
