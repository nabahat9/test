"""Explainability: rule-based, automatically generated node explanations and graph-level statistics.

Every explanation is produced from the *actual* rule results of the graph (no hard-coded text):

    Node 42
      Neural score: 0.81   RuleGAT score: 0.87   [flagged by: both neural and symbolic]
      R2 violated:  account age = 5 days (<= 7) and outgoing transactions = 7 (>= 5)
      R4 violated:  11 unique receivers (>= 8) and outgoing amount = 41,500 (>= 30,000)

Agreement categories (using a decision threshold ``thr`` on the neural probability):

* ``both``          neural probability >= thr AND at least one rule violated
* ``neural-only``   neural probability >= thr, no rule violated        (learned pattern the rules do not cover)
* ``symbolic-only`` neural probability <  thr, at least one rule violated  (the network missed a rule violation)
* ``neither``
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .rules import RULE_IDS, RuleResult, explanations_for_node, rule_flag_matrix, symbolic_score, violated_rules


def agreement_category(neural_flag: bool, symbolic_flag: bool) -> str:
    if neural_flag and symbolic_flag:
        return "both"
    if neural_flag:
        return "neural-only"
    if symbolic_flag:
        return "symbolic-only"
    return "neither"


def agreement_counts(neural_prob: np.ndarray, results: Mapping[str, RuleResult], threshold: float = 0.5,
                     mask: Optional[np.ndarray] = None) -> Dict[str, int]:
    """Counts of nodes per agreement category (optionally restricted to ``mask``)."""
    neural = np.asarray(neural_prob) >= threshold
    symbolic = symbolic_score(results) > 0
    m = np.ones(len(neural), dtype=bool) if mask is None else np.asarray(mask).astype(bool)
    return {
        "both": int((neural & symbolic & m).sum()),
        "neural_only": int((neural & ~symbolic & m).sum()),
        "symbolic_only": int((~neural & symbolic & m).sum()),
        "neither": int((~neural & ~symbolic & m).sum()),
    }


def rule_statistics(results: Mapping[str, RuleResult]) -> Dict[str, Any]:
    """Nodes / violations per rule, the pairwise overlap matrix and the multi-rule node count."""
    flags = rule_flag_matrix(results) > 0
    k = len(RULE_IDS)
    overlap = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            overlap[i, j] = int((flags[:, i] & flags[:, j]).sum())   # diagonal = nodes violating that rule
    return {
        "nodes_per_rule": {rid: int(len(results[rid].nodes)) for rid in RULE_IDS},
        "violations_per_rule": {rid: int(results[rid].n_violations) for rid in RULE_IDS},
        "overlap_matrix": overlap,
        "nodes_violating_multiple_rules": int((flags.sum(axis=1) > 1).sum()),
        "nodes_violating_any_rule": int(flags.any(axis=1).sum()),
    }


def node_report(node_id: int, results: Mapping[str, RuleResult], neural_prob: Optional[np.ndarray] = None,
                final_score: Optional[np.ndarray] = None, threshold: float = 0.5) -> Dict[str, Any]:
    """Structured explanation record for one node."""
    node_id = int(node_id)
    rules = violated_rules(node_id, results)
    record: Dict[str, Any] = {
        "node": node_id,
        "rules_violated": rules,
        "explanations": explanations_for_node(node_id, results),
        "details": {rid: results[rid].details[node_id] for rid in rules},
    }
    if neural_prob is not None:
        record["neural_prob"] = float(neural_prob[node_id])
        record["agreement"] = agreement_category(float(neural_prob[node_id]) >= threshold, bool(rules))
    if final_score is not None:
        record["final_score"] = float(final_score[node_id])
    return record


def format_node_report(record: Mapping[str, Any]) -> str:
    """Pretty multi-line text for a ``node_report`` record."""
    lines = [f"Node {record['node']}"]
    head = []
    if "neural_prob" in record:
        head.append(f"Neural score: {record['neural_prob']:.2f}")
    if "final_score" in record:
        head.append(f"RuleGAT score: {record['final_score']:.2f}")
    if "agreement" in record:
        head.append(f"[{record['agreement']}]")
    if head:
        lines.append("  " + "   ".join(head))
    if record["explanations"]:
        for text in record["explanations"]:
            rid, _, rest = text.partition(": ")
            lines.append(f"  {rid} violated: {rest}")
    else:
        lines.append("  no symbolic rule violated")
    return "\n".join(lines)


def explain_flagged_nodes(results: Mapping[str, RuleResult], neural_prob: np.ndarray, final_score: np.ndarray,
                          threshold: float = 0.5, max_nodes: Optional[int] = None,
                          mask: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Reports for every node flagged by the rules OR by the neural branch (sorted by final score)."""
    n = len(neural_prob)
    m = np.ones(n, dtype=bool) if mask is None else np.asarray(mask).astype(bool)
    flagged = (symbolic_score(results) > 0) | (np.asarray(neural_prob) >= threshold)
    ids = np.flatnonzero(flagged & m)
    ids = ids[np.argsort(-np.asarray(final_score)[ids], kind="stable")]
    if max_nodes is not None:
        ids = ids[:max_nodes]
    return [node_report(int(v), results, neural_prob, final_score, threshold) for v in ids]


def disagreement_report(results: Mapping[str, RuleResult], neural_prob: np.ndarray, threshold: float = 0.5,
                        mask: Optional[np.ndarray] = None, max_nodes: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """Nodes where neural and symbolic reasoning disagree (research question 10)."""
    neural = np.asarray(neural_prob) >= threshold
    symbolic = symbolic_score(results) > 0
    m = np.ones(len(neural), dtype=bool) if mask is None else np.asarray(mask).astype(bool)
    sym_only = np.flatnonzero(~neural & symbolic & m)
    neu_only = np.flatnonzero(neural & ~symbolic & m)
    sym_only = sym_only[np.argsort(np.asarray(neural_prob)[sym_only])][:max_nodes]        # most confidently missed
    neu_only = neu_only[np.argsort(-np.asarray(neural_prob)[neu_only])][:max_nodes]      # most confident extra alarms
    return {
        "symbolic_only": [node_report(int(v), results, neural_prob, threshold=threshold) for v in sym_only],
        "neural_only": [node_report(int(v), results, neural_prob, threshold=threshold) for v in neu_only],
    }
