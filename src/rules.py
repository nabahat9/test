"""Symbolic rule engine (R1-R5).

The rules operate on the *raw graph metadata* of a PyG ``Data`` object -- ``edge_index``, ``edge_amount``,
``age_days`` and ``blocked`` -- and never on the neural feature matrix ``x``.

Public API (all used by ``run_demo.py``)::

    thresholds = RuleThresholds()                       # or RuleThresholds.from_config(cfg["rules"])
    results    = evaluate_all_rules(data, thresholds)   # {"R1": RuleResult, ..., "R5": RuleResult}
    violated_nodes(results)                             # {"R1": [..], ..., "R5": [..]}
    flagged_nodes(results)                              # nodes violating at least one rule
    rule_flag_matrix(results)                           # [N, 5] 0/1 matrix
    symbolic_score(results)                             # [N] 0/1 (OR over rules)
    explanations_for_node(37, results)                  # ["R2: account age = 4 days ...", ...]
    format_node_explanation(37, results)                # multi-line human-readable string

Rule semantics (exactly as specified in the project brief)::

    R1  blocked == True  AND  outgoing_count >= 1
    R2  age_days <= 7    AND  outgoing_count >= 5
    R3  node lies on a directed 3-cycle A->B->C->A in which EVERY edge has amount >= 12000
    R4  unique_receivers >= 8  AND  outgoing_amount >= 30000
    R5  unique_senders   >= 8  AND  incoming_amount >= 30000

Design note for R3: with parallel edges (several transactions on the same ordered pair) the hop
amount is the *largest* transaction on that pair.  All three hops must reach the threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

RULE_IDS: Tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5")

RULE_DESCRIPTIONS: Dict[str, str] = {
    "R1": "blocked account transacts (blocked AND outgoing_count >= 1)",
    "R2": "new account with high outgoing activity (age_days <= 7 AND outgoing_count >= 5)",
    "R3": "large three-node cycle (A->B->C->A, every amount >= threshold)",
    "R4": "many unique receivers AND high outgoing amount",
    "R5": "many unique senders AND high incoming amount",
}


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RuleThresholds:
    """Numeric thresholds of the five rules (defaults = the brief's values)."""

    new_account_max_age: int = 7
    min_outgoing_count: int = 5
    cycle_min_amount: float = 12000.0
    min_unique_receivers: int = 8
    min_outgoing_amount: float = 30000.0
    min_unique_senders: int = 8
    min_incoming_amount: float = 30000.0

    @classmethod
    def from_config(cls, section: Optional[Mapping[str, Any]] = None) -> "RuleThresholds":
        """Build thresholds from the ``rules:`` section of config.yaml (missing keys use defaults)."""
        if not section:
            return cls()
        known = set(cls.__dataclass_fields__)
        unknown = set(section) - known
        if unknown:
            raise ValueError(f"Unknown rule threshold(s) {sorted(unknown)}; valid keys: {sorted(known)}")
        return cls(**dict(section))

    def as_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


# --------------------------------------------------------------------------- #
# Raw graph view + node statistics
# --------------------------------------------------------------------------- #
@dataclass
class GraphArrays:
    """Numpy view of the metadata the rules need."""

    src: np.ndarray
    dst: np.ndarray
    amount: np.ndarray
    age_days: np.ndarray
    blocked: np.ndarray
    num_nodes: int


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def graph_arrays(data: Any) -> GraphArrays:
    """Extract and validate the rule inputs from a PyG ``Data``-like object."""
    for name in ("edge_index", "edge_amount", "age_days", "blocked"):
        if getattr(data, name, None) is None:
            raise ValueError(
                f"Rule engine needs the attribute '{name}' on the graph object but it is missing. "
                "Required attributes: edge_index, edge_amount, age_days, blocked."
            )
    edge_index = _to_numpy(data.edge_index).astype(np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    amount = _to_numpy(data.edge_amount).astype(np.float64).reshape(-1)
    if amount.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_amount has {amount.shape[0]} entries but edge_index has {edge_index.shape[1]} edges; "
            "they must be aligned one-to-one."
        )
    age = _to_numpy(data.age_days).astype(np.float64).reshape(-1)
    blocked = _to_numpy(data.blocked).astype(bool).reshape(-1)
    num_nodes = getattr(data, "num_nodes", None)
    num_nodes = int(num_nodes) if num_nodes is not None else int(age.shape[0])
    if age.shape[0] != num_nodes or blocked.shape[0] != num_nodes:
        raise ValueError(
            f"age_days ({age.shape[0]}) and blocked ({blocked.shape[0]}) must both have num_nodes={num_nodes} entries."
        )
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError("edge_index contains node ids outside [0, num_nodes).")
    return GraphArrays(edge_index[0], edge_index[1], amount, age, blocked, num_nodes)


@dataclass
class NodeStats:
    """Per-node aggregates used by R1, R2, R4 and R5."""

    outgoing_count: np.ndarray
    incoming_count: np.ndarray
    unique_receivers: np.ndarray
    unique_senders: np.ndarray
    outgoing_amount: np.ndarray
    incoming_amount: np.ndarray


def compute_node_stats(g: GraphArrays) -> NodeStats:
    n = g.num_nodes
    outgoing_count = np.bincount(g.src, minlength=n).astype(np.int64)
    incoming_count = np.bincount(g.dst, minlength=n).astype(np.int64)
    outgoing_amount = np.bincount(g.src, weights=g.amount, minlength=n).astype(np.float64)
    incoming_amount = np.bincount(g.dst, weights=g.amount, minlength=n).astype(np.float64)
    if g.src.size:
        pairs = np.unique(np.stack([g.src, g.dst], axis=1), axis=0)  # distinct (sender, receiver)
        unique_receivers = np.bincount(pairs[:, 0], minlength=n).astype(np.int64)
        unique_senders = np.bincount(pairs[:, 1], minlength=n).astype(np.int64)
    else:
        unique_receivers = np.zeros(n, dtype=np.int64)
        unique_senders = np.zeros(n, dtype=np.int64)
    return NodeStats(outgoing_count, incoming_count, unique_receivers, unique_senders,
                     outgoing_amount, incoming_amount)


# --------------------------------------------------------------------------- #
# Result container + explanation text
# --------------------------------------------------------------------------- #
@dataclass
class RuleResult:
    """Outcome of one rule on one graph.

    ``nodes``       sorted ids of nodes violating the rule
    ``details``     node id -> the concrete values responsible for the violation
    ``n_violations`` number of violations; equals ``len(nodes)`` except for R3 where it is the
                    number of distinct large 3-cycles
    ``params``      threshold values used (so explanations state the real condition)
    """

    rule_id: str
    nodes: np.ndarray
    details: Dict[int, Dict[str, Any]]
    n_violations: int
    params: Dict[str, Any]
    num_nodes: int

    def __contains__(self, node_id: int) -> bool:
        return int(node_id) in self.details

    def to_list(self) -> List[int]:
        return [int(v) for v in self.nodes]

    def mask(self) -> np.ndarray:
        m = np.zeros(self.num_nodes, dtype=bool)
        m[self.nodes] = True
        return m

    def explain(self, node_id: int) -> str:
        """Human-readable reason why ``node_id`` violates this rule (generated from stored values)."""
        node_id = int(node_id)
        if node_id not in self.details:
            raise KeyError(f"Node {node_id} does not violate {self.rule_id}.")
        return _EXPLAINERS[self.rule_id](self.details[node_id], self.params)


def _fmt_amount(value: float) -> str:
    return f"{value:,.0f}"


def _explain_r1(d: Dict[str, Any], p: Dict[str, Any]) -> str:
    return f"R1: account is blocked and sends {d['outgoing_count']} outgoing transaction(s) (>= 1)"


def _explain_r2(d: Dict[str, Any], p: Dict[str, Any]) -> str:
    return (f"R2: account age = {d['age_days']:g} days (<= {p['new_account_max_age']}) and "
            f"outgoing transactions = {d['outgoing_count']} (>= {p['min_outgoing_count']})")


def _explain_r3(d: Dict[str, Any], p: Dict[str, Any]) -> str:
    cycles = d["cycles"]
    shown = []
    for cyc in cycles[:3]:
        a, b, c = cyc["nodes"]
        x, y, z = cyc["amounts"]
        shown.append(f"{a}->{b}->{c}->{a} with amounts {_fmt_amount(x)}, {_fmt_amount(y)}, {_fmt_amount(z)}")
    extra = f" (+{len(cycles) - 3} more)" if len(cycles) > 3 else ""
    return (f"R3: on {len(cycles)} directed 3-cycle(s) whose edges are all >= "
            f"{_fmt_amount(p['cycle_min_amount'])}: " + "; ".join(shown) + extra)


def _explain_r4(d: Dict[str, Any], p: Dict[str, Any]) -> str:
    return (f"R4: {d['unique_receivers']} unique receivers (>= {p['min_unique_receivers']}) and outgoing "
            f"amount = {_fmt_amount(d['outgoing_amount'])} (>= {_fmt_amount(p['min_outgoing_amount'])})")


def _explain_r5(d: Dict[str, Any], p: Dict[str, Any]) -> str:
    return (f"R5: {d['unique_senders']} unique senders (>= {p['min_unique_senders']}) and incoming "
            f"amount = {_fmt_amount(d['incoming_amount'])} (>= {_fmt_amount(p['min_incoming_amount'])})")


_EXPLAINERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], str]] = {
    "R1": _explain_r1, "R2": _explain_r2, "R3": _explain_r3, "R4": _explain_r4, "R5": _explain_r5,
}


def _make_result(rule_id: str, mask: np.ndarray, detail_fn: Callable[[int], Dict[str, Any]],
                 params: Dict[str, Any], n_violations: Optional[int] = None) -> RuleResult:
    nodes = np.flatnonzero(mask).astype(np.int64)
    details = {int(v): detail_fn(int(v)) for v in nodes}
    return RuleResult(rule_id=rule_id, nodes=nodes, details=details,
                      n_violations=int(len(nodes) if n_violations is None else n_violations),
                      params=params, num_nodes=int(mask.shape[0]))


# --------------------------------------------------------------------------- #
# Individual rules
# --------------------------------------------------------------------------- #
def _prepare(data: Any, stats: Optional[NodeStats]) -> Tuple[GraphArrays, NodeStats]:
    g = graph_arrays(data)
    return g, (stats if stats is not None else compute_node_stats(g))


def evaluate_r1(data: Any, thresholds: Optional[RuleThresholds] = None,
                stats: Optional[NodeStats] = None) -> RuleResult:
    """R1: blocked account that sends at least one transaction."""
    g, s = _prepare(data, stats)
    mask = g.blocked & (s.outgoing_count >= 1)
    return _make_result("R1", mask, lambda v: {"blocked": True, "outgoing_count": int(s.outgoing_count[v])}, {})


def evaluate_r2(data: Any, thresholds: Optional[RuleThresholds] = None,
                stats: Optional[NodeStats] = None) -> RuleResult:
    """R2: age_days <= 7 AND outgoing_count >= 5."""
    th = thresholds or RuleThresholds()
    g, s = _prepare(data, stats)
    mask = (g.age_days <= th.new_account_max_age) & (s.outgoing_count >= th.min_outgoing_count)
    params = {"new_account_max_age": th.new_account_max_age, "min_outgoing_count": th.min_outgoing_count}
    return _make_result("R2", mask,
                        lambda v: {"age_days": float(g.age_days[v]), "outgoing_count": int(s.outgoing_count[v])},
                        params)


def find_large_cycles(g: GraphArrays, min_amount: float) -> List[Tuple[Tuple[int, int, int], Tuple[float, float, float]]]:
    """All directed 3-cycles A->B->C->A whose three hops each have amount >= ``min_amount``.

    Each cycle is returned once, rotated so its smallest node id comes first.
    """
    best: Dict[Tuple[int, int], float] = {}
    for u, v, a in zip(g.src.tolist(), g.dst.tolist(), g.amount.tolist()):
        if u == v:
            continue
        key = (u, v)
        if key not in best or a > best[key]:
            best[key] = a
    heavy: Dict[int, Dict[int, float]] = {}
    for (u, v), a in best.items():
        if a >= min_amount:
            heavy.setdefault(u, {})[v] = a

    cycles: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
    for a, out_a in heavy.items():
        for b, amt_ab in out_a.items():
            for c, amt_bc in heavy.get(b, {}).items():
                if c == a or c == b:
                    continue
                amt_ca = heavy.get(c, {}).get(a)
                if amt_ca is None:
                    continue
                tri, amts = (a, b, c), (amt_ab, amt_bc, amt_ca)
                k = tri.index(min(tri))
                cycles[tri[k:] + tri[:k]] = amts[k:] + amts[:k]
    return sorted(cycles.items())


def evaluate_r3(data: Any, thresholds: Optional[RuleThresholds] = None,
                stats: Optional[NodeStats] = None) -> RuleResult:
    """R3: node belongs to a directed 3-cycle whose three transactions are all >= cycle_min_amount."""
    th = thresholds or RuleThresholds()
    g = graph_arrays(data)
    cycles = find_large_cycles(g, th.cycle_min_amount)
    per_node: Dict[int, List[Dict[str, Any]]] = {}
    for tri, amts in cycles:
        for node in tri:
            per_node.setdefault(node, []).append({"nodes": tri, "amounts": amts})
    mask = np.zeros(g.num_nodes, dtype=bool)
    if per_node:
        mask[list(per_node)] = True
    return _make_result("R3", mask, lambda v: {"cycles": per_node[v]},
                        {"cycle_min_amount": th.cycle_min_amount}, n_violations=len(cycles))


def evaluate_r4(data: Any, thresholds: Optional[RuleThresholds] = None,
                stats: Optional[NodeStats] = None) -> RuleResult:
    """R4: unique_receivers >= 8 AND outgoing_amount >= 30000."""
    th = thresholds or RuleThresholds()
    g, s = _prepare(data, stats)
    mask = (s.unique_receivers >= th.min_unique_receivers) & (s.outgoing_amount >= th.min_outgoing_amount)
    params = {"min_unique_receivers": th.min_unique_receivers, "min_outgoing_amount": th.min_outgoing_amount}
    return _make_result("R4", mask,
                        lambda v: {"unique_receivers": int(s.unique_receivers[v]),
                                   "outgoing_amount": float(s.outgoing_amount[v])}, params)


def evaluate_r5(data: Any, thresholds: Optional[RuleThresholds] = None,
                stats: Optional[NodeStats] = None) -> RuleResult:
    """R5: unique_senders >= 8 AND incoming_amount >= 30000."""
    th = thresholds or RuleThresholds()
    g, s = _prepare(data, stats)
    mask = (s.unique_senders >= th.min_unique_senders) & (s.incoming_amount >= th.min_incoming_amount)
    params = {"min_unique_senders": th.min_unique_senders, "min_incoming_amount": th.min_incoming_amount}
    return _make_result("R5", mask,
                        lambda v: {"unique_senders": int(s.unique_senders[v]),
                                   "incoming_amount": float(s.incoming_amount[v])}, params)


RULE_FUNCTIONS: Dict[str, Callable[..., RuleResult]] = {
    "R1": evaluate_r1, "R2": evaluate_r2, "R3": evaluate_r3, "R4": evaluate_r4, "R5": evaluate_r5,
}


def evaluate_rule(rule_id: str, data: Any, thresholds: Optional[RuleThresholds] = None) -> RuleResult:
    """Evaluate a single rule by id (``"R1"`` ... ``"R5"``)."""
    if rule_id not in RULE_FUNCTIONS:
        raise ValueError(f"Unknown rule '{rule_id}'. Valid rules: {list(RULE_IDS)}")
    return RULE_FUNCTIONS[rule_id](data, thresholds)


def evaluate_all_rules(data: Any, thresholds: Optional[RuleThresholds] = None) -> Dict[str, RuleResult]:
    """Evaluate R1-R5 on ``data``; node statistics are computed once and shared."""
    th = thresholds or RuleThresholds()
    g = graph_arrays(data)
    stats = compute_node_stats(g)
    return {rid: RULE_FUNCTIONS[rid](data, th, stats) for rid in RULE_IDS}


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def violated_nodes(results: Mapping[str, RuleResult]) -> Dict[str, List[int]]:
    """``{"R1": [node ids], ..., "R5": [node ids]}``."""
    return {rid: results[rid].to_list() for rid in RULE_IDS}


def _num_nodes(results: Mapping[str, RuleResult]) -> int:
    return next(iter(results.values())).num_nodes


def flagged_nodes(results: Mapping[str, RuleResult]) -> np.ndarray:
    """Sorted ids of nodes violating at least one rule."""
    return np.flatnonzero(symbolic_score(results) > 0).astype(np.int64)


def rule_flag_matrix(results: Mapping[str, RuleResult]) -> np.ndarray:
    """``[N, 5]`` float matrix; entry (v, k) is 1 iff node v violates rule ``RULE_IDS[k]``."""
    n = _num_nodes(results)
    out = np.zeros((n, len(RULE_IDS)), dtype=np.float32)
    for k, rid in enumerate(RULE_IDS):
        out[results[rid].nodes, k] = 1.0
    return out


def symbolic_score(results: Mapping[str, RuleResult]) -> np.ndarray:
    """``s_v = 1`` iff node v violates at least one rule (logical OR of R1..R5), else 0."""
    return rule_flag_matrix(results).max(axis=1)


def violated_rules(node_id: int, results: Mapping[str, RuleResult]) -> List[str]:
    """Ids of the rules violated by ``node_id`` (in R1..R5 order)."""
    return [rid for rid in RULE_IDS if int(node_id) in results[rid]]


def explanations_for_node(node_id: int, results: Mapping[str, RuleResult]) -> List[str]:
    """One human-readable line per rule violated by ``node_id``; empty list if none.

    The text is generated from the actual graph values stored in the rule results.
    """
    return [results[rid].explain(node_id) for rid in violated_rules(node_id, results)]


def format_node_explanation(node_id: int, results: Mapping[str, RuleResult]) -> str:
    """Multi-line explanation ("Node 37 violates: * R2: ... * R4: ...")."""
    lines = explanations_for_node(node_id, results)
    if not lines:
        return f"Node {int(node_id)} violates no symbolic rule."
    return f"Node {int(node_id)} violates:\n" + "\n".join(f"  * {line}" for line in lines)
