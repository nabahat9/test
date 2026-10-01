"""Synthetic banking-transaction graph generator.

IMPORTANT: everything here is synthetic. The graph is *not* real banking data and the injected patterns
are illustrative, hand-designed configurations.

Generation procedure
--------------------
1. A **normal background graph**: every account draws a small out-degree (Poisson, capped at
   ``max_normal_out_degree`` < all rule thresholds) and sends small log-normal amounts.
2. **Pattern injection** on disjoint account pools (so patterns do not overwrite each other):

   ============  ==========================================================================
   pattern       injected configuration
   ============  ==========================================================================
   R1            blocked account that sends 1-3 transactions
   R2            age 0-7 days, 5-7 outgoing transactions
   R3 (rings)    ``n_rings`` directed 3-cycles A->B->C->A with every amount in [12000, 25000]
   R4            9-12 receivers, amount 3500-7000 each (total >= 31500)
   R5            9-12 senders,   amount 3500-7000 each (total >= 31500)
   ============  ==========================================================================

3. **Decoys** (near misses that must NOT be flagged): blocked-but-silent accounts, brand-new-but-quiet
   accounts, old-but-busy accounts, 3-cycles with one small edge, 7-receiver / low-amount fan-outs, 7-sender /
   low-amount fan-ins.  They make the neural task less trivial and let us verify the rule engine's negatives.
4. **Behavioural correlates** (synthetic assumption, ``behavioral_signal``): participants of injected patterns
   draw device-change counts and login hours from a shifted distribution with probability
   ``0.6 * behavioral_signal``.  This gives the neural models a *weak* non-structural signal to learn.  The
   rules never use these features.

Three different node sets are tracked -- they are NOT assumed equal:

* ``anchor_mask``       one designated target per injected pattern (ring anchor = first ring member)
* ``participant_mask``  all nodes deliberately made part of an injected pattern (anchors + ring accomplices)
* the set flagged by the symbolic rules (computed later in ``dataset.py`` on the finished graph)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .rules import RuleThresholds

PATTERN_IDS = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}


@dataclass
class GenerationConfig:
    n_nodes: int = 150
    n_rings: int = 8
    instances_per_rule: int = 2
    seed: int = 42
    n_decoys_per_rule: Optional[int] = None
    avg_out_degree: float = 2.2
    max_normal_out_degree: int = 4
    amount_median: float = 400.0
    amount_sigma: float = 0.9
    amount_max: float = 5000.0
    behavioral_signal: float = 1.0
    thresholds: RuleThresholds = field(default_factory=RuleThresholds)

    @classmethod
    def from_config(cls, data_cfg: Optional[Mapping[str, Any]] = None,
                    rules_cfg: Optional[Mapping[str, Any]] = None,
                    seed: Optional[int] = None) -> "GenerationConfig":
        data_cfg = dict(data_cfg or {})
        allowed = set(cls.__dataclass_fields__) - {"thresholds", "seed"}
        kwargs = {k: v for k, v in data_cfg.items() if k in allowed}
        if seed is not None:
            kwargs["seed"] = int(seed)
        elif "seed" in data_cfg:
            kwargs["seed"] = int(data_cfg["seed"])
        return cls(thresholds=RuleThresholds.from_config(rules_cfg), **kwargs)


@dataclass
class RawGraph:
    """Raw generator output (numpy). Edge arrays are aligned and sorted by (src, dst)."""

    n_nodes: int
    src: np.ndarray
    dst: np.ndarray
    amount: np.ndarray
    age_days: np.ndarray
    device_changes: np.ndarray
    login_hour: np.ndarray
    blocked: np.ndarray
    anchor_mask: np.ndarray
    participant_mask: np.ndarray
    decoy_mask: np.ndarray
    pattern_rule: np.ndarray               # 0 = none, 1..5 = rule id of the injected pattern
    decoy_kind: Dict[int, str]
    rings: List[Tuple[int, int, int]]
    config: GenerationConfig


def _validate_config(cfg: GenerationConfig, n_decoys: int) -> int:
    ipr = cfg.instances_per_rule
    if cfg.n_nodes < 1 or cfg.n_rings < 0 or ipr < 0 or n_decoys < 0:
        raise ValueError("n_nodes must be >= 1 and n_rings / instances_per_rule / n_decoys_per_rule >= 0.")
    if not 0.0 <= cfg.behavioral_signal <= 1.0 / 0.6:
        raise ValueError("behavioral_signal must lie in [0, 1/0.6] so that 0.6*signal is a probability.")
    special = 3 * cfg.n_rings + 3 * n_decoys + 4 * ipr + 5 * n_decoys
    # 4 = R1, R2, R4, R5 anchors; decoys: r1, r2-new, r2-busy, r4, r5 (5 kinds) plus 3-cycle decoys above.
    min_plain = 15  # pool used for receivers / senders of the fan patterns
    if cfg.n_nodes < special + min_plain:
        raise ValueError(
            f"n_nodes={cfg.n_nodes} is too small for the requested injections: {special} accounts are "
            f"reserved for patterns/decoys and at least {min_plain} plain accounts are needed as counterparties "
            f"(need n_nodes >= {special + min_plain}). Reduce n_rings / instances_per_rule or increase n_nodes."
        )
    return special


def generate_raw_graph(cfg: GenerationConfig) -> RawGraph:
    """Generate the normal background graph, then inject rule-related patterns and decoys."""
    n = cfg.n_nodes
    ipr = cfg.instances_per_rule
    n_decoys = ipr if cfg.n_decoys_per_rule is None else int(cfg.n_decoys_per_rule)
    _validate_config(cfg, n_decoys)
    th = cfg.thresholds
    rng = np.random.default_rng(cfg.seed)

    # ---- allocate disjoint account pools ---------------------------------------------------------
    perm = [int(v) for v in rng.permutation(n)]
    cursor = 0

    def take(k: int) -> List[int]:
        nonlocal cursor
        out = perm[cursor:cursor + k]
        cursor += k
        return out

    ring_nodes = take(3 * cfg.n_rings)
    rings = [tuple(ring_nodes[3 * i:3 * i + 3]) for i in range(cfg.n_rings)]
    r1_anchor = take(ipr)
    r2_anchor = take(ipr)
    r4_anchor = take(ipr)
    r5_anchor = take(ipr)
    decoy_rings_nodes = take(3 * n_decoys)
    decoy_rings = [tuple(decoy_rings_nodes[3 * i:3 * i + 3]) for i in range(n_decoys)]
    r1_decoy = take(n_decoys)
    r2_decoy_new = take(n_decoys)
    r2_decoy_busy = take(n_decoys)
    r4_decoy = take(n_decoys)
    r5_decoy = take(n_decoys)
    plain = perm[cursor:]                       # everyone else: normal accounts / counterparties

    # ---- node metadata for the normal population ---------------------------------------------------
    age_days = rng.integers(30, 2501, size=n).astype(np.int64)
    device_changes = np.clip(rng.poisson(1.0, size=n), 0, 8).astype(np.int64)
    login_hour = np.mod(np.round(rng.normal(14.0, 4.0, size=n)), 24).astype(np.int64)
    blocked = np.zeros(n, dtype=bool)

    # ---- normal background transactions -------------------------------------------------------------
    edges: Dict[Tuple[int, int], float] = {}

    def draw_amount() -> float:
        raw = rng.lognormal(mean=np.log(cfg.amount_median), sigma=cfg.amount_sigma)
        return round(float(min(raw, cfg.amount_max)), 2)

    fan_in_nodes = set(r5_anchor) | set(r5_decoy)   # their in-degree is fully controlled by the injection
    target_pool = [v for v in range(n) if v not in fan_in_nodes]
    for u in range(n):
        k = int(min(rng.poisson(cfg.avg_out_degree), cfg.max_normal_out_degree))
        candidates = [v for v in target_pool if v != u]
        for v in rng.choice(candidates, size=min(k, len(candidates)), replace=False):
            edges[(u, int(v))] = draw_amount()

    def clear_outgoing(nodes: List[int]) -> None:
        drop = set(nodes)
        for key in [key for key in edges if key[0] in drop]:
            del edges[key]

    # accounts whose outgoing behaviour is fully designed
    clear_outgoing(r1_anchor + r1_decoy + r2_anchor + r2_decoy_new + r2_decoy_busy + r4_anchor + r4_decoy)

    def send_to_plain(u: int, k: int, amount_fn) -> None:
        for v in rng.choice(plain, size=k, replace=False):
            edges[(u, int(v))] = round(float(amount_fn()), 2)

    def receive_from_plain(v: int, k: int, amount_fn) -> None:
        for u in rng.choice(plain, size=k, replace=False):
            edges[(int(u), v)] = round(float(amount_fn()), 2)

    # ---- injected patterns --------------------------------------------------------------------------
    for u in r1_anchor:                                     # R1: blocked and sending
        blocked[u] = True
        send_to_plain(u, int(rng.integers(1, 4)), draw_amount)
    for u in r2_anchor:                                     # R2: new account, busy
        age_days[u] = int(rng.integers(0, th.new_account_max_age + 1))
        send_to_plain(u, int(rng.integers(th.min_outgoing_count, th.min_outgoing_count + 3)), draw_amount)
    for (a, b, c) in rings:                                 # R3: heavy 3-cycles
        for (u, v) in ((a, b), (b, c), (c, a)):
            edges[(u, v)] = round(float(rng.uniform(th.cycle_min_amount, th.cycle_min_amount + 13000.0)), 2)
    for u in r4_anchor:                                     # R4: fan-out with large total
        k = int(rng.integers(th.min_unique_receivers + 1, th.min_unique_receivers + 5))
        send_to_plain(u, k, lambda: rng.uniform(3500.0, 7000.0))
    for v in r5_anchor:                                     # R5: fan-in with large total
        k = int(rng.integers(th.min_unique_senders + 1, th.min_unique_senders + 5))
        receive_from_plain(v, k, lambda: rng.uniform(3500.0, 7000.0))

    # ---- decoys (must NOT trigger a rule) -----------------------------------------------------------
    decoy_kind: Dict[int, str] = {}
    for u in r1_decoy:                                      # blocked but silent
        blocked[u] = True
        decoy_kind[u] = "blocked_but_silent"
    for u in r2_decoy_new:                                  # new but quiet
        age_days[u] = int(rng.integers(0, th.new_account_max_age + 1))
        send_to_plain(u, int(rng.integers(2, th.min_outgoing_count)), draw_amount)
        decoy_kind[u] = "new_but_quiet"
    for u in r2_decoy_busy:                                 # old but busy
        send_to_plain(u, int(rng.integers(th.min_outgoing_count, th.min_outgoing_count + 3)), draw_amount)
        decoy_kind[u] = "old_but_busy"
    for (a, b, c) in decoy_rings:                           # cycle with one small edge
        amounts = [float(rng.uniform(th.cycle_min_amount, th.cycle_min_amount + 13000.0)) for _ in range(3)]
        amounts[int(rng.integers(0, 3))] = float(rng.uniform(2000.0, th.cycle_min_amount * 0.75))
        for (u, v), amt in zip(((a, b), (b, c), (c, a)), amounts):
            edges[(u, v)] = round(amt, 2)
        for u in (a, b, c):
            decoy_kind[u] = "cycle_with_small_edge"
    for i, u in enumerate(r4_decoy):
        if i % 2 == 0:                                      # too few receivers, big amounts
            send_to_plain(u, th.min_unique_receivers - 1, lambda: rng.uniform(5000.0, 9000.0))
            decoy_kind[u] = "few_receivers_large_amount"
        else:                                               # many receivers, tiny amounts
            send_to_plain(u, th.min_unique_receivers + 1, lambda: rng.uniform(100.0, 600.0))
            decoy_kind[u] = "many_receivers_small_amount"
    for i, v in enumerate(r5_decoy):
        if i % 2 == 0:
            receive_from_plain(v, th.min_unique_senders - 1, lambda: rng.uniform(5000.0, 9000.0))
            decoy_kind[v] = "few_senders_large_amount"
        else:
            receive_from_plain(v, th.min_unique_senders + 1, lambda: rng.uniform(100.0, 600.0))
            decoy_kind[v] = "many_senders_small_amount"

    # ---- bookkeeping masks --------------------------------------------------------------------------
    anchor_mask = np.zeros(n, dtype=bool)
    participant_mask = np.zeros(n, dtype=bool)
    decoy_mask = np.zeros(n, dtype=bool)
    pattern_rule = np.zeros(n, dtype=np.int64)
    for rid, nodes in (("R1", r1_anchor), ("R2", r2_anchor), ("R4", r4_anchor), ("R5", r5_anchor)):
        anchor_mask[nodes] = True
        participant_mask[nodes] = True
        pattern_rule[nodes] = PATTERN_IDS[rid]
    for ring in rings:
        anchor_mask[ring[0]] = True
        participant_mask[list(ring)] = True
        pattern_rule[list(ring)] = PATTERN_IDS["R3"]
    decoy_mask[list(decoy_kind)] = True

    # ---- behavioural correlates for participants (synthetic assumption) -------------------------------
    p_shift = 0.6 * cfg.behavioral_signal
    for u in np.flatnonzero(participant_mask):
        if rng.random() < p_shift:
            device_changes[u] = int(np.clip(rng.poisson(3.5), 0, 8))
            login_hour[u] = int(rng.integers(0, 6))          # night-time logins

    # ---- final edge arrays: sorted by (src, dst), amounts stay aligned ---------------------------------
    keys = sorted(edges)
    src = np.array([k[0] for k in keys], dtype=np.int64)
    dst = np.array([k[1] for k in keys], dtype=np.int64)
    amount = np.array([edges[k] for k in keys], dtype=np.float64)

    return RawGraph(n_nodes=n, src=src, dst=dst, amount=amount, age_days=age_days,
                    device_changes=device_changes, login_hour=login_hour, blocked=blocked,
                    anchor_mask=anchor_mask, participant_mask=participant_mask, decoy_mask=decoy_mask,
                    pattern_rule=pattern_rule, decoy_kind=decoy_kind, rings=rings, config=cfg)
