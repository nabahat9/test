"""Graph generation, features, labels, splits, statistics and noise tests."""
import numpy as np
import pytest
import torch

from src.data_generation import GenerationConfig, generate_raw_graph
from src.dataset import (BASE_FEATURE_NAMES, STRUCTURAL_FEATURE_NAMES, assert_no_leaky_features, build_dataset,
                         graph_statistics, load_dataset, make_split_masks, perturb_graph, save_dataset)
from src.rules import evaluate_all_rules, flagged_nodes, symbolic_score


def test_generation_is_deterministic_and_seed_dependent(cfg):
    a, b, c = build_dataset(cfg, seed=42), build_dataset(cfg, seed=42), build_dataset(cfg, seed=43)
    for name in ("x", "edge_index", "edge_amount", "y", "train_mask", "val_mask", "test_mask"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert not torch.equal(a.edge_index, c.edge_index) or not torch.equal(a.edge_amount, c.edge_amount)


def test_required_attributes_and_shapes(default_data):
    d = default_data
    n = 150
    assert d.num_nodes == n and d.x.shape == (n, 4) and d.y.shape == (n,)
    for name in ("x", "edge_index", "edge_amount", "y", "train_mask", "val_mask", "test_mask", "age_days",
                 "blocked", "feature_names"):
        assert getattr(d, name, None) is not None, name
    assert d.edge_index.shape[0] == 2 and d.edge_index.dtype == torch.long
    assert d.edge_amount.shape == (d.edge_index.shape[1],)
    assert d.blocked.dtype == torch.bool and d.age_days.shape == (n,)
    assert set(d.y.tolist()) <= {0, 1}


def test_feature_names_dimensions_and_no_target_leakage(default_data):
    d = default_data
    assert d.feature_names == BASE_FEATURE_NAMES and len(d.feature_names) == d.x.shape[1]
    assert torch.isfinite(d.x).all()
    assert_no_leaky_features(d.feature_names)
    for bad in (["blocked"], ["risk_score"], ["rule_count"], ["label"]):
        with pytest.raises(ValueError, match="leaking"):
            assert_no_leaky_features(bad)
    blocked = d.blocked.float()
    for j in range(d.x.shape[1]):                                # no column is a copy of blocked / y
        assert not torch.equal(d.x[:, j], blocked)
        assert not torch.equal(d.x[:, j], d.y.float())
    # login-hour encoding is on the unit circle
    assert torch.allclose(d.x[:, 2] ** 2 + d.x[:, 3] ** 2, torch.ones(d.num_nodes), atol=1e-5)


def test_edges_are_directed_aligned_and_ring_amounts_stored(default_data):
    d = default_data
    src, dst = d.edge_index.tolist()
    amount = dict(zip(zip(src, dst), d.edge_amount.tolist()))
    assert len(amount) == d.edge_index.shape[1]                                  # no duplicate (src, dst) pairs
    assert all(u != v for u, v in amount)                                       # no self loops
    assert list(zip(src, dst)) == sorted(zip(src, dst))                          # jointly sorted
    for a, b, c in d.rings:
        for u, v in ((a, b), (b, c), (c, a)):
            assert amount[(u, v)] >= 12000                                       # injected heavy ring edges exist
        assert (b, a) not in amount and (c, b) not in amount and (a, c) not in amount   # ring is one-directional
    reciprocal = sum((v, u) in amount for (u, v) in amount)
    assert reciprocal < len(amount) / 2                                          # graph is genuinely directed


def test_labels_are_rule_derived_not_injection_ids(default_data):
    d = default_data
    flagged = flagged_nodes(evaluate_all_rules(d))
    assert d.y.nonzero().flatten().tolist() == flagged.tolist()
    assert symbolic_score(evaluate_all_rules(d)).astype(int).tolist() == d.y.tolist()
    # injection anchors are a strict subset of the labelled positives here (ring accomplices are extra)
    assert int(d.injected_anchor_mask.sum()) < int(d.y.sum())


def test_decoys_are_not_flagged_and_stats_are_consistent(default_data):
    d = default_data
    assert int((d.decoy_mask & (d.y == 1)).sum()) == 0
    st = graph_statistics(d)
    iv = st["injected_vs_flagged"]
    assert st["n_symbolic_flagged_nodes"] == int(d.y.sum()) == st["n_label_positive"]
    assert iv["participants_flagged"] + iv["flagged_but_not_participant"] == st["n_symbolic_flagged_nodes"]
    assert iv["anchors_flagged"] + iv["anchors_not_flagged"] == st["n_injected_anchor_nodes"]
    assert st["n_node_rule_violations"] == sum(v["nodes_violating"] for v in st["per_rule"].values())
    assert st["per_rule"]["R3"]["violations"] == 8 and st["per_rule"]["R3"]["nodes_violating"] == 24
    for rid in ("R1", "R2", "R4", "R5"):
        assert st["per_rule"][rid]["nodes_violating"] == 2


def test_masks_are_disjoint_complete_stratified_and_reproducible(default_data, cfg):
    d = default_data
    tr, va, te = d.train_mask, d.val_mask, d.test_mask
    for m in (tr, va, te):
        assert m.shape == (d.num_nodes,) and m.dtype == torch.bool
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (tr | va | te).all()
    assert (int(tr.sum()), int(va.sum()), int(te.sum())) == (90, 30, 30)
    for m in (tr, va, te):
        assert 0 < int(d.y[m].sum()) < int(m.sum())                              # both classes in every split
    again = make_split_masks(d.y.numpy(), 0.6, 0.2, 0.2, seed=42)
    assert torch.equal(again[2], te)
    with pytest.raises(ValueError, match="sum to 1"):
        make_split_masks(d.y.numpy(), 0.5, 0.2, 0.2, seed=1)


def test_too_small_graph_raises_clear_error():
    with pytest.raises(ValueError, match="too small"):
        generate_raw_graph(GenerationConfig(n_nodes=30))


def test_structural_features_option(cfg):
    conf = dict(cfg)
    conf["data"] = {**cfg["data"], "structural_features": True}
    d = build_dataset(conf, seed=42)
    assert d.x.shape[1] == 6 and d.feature_names == BASE_FEATURE_NAMES + STRUCTURAL_FEATURE_NAMES
    noisy = perturb_graph(d, add_frac=0.3, seed=1)
    assert noisy.x.shape == d.x.shape and not torch.equal(noisy.x[:, 4:], d.x[:, 4:])
    assert torch.equal(noisy.x[:, :4], d.x[:, :4])


def test_perturb_graph_keeps_alignment_labels_and_counts(default_data):
    d = default_data
    e = d.edge_index.shape[1]
    same = perturb_graph(d, 0.0, 0.0, seed=1)
    assert torch.equal(same.edge_index, d.edge_index) and torch.equal(same.edge_amount, d.edge_amount)
    added = perturb_graph(d, add_frac=0.2, seed=1)
    assert added.edge_index.shape[1] == e + round(0.2 * e) and added.edge_amount.shape[0] == added.edge_index.shape[1]
    rewired = perturb_graph(d, rewire_frac=0.1, seed=1)
    assert rewired.edge_index.shape[1] == e
    assert not torch.equal(rewired.edge_index, d.edge_index)
    for g in (added, rewired):
        assert torch.equal(g.y, d.y) and torch.equal(g.test_mask, d.test_mask)      # labels stay clean
        pairs = list(zip(*g.edge_index.tolist()))
        assert pairs == sorted(pairs)
    assert torch.equal(d.edge_index, default_data.edge_index)                        # original untouched
    with pytest.raises(ValueError):
        perturb_graph(d, rewire_frac=1.5)


def test_save_and_load_roundtrip(default_data, tmp_path):
    path = save_dataset(default_data, tmp_path / "g.pt")
    loaded = load_dataset(path)
    assert torch.equal(loaded.edge_index, default_data.edge_index) and loaded.feature_names == default_data.feature_names
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.pt")
