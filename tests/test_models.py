"""Model forward passes, RuleGAT hybrid mechanism, constraint loss, training loop and leakage guards."""
import numpy as np
import pytest
import torch

from src.evaluate import DISPLAY_NAMES, runs_to_rows, run_models
from src.losses import constraint_loss, positive_class_weight, total_loss, weighted_bce
from src.models import GATClassifier, GCNClassifier, MLPClassifier, build_model
from src.rulegat import RuleGAT, combine_scores
from src.rules import evaluate_all_rules, symbolic_score
from src.train import TrainSettings, load_trained_model, train_model
from src.utils import message_passing_edges


@pytest.mark.parametrize("name", ["mlp", "gcn", "gat", "rulegat"])
def test_forward_pass_shapes(name, default_data):
    d = default_data
    model = build_model(name, d.x.shape[1], {"hidden_dim": 16, "heads": 4})
    edges = message_passing_edges(d.edge_index, d.num_nodes, "undirected")
    model.eval()
    logits = model(d.x, edges)
    assert logits.shape == (d.num_nodes,) and torch.isfinite(logits).all()


def test_model_factory_and_validation():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("transformer", 4)
    with pytest.raises(ValueError, match="divisible"):
        GATClassifier(4, hidden_dim=30, heads=4)
    assert isinstance(build_model("mlp", 4), MLPClassifier) and isinstance(build_model("gcn", 4), GCNClassifier)


def test_mlp_ignores_graph_but_gnn_uses_it(default_data):
    d = default_data
    mlp = MLPClassifier(d.x.shape[1]).eval()
    e1 = message_passing_edges(d.edge_index, d.num_nodes)
    assert torch.equal(mlp(d.x, e1), mlp(d.x, e1[:, :10]))
    torch.manual_seed(0)
    gat = GATClassifier(d.x.shape[1]).eval()
    assert not torch.allclose(gat(d.x, e1), gat(d.x, e1[:, :10]))


def test_message_passing_modes(default_data):
    d = default_data
    und = message_passing_edges(d.edge_index, d.num_nodes, "undirected")
    dirc = message_passing_edges(d.edge_index, d.num_nodes, "directed")
    assert torch.equal(dirc, d.edge_index) and und.shape[1] > dirc.shape[1]
    pairs = set(zip(*und.tolist()))
    assert all((v, u) in pairs for (u, v) in pairs)
    with pytest.raises(ValueError):
        message_passing_edges(d.edge_index, d.num_nodes, "sideways")


def test_rulegat_combination_formula_and_range():
    p = torch.tensor([0.4, 0.4, 0.9, 0.0])
    s = torch.tensor([1.0, 0.0, 0.0, 1.0])
    out = combine_scores(p, s, 0.5)
    assert torch.allclose(out, torch.tensor([0.9 / 1.5, 0.4 / 1.5, 0.9 / 1.5, 0.5 / 1.5]))
    assert ((out >= 0) & (out <= 1)).all()
    assert torch.allclose(combine_scores(p, s, 0.0), p)                     # w = 0 -> purely neural
    with pytest.raises(ValueError):
        combine_scores(p, s, -1.0)


def test_rulegat_symbolic_branch_has_no_parameters_and_hybrid_scores(default_data):
    d = default_data
    rg = RuleGAT(d.x.shape[1], 16, 4, symbolic_weight=0.5)
    gat = GATClassifier(d.x.shape[1], 16, 4)
    assert sum(p.numel() for p in rg.parameters()) == sum(p.numel() for p in gat.parameters())
    sym = torch.as_tensor(symbolic_score(evaluate_all_rules(d)), dtype=torch.float32)
    edges = message_passing_edges(d.edge_index, d.num_nodes)
    rg.eval()
    p, final = rg.hybrid_scores(d.x, edges, sym)
    assert torch.allclose(final, (p + 0.5 * sym) / 1.5)
    assert (final[sym > 0] >= 0.5 / 1.5 - 1e-6).all()                        # a rule violation raises the floor
    with pytest.raises(ValueError):
        RuleGAT(4, symbolic_weight=-0.1)


def test_constraint_loss_known_values():
    logits = torch.log(torch.tensor([0.25, 0.25, 0.9]))                      # p = 0.2, 0.2, 0.474...
    p = torch.sigmoid(torch.tensor([-1.3862944, -1.3862944, 5.0]))
    s = torch.tensor([1.0, 0.0, 1.0])
    logits = torch.tensor([-1.3862944, -1.3862944, 5.0])
    expected = ((1 * (1 - p[0])) + 0 + (1 * (1 - p[2]))) / 3
    assert constraint_loss(logits, s).item() == pytest.approx(expected.item(), rel=1e-5)
    assert constraint_loss(logits, s, mask=torch.tensor([True, True, False])).item() == pytest.approx((1 - p[0]).item() / 2, rel=1e-5)
    assert constraint_loss(logits, torch.zeros(3)).item() == 0.0            # never penalises unflagged nodes
    assert constraint_loss(torch.full((3,), 20.0), torch.ones(3)).item() == pytest.approx(0.0, abs=1e-6)
    assert constraint_loss(logits, s, mask=torch.zeros(3, dtype=torch.bool)).item() == 0.0
    lg = logits.clone().requires_grad_(True)
    constraint_loss(lg, s).backward()
    assert lg.grad[0] < 0 and lg.grad[1] == 0                                # pushes flagged logits up


def test_total_loss_composition_and_class_weight():
    logits = torch.tensor([0.3, -0.2, 1.0, -1.0])
    y = torch.tensor([1, 0, 1, 0])
    mask = torch.tensor([True, True, True, False])
    s = torch.tensor([1.0, 0.0, 1.0, 0.0])
    cls = weighted_bce(logits, y, mask)
    t0, c0, k0 = total_loss(logits, y, mask, s, mask, 0.0)
    assert t0.item() == pytest.approx(cls.item()) and k0.item() == 0.0
    t1, c1, k1 = total_loss(logits, y, mask, s, mask, 0.5)
    assert t1.item() == pytest.approx(c1.item() + 0.5 * k1.item()) and k1.item() > 0
    with pytest.raises(ValueError, match="requires the symbolic"):
        total_loss(logits, y, mask, None, mask, 0.5)
    with pytest.raises(ValueError):
        total_loss(logits, y, mask, s, mask, -1.0)
    assert positive_class_weight(torch.tensor([1, 0, 0, 0, 0, 0]), torch.ones(6, dtype=torch.bool)).item() == pytest.approx(5.0)
    heavier = weighted_bce(logits, y, mask, torch.tensor(3.0))
    assert heavier.item() > cls.item()


def test_training_smoke_early_stopping_and_best_state(default_data, cfg):
    d = default_data
    s = TrainSettings.from_config(cfg, "gat", 42, epochs=60, patience=3)
    res = train_model(d, s)
    assert 1 <= res.best_epoch <= len(res.history) <= res.best_epoch + s.patience
    assert {"train_loss", "val_f1", "val_loss", "epoch"} <= set(res.history[0])
    edges = message_passing_edges(d.edge_index, d.num_nodes, s.message_passing)
    res.model.eval()
    from sklearn.metrics import f1_score
    pred = (torch.sigmoid(res.model(d.x, edges)) >= 0.5).numpy()
    vm = d.val_mask.numpy()
    assert f1_score(d.y.numpy()[vm], pred[vm], zero_division=0) == pytest.approx(res.best_val_f1)   # best state restored


def test_training_is_reproducible_and_saves_artifacts(default_data, cfg, tmp_path):
    d = default_data
    s = TrainSettings.from_config(cfg, "gcn", 7, epochs=15)
    a, b = train_model(d, s, save_dir=tmp_path, tag="gcn_test"), train_model(d, s)
    assert a.history == b.history
    for pa, pb in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(pa, pb)
    for suffix in (".pt", "_history.json", "_config.json"):
        assert (tmp_path / f"gcn_test{suffix}").exists()
    reloaded = load_trained_model(tmp_path / "gcn_test.pt", s, d.x.shape[1])
    edges = message_passing_edges(d.edge_index, d.num_nodes, s.message_passing)
    assert torch.allclose(reloaded(d.x, edges), a.model(d.x, edges))


def test_test_labels_do_not_influence_training(default_data, cfg):
    """No test-set leakage: flipping test labels leaves the whole training run unchanged."""
    d = default_data
    flipped = d.clone()
    flipped.y = d.y.clone()
    flipped.y[d.test_mask] = 1 - flipped.y[d.test_mask]
    s = TrainSettings.from_config(cfg, "rulegat", 3, epochs=20, lambda_constraint=0.5)
    sym = symbolic_score(evaluate_all_rules(d))
    a, b = train_model(d, s, symbolic_score=sym), train_model(flipped, s, symbolic_score=sym)
    assert a.history == b.history and a.best_epoch == b.best_epoch


def test_training_setting_validation(default_data, cfg):
    with pytest.raises(ValueError, match="Only the 'rulegat'"):
        TrainSettings.from_config(cfg, "gat", 1, lambda_constraint=0.5)
    with pytest.raises(ValueError, match="constraint_nodes"):
        TrainSettings.from_config(cfg, "rulegat", 1, constraint_nodes="test")
    with pytest.raises(ValueError, match="Unknown model"):
        TrainSettings.from_config(cfg, "svm", 1)
    with pytest.raises(ValueError, match="Unknown training setting"):
        TrainSettings.from_config(cfg, "gat", 1, bogus=1)
    s = TrainSettings.from_config(cfg, "rulegat", 1, epochs=2, lambda_constraint=1.0)
    with pytest.raises(ValueError, match="symbolic_score"):
        train_model(default_data, s, symbolic_score=None)


def test_run_models_rows_and_no_target_leaking_inputs(default_data, cfg):
    d = default_data
    runs = run_models(d, cfg, 42, models=("mlp", "rulegat"), include_symbolic=True)
    assert list(runs) == ["mlp", "rulegat", "rulegat_neural", "symbolic"]
    rows = runs_to_rows(runs)
    assert len(rows) == 8 and {r["setting"] for r in rows} == {"threshold=0.5", "val-selected threshold"}
    for r in rows:
        assert r["model"] in DISPLAY_NAMES.values()
        assert {"accuracy", "precision", "recall", "f1", "roc_auc", "rule_consistency", "R3_detection_rate"} <= set(r)
    assert runs["symbolic"].test_default["f1"] == pytest.approx(1.0)        # rules == labels on the clean graph
    # neural input never contains the blocked flag or labels
    assert not any(k in n for n in d.feature_names for k in ("blocked", "risk", "label"))
