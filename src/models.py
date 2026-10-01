"""Neural models: MLP, GCN and GAT node classifiers (binary, one logit per node).

All models share the same skeleton so that differences come from the layer type only:

    input -> layer -> activation -> dropout -> layer -> activation -> linear head -> logit

Forward signature: ``model(x, edge_index) -> logits [N]``  (``edge_index`` is ignored by the MLP).
The suspicious probability is ``sigmoid(logit)``.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, GCNConv

MODEL_NAMES = ("mlp", "gcn", "gat", "rulegat")


class MLPClassifier(nn.Module):
    """Node-feature-only baseline (no graph information)."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        h = F.dropout(F.relu(self.fc1(x)), p=self.dropout, training=self.training)
        h = F.relu(self.fc2(h))
        return self.head(h).squeeze(-1)


class GCNClassifier(nn.Module):
    """Two GCN layers (symmetric-normalised mean aggregation) + linear head."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.head(h).squeeze(-1)


class GATClassifier(nn.Module):
    r"""Two multi-head GAT layers + linear head.

    Layer:  h_v^{(l+1)} = ||_{k=1..K} ELU( sum_{u in N(v) U {v}} alpha_{vu}^k W^k h_u^{(l)} ),
            alpha_{vu} = softmax_u( LeakyReLU( a^T [W h_v || W h_u] ) ).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, heads: int = 4, dropout: float = 0.3,
                 attn_dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}).")
        per_head = hidden_dim // heads
        self.conv1 = GATConv(in_dim, per_head, heads=heads, dropout=attn_dropout)
        self.conv2 = GATConv(hidden_dim, per_head, heads=heads, dropout=attn_dropout)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = F.dropout(F.elu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        h = F.elu(self.conv2(h, edge_index))
        return self.head(h).squeeze(-1)


def build_model(name: str, in_dim: int, model_cfg: Optional[Mapping[str, Any]] = None,
                symbolic_weight: float = 0.5) -> nn.Module:
    """Factory used by training/evaluation code. ``name`` in ``MODEL_NAMES``."""
    cfg = dict(model_cfg or {})
    hidden = int(cfg.get("hidden_dim", 32))
    heads = int(cfg.get("heads", 4))
    dropout = float(cfg.get("dropout", 0.3))
    attn = float(cfg.get("attn_dropout", 0.0))
    key = name.lower()
    if key == "mlp":
        return MLPClassifier(in_dim, hidden, dropout)
    if key == "gcn":
        return GCNClassifier(in_dim, hidden, dropout)
    if key == "gat":
        return GATClassifier(in_dim, hidden, heads, dropout, attn)
    if key == "rulegat":
        from .rulegat import RuleGAT

        return RuleGAT(in_dim, hidden, heads, dropout, attn, symbolic_weight)
    raise ValueError(f"Unknown model '{name}'. Valid models: {list(MODEL_NAMES)}")
