"""RuleGAT: a GAT (neural branch) combined with the symbolic rule engine (symbolic branch).

Architecture
------------
    neural branch    :  GAT(x, edges) -> logit z_v  ->  p_v = sigmoid(z_v)
    symbolic branch  :  rules R1..R5 on raw graph metadata -> s_v in {0,1}   (no learnable parameters)
    hybrid score     :  final_v = (p_v + w * s_v) / (1 + w)        with symbolic weight  w >= 0

``final_v`` is a convex combination of the neural probability and the symbolic verdict (mixing weight
``w / (1 + w)`` on the symbolic part), so it stays in [0, 1] and can be thresholded like any probability.
Interpretation: a rule violation raises the score of a node by ``w / (1 + w)``; with ``w = 0.5`` a node the
rules flag needs only p >= 0.25 to pass 0.5, whereas an unflagged node needs p >= 0.75.

What happens when
-----------------
TRAINING   the GAT is trained with weighted BCE on the training labels; symbolic information enters ONLY
           through the optional constraint loss ``lambda * L_constraint`` (see ``losses.py``), by default on
           training nodes.  The symbolic verdict is NOT an input feature of the network and the network
           is not trained on ``final_v``.
INFERENCE  ``final_v`` combines the trained network's probability with the rule verdicts computed on the
           graph being scored.  The rules use only ``edge_index``, ``edge_amount``, ``age_days`` and
           ``blocked`` of that graph; they never see ``y``.
CAVEAT     labels are themselves defined by the rules, so on a clean graph the symbolic branch reproduces the
           label function -- the hybrid is aligned with the labels *by construction*.  The neural branch alone
           (``neural_probability``) is therefore also evaluated, and the noise experiment (rules see a noisy
           graph while labels remain clean) is the more informative test.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .models import GATClassifier


def combine_scores(neural_prob: Tensor, symbolic_score: Tensor, symbolic_weight: float) -> Tensor:
    """``(p + w * s) / (1 + w)`` -- convex combination of neural probability and symbolic verdict."""
    if symbolic_weight < 0:
        raise ValueError("symbolic_weight must be >= 0")
    s = symbolic_score.to(neural_prob.dtype)
    return (neural_prob + symbolic_weight * s) / (1.0 + symbolic_weight)


class RuleGAT(nn.Module):
    """GAT classifier + non-parametric symbolic branch, fused at the score level."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, heads: int = 4, dropout: float = 0.3,
                 attn_dropout: float = 0.0, symbolic_weight: float = 0.5) -> None:
        super().__init__()
        if symbolic_weight < 0:
            raise ValueError("symbolic_weight must be >= 0")
        self.neural = GATClassifier(in_dim, hidden_dim, heads, dropout, attn_dropout)
        self.symbolic_weight = float(symbolic_weight)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Neural logits (this is what training optimises)."""
        return self.neural(x, edge_index)

    def neural_probability(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return torch.sigmoid(self.forward(x, edge_index))

    def hybrid_scores(self, x: Tensor, edge_index: Tensor, symbolic_score: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(p_neural, final_score)`` for all nodes."""
        p = self.neural_probability(x, edge_index)
        return p, combine_scores(p, symbolic_score, self.symbolic_weight)
