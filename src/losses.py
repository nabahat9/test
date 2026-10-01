"""Loss functions.

Classification loss (class imbalance)
-------------------------------------
``BCEWithLogitsLoss`` with ``pos_weight = n_neg / n_pos`` computed on the **training** nodes.  Positives are
up-weighted so the minority class is not ignored.  Threshold tuning on validation data is the second
imbalance tool (see ``metrics.select_threshold``).  No resampling / synthetic oversampling is used.

Symbolic constraint loss
------------------------
Let ``p_v = sigmoid(z_v)`` be the neural probability and ``s_v in {0,1}`` the symbolic verdict
(``s_v = 1`` iff node v violates at least one rule R1..R5).  The rules encode the implication

        (v violates a rule)  =>  (v is suspicious),        i.e.   s_v  =>  p_v .

Under the product t-norm the degree to which the implication ``a => b`` is violated is ``a * (1 - b)``, so

        L_constraint = (1 / |S|) * sum_{v in S}  s_v * (1 - p_v)

where ``S`` is the set of nodes the constraint is applied to.  It is zero iff every rule-violating node in ``S``
gets probability 1, it never penalises nodes the rules do not flag (the network stays free to flag nodes for
learned reasons -- rules are *sufficient*, not *necessary*), and it is differentiable.

    L_total = L_classification + lambda * L_constraint

Honest note: when ``S`` is the set of labelled training nodes *and* labels are rule-derived (``y = s``), the
constraint reduces to an additional positive-class term ``mean(y * (1 - p))`` -- i.e. extra emphasis on the
positives.  Its distinct value appears when ``S`` includes *unlabelled* nodes (``constraint_nodes: all``,
transductive) -- but then, for rule-derived labels, it injects label-equivalent information about val/test
nodes.  That is why the default is ``constraint_nodes: train``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def positive_class_weight(y: Tensor, mask: Tensor) -> Tensor:
    """``n_neg / n_pos`` on the masked nodes (1.0 if a class is absent)."""
    y_m = y[mask].float()
    n_pos = y_m.sum()
    n_neg = y_m.numel() - n_pos
    if n_pos.item() == 0 or n_neg.item() == 0:
        return torch.tensor(1.0)
    return n_neg / n_pos


def weighted_bce(logits: Tensor, y: Tensor, mask: Tensor, pos_weight: Optional[Tensor] = None) -> Tensor:
    """Binary cross-entropy with logits over the masked nodes, optional positive-class weight."""
    if mask.sum() == 0:
        raise ValueError("Loss mask selects no nodes.")
    pw = None if pos_weight is None else pos_weight.to(logits.device)
    return F.binary_cross_entropy_with_logits(logits[mask], y[mask].float(), pos_weight=pw)


def constraint_loss(logits: Tensor, symbolic_score: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """``mean_{v in mask} s_v * (1 - sigmoid(z_v))`` -- violation of the implication rule => suspicious."""
    probs = torch.sigmoid(logits)
    violation = symbolic_score.to(probs.dtype) * (1.0 - probs)
    if mask is not None:
        if mask.sum() == 0:
            return logits.new_zeros(())
        violation = violation[mask]
    return violation.mean()


def total_loss(logits: Tensor, y: Tensor, train_mask: Tensor, symbolic_score: Optional[Tensor],
               constraint_mask: Optional[Tensor], lambda_constraint: float,
               pos_weight: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """``L_total = L_cls + lambda * L_constraint``; returns ``(total, cls, constraint)``."""
    if lambda_constraint < 0:
        raise ValueError("lambda_constraint must be >= 0")
    cls = weighted_bce(logits, y, train_mask, pos_weight)
    if lambda_constraint > 0:
        if symbolic_score is None:
            raise ValueError("lambda_constraint > 0 requires the symbolic scores of the rule engine.")
        cons = constraint_loss(logits, symbolic_score, constraint_mask)
    else:
        cons = logits.new_zeros(())
    return cls + lambda_constraint * cons, cls, cons
