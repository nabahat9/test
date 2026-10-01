"""Matplotlib figures for the RuleGAT study (clean, presentation-ready, headless-safe)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .metrics import f1_curve
from .rules import RULE_IDS, RuleResult, rule_flag_matrix
from .utils import resolve_path

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
RULE_COLORS = {"R1": "#C44E52", "R2": "#DD8452", "R3": "#8172B3", "R4": "#4C72B0", "R5": "#55A868"}


def _finish(fig: plt.Figure, path: Optional[Path]) -> Optional[str]:
    fig.tight_layout()
    out = None
    if path is not None:
        p = resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        out = str(p)
    plt.close(fig)
    return out


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def plot_training_curves(histories: Mapping[str, Sequence[Mapping[str, float]]], path: Optional[Path] = None) -> Optional[str]:
    """Training loss vs epoch (left) and validation F1 vs epoch (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, (name, hist) in enumerate(histories.items()):
        if not hist:
            continue
        ep = [h["epoch"] for h in hist]
        c = PALETTE[i % len(PALETTE)]
        axes[0].plot(ep, [h["train_loss"] for h in hist], label=name, color=c)
        axes[1].plot(ep, [h["val_f1"] for h in hist], label=name, color=c)
    axes[0].set(title="Training loss vs epoch", xlabel="epoch", ylabel="total training loss")
    axes[1].set(title="Validation F1 vs epoch (threshold 0.5, neural branch)", xlabel="epoch", ylabel="validation F1")
    for ax in axes:
        _style(ax)
    axes[0].legend(frameon=False)
    return _finish(fig, path)


def plot_roc_curves(y_true: np.ndarray, scores: Mapping[str, np.ndarray], path: Optional[Path] = None,
                    title: str = "ROC curves (test nodes)") -> Optional[str]:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, (name, s) in enumerate(scores.items()):
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=name, color=PALETTE[i % len(PALETTE)])
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(title=title, xlabel="false positive rate", ylabel="true positive rate")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    return _finish(fig, path)


def plot_f1_vs_threshold(y_val: np.ndarray, val_scores: Mapping[str, np.ndarray], y_test: np.ndarray,
                         test_scores: Mapping[str, np.ndarray], selected: Mapping[str, float],
                         path: Optional[Path] = None) -> Optional[str]:
    """F1 vs threshold; solid = validation (used for selection), dashed = test (illustration only)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, name in enumerate(val_scores):
        c = PALETTE[i % len(PALETTE)]
        g, fv = f1_curve(y_val, val_scores[name])
        _, ft = f1_curve(y_test, test_scores[name])
        ax.plot(g, fv, color=c, label=f"{name} (val)")
        ax.plot(g, ft, color=c, ls="--", alpha=0.7)
        ax.axvline(selected[name], color=c, ls=":", alpha=0.6)
    ax.set(title="F1 vs threshold  (solid: validation, used for selection; dashed: test, not used)",
           xlabel="classification threshold", ylabel="F1")
    ax.title.set_fontsize(9)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return _finish(fig, path)


def plot_lambda_performance(df: pd.DataFrame, path: Optional[Path] = None, metrics: Sequence[str] = ("f1", "roc_auc"),
                            group_col: str = "variant") -> Optional[str]:
    """Performance vs lambda. ``df`` has columns lambda_constraint, ``group_col``, mean_/std_<metric>."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        for i, (name, sub) in enumerate(df.groupby(group_col, sort=False)):
            sub = sub.sort_values("lambda_constraint")
            std = sub[f"std_{m}"].fillna(0.0).to_numpy()
            mean = sub[f"mean_{m}"].to_numpy()
            c = PALETTE[i % len(PALETTE)]
            ax.errorbar(sub["lambda_constraint"], mean, yerr=std, marker="o", capsize=3, color=c, label=name)
        ax.set(title=f"{m} vs constraint weight lambda (mean ± std over seeds)", xlabel="lambda", ylabel=m)
        ax.title.set_fontsize(9)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    return _finish(fig, path)


def plot_rule_detection_rates(rates: Mapping[str, Mapping[str, float]], path: Optional[Path] = None,
                             n_violating: Optional[Mapping[str, int]] = None,
                             title: str = "Rule detection rate on test nodes (n/a: no violating test node)") -> Optional[str]:
    """Grouped bars of R1..R5 (and overall) detection rates per model. NaN bars are left empty.

    ``rates[model]`` maps ``"R1_detection_rate"`` ... / ``"rule_violation_detection_rate"`` to a value;
    ``n_violating`` (optional) maps ``"R1"`` ... / ``"overall"`` to the number of violating nodes behind the rates.
    """
    labels = list(RULE_IDS) + ["overall"]
    models = list(rates)
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, name in enumerate(models):
        vals = [rates[name].get(f"{r}_detection_rate", np.nan) for r in RULE_IDS]
        vals.append(rates[name].get("rule_violation_detection_rate", np.nan))
        xs = np.arange(len(labels)) + i * width - 0.4 + width / 2
        ax.bar(xs, np.nan_to_num(vals, nan=0.0), width, label=name, color=PALETTE[i % len(PALETTE)])
        for x_, v in zip(xs, vals):
            if np.isnan(v):
                ax.text(x_, 0.02, "n/a", ha="center", fontsize=6, rotation=90, color="grey")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([f"{lab}\n(n={n_violating[lab]})" if n_violating and lab in n_violating else lab
                        for lab in labels])
    ax.set(ylim=(0, 1.05), ylabel="detection rate", title=title)
    ax.title.set_fontsize(10)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _style(ax)
    return _finish(fig, path)


def plot_rule_overlap(overlap: np.ndarray, path: Optional[Path] = None) -> Optional[str]:
    """Heat-map of the number of nodes violating both rules (diagonal = nodes per rule)."""
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(overlap, cmap="Blues")
    ax.set_xticks(range(len(RULE_IDS)))
    ax.set_yticks(range(len(RULE_IDS)))
    ax.set_xticklabels(RULE_IDS)
    ax.set_yticklabels(RULE_IDS)
    for i in range(overlap.shape[0]):
        for j in range(overlap.shape[1]):
            ax.text(j, i, int(overlap[i, j]), ha="center", va="center",
                    color="white" if overlap[i, j] > overlap.max() / 2 else "black")
    ax.set_title("Rule overlap (nodes violating both rules)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _finish(fig, path)


def plot_robustness(df: pd.DataFrame, path: Optional[Path] = None, metric: str = "f1",
                    order: Optional[Sequence[str]] = None) -> Optional[str]:
    """Grouped bars: mean ± std of ``metric`` per noise setting and model."""
    order = list(order) if order is not None else list(dict.fromkeys(df["noise"]))
    models = list(dict.fromkeys(df["model"]))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    for i, name in enumerate(models):
        sub = df[df["model"] == name].set_index("noise").reindex(order)
        xs = np.arange(len(order)) + i * width - 0.4 + width / 2
        ax.bar(xs, sub[f"mean_{metric}"], width, yerr=sub[f"std_{metric}"].fillna(0.0), capsize=2,
               label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=15)
    ax.set(ylabel=metric, title=f"Robustness to graph noise: {metric} (mean ± std over seeds)", ylim=(0, 1.15))
    ax.title.set_fontsize(10)
    ax.legend(frameon=False, fontsize=8, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.15))
    _style(ax)
    return _finish(fig, path)


def plot_graph(data, results: Mapping[str, RuleResult], path: Optional[Path] = None,
               neural_prob: Optional[np.ndarray] = None, threshold: float = 0.5, seed: int = 0) -> Optional[str]:
    """Transaction graph: nodes coloured by violated rule (multi-rule: dark), grey = no violation.

    A ring around a node marks an injected anchor; dotted grey edge = small transaction, thick = large.
    """
    n = int(data.num_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    src, dst = data.edge_index.numpy()
    amt = data.edge_amount.numpy()
    for u, v, a in zip(src, dst, amt):
        G.add_edge(int(u), int(v), amount=float(a))
    flags = rule_flag_matrix(results)
    colors, sizes = [], []
    for v in range(n):
        rules = [RULE_IDS[k] for k in range(len(RULE_IDS)) if flags[v, k] > 0]
        colors.append("#BBBBBB" if not rules else ("#222222" if len(rules) > 1 else RULE_COLORS[rules[0]]))
        sizes.append(30 if not rules else 90)
    pos = nx.spring_layout(G.to_undirected(), seed=seed, k=0.35, iterations=80)
    fig, ax = plt.subplots(figsize=(9, 8))
    heavy = [(u, v) for u, v, d in G.edges(data=True) if d["amount"] >= 12000]
    light = [(u, v) for u, v, d in G.edges(data=True) if d["amount"] < 12000]
    nx.draw_networkx_edges(G, pos, edgelist=light, ax=ax, alpha=0.12, arrows=False, width=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=heavy, ax=ax, alpha=0.8, width=1.6, arrows=True, arrowsize=8,
                           edge_color="#8172B3")
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, ax=ax, linewidths=0)
    anchors = [int(v) for v in np.flatnonzero(data.injected_anchor_mask.numpy())]
    nx.draw_networkx_nodes(G, pos, nodelist=anchors, node_color="none", edgecolors="black", linewidths=1.0,
                           node_size=170, ax=ax)
    if neural_prob is not None:
        missed = [int(v) for v in np.flatnonzero((flags.max(axis=1) > 0) & (np.asarray(neural_prob) < threshold))]
        nx.draw_networkx_nodes(G, pos, nodelist=missed, node_color="none", edgecolors="#E4A700", linewidths=1.6,
                               node_size=260, ax=ax)
    handles = [plt.Line2D([], [], marker="o", ls="", color=RULE_COLORS[r], label=f"{r} violated") for r in RULE_IDS]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="#BBBBBB", label="no violation"))
    handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", color="black", label="injected anchor"))
    if neural_prob is not None:
        handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", color="#E4A700",
                                  label="rule-violating, missed by neural branch"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Synthetic transaction graph: symbolic rule violations (purple edges: amount >= 12,000)")
    ax.title.set_fontsize(10)
    ax.axis("off")
    return _finish(fig, path)
