# RuleGAT — a neuro-symbolic prototype for explainable inconsistency detection in graphs

> **Status: research prototype / experiment.** All data are **synthetic**. Nothing here is state-of-the-art, and
> nothing here shows effectiveness on real banking data. Read the [limitations](#16-limitations) before the results.

**Research question.** *Can explicit symbolic constraints/rules complement graph neural representations for
explainable detection of inconsistent configurations in graph-structured systems?*

The testbed is a synthetic banking-transaction graph: nodes are accounts, directed edges are transactions with
amounts. Five hand-written rules (R1–R5) describe suspicious configurations. RuleGAT combines a Graph Attention
Network with those rules, and every alarm can be traced back to the rule(s) and the concrete values behind it.

---

## Table of contents
1. [Motivation](#1-project-motivation) · 2. [Problem definition](#2-problem-definition) · 3. [Why GNNs?](#3-why-gnns) ·
4. [Why rules?](#4-why-symbolic-rules) · 5. [Why both?](#5-why-a-neuro-symbolic-combination) ·
6. [Graph representation](#6-graph-representation) · 7. [Rules R1–R5](#7-rule-definitions-r1r5) ·
8. [Architecture](#8-model-architecture) · 9. [Constraint loss](#9-constraint-loss) ·
10. [Explainability](#10-explainability-mechanism) · 11. [Protocol](#11-experimental-protocol) · 12. [Metrics](#12-metrics) ·
13. [Install](#13-how-to-install) · 14. [Run](#14-how-to-run) · 15. [Outputs & measured results](#15-expected-outputs-and-measured-results) ·
16. [Limitations](#16-limitations) · 17. [Future work](#17-possible-future-work)

---

## 1. Project motivation
Systems such as transaction networks, infrastructure graphs or configuration graphs contain *relational* anomalies:
what makes a node suspicious is often how it is connected, not its own attributes. Learned models pick up statistical
regularities but are opaque; hand-written rules are transparent but brittle. This repository is a small, fully
reproducible laboratory for studying how the two interact — with explanations as a first-class output.

## 2. Problem definition
Given a directed attributed graph `G = (V, E)` — `V` = accounts, `E ⊆ V×V` = transactions, each edge `e = (u→v)` with
amount `a_e > 0` — and node metadata, decide for every node `v` whether it is *suspicious*, and say **why**.

Four objects are kept strictly apart (they are **not** the same thing):

| object | meaning | where |
|---|---|---|
| **injection targets** | nodes the generator deliberately made part of a pattern (`injected_anchor_mask`: one designated target per pattern; `injected_participant_mask`: anchors + ring accomplices) | `data_generation.py` |
| **rule violations** | (node, rule) pairs found by the rule engine on a graph | `rules.py` |
| **supervised labels** `y` | `y_v = 1` iff v violates ≥ 1 rule on the **clean** graph (rule-derived, *not* copied from injection ids) | `dataset.py` |
| **symbolic predictions** | rule verdicts on the graph a model is *given* (the observed graph) — equal to `y` on the clean graph, different under noise | `evaluate.py` |

`run_demo.py` prints the counts of all of them (anchors, participants, decoys, flagged nodes, overlaps) and never
assumes they coincide. With the default generator every injected participant is flagged and no decoy is (verified on
30 seeds), while the 16 anchors are a strict subset of the 32 flagged nodes. Under graph noise the symbolic verdicts on the *observed* graph
stop matching the labels (the rules-only F1 drops below 1 in §15).

## 3. Why GNNs?
Message passing lets a node's representation depend on its neighbourhood, which is what relational patterns
(rings, fan-in/out hubs) need. `h_v^(l+1) = σ( Σ_{u∈N(v)∪{v}} α_vu W h_u^(l) )`. A GAT learns the weights α:

```
α_vu = softmax_u( LeakyReLU( aᵀ [W h_v ‖ W h_u] ) ),     h_v^(l+1) = ‖_{k=1..K} ELU( Σ_u α_vu^k W^k h_u^(l) )
```

Honest caveat: attention/mean aggregation normalises over neighbours, so a plain GAT/GCN cannot easily *count*
neighbours or *sum* amounts — exactly what R2/R4/R5 require. That is one reason a symbolic complement is interesting.

## 4. Why symbolic rules?
They encode expert knowledge directly, need no labels, are exact on the configurations they describe and can be
audited: "R4: 10 unique receivers (≥ 8) and outgoing amount 46,432 (≥ 30,000)".

## 5. Why a neuro-symbolic combination?
Rules are precise but only cover what someone thought of; networks can generalise from weak correlated signals but
give no reason. Combining them lets the network raise alarms the rules do not cover, the rules override a network
that misses a known violation, and every disagreement can be inspected (§10).

## 6. Graph representation
`torch_geometric.data.Data` with:

| attribute | shape | notes |
|---|---|---|
| `x` | `[N, 4]` | `age_days/2500`, `device_changes/8`, `sin(2π·hour/24)`, `cos(2π·hour/24)`; names in `feature_names` |
| `edge_index` | `[2, E]` | **directed**, `source → destination`, sorted by (src, dst) |
| `edge_amount` | `[E]` | aligned 1-to-1 with `edge_index` (sorted jointly, checked in tests) |
| `y` | `[N]` | rule-derived label |
| `train_mask/val_mask/test_mask` | `[N]` bool | stratified 60/20/20, disjoint, complete |
| `age_days`, `blocked` | `[N]` | raw metadata for the rules — **never** in `x` |
| `injected_anchor_mask`, `injected_participant_mask`, `decoy_mask`, `injected_pattern` | `[N]` | generator bookkeeping |

**Leakage policy.** `blocked`, `risk_score`, labels, rule counts and symbolic outputs are not neural input;
`assert_no_leaky_features` rejects such names and a test checks that no column of `x` equals `blocked` or `y`. (`age_days`
*is* a feature, as the brief specifies, so the network can learn part of R2 from it; the rules use the raw value.) The
optional `data.structural_features=true` appends log in/out-degree to `x` (computed from observed edges; note this
makes the MLP graph-aware).

**Generator** (`n_nodes=150, n_rings=8, instances_per_rule=2, seed=42`): a sparse normal background graph, then
injections on disjoint account pools — R1: blocked sender; R2: 0–7-day-old account with 5–7 outgoing edges;
R3: 8 heavy 3-cycles; R4: fan-out to 9–12 receivers, total ≥ 31,500; R5: fan-in from 9–12 senders — plus **decoys**
that must *not* fire (blocked-but-silent, new-but-quiet, old-but-busy, cycles with one small edge, 7-receiver /
low-amount hubs). *Synthetic design assumption:* with probability `0.6·behavioral_signal` participants draw shifted
device-change counts and night-time login hours, giving the networks a weak non-structural signal (rules never use it;
set `data.behavioral_signal=0` to remove it).

## 7. Rule definitions R1–R5
Implemented exactly as specified in `src/rules.py` (thresholds in `config.yaml`, defaults shown):

| rule | condition |
|---|---|
| **R1** blocked account transacts | `blocked` ∧ `outgoing_count ≥ 1` |
| **R2** new account, high activity | `age_days ≤ 7` ∧ `outgoing_count ≥ 5` |
| **R3** large 3-cycle | node on a directed cycle A→B→C→A (distinct nodes) with **every** hop amount `≥ 12000` (parallel edges: the largest counts) |
| **R4** many receivers, high outflow | `unique_receivers ≥ 8` ∧ `outgoing_amount ≥ 30000` |
| **R5** many senders, high inflow | `unique_senders ≥ 8` ∧ `incoming_amount ≥ 30000` |

API: `evaluate_all_rules(data)` → `{"R1": RuleResult, …}`; `violated_nodes`, `flagged_nodes`, `rule_flag_matrix`,
`symbolic_score`, `explanations_for_node(node_id, results)`, `format_node_explanation(node_id, results)`.

## 8. Model architecture
All neural models share `input → layer → act → dropout → layer → act → linear head → logit` (hidden 32, 4 heads,
dropout 0.3). `MLP` ignores edges; `GCN` uses `GCNConv`; `GAT` uses `GATConv`. GNN messages flow along both directions
(`model.message_passing: undirected`) so a sender also sees its receivers; the rules and stored `edge_index` stay
directed.

**RuleGAT** = GAT (neural branch, `p_v = σ(z_v)`) + rule engine (symbolic branch, `s_v ∈ {0,1}`, `s_v = 1` iff any rule fires):

```
final_v = ( p_v + w · s_v ) / ( 1 + w ),         w = symbolic_weight (default 0.5)
```

a convex combination in [0,1]; with `w = 0.5` a rule-flagged node needs `p ≥ 0.25` to pass 0.5, an unflagged node `p ≥ 0.75`.

* **Training time:** GAT trained with weighted BCE (+ optional `λ·L_constraint`). Rule verdicts are **not** an input feature
  and the network is not trained on `final_v`.
* **Inference time:** `final_v` combines the trained network's `p_v` with rule verdicts computed on the graph being scored.
  Rules read only `edge_index, edge_amount, age_days, blocked`, never `y`.
* **Circularity warning:** labels are *defined by* the rules, so on a clean graph the symbolic branch reproduces the label
  function. The hybrid is therefore aligned with the labels by construction. To keep this visible every table also contains
  **"RuleGAT (neural branch only)"** (same trained network, no rules at inference) and **"Symbolic rules only (reference)"**.

## 9. Constraint loss
With `p_v = σ(z_v)` and `s_v` the symbolic verdict, the rules say *"v violates a rule ⇒ v is suspicious"*. Under the
product t-norm the violation of `a ⇒ b` is `a·(1−b)`:

```
L_constraint = (1/|S|) Σ_{v∈S} s_v · (1 − p_v)              L_total = L_cls + λ · L_constraint,   λ ∈ {0, 0.1, 0.5, 1.0}
```

It is zero iff every rule-violating node in `S` gets probability 1, never penalises nodes the rules do not flag (rules are
*sufficient*, not *necessary*), and is differentiable. `S` = training nodes by default (`rulegat.constraint_nodes: train`);
`all` applies it to every node (transductive).
**Honest note:** on labelled training nodes with rule-derived labels (`y = s`) this reduces to an extra positive-class term
`mean(y(1−p))`. Its distinct effect can only come from unlabelled nodes (`all`), but there it injects label-equivalent
information about val/test nodes — hence the default. `L_cls` = `BCEWithLogits` with `pos_weight = n_neg/n_pos` (train nodes).

## 10. Explainability mechanism
`src/explain.py` builds, from the actual rule results, a record per node — neural probability, RuleGAT score, violated
rules, the responsible values and thresholds:

```
Node 82
  Neural score: …   RuleGAT score: …   [both]
  R4 violated: 10 unique receivers (>= 8) and outgoing amount = 46,432 (>= 30,000)
```

Graph-level: nodes and violations per rule, pairwise overlap matrix, multi-rule nodes, and the four-way agreement
(both / neural-only / symbolic-only / neither) plus lists of the most confident disagreements (`disagreement_report`) —
this is how question 10 of the brief ("when neural and symbolic disagree, can we inspect it?") is answered.

## 11. Experimental protocol
* Adam, weight decay 5e-4, lr 0.01, ≤ 300 epochs, **early stopping on validation F1** (neural branch, threshold 0.5,
  ties broken by validation loss), best-validation weights restored. Hyper-parameters are the config defaults — **not tuned**.
* **Threshold:** validation-selected (maximise validation F1 on a grid 0.01…0.99; ties → closest to 0.5) and then applied
  *unchanged* to test. Every table reports **both** `threshold=0.5` and `val-selected threshold`.
* The test set is never used for thresholds, checkpoints or hyper-parameters (unit tests flip all test labels and check that
  training history and threshold are unchanged).
* **Experiments:** A baselines (`run_baselines.py`) · B λ-ablation (`run_lambda_ablation.py`) · C threshold comparison
  (in every table) · D seeds 42–46 (`run_multiseed.py`; each seed changes graph, split and initialisation; `--fixed-graph`
  keeps the graph) · robustness (`run_robustness.py`).
* **Robustness:** the *observed* graph gets (i) irrelevant added edges (10 %/30 % of E) or (ii) rewired edge destinations
  (5 %/15 %). Labels stay the clean ground truth; models **and rules** only see the noisy graph.
* Mean ± **sample** std (ddof = 1) over seeds; NaN values are skipped, not zero-filled.

## 12. Metrics
Suspicious nodes are a minority (~21 %): predicting "normal" everywhere already gives ~79 % accuracy. **Recall** (share of
suspicious nodes found), **precision** (share of alarms that are right) and **F1** (their harmonic mean) expose that;
**ROC-AUC** measures ranking independent of a threshold and is `NaN` if only one class is present. Precision/recall/F1
use the `zero_division=0` convention.

Symbolic-rule metrics (on the test nodes; `V_k` = nodes violating Rk on the observed graph, `V = ∪V_k`, `s_v = 1[v∈V]`):

| metric | definition |
|---|---|
| `R{k}_detection_rate` | `|{v ∈ V_k : pred_v=1}| / |V_k|` (NaN if `V_k` empty) |
| `rule_violation_detection_rate` | `|{v ∈ V : pred_v=1}| / |V|` |
| `rule_consistency` | `mean_v 1[pred_v = s_v]` — agreement with the symbolic verdict |

On the clean graph with rule-derived labels these equal *recall* and *accuracy* respectively (tested); they diverge only under
noise. `multiseed_rule_pooled.csv` also pools detections over seeds (Σ detected / Σ violating).

## 13. How to install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch, torch_geometric, numpy, pandas, scikit-learn, matplotlib, networkx, PyYAML, pytest
```
Developed and run on CPU with Python 3.11, torch 2.14, torch_geometric 2.8, scikit-learn 1.8.

## 14. How to run
```bash
python -m pytest tests -q                              # 60 tests
python run_demo.py                                     # full walk-through (+ figures); --no-plots, --reuse-models, --seed N
python -m src.train --model rulegat --seed 42 --epochs 200 --lr 0.01 --hidden-dim 32 --dropout 0.3 --lambda-constraint 0.5 --save

python experiments/run_baselines.py                    # Experiments A + C (seed 42)
python experiments/run_lambda_ablation.py              # Experiment B (seeds 42-46, lambda in {0,0.1,0.5,1}, both constraint scopes)
python experiments/run_multiseed.py                    # Experiment D (seeds 42-46)
python experiments/run_robustness.py                   # graph-noise experiment

# any config value can be overridden:  --set section.key=value  (repeatable), --seeds 42 43 ..., --fixed-graph
python experiments/run_multiseed.py --set data.instances_per_rule=5 --set paths.tables=outputs/more_rules/tables
python experiments/run_multiseed.py --set data.structural_features=true --set paths.tables=outputs/structural/tables
```
Everything runs in about two minutes on a laptop CPU. Runs are deterministic: re-running an experiment reproduced its
report table exactly (checked).

## 15. Expected outputs and measured results
Outputs: `outputs/tables/*.csv|json`, `outputs/figures/*.png`, `outputs/models/*` (weights, history, config),
`outputs/logs/*` (logs + the exact configuration of every experiment), `data/generated/graph_seed42.pt`.

**The numbers below were produced by running this code (CPU, seeds 42–46, default config); they are not the
"historical reference values" of the brief, which are not reproduced (different generator, different graphs).** Each
graph has 150 nodes and 32 suspicious nodes; the **test split has only 30 nodes (6 positives)**, so all figures are noisy.

*Graph statistics, seed 42:* 150 nodes, 435 transactions; 16 injected anchors, 32 injected participants, 16 decoys;
**32 symbolically flagged nodes**, 32 (node, rule) violations, 8 large 3-cycles; R1/R2/R4/R5: 2 nodes each, R3: 24 nodes;
no node violates two rules; 16/16 anchors and 32/32 participants flagged, 0 decoys flagged.

**Experiments A + C + D — test metrics, mean ± std over 5 seeds** (`multiseed_report.csv`; threshold 0.5):

| model | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| MLP | 0.807 ± 0.104 | 0.577 ± 0.282 | 0.600 ± 0.149 | 0.572 ± 0.189 | 0.772 ± 0.131 |
| GCN | 0.780 ± 0.122 | 0.546 ± 0.166 | 0.800 ± 0.298 | 0.596 ± 0.139 | 0.846 ± 0.097 |
| GAT | 0.787 ± 0.124 | 0.609 ± 0.253 | 0.667 ± 0.289 | 0.543 ± 0.179 | 0.887 ± 0.089 |
| RuleGAT (hybrid) | 0.960 ± 0.060 | 0.875 ± 0.177 | 0.967 ± 0.075 | 0.914 ± 0.128 | 0.989 ± 0.021 |
| RuleGAT (neural branch only) | 0.807 ± 0.086 | 0.530 ± 0.114 | 0.800 ± 0.139 | 0.631 ± 0.113 | 0.889 ± 0.099 |
| Symbolic rules only (reference) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

With the **validation-selected threshold** (F1): MLP 0.612 ± 0.135, GCN 0.596 ± 0.139 (selected thresholds 0.50–0.51, same predictions as at 0.5),
GAT 0.578 ± 0.087, RuleGAT 0.935 ± 0.063, neural branch 0.631 ± 0.113 (threshold 0.50 in every seed). The selected thresholds differ a lot between
seeds (MLP 0.50–0.80, GAT 0.30–0.50, RuleGAT 0.36–0.65).

**Experiment B — λ (train-node constraint, threshold 0.5, 5 seeds, F1 / ROC-AUC):**

| λ | RuleGAT hybrid F1 | RuleGAT hybrid AUC | neural branch F1 | neural branch AUC |
|---|---|---|---|---|
| 0 | 0.943 ± 0.078 | 0.999 ± 0.003 | 0.543 ± 0.179 | 0.888 ± 0.089 |
| 0.1 | 0.943 ± 0.078 | 0.999 ± 0.003 | 0.575 ± 0.129 | 0.886 ± 0.089 |
| 0.5 | 0.914 ± 0.128 | 0.989 ± 0.021 | 0.631 ± 0.113 | 0.889 ± 0.099 |
| 1.0 | 0.863 ± 0.132 | 0.979 ± 0.026 | 0.554 ± 0.159 | 0.917 ± 0.054 |

(the transductive `constraint_nodes=all` variant is in `rulegat_lambda_results.csv`; hybrid F1 0.943 → 0.938 → 0.868 → 0.842.)

**Robustness — test F1, threshold 0.5, mean ± std over 5 seeds** (`robustness_report.csv`):

| noise on observed graph | GAT | RuleGAT (hybrid) | RuleGAT (neural only) | rules only |
|---|---|---|---|---|
| clean | 0.543 ± 0.179 | 0.914 ± 0.128 | 0.631 ± 0.113 | 1.000 ± 0.000 |
| +10 % irrelevant edges | 0.519 ± 0.147 | 0.925 ± 0.072 | 0.593 ± 0.134 | 0.971 ± 0.064 |
| +30 % irrelevant edges | 0.615 ± 0.091 | 0.833 ± 0.164 | 0.541 ± 0.085 | 0.956 ± 0.065 |
| 5 % rewired edges | 0.544 ± 0.130 | 0.811 ± 0.070 | 0.565 ± 0.181 | 0.982 ± 0.041 |
| 15 % rewired edges | 0.539 ± 0.198 | 0.789 ± 0.120 | 0.559 ± 0.101 | 0.842 ± 0.208 |

*Supplementary run with degree features in `x`* (`data.structural_features=true`, 5 seeds, threshold 0.5): F1 MLP 0.574 ± 0.144,
GCN 0.610 ± 0.132, GAT 0.636 ± 0.154, RuleGAT hybrid 0.852 ± 0.158, neural branch 0.555 ± 0.178; AUC 0.768 / 0.908 / 0.875 / 0.989 / 0.878.

### What these measurements do and do not say
1. **Graph structure vs MLP.** Mean AUC orders MLP (0.77) < GCN (0.85) < GAT (0.89) but standard deviations are 0.09–0.13, F1 does not
   order the same way, and n = 5 with a 30-node test set. *Suggestive for ranking, not established.*
2. **GAT vs GCN.** No evidence that GAT beats GCN here (higher AUC, lower F1, all within one std).
3. **Do rules improve detection?** Hybrid inference improves a lot (F1 0.54 → 0.91) — but labels are defined by the rules, so this is
   **expected by construction**: the rules alone score 1.000 on the clean graph and *beat* the hybrid, whose neural false positives cost
   precision. The neural branch of RuleGAT (F1 0.631 vs GAT 0.543) is within noise of GAT. The constraint loss did **not** show a
   benefit: neural-branch F1 is non-monotonic in λ within noise, and hybrid F1 falls as λ grows.
4. **Threshold selection.** Small, mixed effects (F1 +0.04 MLP, +0.035 GAT, +0.02 RuleGAT, 0 GCN); with 6 validation positives the
   selected threshold is itself noisy (in the seed-42 demo it lowered RuleGAT's test F1 from 1.00 to 0.91).
5. **Stability.** F1 standard deviations are 0.09–0.19; single runs are not trustworthy — always read the multi-seed tables.
6. **Noise.** RuleGAT (hybrid) beats GAT under every noise setting, but the rules alone have a higher mean F1 than the hybrid in all five
   settings (the gap is within one std only under 15 % rewiring: 0.842 ± 0.208 vs 0.789 ± 0.120). So the results show that the *rules* carry
   the performance and degrade gracefully at these noise levels; they do **not** show that combining them with a GAT adds robustness beyond
   the rules.
7. **Rule frequencies.** R3 has 24 violating nodes and the others 2 each — that is set by `n_rings=8` vs `instances_per_rule=2`, a design
   choice, **not** an empirical finding. With so few instances the per-rule detection rates are thin: over five seeds' test sets there were
   3 R1, 0 R2, 24 R3, 2 R4 and 1 R5 violating test nodes (`multiseed_rule_pooled.csv`, `multiseed_rule_detection.png`), so R2 is never evaluated.
   Use `--set data.instances_per_rule=5` for more (the default is kept as specified).

## 16. Limitations
* **Data are synthetic**; they do not represent real banking data or real fraud, and injected patterns are illustrative.
* **Labels are rule-derived.** Rule-derived labels favour models aligned with those rules; the symbolic-only reference achieves 1.000 on the
  clean graph by construction, and hybrid-vs-neural gaps must be read in that light.
* **These experiments do not establish real-world banking effectiveness**; strong performance on this benchmark does not imply deployment readiness.
* The **rule definitions and thresholds are manually designed** and are not a substitute for domain validation.
* **Results vary across seeds** (std 0.09–0.19 on F1); the test split is 30 nodes with 6 positives; five seeds is a minimum, not a statistical test.
* Hyper-parameters are untuned defaults; early stopping uses a validation set with only 6 positives.
* Synthetic assumption: injected accounts have a mild behavioural shift; results depend on `behavioral_signal`.
* The hybrid weight `w` and the constraint scope are design choices, not learned; `constraint_nodes=all` leaks label-equivalent information.
* GCN/GAT normalise over neighbours and cannot count neighbours or sum amounts; the architecture is deliberately small and simple.
* Determinism is requested (`use_deterministic_algorithms(warn_only=True)`) and observed on CPU; other hardware/versions may differ slightly.

## 17. Possible future work
Hold out rules (the network/hybrid only knows a subset) so that the labels are no longer identical to the symbolic branch; independent labels
(noisy/partial or an external process) so that the rules become weak supervision; learnable per-rule weights and soft rule relaxations
(fuzzy cycle amounts, margins); rule-specific constraint losses; count-aware aggregation (sum/degree encodings, PNA); larger graphs and
more instances per rule with proper significance tests; temporal transactions; attention-based explanations next to rule-based ones.

## Repository layout
```
rulegat/
├── README.md  requirements.txt  config.yaml  .gitignore  run_demo.py
├── data/generated/                      # saved graphs
├── src/
│   ├── data_generation.py  dataset.py   # generator, PyG dataset, labels, splits, statistics, graph noise
│   ├── rules.py                         # R1-R5, aggregation, explanations
│   ├── models.py  rulegat.py  losses.py # MLP/GCN/GAT, RuleGAT hybrid, weighted BCE + constraint loss
│   ├── train.py  evaluate.py            # training loop + CLI, protocol/orchestration
│   ├── metrics.py  explain.py  visualization.py  utils.py
├── experiments/  run_baselines.py  run_lambda_ablation.py  run_multiseed.py  run_robustness.py  _common.py
├── tests/  conftest.py  test_rules.py  test_data.py  test_models.py  test_metrics.py  test_explain.py
└── outputs/  models/  figures/  tables/  logs/
```
