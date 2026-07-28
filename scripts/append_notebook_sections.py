"""Append the corrected ensemble + deep-analysis sections to the pipeline
notebook.  Idempotent: re-running replaces the previously appended Part II.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MARKER = "Part II — Error-decorrelated stacking ensemble"


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


CELLS = [
    md(f"""
---

# {MARKER}

Part I compared three prior-correction settings on one GCN-GRU backbone.
Part II adds the ensemble extension and the statistical work that decides
whether its gains are real.

This is the **corrected** implementation. It keeps the design of
`elliptic_stacking_ensemble.ipynb` — five base learners, a meta-learner
fitted on a held-out validation window, the online Saerens-EM tracking head
— and fixes six things:

| # | Original | Here |
|---|----------|------|
| 1 | wavelet block decomposed over all 49 weeks | expanding-prefix, causal |
| 2 | one frozen GCN-GRU reused for all ten seeds | retrained per seed |
| 3 | two threshold conventions in adjacent columns | four conventions, reported separately |
| 4 | tracking head ran at the Beta(0.2, 1.8) defaults | Beta(5, 10) and Beta(0.2, 1.8), both reported |
| 5 | time-blocked OOF computed, never used | validation-only, stated plainly |
| 6 | across-seed interval labelled a confidence interval | across-seed **and** node-level bootstrap, both named |

`AUDIT_ENSEMBLE.md` documents each one with the evidence.

| Section | What it establishes |
|---------|---------------------|
| 12 | The leak-aware split, the five base learners, the reference gate |
| 13 | The stacking ensemble and the meta-learner ablation |
| 14 | The head-to-head, under one convention at a time |
| 15 | Where the decorrelation comes from |
| 16 | What the tracking head is sensitive to |
| 17 | Deeper analysis — enrichment, calibration, cost, drift, window |
| 18 | Reconciliation against the paper and the original run |
"""),
    md("""
## 12. Leak-aware split, base learners, reference gate

```
train      weeks  1 - 34    base learners
validate   weeks 35 - 41    meta-learner + deployable thresholds
test       weeks 42 - 49    reported (shutdown at 43)
```

The legacy split (train ≤ 34, test ≥ 35) has no validation window, so a
meta-learner fitted under it would have to see test data. It is kept only as
the reference gate: the Random Forest has to reproduce F1 ≈ 0.830 there, or
nothing downstream is interpretable.
"""),
    code("""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/Users/aarthy/projects/nwu/stgnn_fraud")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

STUDY = PROJECT_ROOT / "artefacts" / "ensemble_study.json"
A = json.load(open(STUDY)) if STUDY.exists() else None
if A is None:
    print("No study found. Build it with:\\n"
          "  python -m scripts.build_enrichment_cache\\n"
          "  python -m scripts.train_gnn_bases\\n"
          "  python -m scripts.run_ensemble_study\\n"
          "  python -m scripts.aggregate_ensemble_study "
          "--out artefacts/ensemble_study.json")
else:
    m = A["meta"]
    print(f"{A['n_seeds']} seeds | {m['n_nodes']:,} nodes, {m['n_edges']:,} edges, "
          f"{m['n_illicit']:,} illicit")
    print(f"features: {m['n_features_total']} ({m['feature_blocks']}), "
          f"causal wavelet = {m['causal_wavelet']}")
    print(f"nodes appearing in more than one timestep: {m['multi_timestep_nodes']}")
    print(f"split: {m['split']}")

S = (A or {}).get("summary", {})
def q(key, field="median"):
    return S.get(key, {}).get(field, float("nan"))
def ci(key):
    return f"[{q(key,'lo'):.3f}, {q(key,'hi'):.3f}]"
"""),
    code("""
if A:
    rows = []
    for k, lbl in [("gate.rf_f1", "Random Forest"), ("gate.xgb_f1", "XGBoost"),
                   ("gate.lgbm_f1", "LightGBM"), ("gate.gnn_f1", "GCN-GRU")]:
        rows.append({"learner": lbl, "F1 (legacy, oracle)": q(k),
                     "95% CI": ci(k), "seeds": S[k]["n_seeds"]})
    print("Reference gate — legacy split (train 1-34, test 35-49)")
    display(pd.DataFrame(rows).set_index("learner").round(4))
    print("\\npaper reference: RF = 0.830 | original notebook run: 0.8304")
    assert 0.78 <= q("gate.rf_f1") <= 0.86, "GATE FAILED"
    print("GATE PASSED")
"""),
    md("""
## 13. The stacking ensemble and the meta-learner ablation

Five base learners feed the combiner: three tree models that are nearly
redundant with each other, one linear model, and the GCN-GRU. The
meta-learner is fitted on the **validation window only** — never on
base-model training predictions — so it cannot inherit base-model
overfitting.

PR-AUC is the ablation axis because it measures ranking quality
independently of where the threshold lands, and ranking quality is what the
tracking head consumes downstream.
"""),
    code("""
if A:
    rows = []
    for k, lbl in [("meta.DT.pr_auc", "Decision tree"),
                   ("meta.LR.pr_auc", "Logistic regression"),
                   ("meta.RF.pr_auc", "Random Forest"),
                   ("meta.XGBs.pr_auc", "XGBoost"),
                   ("la.lgbm_pr_auc", "Best single base (LightGBM)")]:
        rows.append({"meta-learner": lbl, "PR-AUC": q(k), "95% CI": ci(k)})
    display(pd.DataFrame(rows).set_index("meta-learner").round(4))

    paper = {r["key"]: r["paper"] for r in A["reconciliation"]}
    orig  = {r["key"]: r["original"] for r in A["reconciliation"]}
    keys = ["meta.DT.pr_auc","meta.LR.pr_auc","meta.RF.pr_auc",
            "meta.XGBs.pr_auc","la.lgbm_pr_auc"]
    labels = [r["meta-learner"] for r in rows]
    x = np.arange(len(keys)); w = 0.27
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(x - w, [paper.get(k) or np.nan for k in keys], w, label="paper",
           color="#898781")
    ax.bar(x,     [orig.get(k)  or np.nan for k in keys], w, label="original run",
           color="#eb6834")
    ax.bar(x + w, [q(k) for k in keys], w, label="corrected run", color="#2a78d6")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylabel("PR-AUC"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Meta-learner ablation, leak-aware test window")
    plt.tight_layout(); plt.show()
"""),
    md("""
## 14. The head-to-head, under one convention at a time

The published claim is that feeding the tracking head a decorrelated score
improves the post-shutdown rare-class decision. That comparison is only
meaningful if both sides are scored the same way.

The original notebook printed `F1 post-43(oracle)` — an oracle threshold
fitted on the **whole** test window — next to `F1 post-43 +track`, fitted on
the **post-43 subset**. Read across the row, the tracking head looks like it
quadruples F1; part of that is the head and part is the convention. Four
conventions are reported here:

| convention | uses test labels? | what it means |
|---|---|---|
| whole-window oracle | yes | threshold tuned on weeks 42–49, applied post-43 |
| post-subset oracle | yes | threshold tuned on weeks 43–49 — the upper bound |
| validation-fitted | **no** | threshold tuned on weeks 35–41 — deployable |
| prior-matched | **no** | flag the top q<sub>t</sub> of each week from the tracked rate |
"""),
    code("""
if A:
    systems = [("track.stack_RF", "Stack (RF meta) + tracking"),
               ("track.stack_XGBs", "Stack (XGB meta) + tracking"),
               ("track.rf_alone", "Random Forest + tracking"),
               ("track.gnn_alone", "GCN-GRU + tracking")]
    conv = [("f1_post_wholewin", "whole-window oracle"),
            ("f1_post", "post-subset oracle"),
            ("f1_post_deployable", "validation-fitted"),
            ("f1_post_prior_matched", "prior-matched")]
    tbl = pd.DataFrame(
        {lbl: {cl: q(f"{sk}.{ck}") for ck, cl in conv} for sk, lbl in systems}).T
    display(tbl.round(4))
    print("\\nCompare DOWN a column, never across a row.")

    for name, t in A["paired_seed_tests"].items():
        print(f"{name:36s} median diff {t['median_diff']:+.4f}  "
              f"wins {t['wins']}/{t['n']}  sign p = {t['sign_p']:.4f}")
"""),
    md("""
## 15. Where the decorrelation comes from

Stacking pays only when the base learners make *different* mistakes. Unlike
the original — which reused one frozen GCN-GRU artefact for every seed, so
the GNN's error pattern came from a single draw — the encoder is retrained
per seed here, and the matrix below is the median over those seeds.
"""),
    code("""
if A:
    ec = A["error_correlation"]
    C = np.array(ec["matrix"]); labels = ec["names"]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(C, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    color="white" if C[i,j] > 0.6 else "#0b0b0b", fontsize=9)
    ax.set_title("Base-learner error correlation (median over seeds)")
    fig.colorbar(im, shrink=0.8); plt.tight_layout(); plt.show()
    print(f"disagreement fraction: {ec['disagreement_fraction']:.4f}")
    display(pd.Series(ec["error_rate"], name="test error rate").to_frame().round(4))
"""),
    md("""
## 16. What the tracking head is sensitive to

Two choices were implicit in the original and are made explicit here.

**The regulariser.** The paper's Methods states Beta(5, 10). The original
code calls `online_per_timestep_tracker(p_per_t, p_train)` with no α or β,
so it ran at the Beta(0.2, 1.8) defaults. Both are reported below.

**The EM population.** The prior was estimated from labelled test rows only.
Elliptic labels 23% of nodes and the labelled subset is not a random sample
of the stream, so this estimates the labelled subsample's prior rather than
the deployment prior. It is self-consistent — the target rate is also the
labelled rate — but it is a choice, and the all-node estimator is the one
that could actually be deployed.

**On ρ.** The test window is eight weeks, so Spearman ρ lives on a lattice
of 84 steps and the exact two-sided permutation p-value for |ρ| ≥ 0.333 at
n = 8 is 0.428. Rank correlation cannot separate these systems. The mean
absolute error between the tracked and true weekly rate uses the magnitudes
and does not degrade at this n, so it is reported alongside.
"""),
    code("""
if A:
    rows = []
    for k, v in A["tracker_sensitivity"].items():
        sysname, cfg, pop = k.split("|")
        rows.append({"system": sysname, "regulariser": cfg,
                     "EM population": pop, "post-43 F1": v["f1_post"],
                     "tracking rho": v["rho"], "rate MAE": v["rate_mae"]})
    display(pd.DataFrame(rows).set_index(
        ["system", "regulariser", "EM population"]).round(4))

    traj = A["trajectories"]; true_rate = A["meta"]["true_rate"]
    ts = sorted(int(t) for t in true_rate if int(t) >= 42)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(ts, [true_rate[str(t)] for t in ts], "o-", color="#0b0b0b",
            lw=2, label="true illicit rate")
    for key, lbl, c in [("stack_RF", "tracked — stack (RF meta)", "#4a3aa7"),
                        ("rf_alone", "tracked — Random Forest", "#eb6834"),
                        ("gnn_alone", "tracked — GCN-GRU", "#1baf7a")]:
        srs = traj.get(key) or {}
        xs = sorted(int(t) for t in srs)
        if xs:
            ax.plot(xs, [srs[str(t)] for t in xs], "s--", color=c, label=lbl)
    ax.axvline(42.5, color="#d03b3b", ls=":", lw=1)
    ax.set_xlabel("timestep (week)"); ax.set_ylabel("illicit rate")
    ax.set_title("Tracked vs. true illicit rate (median across seeds)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
"""),
    md("""
## 17. Deeper analysis

**17.1 Feature enrichment.** Topological, node2vec, delta and *causal*
wavelet blocks against the raw 166 columns, paired across seeds. The
original's wavelet block leaked one week: perturbing only weeks 42–49
changed the feature values for validation weeks 40 and 41, and the
meta-learner trains on the validation window.

**17.2 Calibration.** Prior shift moves the decision boundary; it does not
by itself miscalibrate the score. Separating the two is what rules out
temperature scaling as a fix.

**17.3 Decision economics.** F1 weights a missed illicit transaction and a
frozen legitimate customer equally, which no compliance desk does.

**17.4 Temporal window.** Zero Elliptic transaction nodes appear in more
than one timestep, so there is no per-node history to sweep. The original
prepended zero vectors; here the preceding weeks' population-mean embeddings
are prepended, so W carries actual context.
"""),
    code("""
if A:
    rows = []
    for nm in ("RF", "XGB", "LGBM", "LR", "GNN"):
        rows.append({"learner": nm,
                     "raw166 F1": q(f"base.raw166.{nm}.f1"),
                     "enriched F1": q(f"base.enriched.{nm}.f1"),
                     "lift": q(f"lift.{nm}.f1"),
                     "raw166 PR-AUC": q(f"base.raw166.{nm}.pr_auc"),
                     "enriched PR-AUC": q(f"base.enriched.{nm}.pr_auc")})
    print("17.1 Feature enrichment (leak-aware test window)")
    display(pd.DataFrame(rows).set_index("learner").round(4))
"""),
    code("""
if A:
    cal = A["calibration"]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot([0, 1], [0, 1], ":", color="#c3c2b7", label="perfect")
    for k, lbl, c in [("rf", "Random Forest", "#eb6834"),
                      ("gnn", "GCN-GRU", "#1baf7a"),
                      ("stack_tracking", "stack + tracking", "#4a3aa7")]:
        d = cal.get(k)
        if not d:
            continue
        ax.plot(d["bin_conf"], d["bin_acc"], "o-", color=c,
                label=f"{lbl} (ECE {d['ece']:.3f}, Brier {d['brier']:.3f})")
    ax.set_xlabel("predicted illicit probability")
    ax.set_ylabel("observed illicit fraction")
    ax.set_title("17.2 Reliability, post-shutdown window")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

    cc = A.get("cost_curves", {})
    if cc:
        print("17.3 Cost-minimising threshold, stack + tracking, post-shutdown")
        display(pd.DataFrame([
            {"FN:FP": int(k), "best threshold": v["best_threshold"],
             "cost at best": v["best_cost"], "cost at 0.5": v["cost_at_0p5"]}
            for k, v in sorted(cc.items(), key=lambda kv: int(kv[0]))
        ]).set_index("FN:FP").round(4))

    print("\\n17.4 Temporal window sweep")
    display(pd.DataFrame([
        {"W": w, "corrected (population context)": q(f"win.{w}"),
         "original (zero padding)": next(
             (r["original"] for r in A["reconciliation"]
              if r["key"] == f"win.{w}"), np.nan)}
        for w in (1, 3, 5, 8)]).set_index("W").round(4))
"""),
    md("""
## 18. Reconciliation

Every published number, re-measured. Three columns: the paper, the original
notebook's own outputs, and this corrected run with its across-seed 95%
interval.

* **consistent** — the reference value lies inside this run's interval
* **close** — within 5% relative
* **shifted** — 5 to 20% away
* **diverges** — more than 20% away
"""),
    code("""
if A:
    R = pd.DataFrame(A["reconciliation"])
    R["95% CI"] = R.apply(lambda r: f"[{r['lo']:.3f}, {r['hi']:.3f}]", axis=1)
    view = R[["label", "paper", "original", "median", "95% CI",
              "vs_paper", "vs_original"]].rename(columns={
        "label": "quantity", "median": "corrected"})
    display(view.round(4))
    print()
    print("vs paper:"); print(view["vs_paper"].value_counts().to_string())
    print(); print("vs original run:")
    print(view["vs_original"].value_counts().to_string())
"""),
    md("""
### How to read this

**Apparatus.** The reference checks — Random Forest near F1 0.83 on the
legacy split, the boosting learners just behind, the same forest dropping to
≈0.63 once evaluation crosses week 43 — certify that the pipeline and
protocol are the ones the paper describes. They reproduce to three decimals
in both the original and the corrected run. If they had moved, nothing below
them would be interpretable.

**The ensemble claim.** Bounded and specific: a post-shutdown F1 gain when
the stack's score feeds the tracking head, measured under one convention on
both sides, and *not* an improvement in how faithfully the per-week rate is
tracked. The paired seed tests in §14 keep those apart; a raw median
comparison would blur them.

**The negatives.** The absolute-F1 targets fail with their intervals on the
wrong side, and they fail structurally rather than by tuning — the training
window sits entirely before the shutdown, so the regime under test was never
in the training data. No encoder and no feature block recovers a regime that
was never trained on, which is exactly what §17.1 shows.

**What the corrections changed.** See `AUDIT_ENSEMBLE.md`. The two verified
contributions of the paper — the Random Forest reproducing the
strict-inductive benchmark, and the tracking head being
encoder-independent — survive every fix. What the corrections bear on is how
the ensemble result should be *stated*, and how much confidence the seed
intervals carry.
"""),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default="stgnn_fraud_pipeline.ipynb")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    nb = json.load(open(args.notebook))
    keep, dropping = [], False
    for c in nb["cells"]:
        src = "".join(c["source"])
        if MARKER in src:
            dropping = True
        if not dropping:
            keep.append(c)
    nb["cells"] = keep + CELLS
    out = Path(args.out or args.notebook)
    json.dump(nb, open(out, "w"), indent=1)
    print(f"notebook now has {len(nb['cells'])} cells "
          f"({len(CELLS)} appended) -> {out}")


if __name__ == "__main__":
    main()
