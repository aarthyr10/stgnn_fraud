"""Corrected stacking-ensemble + prior-tracking study on the real Elliptic
graph.

This is the methodologically-fixed successor to
``elliptic_stacking_ensemble.ipynb``.  It keeps that notebook's design --
five base learners, a meta-learner fitted on a held-out validation window,
the online Saerens-EM tracking head, the same reference gates -- and
corrects six things the original got wrong or left implicit:

1. **Causal feature enrichment.**  The wavelet block is recomputed on the
   expanding prefix instead of decomposing the whole 49-week series, so no
   test-period information reaches a training row.
2. **Honest seed protocol.**  The GCN-GRU base learner is retrained per
   seed rather than reused from one frozen artefact, so seed intervals
   cover the GNN too.
3. **One thresholding convention at a time.**  Every system reports the
   deployable (validation-fitted), prior-matched and oracle numbers side by
   side, and post-shutdown F1 is never compared across conventions.
4. **Explicit tracker hyper-parameters.**  The paper states Beta(5, 10);
   the original code silently used the Beta(0.2, 1.8) defaults.  Both are
   run and reported.
5. **Explicit EM population.**  Estimating the deployment prior from
   labelled rows only is a choice, not a default; the all-node estimate is
   run alongside it.
6. **Scaled linear model and data-derived class weighting** instead of an
   unscaled logistic regression and a hard-coded ``scale_pos_weight``.

Writes one JSON per seed so the run is resumable.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import deep_analysis as da  # noqa: E402
from app.services.enrichment import enrich  # noqa: E402

log = logging.getLogger("study")

TRAIN_MAX, VAL_MAX, SHUTDOWN, LAST_T = 34, 41, 43, 49
BASE_LEARNERS = ["RF", "XGB", "LGBM", "LR", "GNN"]
META_LEARNERS = ["DT", "LR", "RF", "XGBs"]
TRACKER_CONFIGS = {
    "paper_beta_5_10": dict(alpha=5.0, beta=10.0),
    "code_beta_0.2_1.8": dict(alpha=0.2, beta=1.8),
}


# ==========================================================================
# data
# ==========================================================================
def load_frame(data_dir: Path):
    feat = np.load(data_dir / "feat.npy")
    ts = np.load(data_dir / "ts.npy").astype(np.int64)
    txid = np.load(data_dir / "txid.npy").astype(np.int64)
    raw = [f"f{i}" for i in range(feat.shape[1])]
    df = pd.DataFrame(feat, columns=raw)
    df.insert(0, "t", ts)
    df.insert(0, "txId", txid)
    cls = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    cls.columns = ["txId", "class"]
    cls["y"] = (cls["class"].astype(str)
                .map({"1": 1, "2": 0, "unknown": -1}).fillna(-1).astype(int))
    df = df.merge(cls[["txId", "y"]], on="txId", how="left")
    df["y"] = df["y"].fillna(-1).astype(int)
    df = df.sort_values("txId").reset_index(drop=True)
    edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
    edges.columns = ["src", "dst"]
    return df, edges, raw


def masks_for(frame, scheme="threeway"):
    lab = frame.y.isin([0, 1]).to_numpy()
    t = frame.t.to_numpy()
    if scheme == "threeway":
        return (lab & (t <= TRAIN_MAX), lab & (t > TRAIN_MAX) & (t <= VAL_MAX),
                lab & (t > VAL_MAX))
    if scheme == "legacy":
        return lab & (t <= TRAIN_MAX), None, lab & (t > TRAIN_MAX)
    raise ValueError(scheme)


# ==========================================================================
# learners
# ==========================================================================
def make_base(name, seed, pos_weight):
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, max_depth=None,
                                      n_jobs=-1, random_state=seed)
    if name == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=6,
                             learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="aucpr",
                             scale_pos_weight=pos_weight, tree_method="hist",
                             random_state=seed, n_jobs=-1)
    if name == "LGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=500, max_depth=-1,
                              learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, class_weight="balanced",
                              random_state=seed, n_jobs=-1, verbose=-1)
    if name == "LR":
        # scaled: an unscaled logistic regression on 213 heterogeneous
        # columns does not converge inside max_iter and is not comparable
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0,
                               random_state=seed))
    raise ValueError(name)


def make_meta(name, seed, pos_weight):
    if name == "DT":
        return DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                                      random_state=seed)
    if name == "LR":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=seed))
    if name == "RF":
        return RandomForestClassifier(n_estimators=300,
                                      class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed)
    if name == "XGBs":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=3,
                             learning_rate=0.1, scale_pos_weight=pos_weight,
                             tree_method="hist", random_state=seed, n_jobs=-1)
    raise ValueError(name)


def fit_predict(model, X, y, tr):
    model.fit(X[tr], y[tr])
    classes = np.asarray(model.classes_)
    col = int(np.where(classes == 1)[0][0])
    return model.predict_proba(X)[:, col].astype(np.float64)


# ==========================================================================
# prior-tracking head
# ==========================================================================
def _correct(p2, q, p_train):
    w = p2 * (q / np.clip(p_train, 1e-8, None))[None, :]
    return w / np.clip(w.sum(1, keepdims=True), 1e-12, None)


def _em_step(p2, q, p_train, alpha, beta):
    c = _correct(p2, q, p_train)
    qi = float(np.clip((c[:, 1].sum() + alpha) / (c.shape[0] + alpha + beta),
                       1e-6, 1 - 1e-6))
    return np.array([1.0 - qi, qi])


def track(prob, weeks, p_train_illicit, *, alpha, beta, est_mask, apply_mask,
          max_iter=12, tol=1e-5, blend=0.5, floor=0.005):
    """Online per-week Saerens-EM.

    ``est_mask`` chooses the rows the EM estimates the prior from;
    ``apply_mask`` chooses the rows whose posteriors get corrected.  Keeping
    them separate is what makes "estimate on the whole weekly stream, score
    the labelled subset" expressible.
    """
    p_train = np.array([1.0 - p_train_illicit, p_train_illicit])
    p_train = p_train / p_train.sum()
    q_prev = p_train.copy()
    tracked = {}
    for t in sorted(set(weeks[est_mask].tolist())):
        pe = prob[est_mask][weeks[est_mask] == t]
        if pe.size == 0:
            continue
        P = np.stack([1 - pe, pe], 1)
        q = blend * q_prev + (1 - blend) * p_train
        q = q / q.sum()
        if floor > 0 and q[1] < floor:
            q = np.array([1 - floor, floor])
        for _ in range(max_iter):
            qn = _em_step(P, q, p_train, alpha, beta)
            if np.abs(qn - q).max() < tol:
                q = qn
                break
            q = qn
        tracked[int(t)] = float(q[1])
        q_prev = q

    wa = weeks[apply_mask]
    pa = prob[apply_mask].copy()
    for t in np.unique(wa):
        q = tracked.get(int(t))
        if q is None:
            continue
        m = wa == t
        P = np.stack([1 - pa[m], pa[m]], 1)
        pa[m] = _correct(P, np.array([1 - q, q]), p_train)[:, 1]
    return pa, tracked


# ==========================================================================
# scoring -- every convention side by side, never mixed
# ==========================================================================
def score(prob_test, y_test, weeks_test, *, val_threshold, tracked,
          true_rate, boot=400, seed=0, calib=True):
    post = weeks_test >= SHUTDOWN
    f1_oracle_test, thr_test = da.oracle_f1(y_test, prob_test)
    out = {
        "n_test": int(y_test.size), "n_post": int(post.sum()),
        "n_illicit_post": int((y_test[post] == 1).sum()),
        # --- deployable: threshold fitted on the validation window ---
        "f1_deployable": da.f1_at(y_test, prob_test, val_threshold),
        "f1_post_deployable": (da.f1_at(y_test[post], prob_test[post],
                                        val_threshold)
                               if post.any() else float("nan")),
        "val_threshold": float(val_threshold),
        # --- oracle: threshold fitted on the window being scored ---
        "f1_oracle_test": f1_oracle_test,
        "oracle_threshold_test": float(thr_test),
        "f1_post_at_test_threshold": (da.f1_at(y_test[post], prob_test[post],
                                               thr_test)
                                      if post.any() else float("nan")),
        "f1_post_oracle": (da.oracle_f1(y_test[post], prob_test[post])[0]
                           if post.any() else float("nan")),
        "f1_pre_oracle": (da.oracle_f1(y_test[~post], prob_test[~post])[0]
                          if (~post).any() else float("nan")),
        "n_pre_weeks": int(len(set(weeks_test[~post].tolist()))),
        # --- ranking ---
        "pr_auc": da.pr_auc(y_test, prob_test),
        "pr_auc_post": (da.pr_auc(y_test[post], prob_test[post])
                        if post.any() else float("nan")),
        "roc_auc": da.roc_auc(y_test, prob_test),
        "recall_at_5pct_fpr": da.recall_at_fpr(y_test, prob_test, 0.05),
    }
    if tracked:
        p_per_t = {int(t): prob_test[weeks_test == t]
                   for t in np.unique(weeks_test)}
        y_per_t = {int(t): y_test[weeks_test == t]
                   for t in np.unique(weeks_test)}
        pred, yy = da.prior_matched_predictions(
            p_per_t, y_per_t, tracked, sorted(p_per_t))
        pred_p, yy_p = da.prior_matched_predictions(
            p_per_t, y_per_t, tracked,
            [t for t in sorted(p_per_t) if t >= SHUTDOWN])
        from sklearn.metrics import f1_score as _f1
        out["f1_prior_matched"] = (float(_f1(yy, pred, pos_label=1,
                                             zero_division=0))
                                   if yy.size else float("nan"))
        out["f1_post_prior_matched"] = (float(_f1(yy_p, pred_p, pos_label=1,
                                                  zero_division=0))
                                        if yy_p.size else float("nan"))
        ks = sorted(t for t in tracked if t in true_rate)
        est = np.array([tracked[t] for t in ks])
        tru = np.array([true_rate[t] for t in ks])
        out["tracked"] = {str(t): tracked[t] for t in ks}
        out["rho_n_weeks"] = len(ks)
        out["rho"] = (float(spearmanr(tru, est).correlation)
                      if len(ks) >= 3 else float("nan"))
        out["rho_p_value"] = (float(spearmanr(tru, est).pvalue)
                              if len(ks) >= 3 else float("nan"))
        post_ks = [t for t in ks if t >= SHUTDOWN]
        out["rho_post"] = (float(spearmanr(
            [true_rate[t] for t in post_ks],
            [tracked[t] for t in post_ks]).correlation)
            if len(post_ks) >= 3 else float("nan"))
        out["rate_mae"] = float(np.mean(np.abs(est - tru)))
        out["rate_mae_post"] = (float(np.mean(np.abs(
            np.array([tracked[t] for t in post_ks])
            - np.array([true_rate[t] for t in post_ks]))))
            if post_ks else float("nan"))
    if calib:
        out["calibration_test"] = da.calibration(y_test, prob_test).to_dict()
        if post.any():
            out["calibration_post"] = da.calibration(
                y_test[post], prob_test[post]).to_dict()
            out["cost_curve_post"] = da.cost_curve(y_test[post],
                                                   prob_test[post])
        out["per_week"] = da.per_week_report(
            {int(t): prob_test[weeks_test == t] for t in np.unique(weeks_test)},
            {int(t): y_test[weeks_test == t] for t in np.unique(weeks_test)},
            sorted(int(t) for t in np.unique(weeks_test)), thr_test)
    if boot:
        out["boot_pr_auc"] = da.bootstrap_ci(y_test, prob_test, da.pr_auc,
                                             n_boot=boot, seed=seed)
        if post.any():
            out["boot_f1_post_oracle"] = da.bootstrap_ci(
                y_test[post], prob_test[post],
                lambda y, p: da.oracle_f1(y, p)[0], n_boot=boot, seed=seed)
    return out


# ==========================================================================
def run_seed(seed, ctx, args) -> dict:
    t0 = time.time()
    y, weeks = ctx["y"], ctx["weeks"]
    tr, va, te = ctx["masks"]
    true_rate = ctx["true_rate"]
    res = {"seed": seed}

    pos_weight = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    res["train_pos_weight"] = pos_weight
    res["p_train_illicit"] = float((y[tr] == 1).mean())

    gnn = np.load(Path(args.gnn_dir) / f"gnn_seed_{seed}.npz")
    gnn_prob = gnn[f"w{args.stack_window}"].astype(np.float64)

    # ---- base learners on the enriched and the raw column sets ----------
    probs = {"enriched": {}, "raw166": {}}
    for tag, X in (("enriched", ctx["X_all"]), ("raw166", ctx["X_raw"])):
        for name in ("RF", "XGB", "LGBM", "LR"):
            probs[tag][name] = fit_predict(make_base(name, seed, pos_weight),
                                           X, y, tr)
            log.info("  [seed %d] %s/%s fitted", seed, tag, name)
        probs[tag]["GNN"] = gnn_prob

    yt, wt = y[te], weeks[te]
    yv = y[va]

    def val_thr(p):
        return da.oracle_f1(yv, p[va])[1]

    # ---- reference gate on the legacy split ----------------------------
    # The legacy training window (labelled, t <= 34) is byte-identical to the
    # three-way one, so the raw-166 models above *are* the legacy models --
    # only the evaluation window differs (35-49 rather than 42-49).  Refitting
    # them would burn four model fits per seed to reproduce the same weights.
    ltr, _, lte = ctx["legacy_masks"]
    assert np.array_equal(ltr, tr), "legacy and three-way train windows differ"
    legacy = {}
    for name in BASE_LEARNERS:
        p = probs["raw166"][name]
        f1, _ = da.oracle_f1(y[lte], p[lte])
        legacy[name] = {"f1_oracle": f1,
                        "pr_auc": da.pr_auc(y[lte], p[lte]),
                        "recall_at_5pct_fpr": da.recall_at_fpr(
                            y[lte], p[lte], 0.05)}
    res["legacy_gate"] = legacy

    # ---- base learners, leak-aware split -------------------------------
    res["base"] = {}
    for tag in ("enriched", "raw166"):
        res["base"][tag] = {
            name: score(probs[tag][name][te], yt, wt,
                        val_threshold=val_thr(probs[tag][name]), tracked=None,
                        true_rate=true_rate, boot=0, seed=seed,
                        calib=(tag == "enriched"))
            for name in BASE_LEARNERS}

    # ---- stacks ---------------------------------------------------------
    Z = np.column_stack([probs["enriched"][n] for n in BASE_LEARNERS])
    res["stacks"] = {}
    stack_probs = {}
    for m in META_LEARNERS:
        meta = make_meta(m, seed, pos_weight)
        meta.fit(Z[va], yv)
        classes = np.asarray(meta.classes_)
        col = int(np.where(classes == 1)[0][0])
        p = meta.predict_proba(Z)[:, col].astype(np.float64)
        stack_probs[m] = p
        res["stacks"][m] = score(p[te], yt, wt, val_threshold=val_thr(p),
                                 tracked=None, true_rate=true_rate, boot=0,
                                 seed=seed)
    log.info("  [seed %d] stacks fitted", seed)

    # ---- tracking head --------------------------------------------------
    est_labelled = te
    est_all = (weeks > VAL_MAX)
    res["tracking"] = {}
    systems = {f"stack_{m}": stack_probs[m] for m in META_LEARNERS}
    systems["rf_alone"] = probs["enriched"]["RF"]
    systems["gnn_alone"] = probs["enriched"]["GNN"]
    systems["lgbm_alone"] = probs["enriched"]["LGBM"]

    headline = {"stack_RF", "rf_alone"}
    for sysname, p in systems.items():
        for cfgname, cfg in TRACKER_CONFIGS.items():
            for popname, est in (("labelled", est_labelled),
                                 ("all_nodes", est_all)):
                primary = (cfgname == "paper_beta_5_10"
                           and popname == "labelled")
                if not primary and sysname not in headline:
                    continue
                adj, tracked = track(p, weeks, res["p_train_illicit"],
                                     est_mask=est, apply_mask=te, **cfg)
                key = (sysname if primary
                       else f"{sysname}|{cfgname}|{popname}")
                res["tracking"][key] = score(
                    adj, yt, wt, val_threshold=da.oracle_f1(
                        yv, track(p, weeks, res["p_train_illicit"],
                                  est_mask=va, apply_mask=va, **cfg)[0])[1],
                    tracked=tracked, true_rate=true_rate,
                    boot=args.boot if primary else 0, seed=seed,
                    calib=primary)
                res["tracking"][key]["tracker"] = {
                    "config": cfgname, "population": popname, **cfg}
    log.info("  [seed %d] tracking done", seed)

    # ---- error decorrelation -------------------------------------------
    err = {}
    for name in BASE_LEARNERS:
        p = probs["enriched"][name]
        thr = val_thr(p)
        err[name] = ((p[te] >= thr).astype(int) != yt).astype(int)
    E = pd.DataFrame(err)
    res["error_correlation"] = {
        "names": BASE_LEARNERS, "matrix": E.corr().to_numpy().tolist(),
        "disagreement_fraction": float((E.nunique(axis=1) > 1).mean()),
        "error_rate": {k: float(v.mean()) for k, v in err.items()}}

    # ---- window sweep ---------------------------------------------------
    res["window_sweep"] = {}
    for w in args.windows:
        key = f"w{w}"
        if key not in gnn:
            continue
        p = gnn[key].astype(np.float64)
        f1o, _ = da.oracle_f1(yt, p[te])
        res["window_sweep"][str(w)] = {
            "f1_oracle_test": f1o,
            "f1_deployable": da.f1_at(yt, p[te], da.oracle_f1(yv, p[va])[1]),
            "pr_auc": da.pr_auc(yt, p[te])}

    res["runtime_sec"] = time.time() - t0
    log.info("[seed %d] complete in %.0fs", seed, res["runtime_sec"])
    return res


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="elldata")
    ap.add_argument("--enrich-cache", default="out/enriched_cache.pkl")
    ap.add_argument("--gnn-dir", default="out/gnn")
    ap.add_argument("--out-dir", default="out/study")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--stack-window", type=int, default=1)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 3, 5, 8])
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--non-causal-wavelet", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    df, edges, raw_cols = load_frame(Path(args.data_dir))
    base_cols = ["t"] + raw_cols
    assert len(base_cols) == 166
    log.info("nodes=%d edges=%d illicit=%d licit=%d unknown=%d", len(df),
             len(edges), int((df.y == 1).sum()), int((df.y == 0).sum()),
             int((df.y == -1).sum()))
    multi = int((df.groupby("txId").t.nunique() > 1).sum())
    log.info("nodes appearing in more than one timestep: %d", multi)

    blob = pickle.load(open(args.enrich_cache, "rb"))
    topo, n2v = blob.get("topo"), blob.get("n2v")
    if topo is None and blob.get("topo_rows"):
        topo = pd.DataFrame.from_dict(blob["topo_rows"], orient="index")
        topo.index.name = "txId"
        topo = topo.reset_index()
    if n2v is None and blob.get("n2v_rows"):
        n2v = pd.DataFrame.from_dict(
            blob["n2v_rows"], orient="index",
            columns=[f"n2v{i}" for i in range(16)])
        n2v.index.name = "txId"
        n2v = n2v.reset_index()
    done = len(blob.get("weeks_topo", []))
    if done and done < LAST_T:
        log.warning("enrichment cache covers only %d/%d weeks -- nodes in the "
                    "remaining weeks get zero-filled enriched columns",
                    done, LAST_T)
    ef = enrich(df, edges, base_cols, raw_cols, topo=topo, n2v=n2v,
                causal_wavelet=not args.non_causal_wavelet)
    frame = ef.frame
    log.info("features: %d base + %d enriched (%s)", len(ef.base_cols),
             len(ef.enriched_cols),
             {k: len(v) for k, v in ef.blocks().items()})

    ctx = {
        "df": frame,
        "y": frame.y.to_numpy(),
        "weeks": frame.t.to_numpy(),
        "X_all": frame[ef.all_cols].to_numpy(dtype=np.float32),
        "X_raw": frame[ef.base_cols].to_numpy(dtype=np.float32),
        "masks": masks_for(frame, "threeway"),
        "legacy_masks": masks_for(frame, "legacy"),
    }
    lab = frame.y.isin([0, 1]).to_numpy()
    ctx["true_rate"] = {
        int(w): float((ctx["y"][(ctx["weeks"] == w) & lab] == 1).mean())
        for w in range(1, LAST_T + 1)
        if ((ctx["weeks"] == w) & lab).any()}

    meta = {"n_nodes": len(frame), "n_edges": len(edges),
            "n_illicit": int((frame.y == 1).sum()),
            "n_licit": int((frame.y == 0).sum()),
            "multi_timestep_nodes": multi,
            "feature_blocks": {k: len(v) for k, v in ef.blocks().items()},
            "n_features_total": len(ef.all_cols),
            "causal_wavelet": not args.non_causal_wavelet,
            "true_rate": ctx["true_rate"],
            "split": {"train": [1, TRAIN_MAX], "val": [TRAIN_MAX + 1, VAL_MAX],
                      "test": [VAL_MAX + 1, LAST_T], "shutdown": SHUTDOWN}}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)

    for seed in args.seeds:
        path = out_dir / f"seed_{seed}.json"
        if path.exists():
            log.info("seed %d done, skipping", seed)
            continue
        np.random.seed(seed)
        json.dump(run_seed(seed, ctx, args), open(path, "w"))
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
