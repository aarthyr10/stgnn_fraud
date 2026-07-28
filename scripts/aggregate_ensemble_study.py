"""Aggregate the corrected study and reconcile it three ways: against the
paper's published numbers, against the original notebook's own outputs, and
against itself across seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.deep_analysis import paired_seed_test, seed_summary  # noqa: E402

META = ["DT", "LR", "RF", "XGBs"]
META_KEY = {"DT": "DT", "LR": "LR", "RF": "RF", "XGBs": "XGBs"}
BASE = ["RF", "XGB", "LGBM", "LR", "GNN"]

# key -> (label, paper value, original-notebook value)
TARGETS: dict[str, tuple[str, float | None, float | None]] = {
    "gate.rf_f1": ("RF F1, legacy split, oracle threshold", 0.830, 0.83039),
    "gate.rf_pr_auc": ("RF PR-AUC, legacy split", 0.793, 0.79298),
    "gate.rf_recall5": ("RF recall @ 5% FPR, legacy", 0.746, None),
    "gate.xgb_f1": ("XGBoost F1, legacy split", 0.828, 0.82751),
    "gate.lgbm_f1": ("LightGBM F1, legacy split", 0.827, 0.82724),
    "la.rf_f1": ("RF F1, leak-aware three-way split", 0.634, 0.63366),
    "la.rf_pr_auc": ("RF PR-AUC, three-way split", 0.541, 0.54134),
    "la.lgbm_pr_auc": ("LightGBM PR-AUC, three-way (best base)", 0.567,
                       0.56721),
    "meta.DT.pr_auc": ("Meta PR-AUC — decision tree", 0.522, 0.52201),
    "meta.LR.pr_auc": ("Meta PR-AUC — logistic regression", 0.562, 0.56236),
    "meta.RF.pr_auc": ("Meta PR-AUC — random forest", 0.590, 0.58990),
    "meta.XGBs.pr_auc": ("Meta PR-AUC — XGBoost", 0.608, 0.60797),
    "meta.RF.f1_deploy": ("RF-stacker F1, validation-fitted threshold", 0.629,
                          0.62942),
    "meta.DT.f1_deploy": ("DT-stacker F1, validation-fitted threshold", None,
                          0.62449),
    "meta.LR.f1_deploy": ("LR-stacker F1, validation-fitted threshold", None,
                          0.62429),
    "meta.XGBs.f1_deploy": ("XGB-stacker F1, validation-fitted threshold",
                            None, 0.60515),
    "meta.RF.f1_post_wholewin": (
        "RF-stacker post-43 F1 @ whole-window threshold", None, 0.03352),
    "meta.XGBs.f1_post_wholewin": (
        "XGB-stacker post-43 F1 @ whole-window threshold", None, 0.03390),
    "track.stack_RF.f1_post": ("Stack (RF meta) + tracking, post-43 F1",
                               0.149, 0.14889),
    "track.stack_RF.rho": ("Stack (RF meta) + tracking, tracking rho", 0.274,
                           0.27381),
    "track.stack_XGBs.f1_post": ("Stack (XGB meta) + tracking, post-43 F1",
                                 0.135, 0.13451),
    "track.stack_XGBs.rho": ("Stack (XGB meta) + tracking, tracking rho",
                             0.214, 0.21429),
    "track.rf_alone.f1_post": ("RF alone + tracking, post-43 F1", 0.064,
                               0.06436),
    "track.rf_alone.rho": ("RF alone + tracking, tracking rho", 0.333,
                           0.33333),
    "corr.tree_tree": ("Error correlation, tree-tree", 0.965, None),
    "corr.lr_tree": ("Error correlation, linear-tree", 0.525, None),
    "corr.gnn_other": ("Error correlation, GNN-other", 0.485, None),
    "corr.disagreement": ("Base-learner error disagreement fraction", 0.086,
                          0.08608),
    "win.1": ("Window sweep F1, W = 1", 0.272, 0.27169),
    "win.3": ("Window sweep F1, W = 3", 0.273, 0.27258),
    "win.5": ("Window sweep F1, W = 5", 0.251, 0.25081),
    "win.8": ("Window sweep F1, W = 8", 0.247, 0.24708),
}


def collect(seeds: list[dict]) -> dict[str, list[float]]:
    v: dict[str, list[float]] = {}

    def push(k, x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return
        v.setdefault(k, []).append(x)

    for r in seeds:
        g = r["legacy_gate"]
        push("gate.rf_f1", g["RF"]["f1_oracle"])
        push("gate.rf_pr_auc", g["RF"]["pr_auc"])
        push("gate.rf_recall5", g["RF"]["recall_at_5pct_fpr"])
        push("gate.xgb_f1", g["XGB"]["f1_oracle"])
        push("gate.lgbm_f1", g["LGBM"]["f1_oracle"])
        push("gate.gnn_f1", g["GNN"]["f1_oracle"])

        be, br = r["base"]["enriched"], r["base"]["raw166"]
        push("la.rf_f1", br["RF"]["f1_oracle_test"])
        push("la.rf_pr_auc", br["RF"]["pr_auc"])
        push("la.lgbm_pr_auc", br["LGBM"]["pr_auc"])
        for n in BASE:
            push(f"base.enriched.{n}.pr_auc", be[n]["pr_auc"])
            push(f"base.raw166.{n}.pr_auc", br[n]["pr_auc"])
            push(f"base.enriched.{n}.f1", be[n]["f1_oracle_test"])
            push(f"base.raw166.{n}.f1", br[n]["f1_oracle_test"])
            push(f"lift.{n}.f1", be[n]["f1_oracle_test"]
                 - br[n]["f1_oracle_test"])

        for m in META:
            s = r["stacks"][m]
            push(f"meta.{m}.pr_auc", s["pr_auc"])
            push(f"meta.{m}.f1_deploy", s["f1_deployable"])
            push(f"meta.{m}.f1_oracle", s["f1_oracle_test"])
            push(f"meta.{m}.f1_post_wholewin", s["f1_post_at_test_threshold"])
            push(f"meta.{m}.f1_post_subset", s["f1_post_oracle"])

        for key, t in r["tracking"].items():
            tag = key.replace("|", ".")
            push(f"track.{tag}.f1_post", t["f1_post_oracle"])
            push(f"track.{tag}.f1_post_wholewin",
                 t["f1_post_at_test_threshold"])
            push(f"track.{tag}.f1_post_prior_matched",
                 t.get("f1_post_prior_matched"))
            push(f"track.{tag}.f1_post_deployable", t["f1_post_deployable"])
            push(f"track.{tag}.rho", t.get("rho"))
            push(f"track.{tag}.rho_p", t.get("rho_p_value"))
            push(f"track.{tag}.rate_mae", t.get("rate_mae"))
            push(f"track.{tag}.rate_mae_post", t.get("rate_mae_post"))
            push(f"track.{tag}.pr_auc", t["pr_auc"])

        C = np.array(r["error_correlation"]["matrix"])
        names = r["error_correlation"]["names"]
        i = {n: k for k, n in enumerate(names)}
        trees = ["RF", "XGB", "LGBM"]
        push("corr.tree_tree", np.mean([C[i[a], i[b]]
                                        for k, a in enumerate(trees)
                                        for b in trees[k + 1:]]))
        push("corr.lr_tree", np.mean([C[i["LR"], i[t]] for t in trees]))
        push("corr.gnn_other", np.mean([C[i["GNN"], i[n]] for n in names
                                        if n != "GNN"]))
        push("corr.min_lr_gnn", C[i["LR"], i["GNN"]])
        push("corr.disagreement",
             r["error_correlation"]["disagreement_fraction"])

        for w, d in r.get("window_sweep", {}).items():
            push(f"win.{w}", d["f1_oracle_test"])
            push(f"win.{w}.deployable", d["f1_deployable"])
    return v


def verdict(obs: dict, ref: float | None) -> str:
    m, lo, hi = obs["median"], obs["lo"], obs["hi"]
    if ref is None or not np.isfinite(m):
        return "n/a"
    if lo <= ref <= hi:
        return "consistent"
    rel = abs(m - ref) / max(abs(ref), 1e-9)
    if rel <= 0.05:
        return "close"
    if rel <= 0.20:
        return "shifted"
    return "diverges"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", default="out/study")
    ap.add_argument("--out", default="out/aggregate.json")
    args = ap.parse_args()

    d = Path(args.seed_dir)
    files = sorted(d.glob("seed_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    seeds = [json.load(open(f)) for f in files]
    meta = json.load(open(d / "meta.json")) if (d / "meta.json").exists() else {}
    print(f"loaded {len(seeds)} seeds")

    raw = collect(seeds)
    summ = {k: seed_summary(v) for k, v in raw.items()}

    rows = []
    for key, (label, paper, orig) in TARGETS.items():
        o = summ.get(key, seed_summary([]))
        rows.append({"key": key, "label": label, "paper": paper,
                     "original": orig, "median": o["median"],
                     "lo": o["lo"], "hi": o["hi"], "min": o["min"],
                     "max": o["max"], "n_seeds": o["n_seeds"],
                     "vs_paper": verdict(o, paper),
                     "vs_original": verdict(o, orig)})

    paired = {
        "stackRF_vs_rfalone__post_f1": paired_seed_test(
            raw.get("track.stack_RF.f1_post", []),
            raw.get("track.rf_alone.f1_post", [])),
        "stackRF_vs_rfalone__rho": paired_seed_test(
            raw.get("track.stack_RF.rho", []),
            raw.get("track.rf_alone.rho", [])),
        "stackRF_vs_rfalone__rate_mae": paired_seed_test(
            raw.get("track.rf_alone.rate_mae", []),
            raw.get("track.stack_RF.rate_mae", [])),
        "stackXGBs_vs_rfalone__post_f1": paired_seed_test(
            raw.get("track.stack_XGBs.f1_post", []),
            raw.get("track.rf_alone.f1_post", [])),
        "enriched_vs_raw__RF_f1": paired_seed_test(
            raw.get("base.enriched.RF.f1", []),
            raw.get("base.raw166.RF.f1", [])),
        "enriched_vs_raw__XGB_f1": paired_seed_test(
            raw.get("base.enriched.XGB.f1", []),
            raw.get("base.raw166.XGB.f1", [])),
        "enriched_vs_raw__LGBM_f1": paired_seed_test(
            raw.get("base.enriched.LGBM.f1", []),
            raw.get("base.raw166.LGBM.f1", [])),
    }

    # tracker-config sensitivity (headline systems only)
    sens = {}
    for sysname in ("stack_RF", "rf_alone"):
        for cfg in ("paper_beta_5_10", "code_beta_0.2_1.8"):
            for pop in ("labelled", "all_nodes"):
                k = (f"track.{sysname}" if (cfg == "paper_beta_5_10"
                                            and pop == "labelled")
                     else f"track.{sysname}.{cfg}.{pop}")
                sens[f"{sysname}|{cfg}|{pop}"] = {
                    "f1_post": summ.get(f"{k}.f1_post", {}).get("median"),
                    "rho": summ.get(f"{k}.rho", {}).get("median"),
                    "rate_mae": summ.get(f"{k}.rate_mae", {}).get("median"),
                }

    payload = {"n_seeds": len(seeds), "meta": meta, "reconciliation": rows,
               "summary": summ, "raw": raw, "paired_seed_tests": paired,
               "tracker_sensitivity": sens}
    payload.update(render_payload(seeds))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(args.out, "w"))

    w = max(len(r["label"]) for r in rows)
    print(f"\n{'quantity':{w}s} {'paper':>8s} {'orig':>8s} {'ours':>8s} "
          f"{'95% CI':>18s}  {'vs paper':<11s} vs original")
    print("-" * (w + 70))
    for r in rows:
        ci = f"[{r['lo']:.3f}, {r['hi']:.3f}]"
        p = f"{r['paper']:8.3f}" if r["paper"] is not None else "       —"
        o = f"{r['original']:8.3f}" if r["original"] is not None else "       —"
        print(f"{r['label']:{w}s} {p} {o} {r['median']:8.3f} {ci:>18s}  "
              f"{r['vs_paper']:<11s} {r['vs_original']}")
    print("\npaired across-seed tests:")
    for k, s in paired.items():
        print(f"  {k:34s} median diff {s['median_diff']:+.4f}  "
              f"wins {s['wins']}/{s['n']}  sign p={s['sign_p']:.4f}")
    print("\ntracker sensitivity (post-43 F1 / rho / rate MAE):")
    for k, s in sens.items():
        def _f(x):
            return "—" if x is None or x != x else f"{x:.4f}"
        print(f"  {k:44s} {_f(s['f1_post'])}  {_f(s['rho'])}  "
              f"{_f(s['rate_mae'])}")
    print(f"\nwrote {args.out}")


def _median_series(dicts):
    keys = sorted({k for d in dicts for k in d})
    out = {}
    for k in keys:
        v = [d[k] for d in dicts if k in d and np.isfinite(d[k])]
        if v:
            out[k] = float(np.median(v))
    return out


def _median_cal(cals):
    cals = [c for c in cals if c]
    if not cals:
        return {}
    n = len(cals[0].get("bin_conf", []))
    conf, acc = [], []
    for b in range(n):
        c = [x["bin_conf"][b] for x in cals if np.isfinite(x["bin_conf"][b])]
        a = [x["bin_acc"][b] for x in cals if np.isfinite(x["bin_acc"][b])]
        conf.append(float(np.median(c)) if c else float("nan"))
        acc.append(float(np.median(a)) if a else float("nan"))
    return {"bin_conf": conf, "bin_acc": acc,
            "ece": float(np.median([x["ece"] for x in cals])),
            "brier": float(np.median([x["brier"] for x in cals]))}


def render_payload(seeds: list[dict]) -> dict:
    if not seeds:
        return {}
    out = {}
    out["trajectories"] = {
        "stack_RF": _median_series([s["tracking"]["stack_RF"]["tracked"]
                                    for s in seeds]),
        "rf_alone": _median_series([s["tracking"]["rf_alone"]["tracked"]
                                    for s in seeds]),
        "gnn_alone": _median_series([s["tracking"]["gnn_alone"]["tracked"]
                                     for s in seeds]),
    }

    def wk(rows):
        return {int(x["t"]): x["f1"] for x in rows}

    series = {
        "rf": [wk(s["base"]["enriched"]["RF"]["per_week"]) for s in seeds],
        "gnn": [wk(s["base"]["enriched"]["GNN"]["per_week"]) for s in seeds],
        "stack": [wk(s["stacks"]["RF"]["per_week"]) for s in seeds],
        "stack_tracking": [wk(s["tracking"]["stack_RF"]["per_week"])
                           for s in seeds],
    }
    weeks = sorted({t for lst in series.values() for d in lst for t in d})
    out["per_week_median"] = [
        {"t": t, **{k: (float(np.median([d[t] for d in lst if t in d]))
                        if any(t in d for d in lst) else None)
                    for k, lst in series.items()}} for t in weeks]

    out["calibration"] = {
        "rf": _median_cal([s["base"]["enriched"]["RF"].get("calibration_post")
                           for s in seeds]),
        "gnn": _median_cal([s["base"]["enriched"]["GNN"].get(
            "calibration_post") for s in seeds]),
        "stack_tracking": _median_cal([s["tracking"]["stack_RF"].get(
            "calibration_post") for s in seeds]),
    }
    names = seeds[0]["error_correlation"]["names"]
    mats = np.array([s["error_correlation"]["matrix"] for s in seeds])
    out["error_correlation"] = {
        "names": names, "matrix": np.nanmedian(mats, axis=0).tolist(),
        "disagreement_fraction": float(np.median(
            [s["error_correlation"]["disagreement_fraction"]
             for s in seeds])),
        "error_rate": {k: float(np.median(
            [s["error_correlation"]["error_rate"][k] for s in seeds]))
            for k in names}}
    cc = {}
    for ratio in ("1", "5", "10", "25", "50", "100"):
        vals = [s["tracking"]["stack_RF"].get("cost_curve_post", {}).get(ratio)
                for s in seeds]
        vals = [v for v in vals if v]
        if vals:
            cc[ratio] = {"best_threshold": float(np.median(
                [v["best_threshold"] for v in vals])),
                "best_cost": float(np.median([v["best_cost"] for v in vals])),
                "cost_at_0p5": float(np.median(
                    [v["cost_at_0p5"] for v in vals]))}
    out["cost_curves"] = cc
    return out


if __name__ == "__main__":
    main()
