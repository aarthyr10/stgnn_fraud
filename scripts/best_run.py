from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent

CRITERIA = ("rho_full", "rho_post", "C1_F1", "C3_postF1", "composite")


def _load() -> list[dict]:
    p = ROOT / "artefacts" / "run_history.jsonl"
    out = []
    if not p.exists():
        return out
    with open(p) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = r.get("metrics", {})

            def g(k, default=float("nan")):
                v = m.get(k)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default

            out.append({
                "ts": r["ts"], "note": r["note"], "params": r["params"],
                "C1_F1":     g("gcn_gru_none.f1_illicit"),
                "C2_postF1": g("gcn_gru_batch.f1_post_shutdown"),
                "C3_postF1": g("gcn_gru_online.f1_post_shutdown"),
                "rho_post":  g("gcn_gru_online.spearman_rho_prior"),
                "rho_full":  g("gcn_gru_online.spearman_rho_prior_full"),
                "RF_F1":     g("rf_none.f1_illicit"),
                "RFplus_rho": g("rf_online.spearman_rho_prior"),
                "PR_AUC":    g("gcn_gru_online.pr_auc"),
            })
    return out


def composite(r) -> float:
    c1 = max(0.0, min(1.0, r["C1_F1"] / 0.69))
    c2 = max(0.0, min(1.0, r["C2_postF1"] / 0.08))
    c3 = max(0.0, min(1.0, r["C3_postF1"] / 0.18))
    rf = max(0.0, min(1.0, r["RF_F1"] / 0.82))
    rho = (r["rho_full"] + r["rho_post"]) / 2.0
    rho_n = max(0.0, min(1.0, (rho + 1.0) / 2.0))
    return c1 + c2 + c3 + rf + rho_n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--by", choices=CRITERIA, default="rho_full")
    ap.add_argument("--top", type=int, default=1)
    ap.add_argument("--filter", default="",
                    help="Substring match on 'note' (e.g. 'seed:' to "
                         "restrict to seed-sweep rows).")
    args = ap.parse_args()

    rows = _load()
    if args.filter:
        rows = [r for r in rows if args.filter in r["note"]]
    if not rows:
        print("(no rows match)")
        return

    if args.by == "composite":
        rows.sort(key=composite, reverse=True)
    else:
        rows.sort(key=lambda r: r[args.by], reverse=True)

    print(f"# Best {args.top} run(s) by {args.by}  "
          f"(filter={args.filter or '<none>'},  pool={len(rows)})")
    print()
    for i, r in enumerate(rows[:args.top], 1):
        p = r["params"]
        ts = r["ts"]
        print(f"## #{i}  {r['note']}")
        print(f"    timestamp: {ts}")
        print(f"    params:    alpha={p.get('alpha')}  beta={p.get('beta')} "
              f"em_iter={p.get('em_iter')}  seed={p.get('seed')}")
        print(f"    C1 F1     = {r['C1_F1']:.3f}    (target 0.69)")
        print(f"    C2 postF1 = {r['C2_postF1']:.3f}    (target 0.08-0.20)")
        print(f"    C3 postF1 = {r['C3_postF1']:.3f}    (target >= 0.18)")
        print(f"    rho_post  = {r['rho_post']:+.3f}   (target >= 0.7)")
        print(f"    rho_full  = {r['rho_full']:+.3f}   (target >= 0.7 on 15-pt window)")
        print(f"    PR-AUC    = {r['PR_AUC']:.3f}")
        print(f"    RF F1     = {r['RF_F1']:.3f}    (target 0.82)")
        print(f"    RF+ rho   = {r['RFplus_rho']:+.3f}   (target > 0)")
        passes = (
            (abs(r["C1_F1"] - 0.69) <= 0.05)
            + (0.08 <= r["C2_postF1"] <= 0.20)
            + (r["C3_postF1"] >= 0.18)
            + (r["rho_post"] >= 0.7)
            + (r["RFplus_rho"] > 0)
            + (abs(r["RF_F1"] - 0.82) <= 0.03)
        )
        print(f"    PASS: {passes}/6  (composite score {composite(r):.3f})")
        print()

    seed_rows = [r for r in rows if r["note"].startswith("seed:")]
    if seed_rows:
        print(f"# Median across {len(seed_rows)} seed-sweep rows "
              f"(proposal §3 bootstrap reporting):")
        for k in ("C1_F1", "C2_postF1", "C3_postF1",
                  "rho_post", "rho_full", "RF_F1"):
            vals = sorted(r[k] for r in seed_rows)
            print(f"    {k:10s} median={median(vals):+.3f}  "
                  f"min={min(vals):+.3f}  max={max(vals):+.3f}")


if __name__ == "__main__":
    main()
