from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fmt_pct(x: float) -> str:
    if x != x:
        return "  --  "
    return f"{x * 100:6.2f}"


def _bar(p: float, width: int = 12) -> str:
    p = max(0.0, min(1.0, p))
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def trajectory_table(metrics: dict) -> None:
    true_prior = metrics.get("true_prior", {})
    if not true_prior:
        print("(no true_prior in metrics.json — re-run the pipeline)")
        return

    test_ts = sorted(int(t) for t in true_prior.keys())
    print(f"\n[1] q-trajectory across t = {test_ts[0]}..{test_ts[-1]} "
          f"(post-shutdown marker at t=43)")
    print()
    print(f"  {'t':>3}  {'true %':>7}  {'gru raw':>8}  {'gru C2':>7}  "
          f"{'gru C3':>7}  {'rf raw':>7}  {'rf C3':>7}  | bar (true vs C3)")
    print("  " + "-" * 86)

    def _q(cond_key: str, t: int) -> float:
        cond = metrics.get(cond_key, {})
        qmap = cond.get("estimated_q_illicit") or {}
        v = qmap.get(str(t))
        return float(v) if v is not None and v == v else float("nan")

    for t in test_ts:
        true_p = float(true_prior[str(t)])
        gru_raw = _q("gcn_gru_none", t)
        gru_c2 = _q("gcn_gru_batch", t)
        gru_c3 = _q("gcn_gru_online", t)
        rf_raw = _q("rf_none", t)
        rf_c3 = _q("rf_online", t)
        marker = "*" if t >= 43 else " "
        bar_t = _bar(true_p, 10)
        bar_c3 = _bar(gru_c3, 10)
        print(f"  {t:>3}{marker} {fmt_pct(true_p):>7}  {fmt_pct(gru_raw):>8}  "
              f"{fmt_pct(gru_c2):>7}  {fmt_pct(gru_c3):>7}  "
              f"{fmt_pct(rf_raw):>7}  {fmt_pct(rf_c3):>7}  | {bar_t} {bar_c3}")
    print("\n  (`*` post-shutdown.  C3 should follow true closely if the tracker works.)")


def post_shutdown_stats(metrics: dict) -> None:
    print("\n[2] Post-shutdown (t>=43) evaluation, all conditions")
    print()
    print(f"  {'condition':<25s}  {'PR-AUC':>7s}  {'F1':>6s}  "
          f"{'post-F1':>8s}  {'ρ post':>7s}  {'ρ full':>7s}")
    print("  " + "-" * 80)
    for cond in ("gcn_only", "gcn_gru_none", "gcn_gru_batch",
                 "gcn_gru_online", "rf_none", "rf_batch", "rf_online"):
        m = metrics.get(cond, {})
        if not m:
            continue
        pr = m.get("pr_auc", float("nan"))
        f1 = m.get("f1_illicit", float("nan"))
        f1p = m.get("f1_post_shutdown", float("nan"))
        rho = m.get("spearman_rho_prior", float("nan"))
        rho_full = m.get("spearman_rho_prior_full", float("nan"))
        for v in (rho, rho_full):
            pass

        def f(x, n=3):
            if x is None:
                return "  --"
            try:
                xf = float(x)
                if xf != xf:
                    return "  --"
                return f"{xf:.{n}f}"
            except (TypeError, ValueError):
                return "  --"

        print(f"  {cond:<25s}  {f(pr,4):>7s}  {f(f1):>6s}  "
              f"{f(f1p):>8s}  {f(rho):>7s}  {f(rho_full):>7s}")


def verdict(metrics: dict) -> None:
    def g(path, default=float("nan")):
        cur = metrics
        for k in path.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        try:
            return float(cur)
        except (TypeError, ValueError):
            return default

    c1_f1 = g("gcn_gru_none.f1_illicit")
    c2_post = g("gcn_gru_batch.f1_post_shutdown")
    c3_post = g("gcn_gru_online.f1_post_shutdown")
    rho_post = g("gcn_gru_online.spearman_rho_prior")
    rf_plus_rho = g("rf_online.spearman_rho_prior")
    rf_f1 = g("rf_none.f1_illicit")

    checks = [
        ("(a) C1 F1 ~ 0.69", abs(c1_f1 - 0.69) <= 0.05, f"F1 = {c1_f1:.3f}"),
        ("(b) C2 post-F1 in [.08,.20]",
            0.08 <= c2_post <= 0.20, f"C2 post-F1 = {c2_post:.3f}"),
        ("(c) C3 post-F1 >= 0.18", c3_post >= 0.18,
            f"C3 post-F1 = {c3_post:.3f}"),
        ("(d) rho_post >= 0.7", rho_post >= 0.7,
            f"rho_post = {rho_post:.3f}"),
        ("(e) RF+ rho > 0", rf_plus_rho > 0.0,
            f"RF+ rho = {rf_plus_rho:.3f}"),
        ("ref: RF F1 ~ 0.82", abs(rf_f1 - 0.82) <= 0.03,
            f"RF F1 = {rf_f1:.3f}"),
    ]
    n_pass = sum(1 for _, p, _ in checks if p)
    print(f"\n[3] Verdict: {n_pass}/{len(checks)} PASS")
    for name, ok, det in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:<35s}  {det}")


def history_summary(history_path: Path, top: int) -> None:
    if not history_path.exists():
        print(f"(no history at {history_path})")
        return
    rows = []
    with open(history_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = r.get("metrics", {})

            def s(k, default=float("nan")):
                v = m.get(k)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default

            rows.append({
                "ts": r.get("ts"),
                "note": r.get("note", ""),
                "params": r.get("params", {}),
                "C1 F1": s("gcn_gru_none.f1_illicit"),
                "C2 postF1": s("gcn_gru_batch.f1_post_shutdown"),
                "C3 postF1": s("gcn_gru_online.f1_post_shutdown"),
                "C3 rho": s("gcn_gru_online.spearman_rho_prior"),
                "C3 rho_full": s("gcn_gru_online.spearman_rho_prior_full"),
                "RF F1": s("rf_none.f1_illicit"),
                "RF+ rho": s("rf_online.spearman_rho_prior"),
            })

    if not rows:
        print("(history file is empty)")
        return

    def passes(r):
        n = 0
        if abs(r["C1 F1"] - 0.69) <= 0.05:
            n += 1
        if 0.08 <= r["C2 postF1"] <= 0.20:
            n += 1
        if r["C3 postF1"] >= 0.18:
            n += 1
        if r["C3 rho"] >= 0.7:
            n += 1
        if r["RF+ rho"] > 0:
            n += 1
        if abs(r["RF F1"] - 0.82) <= 0.03:
            n += 1
        return n

    for r in rows:
        r["pass"] = passes(r)

    print(f"\n[4] History summary ({len(rows)} runs)")
    print()
    rows.sort(key=lambda r: (r["pass"], r["C3 rho"], r["C3 postF1"]),
              reverse=True)
    show = rows[:top] if top else rows
    print(f"  {'pass':>4s} {'C1 F1':>6s} {'C2 pF1':>6s} {'C3 pF1':>6s} "
          f"{'ρpost':>5s} {'ρfull':>5s} {'RF F1':>6s} {'RF+ρ':>5s}  note")
    print("  " + "-" * 78)
    for r in show:
        p = r["params"]
        note = (r["note"] or "")[:38]
        print(f"  {r['pass']}/6  {r['C1 F1']:.3f}  {r['C2 postF1']:.3f}  "
              f"{r['C3 postF1']:.3f}  {r['C3 rho']:.2f}  "
              f"{r['C3 rho_full']:.2f}  {r['RF F1']:.3f}  "
              f"{r['RF+ rho']:.2f}  {note}  "
              f"α={p.get('alpha')} β={p.get('beta')} em={p.get('em_iter')} "
              f"seed={p.get('seed')}")
    if top and len(rows) > top:
        print(f"  ... ({len(rows) - top} more)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default="artefacts/metrics.json")
    ap.add_argument("--history", action="store_true",
                    help="Also summarise artefacts/run_history.jsonl")
    ap.add_argument("--best", type=int, default=10,
                    help="When --history, show this many top rows.")
    args = ap.parse_args()

    metrics_path = ROOT / args.metrics if not Path(args.metrics).is_absolute()\
        else Path(args.metrics)
    if not metrics_path.exists():
        print(f"No metrics at {metrics_path}.")
        sys.exit(1)
    with open(metrics_path) as fh:
        metrics = json.load(fh)

    print(f"# Diagnosing {metrics_path}")
    meta = metrics.get("_meta", {})
    if meta:
        print(f"# params: {meta}")
    trajectory_table(metrics)
    post_shutdown_stats(metrics)
    verdict(metrics)

    if args.history:
        history_summary(ROOT / "artefacts" / "run_history.jsonl", args.best)


if __name__ == "__main__":
    main()
