from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app.services.history import history_rows, load_history
from app.utils.theme import section_close, section_open

C1_F1_THRESHOLD = 0.60
C2_POST_F1_LOW = 0.08
C2_POST_F1_HIGH = 0.20
C3_POST_F1_THRESHOLD = 0.18
RHO_THRESHOLD = 0.7
RF_F1_THRESHOLD = 0.78

VERDICT_COLUMNS = [
    ("C1 F1 ~ 0.69", "Proposal (a): uncorrected aggregate F1(illicit) approx 0.69"),
    ("C2 post-F1 in [.08,.20]", "Proposal (b): batch EM partial recovery"),
    ("C3 post-F1 >= .18", "Proposal (c): online beats batch on post-shutdown F1"),
    ("rho_post >= 0.7", "Proposal (d): C3 rho on t>=43 >= 0.7"),
    ("RF+ rho > 0", "Proposal (e): tracker architecture-agnostic"),
    ("RF F1 ~ 0.82", "Maganti reference: RF F1(illicit) approx 0.82"),
]


def _fmt_ts(ts: int) -> str:
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"


def _as_float(value) -> float:
    try:
        f = float(value)
        if f != f:
            return float("nan")
        return f
    except (TypeError, ValueError):
        return float("nan")


def _fmt_num(value, digits: int = 3) -> str:
    f = _as_float(value)
    if f != f:
        return "—"
    return f"{f:.{digits}f}"


def _row_verdicts(row: dict) -> list[tuple[bool, str]]:
    c1_f1 = _as_float(row.get("C1 F1"))
    c2_post_f1 = _as_float(row.get("C2 F1@t>=43"))
    c3_post_f1 = _as_float(row.get("C3 F1@t>=43"))
    rho = _as_float(row.get("C3 rho"))
    rf_rho = _as_float(row.get("RF+ rho"))
    rf_f1 = _as_float(row.get("RF F1"))

    c1_pass = c1_f1 == c1_f1 and c1_f1 >= C1_F1_THRESHOLD
    c2_pass = (
        c2_post_f1 == c2_post_f1
        and C2_POST_F1_LOW <= c2_post_f1 <= C2_POST_F1_HIGH
    )
    c3_pass = c3_post_f1 == c3_post_f1 and c3_post_f1 >= C3_POST_F1_THRESHOLD
    rho_pass = rho == rho and rho >= RHO_THRESHOLD
    rf_rho_pass = rf_rho == rf_rho and rf_rho > 0
    rf_f1_pass = rf_f1 == rf_f1 and rf_f1 >= RF_F1_THRESHOLD

    return [
        (c1_pass, f"F1 = {_fmt_num(c1_f1)}"),
        (c2_pass, f"C2 post-F1 = {_fmt_num(c2_post_f1)}"),
        (c3_pass, f"C3 post-F1 = {_fmt_num(c3_post_f1)}"),
        (rho_pass, f"rho_post = {_fmt_num(rho, 2)}"),
        (rf_rho_pass, f"RF+ rho = {_fmt_num(rf_rho, 2)}"),
        (rf_f1_pass, f"RF F1 = {_fmt_num(rf_f1)}"),
    ]


def _badge_html(passed: bool, observed: str) -> str:
    label = "PASS" if passed else "FAIL"
    colour = "#0F6E56" if passed else "#993556"
    return (
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;'
        f'gap:0.35rem 0.5rem;font-size:0.78rem;line-height:1.25">'
        f'<span style="flex:0 0 auto;display:inline-block;'
        f'padding:0.12rem 0.55rem;border-radius:4px;background:{colour};'
        f'color:#FFFFFF;font-size:0.7rem;font-weight:700;'
        f'letter-spacing:0.05em;text-align:center;white-space:nowrap;'
        f'font-family:&quot;Anthropic Sans&quot;, -apple-system, '
        f'sans-serif">{label}</span>'
        f'<code style="font-size:0.72rem;color:#3C3A33;'
        f'word-break:break-word;white-space:normal">{observed}</code>'
        f'</div>'
    )


def _verdict_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    meta_cols = [c for c in ("Time", "Note", "alpha", "beta") if c in df.columns]
    head_cells = []
    for col in meta_cols:
        head_cells.append(
            f'<th style="text-align:left;padding:0.5rem 0.7rem;'
            f'background:#F6F5F1;border-bottom:1px solid #ECEBE3;'
            f'font-weight:600;font-size:0.78rem">{col}</th>'
        )
    for short, full in VERDICT_COLUMNS:
        head_cells.append(
            f'<th style="text-align:left;padding:0.5rem 0.7rem;'
            f'background:#F6F5F1;border-bottom:1px solid #ECEBE3;'
            f'font-weight:600;font-size:0.78rem" title="{full}">'
            f'{short}</th>'
        )

    body_rows: list[str] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in meta_cols:
            value = row[col]
            if col in ("alpha", "beta"):
                value = _fmt_num(value, 2)
            elif pd.isna(value):
                value = "—"
            cells.append(
                f'<td style="padding:0.45rem 0.7rem;border-bottom:1px '
                f'solid #ECEBE3;font-size:0.82rem;vertical-align:top;'
                f'word-break:break-word;white-space:normal">'
                f'{value}</td>'
            )
        verdicts = _row_verdicts(row.to_dict())
        for passed, observed in verdicts:
            cells.append(
                f'<td style="padding:0.45rem 0.7rem;border-bottom:1px '
                f'solid #ECEBE3;vertical-align:top;'
                f'word-break:break-word;white-space:normal">'
                f'{_badge_html(passed, observed)}</td>'
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<div style="border:1px solid #ECEBE3;border-radius:10px;'
        'overflow:hidden;margin:0.4rem 0 0.8rem 0">'
        '<table style="width:100%;border-collapse:collapse;'
        'table-layout:fixed;'
        'font-family:&quot;Anthropic Sans&quot;, -apple-system, '
        'sans-serif">'
        f"<thead><tr>{''.join(head_cells)}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _df_from_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["Time"] = df["ts"].apply(_fmt_ts)
    df = df.rename(columns={
        "alpha": "alpha", "beta": "beta",
        "em_iter": "EM iter", "seed": "seed", "note": "Note",
    })
    ordered = [
        "Time", "Note", "alpha", "beta", "EM iter", "seed",
        "C1 F1", "C2 F1@t>=43", "C3 F1@t>=43", "RF F1", "RF+ F1@t>=43",
        "C1 PR-AUC", "C2 PR-AUC", "C3 PR-AUC",
        "RF PR-AUC", "RF+ PR-AUC",
        "C3 rho", "RF+ rho",
    ]
    return df[[c for c in ordered if c in df.columns]]


def render_history_tab(artefact_paths: dict) -> None:
    st.markdown('<h2 style="margin-top:0">History</h2>',
                unsafe_allow_html=True)

    history_path = artefact_paths.get("history")
    if not history_path:
        st.warning("History path is not configured.")
        return

    records = load_history(history_path)
    if not records:
        st.info("No runs yet. Use the Pipeline tab.")
        return

    rows = history_rows(records)
    df = _df_from_rows(rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total runs", len(df))
    if not df.empty:
        c2.metric("Latest run", df.iloc[-1]["Time"])
        best_c3 = df["C3 PR-AUC"].astype(float).max()
        c3.metric("Best C3 PR-AUC", f"{best_c3:.3f}"
                  if best_c3 == best_c3 else "—")
        best_rho = df["C3 rho"].astype(float).max()
        c4.metric("Best C3 rho", f"{best_rho:.2f}"
                  if best_rho == best_rho else "—")

    st.markdown(section_open(
        "Verdict table",
        eyebrow="PASS / FAIL on the five headline claims, one row per run",
        tint="model",
    ), unsafe_allow_html=True)

    st.markdown(_verdict_table_html(df), unsafe_allow_html=True)

    pass_counts = {short: 0 for short, _ in VERDICT_COLUMNS}
    total = 0
    for _, row in df.iterrows():
        verdicts = _row_verdicts(row.to_dict())
        total += 1
        for (short, _full), (passed, _obs) in zip(VERDICT_COLUMNS, verdicts):
            if passed:
                pass_counts[short] += 1

    if total > 0:
        summary_bits = [
            f"<b>{short}</b>: {pass_counts[short]}/{total} pass"
            for short, _full in VERDICT_COLUMNS
        ]
        st.markdown(
            "<div class='app-caption'>"
            + " &nbsp;·&nbsp; ".join(summary_bits)
            + "</div>",
            unsafe_allow_html=True,
        )

    csv = df.to_csv(index=False)
    st.download_button(
        "Download history as CSV",
        data=csv,
        file_name="run_history.csv",
        mime="text/csv",
        width="stretch",
    )
    st.markdown(section_close(), unsafe_allow_html=True)

    with st.expander("Clear history", expanded=False):
        if st.button("Clear all runs", width="stretch"):
            try:
                p = Path(history_path)
                if p.exists():
                    p.unlink()
                st.success("Cleared.")
            except Exception as exc:
                st.error(str(exc))
