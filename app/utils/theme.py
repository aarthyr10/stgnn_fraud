from __future__ import annotations

import streamlit as st

PALETTE = {
    "presentation": {"bg": "#E6F1FB", "stroke": "#185FA5", "text": "#0C447C"},
    "application":  {"bg": "#EEEDFE", "stroke": "#534AB7", "text": "#3C3489"},
    "model":        {"bg": "#FAECE7", "stroke": "#993C1D", "text": "#712B13"},
    "data":         {"bg": "#EAF3DE", "stroke": "#3B6D11", "text": "#27500A"},
    "storage":      {"bg": "#F1EFE8", "stroke": "#5F5E5A", "text": "#44443F"},
    "viz":          {"bg": "#FAEEDA", "stroke": "#854F0B", "text": "#633806"},
    "post":         {"bg": "#FBEAF0", "stroke": "#993556", "text": "#72243E"},
    "cache":        {"bg": "#FBEAF0", "stroke": "#993556", "text": "#72243E"},
    "ok":           {"bg": "#E1F5EE", "stroke": "#0F6E56", "text": "#085041"},
    "warn":         {"bg": "#FBEAF0", "stroke": "#993556", "text": "#72243E"},
}


CHART = {
    "ink": "#0B0F1A",
    "muted": "#6A7286",
    "surface": "#FFFFFF",
    "grid": "#EFF1F6",
    "line": "#E4E7EF",
    "violet": "#5B4BE3",
    "emerald": "#0F9E73",
    "amber": "#E08A1F",
    "sky": "#2872C8",
    "crimson": "#D1366B",
    "slate": "#5F6B85",
}

PLOTLY_COLOURS = {
    "gcn": CHART["amber"],
    "tgat": CHART["sky"],
    "gcn_gru": CHART["violet"],
    "gcn_gru_no_gru": CHART["crimson"],
    "gcn_gru_no_gcn": CHART["slate"],
    "illicit": CHART["crimson"],
    "licit": CHART["emerald"],
    "unknown": "#B7BCC9",
}

CONDITION_COLOURS = {
    "none": CHART["amber"],
    "batch": CHART["sky"],
    "online": CHART["violet"],
}

CHART_FONT = '"Inter", "Anthropic Sans", -apple-system, "Segoe UI", sans-serif'
CHART_MONO = '"JetBrains Mono", "SFMono-Regular", Menlo, monospace'

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "autoScale2d", "hoverClosestCartesian",
        "hoverCompareCartesian", "toggleSpikelines",
    ],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def inject_css() -> None:
    st.markdown(
        """
<style>
/* ---------- Streamlit chrome ------------------------------------------ */
/* The default toolbar sits at the very top with z-index 999 and a
   semi-transparent background. On wide pages the running indicator
   visibly slides over the H1. We give the app some breathing room by
   pushing the main container down and toning the header back. */
header[data-testid="stHeader"] {
    background: transparent;
    height: 2.1rem;
}
header[data-testid="stHeader"]::before {
    /* Soft divider so the page does not feel headerless. */
    content: "";
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 1px;
    background: #ECEBE3;
}
[data-testid="stToolbar"] { right: 0.5rem; }

/* Hide the deploy menu but keep the hamburger so users can still
   reach Settings if needed. */
[data-testid="stDeployButton"] { display: none !important; }

/* Reserve space so the H1 never tucks under the chrome. */
.block-container {
    padding-top: 2.4rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

/* Footer is empty by default; collapse it so the page ends flush. */
footer { visibility: hidden; height: 0; }
#MainMenu { visibility: visible; }

/* ---------- Typography ------------------------------------------------- */
html, body, [class*="css"] {
    font-family: "Anthropic Sans", -apple-system, "system-ui",
                 "Segoe UI", sans-serif;
    color: #14140F;
}
h1, h2, h3, h4 { color: #14140F; font-weight: 600; letter-spacing: -0.01em; }
h1 { font-size: 1.7rem; }
h2 { font-size: 1.25rem; }
h3 { font-size: 1.05rem; }
.app-caption { color: #6D6C66; font-size: 0.92rem; margin-top: -0.4rem; }

/* ---------- Sidebar --------------------------------------------------- */
section[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}
.block-container {
    padding-left: 2.4rem !important;
    padding-right: 2.4rem !important;
}

/* ---------- Tabs ------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    border-bottom: 1px solid #ECEBE3;
}
.stTabs [data-baseweb="tab"] {
    height: 2.2rem;
    padding: 0 0.9rem;
    background: transparent;
    border-radius: 6px 6px 0 0;
    color: #44443F;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF;
    color: #534AB7;
    border: 1px solid #ECEBE3;
    border-bottom-color: #FFFFFF;
    font-weight: 600;
}

/* ---------- Stage cards (the pipeline visualisation primitive) -------- */
.stage-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.6rem;
    margin: 0.6rem 0 1rem 0;
}
.stage-card {
    border: 1px solid #ECEBE3;
    border-radius: 10px;
    padding: 0.7rem 0.85rem;
    background: #FFFFFF;
    box-shadow: 0 1px 2px rgba(20, 20, 15, 0.04);
    position: relative;
}
.stage-card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #6D6C66;
    margin-bottom: 0.15rem;
}
.stage-card .title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #14140F;
    margin-bottom: 0.15rem;
}
.stage-card .detail {
    font-size: 0.8rem;
    color: #44443F;
    line-height: 1.3;
}
.stage-card .badge {
    position: absolute;
    top: 0.55rem;
    right: 0.6rem;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.1rem 0.45rem;
    border-radius: 999px;
}
.stage-card.done   .badge { background: #E1F5EE; color: #0F6E56; }
.stage-card.active .badge { background: #EEEDFE; color: #534AB7; }
.stage-card.idle   .badge { background: #F1EFE8; color: #6D6C66; }
.stage-card.warn   .badge { background: #FBEAF0; color: #993556; }
.stage-card.done   { border-left: 3px solid #0F6E56; }
.stage-card.active { border-left: 3px solid #534AB7; }
.stage-card.idle   { border-left: 3px solid #B4B2A9; }
.stage-card.warn   { border-left: 3px solid #993556; }

/* Layer-tinted variants used by the architecture page. */
.stage-card.presentation { background: #E6F1FB; border-color: #C7DEF1; }
.stage-card.application  { background: #EEEDFE; border-color: #D5D2F1; }
.stage-card.model        { background: #FAECE7; border-color: #ECCBBD; }
.stage-card.data         { background: #EAF3DE; border-color: #CDDDB7; }
.stage-card.storage      { background: #F1EFE8; border-color: #DCD9CE; }
.stage-card.viz          { background: #FAEEDA; border-color: #E7D1AA; }
.stage-card.post         { background: #FBEAF0; border-color: #F1CFDB; }

/* ---------- Section wrappers ----------------------------------------- */
.section {
    border: 1px solid #ECEBE3;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    background: #FFFFFF;
}
.section.tint-data        { background: #F6FAEF; border-color: #DDE9C9; }
.section.tint-application { background: #F6F5FE; border-color: #D5D2F1; }
.section.tint-model       { background: #FBF1EB; border-color: #ECCBBD; }
.section.tint-viz         { background: #FCF5E6; border-color: #E7D1AA; }
.section.tint-post        { background: #FCEEF3; border-color: #F1CFDB; }
.section .section-eyebrow {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #534AB7;
    margin-bottom: 0.15rem;
    font-weight: 600;
}
.section h3 { margin: 0 0 0.2rem 0; }

/* ---------- Pipeline arrow ------------------------------------------- */
.pipe-arrow {
    text-align: center;
    color: #B4B2A9;
    font-size: 1.1rem;
    margin: -0.1rem 0;
}

/* ---------- Metrics ---------------------------------------------------- */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #ECEBE3;
    border-radius: 10px;
    padding: 0.5rem 0.8rem;
}
[data-testid="stMetricLabel"] {
    color: #6D6C66;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    color: #14140F;
    font-size: 1.45rem;
}

/* ---------- Buttons ---------------------------------------------------- */
button[kind="primary"] {
    background: #534AB7 !important;
    border-color: #534AB7 !important;
}
button[kind="primary"]:hover { background: #423A99 !important; }

/* ---------- Tables ---------------------------------------------------- */
[data-testid="stDataFrame"] thead tr {
    background: #F6F5F1;
}

/* ---------- Code blocks ---------------------------------------------- */
.stCode, pre, code {
    font-family: "JetBrains Mono", "SF Mono", Menlo, monospace !important;
    font-size: 0.82rem;
}

/* ---------- Animated pipeline step ------------------------------------ */
.step-row {
    display: grid;
    grid-template-columns: 56px 1fr 1.1fr 1.3fr;
    gap: 0.6rem;
    align-items: stretch;
    margin: 0.45rem 0;
    animation: stepFadeIn 0.45s ease-out both;
}
@keyframes stepFadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.step-num {
    background: #534AB7;
    color: #FFFFFF;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    font-weight: 700;
    box-shadow: 0 2px 5px rgba(83, 74, 183, 0.25);
}
.step-num.idle    { background: #C5C2BA; box-shadow: none; }
.step-num.running {
    background: #185FA5;
    animation: pulseRing 1.2s ease-in-out infinite;
}
.step-num.done    { background: #0F6E56; }
.step-num.warn    { background: #993556; }
@keyframes pulseRing {
    0%   { box-shadow: 0 0 0 0 rgba(24, 95, 165, 0.45); }
    70%  { box-shadow: 0 0 0 10px rgba(24, 95, 165, 0); }
    100% { box-shadow: 0 0 0 0 rgba(24, 95, 165, 0); }
}
.step-card {
    border: 1px solid #ECEBE3;
    border-radius: 10px;
    padding: 0.65rem 0.85rem;
    background: #FFFFFF;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}
.step-card .eyebrow {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #6D6C66;
    font-weight: 600;
}
.step-card .title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #14140F;
}
.step-card .detail {
    font-size: 0.8rem;
    color: #44443F;
    line-height: 1.35;
}
.step-card .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
    gap: 0.4rem;
    margin-top: 0.2rem;
}
.step-card .stat-grid .stat .lbl {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6D6C66;
}
.step-card .stat-grid .stat .val {
    font-size: 0.9rem;
    font-weight: 600;
    color: #14140F;
}
.step-card.tint-data        { background: #F6FAEF; border-color: #DDE9C9; }
.step-card.tint-application { background: #F6F5FE; border-color: #D5D2F1; }
.step-card.tint-model       { background: #FBF1EB; border-color: #ECCBBD; }
.step-card.tint-post        { background: #FCEEF3; border-color: #F1CFDB; }
.step-card.tint-viz         { background: #FCF5E6; border-color: #E7D1AA; }
.step-card.tint-storage     { background: #F4F2EC; border-color: #DCD9CE; }

.encoder-banner {
    margin: 1.2rem 0 0.6rem 0;
    padding: 0.8rem 1rem;
    border-radius: 12px;
    background: linear-gradient(95deg, #EEEDFE 0%, #F6F5FE 100%);
    border: 1px solid #D5D2F1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    animation: stepFadeIn 0.4s ease-out both;
}
.encoder-banner .name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #3C3489;
}
.encoder-banner .role {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6D6C66;
    margin-right: 0.5rem;
}
.encoder-banner .pill {
    font-size: 0.72rem;
    font-weight: 600;
    background: #FFFFFF;
    color: #534AB7;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    border: 1px solid #D5D2F1;
}

.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0.6rem;
}
.result-cell {
    border: 1px solid #ECEBE3;
    border-radius: 10px;
    padding: 0.7rem 0.85rem;
    background: #FFFFFF;
}
.result-cell.highlight {
    border-color: #534AB7;
    box-shadow: 0 1px 6px rgba(83, 74, 183, 0.15);
}
.result-cell .name {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #534AB7;
    font-weight: 700;
}
.result-cell .big {
    font-size: 1.4rem;
    font-weight: 700;
    color: #14140F;
    line-height: 1.2;
    margin-top: 0.15rem;
}
.result-cell .sub {
    font-size: 0.75rem;
    color: #6D6C66;
}
</style>
""",
        unsafe_allow_html=True,
    )


def stage_card(
    *,
    label: str,
    title: str,
    detail: str = "",
    state: str = "idle",
    layer: str | None = None,
) -> str:
    layer_cls = f" {layer}" if layer else ""
    badge_text = {
        "done": "done", "active": "live", "idle": "pending", "warn": "fallback",
    }.get(state, state)
    return (
        f'<div class="stage-card {state}{layer_cls}">'
        f'<div class="badge">{badge_text}</div>'
        f'<div class="label">{label}</div>'
        f'<div class="title">{title}</div>'
        f'<div class="detail">{detail}</div>'
        f"</div>"
    )


def stage_row(cards: list[str]) -> str:
    return '<div class="stage-row">' + "".join(cards) + "</div>"


def section_open(title: str, eyebrow: str = "", tint: str = "") -> str:
    cls = f"section{' tint-' + tint if tint else ''}"
    eb = f'<div class="section-eyebrow">{eyebrow}</div>' if eyebrow else ""
    return f'<div class="{cls}">{eb}<h3>{title}</h3>'


def section_close() -> str:
    return "</div>"


def pipe_arrow() -> str:
    return '<div class="pipe-arrow">&#9660;</div>'


def step_num_html(number: int, state: str) -> str:
    return f'<div class="step-num {state}">{number}</div>'


def step_row(
    number: int, state: str,
    status_html: str, data_html: str, result_html: str,
) -> str:
    return (
        f'<div class="step-row">'
        f'{step_num_html(number, state)}'
        f'<div class="step-card">{status_html}</div>'
        f'<div class="step-card tint-data">{data_html}</div>'
        f'<div class="step-card tint-post">{result_html}</div>'
        f'</div>'
    )


def step_card_body(eyebrow: str, title: str, detail: str = "",
                   stats: list[tuple[str, str]] | None = None) -> str:
    body = (
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<div class="title">{title}</div>'
    )
    if detail:
        body += f'<div class="detail">{detail}</div>'
    if stats:
        body += '<div class="stat-grid">'
        for lbl, val in stats:
            body += (
                f'<div class="stat"><div class="lbl">{lbl}</div>'
                f'<div class="val">{val}</div></div>'
            )
        body += "</div>"
    return body


def encoder_banner(name: str, role: str, pill: str) -> str:
    return (
        f'<div class="encoder-banner">'
        f'<div><span class="role">{role}</span>'
        f'<span class="name">{name}</span></div>'
        f'<div class="pill">{pill}</div>'
        f'</div>'
    )


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int],
              t: float) -> tuple[int, int, int]:
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


SHADING_RAMPS: dict[str, tuple[str, str]] = {
    "Purples": ("#FFFFFF", "#534AB7"),
    "Greens":  ("#FFFFFF", "#0F6E56"),
    "Oranges": ("#FFFFFF", "#993C1D"),
    "Reds":    ("#FFFFFF", "#993556"),
    "Blues":   ("#FFFFFF", "#185FA5"),
}


def shaded_table_html(
    df,
    *,
    subset: list[str] | None = None,
    ramp_per_column: dict[str, str] | None = None,
    formats: dict[str, str] | None = None,
) -> str:
    import html as _html

    import pandas as pd

    subset = subset or []
    ramp_per_column = ramp_per_column or {}
    formats = formats or {}

    bounds: dict[str, tuple[float, float, tuple[int, int, int],
                            tuple[int, int, int]]] = {}
    for col in subset:
        if col not in df.columns:
            continue
        nums = pd.to_numeric(df[col], errors="coerce")
        if not nums.notna().any():
            continue
        ramp_name = ramp_per_column.get(col, "Purples")
        lo_hex, hi_hex = SHADING_RAMPS.get(ramp_name,
                                          SHADING_RAMPS["Purples"])
        bounds[col] = (float(nums.min()), float(nums.max()),
                       _hex_to_rgb(lo_hex), _hex_to_rgb(hi_hex))

    def _cell_style(col: str, v) -> str:
        if col not in bounds or pd.isna(v):
            return ""
        vmin, vmax, lo, hi = bounds[col]
        span = vmax - vmin
        if span <= 0:
            return ""
        try:
            t = max(0.0, min(1.0, (float(v) - vmin) / span))
        except (TypeError, ValueError):
            return ""
        r, g, b = _lerp_rgb(lo, hi, t)
        text = "#FFFFFF" if t > 0.55 else "#14140F"
        return f"background:rgb({r},{g},{b});color:{text};"

    def _fmt(col: str, v) -> str:
        if pd.isna(v):
            return "—"
        spec = formats.get(col)
        if spec:
            try:
                return spec.format(v)
            except (TypeError, ValueError):
                pass
        return _html.escape(str(v))

    head = "".join(
        f'<th style="text-align:left;padding:0.4rem 0.6rem;'
        f'background:#F6F5F1;border-bottom:1px solid #ECEBE3;'
        f'font-weight:600;font-size:0.8rem">{_html.escape(str(c))}</th>'
        for c in df.columns
    )

    body_rows: list[str] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in df.columns:
            v = row[col]
            cells.append(
                f'<td style="padding:0.35rem 0.6rem;border-bottom:1px '
                f'solid #ECEBE3;font-size:0.85rem;{_cell_style(col, v)}">'
                f'{_fmt(col, v)}</td>'
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<div style="border:1px solid #ECEBE3;border-radius:10px;'
        'overflow:hidden;margin:0.4rem 0 0.8rem 0">'
        '<table style="width:100%;border-collapse:collapse;'
        'font-family:&quot;Anthropic Sans&quot;, -apple-system, '
        'sans-serif">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def gradient_styler(
    df,
    subset: list[str],
    ramp: str = "Purples",
    ramp_per_column: dict[str, str] | None = None,
):
    import pandas as pd

    ramp_per_column = ramp_per_column or {}

    def _shade(series: pd.Series) -> list[str]:
        if series.name not in subset:
            return [""] * len(series)
        ramp_name = ramp_per_column.get(series.name, ramp)
        lo_hex, hi_hex = SHADING_RAMPS.get(ramp_name, SHADING_RAMPS["Purples"])
        lo = _hex_to_rgb(lo_hex)
        hi = _hex_to_rgb(hi_hex)
        nums = pd.to_numeric(series, errors="coerce")
        if not nums.notna().any():
            return [""] * len(series)
        vmin = float(nums.min())
        vmax = float(nums.max())
        span = vmax - vmin
        out: list[str] = []
        for v in nums:
            if pd.isna(v) or span <= 0:
                out.append("")
                continue
            t = max(0.0, min(1.0, (float(v) - vmin) / span))
            r, g, b = _lerp_rgb(lo, hi, t)
            text = "#FFFFFF" if t > 0.55 else "#14140F"
            out.append(f"background-color: rgb({r},{g},{b}); color: {text}")
        return out

    return df.style.apply(_shade, axis=0)


def apply_plotly_layout(fig, height: int = 320, title: str | None = None):
    fig.update_layout(
        height=height,
        title=dict(
            text=title or "",
            font=dict(size=14, color=CHART["ink"]),
            x=0.0, xanchor="left", y=0.97,
        ) if title else None,
        margin=dict(l=16, r=16, t=48 if title else 26, b=44),
        paper_bgcolor=CHART["surface"],
        plot_bgcolor=CHART["surface"],
        font=dict(family=CHART_FONT, color=CHART["ink"], size=12),
        colorway=[
            CHART["violet"], CHART["sky"], CHART["emerald"],
            CHART["amber"], CHART["crimson"], CHART["slate"],
        ],
        hoverlabel=dict(
            bgcolor=CHART["ink"], bordercolor=CHART["ink"],
            font=dict(color="#FFFFFF", family=CHART_MONO, size=12),
        ),
        xaxis=dict(
            gridcolor=CHART["grid"], zerolinecolor=CHART["line"],
            linecolor=CHART["line"], ticks="outside",
            tickcolor=CHART["line"], tickfont=dict(color=CHART["muted"]),
        ),
        yaxis=dict(
            gridcolor=CHART["grid"], zerolinecolor=CHART["line"],
            linecolor=CHART["line"], ticks="outside",
            tickcolor=CHART["line"], tickfont=dict(color=CHART["muted"]),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0.6)", bordercolor=CHART["line"],
            borderwidth=1, font=dict(size=11, color=CHART["muted"]),
        ),
    )
    return fig
