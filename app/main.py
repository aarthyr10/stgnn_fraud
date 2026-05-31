from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
except ImportError:
    pass

import streamlit as st

from app.components.dashboard_tab import render_dashboard_tab
from app.components.dataset_tab import render_dataset_tab
from app.components.history_tab import render_history_tab
from app.components.pipeline_tab import render_pipeline_tab
from app.components.results_tab import render_results_tab
from app.utils.theme import inject_css

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def get_artefact_paths() -> dict:
    art = ROOT / "artefacts"
    return {
        "gcn": os.getenv("STGNN_GCN_PATH", str(art / "gcn_subnet.pt")),
        "tgat": os.getenv("STGNN_TGAT_PATH", str(art / "tgat.pt")),
        "hybrid_head": os.getenv("STGNN_HYBRID_PATH", str(art / "gru_head.pt")),
        "embeddings": os.getenv(
            "STGNN_EMBED_PATH", str(art / "embeddings.parquet"),
        ),
        "metrics": os.getenv("STGNN_METRICS_PATH", str(art / "metrics.json")),
        "rf": os.getenv("STGNN_RF_PATH", str(art / "rf_baseline.pkl")),
        "history": os.getenv(
            "STGNN_HISTORY_PATH", str(art / "run_history.jsonl"),
        ),
    }


def _hero_header() -> None:
    st.markdown("""
<style>
.brand-row {
    display: flex; align-items: center; gap: 14px;
    margin: 0 0 14px 0; padding-bottom: 12px;
    border-bottom: 1px solid #ECEBE3;
}
.brand-mark {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, #534AB7 0%, #185FA5 100%);
    box-shadow: 0 2px 8px rgba(83,74,183,0.3);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 800; font-size: 18px; letter-spacing: -1px;
}
.brand-text h1 {
    margin: 0; font-size: 22px; font-weight: 700; color: #14140F;
    letter-spacing: -0.01em;
}
.brand-text .sub {
    margin: 2px 0 0 0; font-size: 12px; color: #6D6C66;
    text-transform: uppercase; letter-spacing: 0.08em;
}
</style>
<div class="brand-row">
  <div class="brand-mark">PT</div>
  <div class="brand-text">
    <h1>Prior-Tracking STGNN</h1>
    <div class="sub">Elliptic Bitcoin · GCN-GRU · Saerens-EM</div>
  </div>
</div>
""", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="Prior-Tracking STGNN",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_css()
    _hero_header()

    artefact_paths = get_artefact_paths()
    data_dir = os.getenv("STGNN_DATA_DIR", str(ROOT / "data" / "elliptic"))
    graph_cache = os.getenv(
        "STGNN_GRAPH_CACHE", str(ROOT / "artefacts" / "graph.pkl"),
    )

    tab_dash, tab_pipe, tab_results, tab_hist, tab_data = st.tabs(
        ["Dashboard", "Pipeline", "Results", "History", "Dataset"],
    )
    with tab_dash:
        render_dashboard_tab(artefact_paths)
    with tab_pipe:
        render_pipeline_tab(artefact_paths, data_dir, graph_cache)
    with tab_results:
        render_results_tab(artefact_paths)
    with tab_hist:
        render_history_tab(artefact_paths)
    with tab_data:
        render_dataset_tab(artefact_paths, data_dir, graph_cache)


if __name__ == "__main__":
    main()
