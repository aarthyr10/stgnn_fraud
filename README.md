# Online Prior-Tracking Spatio-Temporal GNN for Cryptocurrency Fraud Detection

End-to-end Streamlit app and training pipeline that detects illicit
transactions in the Elliptic Bitcoin dataset using a frozen-GCN → GRU
backbone (Pareja et al. 2020), with an **online per-timestep
prior-tracking head** as the novel contribution. A Random Forest on raw
features (the Maganti 2026 reference at F1 ≈ 0.82) is included with and
without the same correction head to test whether the tracker is
architecture-agnostic.

## What the app shows

Five tabs, navigated left to right:

| Tab | What it visualises |
|-----|--------------------|
| **Dashboard** | Headline KPIs across all logged runs, the §6 verdict for the latest run, a spotlight radar for the best run, a claims summary, and the estimated-vs-true prior trajectory. |
| **Pipeline** | The place to actually run training and evaluation. Tracker settings (α, β, EM iterations), a live eight-stage request ribbon, per-condition result cards (C1/C2/C3) for both encoders, plus trajectory and per-timestep F1 charts. |
| **Results** | The scoreboard: two encoders × three prior-correction conditions with PR-AUC, recall @ 5% FPR, F1, and Spearman ρ; PR-AUC and recall bar charts; the tracker trajectory; and a pass/fail verdict against the proposal's claims. |
| **History** | Tabular log of every run with its verdict, so grid sweeps and seed sweeps can be compared at a glance. |
| **Dataset** | The non-stationary illicit-rate trajectory that motivates the work, plus per-timestep label composition and a dataset summary. |

All charts share one theme: a light surface, Inter typography, a
violet/sky/emerald/amber palette, and consistent gridlines, legends, and
hover labels (`app/utils/theme.py`).

## Architecture

| Layer | Files |
|-------|-------|
| Presentation | `app/main.py`, `app/components/*`, `app/utils/theme.py` |
| Application | `app/services/cache.py`, `app/services/inference.py` |
| Correction head | `app/services/prior_tracker.py` (Saerens-EM batch + online per-t) |
| Model serving | `app/models/gcn.py`, `gcn_gru.py`, `app/services/rf_baseline.py` |
| Data | `app/data/loader.py`, `snapshots.py`, `preprocess.py`, `demo.py` |
| Storage | `artefacts/*.pt`, `rf_baseline.pkl`, `*.parquet`, `*.json` |

The static GCN is pretrained, frozen, and reused as the spatial subnet.
The GRU consumes the frozen embeddings; the **prior tracker** sits
between the GRU softmax and the decision threshold.

## Requirements

- Python 3.10 or newer (developed on 3.13)
- The packages in `requirements.txt` (PyTorch, PyTorch Geometric,
  Streamlit, Plotly, pandas, scikit-learn)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

On the **first run** the app will:

1. detect that the Elliptic CSVs are missing,
2. generate a structurally faithful synthetic graph (49 weeks, 166
   features per node, ~10k nodes, illicit clusters wired as peeling
   chains and mixer stars),
3. train a demo version of all models (~30s total on CPU),
4. cache embeddings to `artefacts/embeddings.parquet` and metrics to
   `artefacts/metrics.json`,
5. open the app at <http://localhost:8501>.

## Using the real Elliptic dataset

Drop the three real CSVs into `data/elliptic/` to switch over:

```
data/elliptic/elliptic_txs_features.csv
data/elliptic/elliptic_txs_edgelist.csv
data/elliptic/elliptic_txs_classes.csv
```

Then run the full training pipeline (the demo artefacts are ignored once
real weights exist):

```bash
python -m training.train_gcn --epochs 200 --seed 42
python -m training.precompute_embeddings
python -m training.train_gru --epochs 60 --seed 42
python -m training.train_tgat --epochs 150 --seed 42
python -m training.evaluate
```

## Tests and linting

```bash
pytest tests/
ruff check .
```

`ruff.toml` holds the lint configuration. Entry-point and training
scripts ignore `E402` because they must extend `sys.path` before
importing the `app` package.

## Deploy with Docker on Render (display-only)

The deployed site is **display-only**: it never trains. You run the
pipeline locally, commit the results, push, and the site shows them.
This is controlled by `STGNN_DISPLAY_ONLY=true` (set in `Dockerfile` and
`render.yaml`); in that mode the Pipeline tab hides its run controls and
the Dashboard, Results, and History tabs render the published metrics.

Only two artefacts are needed for display and they are git-tracked:
`artefacts/metrics.json` and `artefacts/run_history.jsonl`. The heavy
model weights stay ignored — the display tabs do not load them.

### Publish a new result

```bash
python -m scripts.run_pipeline --alpha 5.0 --beta 10.0 --em-iter 12 \
    --init-mode blend --seed 1337 --force-retrain --note "frozen-seed1337"

git add artefacts/metrics.json artefacts/run_history.jsonl
git commit -m "Publish latest run"
git push
```

Render redeploys automatically (`autoDeploy: true`) and the site updates.

### Run the container locally

```bash
docker build -t stgnn-fraud .
docker run -p 8501:8501 stgnn-fraud
```

Open <http://localhost:8501>.

### First-time Render setup

1. Push the repo to GitHub:

   ```bash
   git init
   git add .
   git commit -m "Prior-tracking STGNN app"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```

2. On <https://render.com>, choose **New → Blueprint** and point it at
   the repo. Render reads `render.yaml` (a Docker web service on the
   **Standard** instance, 2 GB RAM). Or use **New → Web Service**, pick
   the repo, and select the **Docker** runtime.

3. Deploy. Render builds the image, injects `PORT`, binds Streamlit to
   it, and health-checks `/_stcore/health`.

Notes:

- `requirements.txt` pins the CPU-only PyTorch wheel index, which keeps
  the image small and the build fast.
- The display-only site is light; the Standard plan is comfortable. Drop
  `STGNN_DISPLAY_ONLY` (or set it to `false`) only if you intend to run
  training inside the container, which needs ~2 GB or more.

## Environment overrides

| Variable | Default |
|----------|---------|
| `STGNN_DATA_DIR` | `data/elliptic` |
| `STGNN_GRAPH_CACHE` | `artefacts/graph.pkl` |
| `STGNN_GCN_PATH` | `artefacts/gcn_subnet.pt` |
| `STGNN_TGAT_PATH` | `artefacts/tgat.pt` |
| `STGNN_HYBRID_PATH` | `artefacts/gru_head.pt` |
| `STGNN_EMBED_PATH` | `artefacts/embeddings.parquet` |
| `STGNN_METRICS_PATH` | `artefacts/metrics.json` |
| `STGNN_RF_PATH` | `artefacts/rf_baseline.pkl` |
| `STGNN_HISTORY_PATH` | `artefacts/run_history.jsonl` |
| `STGNN_DISPLAY_ONLY` | `false` (set `true` to disable in-app training) |
