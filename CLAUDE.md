# AI Model Evolution Explorer

## Goal
Streamlit app that visualizes the history and growth of AI models (scale, compute, cost, openness, organizations, domains) using Epoch AI's Notable AI Models dataset.

## How to run
```bash
source .venv/bin/activate            # Python 3.12 venv (created with uv)
python scripts/download_data.py      # writes data/notable_ai_models.csv; --force to refresh
streamlit run app.py
```

## Dataset: `data/notable_ai_models.csv`
Snapshot downloaded 2026-09-24: 1,072 rows, 47 columns, dates 1950-07-02 → 2026-09-21.

Key columns (exact names):

| Concept | Column | Type / notes | Null % |
|---|---|---|---|
| Date | `Publication date` | `YYYY-MM-DD` string → parse with `pd.to_datetime` | 0 |
| Organization | `Organization` | str; **comma-separated when multiple orgs** (e.g. `Meta AI,New York University (NYU)`) | 1.7 |
| Domain | `Domain` | str; **comma-separated multi-values** (e.g. `Multimodal,Language,Vision`) | 0.3 |
| Parameters | `Parameters` | float | 32.4 |
| Training compute | `Training compute (FLOP)` | float, spans many orders of magnitude → log axes | 49.6 |
| Cost | `Training compute cost (2023 USD)` | float, sparse | 83.2 |
| Accessibility | `Model accessibility` | `Unreleased`, `Open weights (unrestricted)`, `API access`, `Open weights (non-commercial)`, `Open weights (restricted use)`, `Hosted access (no API)`, `Limited access` | 23.4 |
| Frontier | `Frontier model` | only `True` or blank → treat blank as `False` (124 True) | 88.4 |

Other useful columns: `Model`, `Task`, `Organization categorization` (Industry/Academia, also comma-separated), `Country (of organization)`, `Open model weights?` (Yes/No), `Training hardware`, `Training dataset size (total)` (str), `Confidence`, `Notability criteria`, `Training compute cost (cloud)`, `Training compute cost (upfront)`.

`utils/data_loader.py` exposes these as constants (`DATE_COL`, `ORG_COL`, `DOMAIN_COL`, `PARAMS_COL`, `COMPUTE_COL`, `COST_COL`, `ACCESS_COL`, `FRONTIER_COL`); use them instead of retyping strings.

## Conventions
- Plotly (`plotly.express` / `graph_objects`) for all charts.
- All chart functions live in `utils/charts.py`; they take a DataFrame and return a `go.Figure`. `app.py` only lays out UI.
- Data loading goes through `@st.cache_data` functions in `utils/data_loader.py`.
- Split comma-separated multi-value columns (`str.split(",")` + `explode`) before grouping by org/domain.
- Never hardcode API keys. Read `ANTHROPIC_API_KEY` from `st.secrets` (`.streamlit/secrets.toml`, gitignored) or the environment; see `utils/llm.py`.
- Every page/chart area credits: **"Data: Epoch AI (CC-BY 4.0)"**.
- Dependencies are pinned in `requirements.txt`.
