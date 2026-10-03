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

Derived columns added by `load_data()`: `year` (int), `primary_org` / `primary_domain` (first value of the comma-separated field, `Unknown` if missing), `open_status` (`Open` = any "Open weights…" accessibility, `Closed`, `Unknown`), and `Frontier model` as bool.

`data/milestones.csv` (date, title, description) is curated by hand; quote any field containing commas.

## Tests
```bash
python tests/test_growth.py   # growth-rate fit
python tests/test_llm.py      # query functions + tool-use loop (fake client, no API calls)
python tests/test_surprise.py # the 25 static facts
python tests/test_insights.py # model profiles, compare, race frames, downloader change log
```

## Conventions
- Plotly (`plotly.express` / `graph_objects`) for all charts.
- All chart functions live in `utils/charts.py`; they take a DataFrame and return a `go.Figure`.
- App structure (same as the AI History RAG and Travel Agent Lab projects): `app.py` only registers pages with `st.navigation` in four groups: **App** (`lab/home.py`, `lab/app_pages.py`, `lab/explore_pages.py`: race, find a model, compare), **Overview** (`lab/overview.py`, plus the cost page from `lab/explore_pages.py`), **Explorer Lab** (`lab/lab_pages.py`, order in `lab/nav.TOUR`), **Surprise me** (`lab/surprise.py`, 25 static facts; never generate them at run time). `ui/chrome.py` adds the author card + LinkedIn link at the sidebar bottom and the footer on every page.
- Theme: `.streamlit/config.toml` (Claude-style light + dark, terracotta accent, Source Serif 4 headings, Hanken Grotesk body) and `ui/theme.py` (CSS tokens + shared classes: `hero`, `lede`, `eyebrow`, `badge`, `steps`, `ladder`, `learnbox`, `ltable`). Never hard-code colours in pages.
- Data pages call `ui.filters.sidebar_filters()` to draw the shared sidebar filters and get the filtered `df`; filter values persist across pages via `filters.keep_state()`.
- Data loading goes through `@st.cache_data` functions in `utils/data_loader.py`. The source CSV has double-encoded UTF-8 (`UniversitÃ©`); `load_data()` repairs it with `_fix_mojibake`, so always load through it.
- Derived per-model facts (profiles, similar models, comparisons, race frames, live growth, what's new) live in `utils/insights.py`: pure pandas, no Streamlit, tested in `tests/test_insights.py`.
- `scripts/download_data.py --force` writes `data/changes.json` (models added/removed); the Start page's What's new reads it.
- Split comma-separated multi-value columns (`str.split(",")` + `explode`) before grouping by org/domain.
- Never hardcode API keys. Read `ANTHROPIC_API_KEY` from `st.secrets` (`.streamlit/secrets.toml`, gitignored) or the environment; see `utils/llm.py`.
- Never execute LLM-generated code. Ask the Data only dispatches to the functions registered in `utils/llm.QUERIES`; add new capabilities there.
- Every chart, table, and LLM call uses the sidebar-filtered `df`. Handle empty results with a friendly message.
- Chart colors: use `palette()` / `color_map` / `chart_tokens()` in `utils/charts.py` (they follow the light/dark mode) (fixed order, validated palette). Assign domain colors from the unfiltered data so filters never repaint a series; fold beyond 7 series into gray "Other".
- Every page/chart area credits: **"Data: Epoch AI (CC-BY 4.0)"**.
- Dependencies are pinned in `requirements.txt`.
