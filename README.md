# AI Model Evolution Explorer

A Streamlit app that visualizes the history and growth of AI models — parameters, training compute, cost, openness, and who built them — using Epoch AI's [Notable AI Models](https://epoch.ai/data/notable-ai-models) dataset.

## Setup

```bash
uv venv --python 3.12 .venv        # or: python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py     # add --force to re-download
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # optional, for LLM features
streamlit run app.py
```

## Layout

- `app.py` — Streamlit entry point
- `utils/data_loader.py` — cached CSV loading and cleaning
- `utils/charts.py` — Plotly chart builders
- `utils/llm.py` — Anthropic client helper
- `scripts/download_data.py` — fetches the dataset (Epoch AI, with DataHub mirror fallback)

## Credits

Data: Epoch AI (CC-BY 4.0) — https://epoch.ai/data/notable-ai-models
