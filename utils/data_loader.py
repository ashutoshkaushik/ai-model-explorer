"""Load and clean the Epoch AI Notable AI Models dataset."""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "notable_ai_models.csv"

DATE_COL = "Publication date"
ORG_COL = "Organization"
DOMAIN_COL = "Domain"
PARAMS_COL = "Parameters"
COMPUTE_COL = "Training compute (FLOP)"
COST_COL = "Training compute cost (2023 USD)"
ACCESS_COL = "Model accessibility"
FRONTIER_COL = "Frontier model"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[FRONTIER_COL] = df[FRONTIER_COL].fillna(False).astype(bool)
    df["Year"] = df[DATE_COL].dt.year
    return df
