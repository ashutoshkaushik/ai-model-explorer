"""Load and clean the Epoch AI Notable AI Models dataset."""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_PATH = DATA_DIR / "notable_ai_models.csv"

MODEL_COL = "Model"
DATE_COL = "Publication date"
ORG_COL = "Organization"
DOMAIN_COL = "Domain"
PARAMS_COL = "Parameters"
COMPUTE_COL = "Training compute (FLOP)"
COST_COL = "Training compute cost (2023 USD)"
ACCESS_COL = "Model accessibility"
FRONTIER_COL = "Frontier model"

NUMERIC_COLS = [PARAMS_COL, COMPUTE_COL, COST_COL]


def _first_value(series: pd.Series) -> pd.Series:
    """'Meta AI,NYU' -> 'Meta AI'. Missing stays 'Unknown'."""
    return series.fillna("Unknown").str.split(",").str[0].str.strip().replace("", "Unknown")


def _open_status(access: pd.Series) -> pd.Series:
    status = pd.Series("Closed", index=access.index)
    status[access.str.startswith("Open weights", na=False)] = "Open"
    status[access.isna()] = "Unknown"
    return status


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    df["year"] = df[DATE_COL].dt.year.astype(int)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["primary_org"] = _first_value(df[ORG_COL])
    df["primary_domain"] = _first_value(df[DOMAIN_COL])
    df[FRONTIER_COL] = df[FRONTIER_COL].astype("string").str.lower().eq("true").fillna(False).astype(bool)
    df["open_status"] = _open_status(df[ACCESS_COL])
    return df.sort_values(DATE_COL, ascending=False).reset_index(drop=True)
