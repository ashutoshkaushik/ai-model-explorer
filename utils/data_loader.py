"""Load and clean the Epoch AI Notable AI Models dataset."""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_PATH = DATA_DIR / "notable_ai_models.csv"
MILESTONES_PATH = DATA_DIR / "milestones.csv"

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


def _fix_mojibake(text):
    """'UniversitÃ© de MontrÃ©al' -> 'Université de Montréal'.

    The published CSV has some UTF-8 text that was decoded as Latin-1/Windows-1252 and re-encoded. Undo that
    per value, and keep the original whenever the round trip doesn't produce valid UTF-8.
    """
    if not isinstance(text, str) or ("Ã" not in text and "Â" not in text):
        return text
    for codec in ("latin-1", "cp1252"):
        try:
            return text.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def _open_status(access: pd.Series) -> pd.Series:
    status = pd.Series("Closed", index=access.index)
    status[access.str.startswith("Open weights", na=False)] = "Open"
    status[access.isna()] = "Unknown"
    return status


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(_fix_mojibake)
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


def top_orgs(df: pd.DataFrame, n: int = 15) -> list[str]:
    counts = df.loc[df["primary_org"] != "Unknown", "primary_org"].value_counts()
    return counts.index[:n].tolist()


def apply_filters(
    df: pd.DataFrame,
    years: tuple[int, int],
    domains: list[str],
    orgs: list[str],
    org_options: list[str],
    access: list[str],
    frontier_only: bool,
) -> pd.DataFrame:
    """Empty selection lists mean 'no filter'. 'Other' in orgs matches every org outside org_options."""
    mask = df["year"].between(*years)
    if domains:
        mask &= df["primary_domain"].isin(domains)
    if orgs:
        org_mask = df["primary_org"].isin(orgs)
        if "Other" in orgs:
            org_mask |= ~df["primary_org"].isin(org_options)
        mask &= org_mask
    if access:
        mask &= df[ACCESS_COL].fillna("Unknown").isin(access)
    if frontier_only:
        mask &= df[FRONTIER_COL]
    return df[mask]


@st.cache_data
def load_milestones() -> pd.DataFrame:
    m = pd.read_csv(MILESTONES_PATH, parse_dates=["date"])
    return m.sort_values("date").reset_index(drop=True)


def models_near(df: pd.DataFrame, date: pd.Timestamp, months: int = 6, n: int = 3) -> pd.DataFrame:
    """Top-n models by training compute published within ±months of date."""
    window = df[
        df[DATE_COL].between(date - pd.DateOffset(months=months), date + pd.DateOffset(months=months))
        & df[COMPUTE_COL].notna()
    ]
    return window.nlargest(n, COMPUTE_COL)


def frontier_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """The single highest-compute model for each year, with growth vs. the previous listed year."""
    known = df[df[COMPUTE_COL].notna()]
    if known.empty:
        return known.assign(growth=pd.Series(dtype=float))
    top = known.loc[known.groupby("year")[COMPUTE_COL].idxmax()].sort_values("year")
    return top.assign(growth=top[COMPUTE_COL] / top[COMPUTE_COL].shift())
