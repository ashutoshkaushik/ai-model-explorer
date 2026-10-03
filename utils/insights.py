"""Derived facts for the model pages: profiles, similar models, comparisons, the compute race, live growth.

Pure pandas/numpy (no Streamlit), so it can be tested offline. Every function takes the (filtered) DataFrame.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.charts import GrowthFit, decimal_year
from utils.data_loader import COMPUTE_COL, COST_COL, DATE_COL, MODEL_COL, PARAMS_COL

SECONDS_PER_YEAR = 365.25 * 24 * 3600
CHANGES_PATH = Path(__file__).resolve().parent.parent / "data" / "changes.json"

# Familiar price tags for the training-cost page (approximate, rounded on purpose)
REFERENCE_COSTS = [
    (1_000, "a good laptop"),
    (400_000, "a typical US home"),
    (200_000_000, "a big-budget Hollywood film"),
]

COMPARE_METRICS = [  # (column, label, unit)
    (COMPUTE_COL, "Training compute", "FLOP"),
    (PARAMS_COL, "Parameters", "parameters"),
    (COST_COL, "Training cost", "2023 USD"),
]


def trend_value(fit: GrowthFit, when: pd.Timestamp) -> float:
    """Value of the fitted trend line at a date."""
    return float(10 ** (fit.intercept + fit.slope * decimal_year(pd.Series([when])).iloc[0]))


@dataclass
class Profile:
    row: pd.Series
    rank_in_year: int | None      # by training compute among the year's models with known compute
    known_in_year: int
    percentile: float | None      # share of all models with known compute that used less
    vs_trend: float | None        # compute ÷ trend line at its publication date


def model_profile(df: pd.DataFrame, name: str, fit: GrowthFit | None) -> Profile:
    row = df.loc[df[MODEL_COL] == name].iloc[0]
    compute = row[COMPUTE_COL]
    known = df[df[COMPUTE_COL] > 0]
    year_known = known[known["year"] == row["year"]]
    if pd.notna(compute) and compute > 0:
        rank = int((year_known[COMPUTE_COL] > compute).sum()) + 1
        pct = float((known[COMPUTE_COL] < compute).mean())
        vs = compute / trend_value(fit, row[DATE_COL]) if fit and row["year"] >= fit.start_year else None
    else:
        rank = pct = vs = None
    return Profile(row, rank, len(year_known), pct, vs)


def similar_models(df: pd.DataFrame, name: str, n: int = 5) -> pd.DataFrame:
    """Models closest in (log compute, publication date), preferring the same domain.

    Distance = |Δ log10 compute| + |Δ years| / 2, plus 1 for a different primary domain. Falls back to
    parameters when compute is unknown, and to publication date alone when both are.
    """
    row = df.loc[df[MODEL_COL] == name].iloc[0]
    others = df[df[MODEL_COL] != name]
    years = (decimal_year(others[DATE_COL]) - decimal_year(pd.Series([row[DATE_COL]])).iloc[0]).abs() / 2
    dist = years + (others["primary_domain"] != row["primary_domain"]).astype(float)
    for col in (COMPUTE_COL, PARAMS_COL):
        if pd.notna(row[col]) and row[col] > 0:
            dist = dist + (np.log10(others[col].where(others[col] > 0)) - np.log10(row[col])).abs()
            break
    return others.assign(distance=dist).dropna(subset=["distance"]).nsmallest(n, "distance")


def compare(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """One row per metric: both values and B ÷ A (NaN when either is unknown)."""
    rows = []
    for col, label, unit in COMPARE_METRICS:
        va, vb = a[col], b[col]
        ratio = vb / va if pd.notna(va) and pd.notna(vb) and va > 0 else np.nan
        rows.append({"metric": label, "unit": unit, "a": va, "b": vb, "ratio": ratio})
    return pd.DataFrame(rows)


def times(ratio: float) -> str:
    """4.2 -> '4.2×', 64000 -> '64,000×', 6.9e8 -> '690 million×', 0.25 -> '1/4.0'."""
    if ratio >= 1e9:
        return f"{ratio / 1e9:,.1f} billion×"
    if ratio >= 1e6:
        return f"{ratio / 1e6:,.0f} million×"
    if ratio >= 1:
        return f"{ratio:,.1f}×" if ratio < 100 else f"{ratio:,.0f}×"
    return f"1/{1 / ratio:,.1f}" if 1 / ratio < 100 else f"1/{1 / ratio:,.0f}"


def race_frames(df: pd.DataFrame, start_year: int, top_n: int = 10) -> pd.DataFrame:
    """For each year from start_year: the top_n largest training runs published up to the end of that year."""
    known = df[df[COMPUTE_COL] > 0]
    if known.empty:
        return known.assign(frame=pd.Series(dtype=int), rank=pd.Series(dtype=int))
    frames = []
    for year in range(max(start_year, int(known["year"].min())), int(known["year"].max()) + 1):
        top = known[known["year"] <= year].nlargest(top_n, COMPUTE_COL)
        frames.append(top.assign(frame=year, rank=range(1, len(top) + 1)))
    return pd.concat(frames, ignore_index=True)


def growth_per_second(fit: GrowthFit) -> float:
    """Fractional growth of the trend per second (continuous rate)."""
    return fit.slope * np.log(10) / SECONDS_PER_YEAR


def growth_since(fit: GrowthFit, seconds: float) -> float:
    """Fractional growth of the trend over `seconds` (0.01 = 1%)."""
    return fit.factor_per_year ** (seconds / SECONDS_PER_YEAR) - 1


def recent_models(df: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    """Models published in the `days` before the newest date in the data."""
    newest = df[DATE_COL].max()
    return df[df[DATE_COL] > newest - pd.Timedelta(days=days)].sort_values(DATE_COL, ascending=False)


def load_changes() -> dict | None:
    """What the last `download_data.py --force` added, if it recorded anything."""
    try:
        return json.loads(CHANGES_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
