import math

import pandas as pd
import streamlit as st

from utils.charts import METRICS, fit_growth, over_time_scatter, top_categories
from utils.data_loader import (
    ACCESS_COL,
    COMPUTE_COL,
    COST_COL,
    DATE_COL,
    MODEL_COL,
    PARAMS_COL,
    load_data,
)

CREDIT = "Data: Epoch AI (CC-BY 4.0)"
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float) -> str:
    """1.0001e27 -> '1.0 × 10²⁷'."""
    exp = math.floor(math.log10(x))
    return f"{x / 10**exp:.1f} × 10{str(exp).translate(SUPERSCRIPT)}"


st.set_page_config(page_title="AI Model Evolution Explorer", page_icon="📈", layout="wide")
st.title("AI Model Evolution Explorer")
st.markdown("How notable AI models have grown in scale, compute, and cost from 1950 to today.")

df = load_data()

# --- KPIs ---
biggest = df.loc[df[COMPUTE_COL].idxmax()]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Models", f"{len(df):,}", border=True)
k2.metric("Year range", f"{df['year'].min()}–{df['year'].max()}", border=True)
k3.metric("Organizations", f"{df['primary_org'].nunique():,}", border=True)
k4.metric(
    "Largest training compute",
    f"{sci(biggest[COMPUTE_COL])} FLOP",
    help=f"{biggest[MODEL_COL]} ({biggest['primary_org']}, {biggest[DATE_COL]:%b %Y})",
    border=True,
)

# --- Compute over time ---
st.subheader("Compute Over Time")
metric = st.radio("Y-axis metric", list(METRICS), horizontal=True)
metric_col = METRICS[metric]["col"]
plotted = df[df[metric_col] > 0]
excluded = len(df) - len(plotted)
fit = fit_growth(plotted, metric_col)
domains = top_categories(df["primary_domain"])
if plotted.empty:
    st.info("No models with this metric match the current selection.")
else:
    if fit:
        st.info(
            f"**{METRICS[metric]['noun']} ~{fit.factor_per_year:.1f}× per year** "
            f"(log-linear fit over {fit.n:,} models published since {fit.start_year})."
        )
    st.plotly_chart(over_time_scatter(plotted, metric, fit, domains), width="stretch")
    st.caption(f"{excluded:,} models excluded (missing data). Marker size ∝ log(parameters).")

# --- Searchable table ---
st.subheader("Browse models")
query = st.text_input("Search", placeholder="Model, organization, or domain…")
table = df[[MODEL_COL, "primary_org", "primary_domain", DATE_COL, PARAMS_COL, COMPUTE_COL, COST_COL, ACCESS_COL]]
if query:
    haystack = table[[MODEL_COL, "primary_org", "primary_domain"]].astype(str).agg(" ".join, axis=1)
    table = table[haystack.str.contains(query, case=False, regex=False)]
st.caption(f"{len(table):,} models")
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "primary_org": "Organization",
        "primary_domain": "Domain",
        DATE_COL: st.column_config.DateColumn("Published", format="YYYY-MM-DD"),
        PARAMS_COL: st.column_config.NumberColumn("Parameters", format="%.2e"),
        COMPUTE_COL: st.column_config.NumberColumn("Compute (FLOP)", format="%.2e"),
        COST_COL: st.column_config.NumberColumn("Cost (2023 USD)", format="dollar"),
        ACCESS_COL: "Accessibility",
    },
)

st.divider()
st.caption(CREDIT)
