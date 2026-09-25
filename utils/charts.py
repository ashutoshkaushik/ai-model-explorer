"""All Plotly chart builders for the app live here.

Every function takes a DataFrame (already filtered) and returns a go.Figure.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.data_loader import (
    COMPUTE_COL,
    COST_COL,
    DATE_COL,
    MODEL_COL,
    PARAMS_COL,
)

# Categorical palette (validated for CVD separation on a dark surface), assigned in fixed order.
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
OTHER_COLOR = "#8a8984"
TEXT_MUTED = "#c3c2b7"
GRID = "rgba(255,255,255,0.08)"
MAX_SERIES = 7  # beyond this, fold into "Other"

METRICS = {
    "Training compute": {"col": COMPUTE_COL, "unit": "FLOP", "noun": "Compute grows"},
    "Parameters": {"col": PARAMS_COL, "unit": "parameters", "noun": "Parameter counts grow"},
    "Training cost": {"col": COST_COL, "unit": "2023 USD", "noun": "Training cost grows"},
}


def _base_layout(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#242422"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def color_map(categories: list[str]) -> dict[str, str]:
    """Stable color per category; 'Other'/'Unknown' always gray."""
    colors, i = {}, 0
    for cat in categories:
        if cat in ("Other", "Unknown"):
            colors[cat] = OTHER_COLOR
        else:
            colors[cat] = PALETTE[i % len(PALETTE)]
            i += 1
    return colors


def top_categories(series: pd.Series, n: int = MAX_SERIES) -> list[str]:
    """Most frequent values (excluding Unknown/Other), used to fix color assignment."""
    counts = series[~series.isin(["Unknown", "Other"])].value_counts()
    return counts.index[:n].tolist()


def fold_other(series: pd.Series, keep: list[str]) -> pd.Series:
    return series.where(series.isin(keep), "Other")


# --- Growth-rate fit -------------------------------------------------------------------------


@dataclass
class GrowthFit:
    factor_per_year: float  # multiplicative growth per year
    slope: float  # log10 units per year
    intercept: float
    n: int
    start_year: int


def decimal_year(dates: pd.Series) -> pd.Series:
    return dates.dt.year + (dates.dt.dayofyear - 1) / 365.25


def fit_growth(df: pd.DataFrame, value_col: str, start_year: int = 2010) -> GrowthFit | None:
    """Least-squares fit of log10(value) ~ year for models published on/after start_year."""
    d = df[(df[DATE_COL].dt.year >= start_year) & (df[value_col] > 0)]
    if len(d) < 3:
        return None
    x = decimal_year(d[DATE_COL]).to_numpy()
    y = np.log10(d[value_col].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    return GrowthFit(10**slope, slope, intercept, len(d), start_year)


# --- Compute over time -----------------------------------------------------------------------


def over_time_scatter(df: pd.DataFrame, metric: str, fit: GrowthFit | None, domains: list[str]) -> go.Figure:
    """Log-scale scatter of the chosen metric over time, colored by primary domain.

    `domains` is the fixed ordered list of domains that get their own color; others fold into "Other".
    """
    cfg = METRICS[metric]
    col = cfg["col"]
    d = df.copy()
    d["domain"] = fold_other(d["primary_domain"], domains)
    # Marker size: log10(parameters), rescaled to 5-22px; unknown params get the smallest size.
    logp = np.log10(d[PARAMS_COL].where(d[PARAMS_COL] > 0))
    lo, hi = logp.min(), logp.max()
    scaled = (logp - lo) / (hi - lo) if hi > lo else logp * 0
    d["size"] = (5 + 17 * scaled).fillna(5)

    colors = color_map(domains + ["Other"])
    fig = go.Figure()
    for dom in domains + ["Other"]:
        s = d[d["domain"] == dom]
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s[DATE_COL],
                y=s[col],
                mode="markers",
                name=dom,
                marker=dict(
                    size=s["size"],
                    color=colors[dom],
                    opacity=0.8,
                    line=dict(width=1, color="#1a1a19"),
                ),
                customdata=np.stack(
                    [s[MODEL_COL], s["primary_org"], s[COMPUTE_COL], s[PARAMS_COL], s["primary_domain"]], axis=-1
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[4]}<br>"
                    "%{x|%b %d, %Y}<br>Compute: %{customdata[2]:.2e} FLOP<br>"
                    "Parameters: %{customdata[3]:.2e}<extra></extra>"
                ),
            )
        )

    if fit is not None:
        start = pd.Timestamp(year=fit.start_year, month=1, day=1)
        end = d[DATE_COL].max()
        xs = pd.date_range(start, end, periods=50)
        ys = 10 ** (fit.intercept + fit.slope * decimal_year(pd.Series(xs)))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"Trend since {fit.start_year} (~{fit.factor_per_year:.1f}×/yr)",
                line=dict(color="#ffffff", width=2.5, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_yaxes(type="log", title=f"{metric} ({cfg['unit']}, log scale)", exponentformat="power")
    fig.update_xaxes(title="Publication date")
    fig.update_layout(legend=dict(itemsizing="constant"))
    return _base_layout(fig, height=560)


# --- Breakdown charts ------------------------------------------------------------------------

OPEN_COLORS = {"Open": PALETTE[0], "Closed": PALETTE[1], "Unknown": OTHER_COLOR}


def top_orgs_bar(df: pd.DataFrame, n: int = 10) -> go.Figure:
    counts = df.loc[df["primary_org"] != "Unknown", "primary_org"].value_counts().head(n)[::-1]
    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=counts.index,
            orientation="h",
            marker=dict(color=PALETTE[0], cornerradius=4),
            text=counts.values,
            textposition="outside",
            textfont=dict(color=TEXT_MUTED),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x} models<extra></extra>",
        )
    )
    fig.update_xaxes(title="Models", showgrid=False, showticklabels=False)
    fig.update_yaxes(title=None)
    return _base_layout(fig, f"Top {len(counts)} organizations by model count", height=400)


def domain_area(df: pd.DataFrame, domains: list[str]) -> go.Figure:
    d = df.assign(domain=fold_other(df["primary_domain"], domains))
    counts = d.groupby(["year", "domain"]).size().unstack(fill_value=0)
    counts = counts.reindex(range(counts.index.min(), counts.index.max() + 1), fill_value=0)
    colors = color_map(domains + ["Other"])
    fig = go.Figure()
    for dom in domains + ["Other"]:
        if dom not in counts:
            continue
        fig.add_trace(
            go.Scatter(
                x=counts.index,
                y=counts[dom],
                name=dom,
                stackgroup="one",
                mode="lines",
                line=dict(width=0.5, color=colors[dom]),
                fillcolor=colors[dom],
                hovertemplate=f"{dom}: %{{y}}<extra></extra>",
            )
        )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Models per year")
    return _base_layout(fig, "Models per year by domain", height=400)


def open_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["open_status"].value_counts().reindex(["Open", "Closed", "Unknown"]).dropna()
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.6,
            sort=False,
            marker=dict(colors=[OPEN_COLORS[k] for k in counts.index], line=dict(color="#1a1a19", width=2)),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value} models (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    return _base_layout(fig, "Open vs. closed weights", height=380)


def open_share_line(df: pd.DataFrame, min_models: int = 5) -> go.Figure | None:
    """Share of open-weight models per year, among models with known accessibility.

    Years with fewer than `min_models` known models are dropped to avoid noisy 0%/100% spikes.
    """
    known = df[df["open_status"] != "Unknown"]
    by_year = known.groupby("year")["open_status"].agg(total="size", open=lambda s: (s == "Open").sum())
    by_year = by_year[by_year["total"] >= min_models]
    if by_year.empty:
        return None
    share = by_year["open"] / by_year["total"]
    fig = go.Figure(
        go.Scatter(
            x=share.index,
            y=share.values,
            mode="lines+markers",
            line=dict(color=PALETTE[0], width=2),
            marker=dict(size=8, line=dict(width=2, color="#1a1a19")),
            customdata=np.stack([by_year["open"], by_year["total"]], axis=-1),
            hovertemplate="%{x}: %{y:.0%} open (%{customdata[0]} of %{customdata[1]})<extra></extra>",
        )
    )
    fig.update_yaxes(title="Open-weight share", tickformat=".0%", range=[0, 1.05])
    fig.update_xaxes(title=None)
    return _base_layout(fig, "Open-weight share over time", height=380)
