"""All Plotly chart builders for the app live here.

Every function takes a DataFrame (already filtered) and returns a go.Figure.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ui import theme
from utils.data_loader import (
    COMPUTE_COL,
    COST_COL,
    DATE_COL,
    MODEL_COL,
    PARAMS_COL,
)

# Categorical palettes (validated for CVD separation on each surface), assigned in fixed order.
# They mirror chartCategoricalColors in .streamlit/config.toml; the site follows the visitor's light/dark setting.
PALETTE_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
OTHER_COLOR = "#8a8984"
MAX_SERIES = 7  # beyond this, fold into "Other"

# Chart-only colours per mode; surface/ink/muted come from ui.theme so charts match the page.
CHART = {
    "light": {"grid": "rgba(31,30,29,0.08)", "era": ("rgba(31,30,29,0)", "rgba(31,30,29,0.03)", "rgba(31,30,29,0.06)"),
              "rule": "rgba(31,30,29,0.30)", "label_bg": "rgba(250,249,245,0.85)", "hover_bg": "#ffffff"},
    "dark": {"grid": "rgba(242,240,232,0.08)", "era": ("rgba(242,240,232,0)", "rgba(242,240,232,0.035)", "rgba(242,240,232,0.07)"),
             "rule": "rgba(242,240,232,0.35)", "label_bg": "rgba(38,38,36,0.80)", "hover_bg": "#30302E"},
}


def palette() -> list[str]:
    return PALETTE_DARK if theme.mode() == "dark" else PALETTE_LIGHT


def chart_tokens() -> dict:
    """Page tokens (surface, ink, muted) plus chart-only colours, for the current light/dark mode."""
    return {**theme.tokens(), **CHART[theme.mode()]}

METRICS = {
    "Training compute": {"col": COMPUTE_COL, "unit": "FLOP", "noun": "Compute grows"},
    "Parameters": {"col": PARAMS_COL, "unit": "parameters", "noun": "Parameter counts grow"},
    "Training cost": {"col": COST_COL, "unit": "2023 USD", "noun": "Training cost grows"},
}


def _base_layout(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    t = chart_tokens()
    fig.update_layout(
        template="plotly_dark" if theme.mode() == "dark" else "plotly_white",
        title=dict(text=title, font=dict(family=theme.FONT_HEADING, size=17, color=t["ink"])) if title else None,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=theme.FONT_BODY, color=t["muted"]),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=t["hover_bg"], font=dict(family=theme.FONT_BODY, color=t["ink"])),
    )
    fig.update_xaxes(gridcolor=t["grid"], linecolor=t["line"], zeroline=False)
    fig.update_yaxes(gridcolor=t["grid"], linecolor=t["line"], zeroline=False)
    return fig


def color_map(categories: list[str]) -> dict[str, str]:
    """Stable color per category; 'Other'/'Unknown' always gray."""
    colors, i = {}, 0
    for cat in categories:
        if cat in ("Other", "Unknown"):
            colors[cat] = OTHER_COLOR
        else:
            colors[cat] = palette()[i % len(palette())]
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


ERAS = [  # (label, start, end, index into the mode's era fills)
    ("Pre-Deep<br>Learning", None, "2010-01-01", 0),
    ("Deep<br>Learning", "2010-01-01", "2018-01-01", 1),
    ("Large-Scale<br>/ LLM Era", "2018-01-01", None, 2),
]


def add_eras(fig: go.Figure, x_min: pd.Timestamp, x_max: pd.Timestamp) -> None:
    """Shade the three eras (clipped to the visible date range) and label each at the bottom."""
    t = chart_tokens()
    for name, start, end, fill in ERAS:
        x0 = max(pd.Timestamp(start), x_min) if start else x_min
        x1 = min(pd.Timestamp(end), x_max) if end else x_max
        if x0 >= x1:
            continue
        fig.add_vrect(x0=x0, x1=x1, fillcolor=t["era"][fill], line_width=0, layer="below")
        fig.add_annotation(
            x=x0, y=0, xref="x", yref="paper", text=name, showarrow=False,
            xanchor="left", yanchor="bottom", xshift=4, yshift=4, align="left",
            font=dict(size=11, color=t["muted"]),
        )


def add_milestones(fig: go.Figure, milestones: pd.DataFrame, x_min: pd.Timestamp, x_max: pd.Timestamp) -> None:
    """Dashed vertical line per milestone, with a hoverable marker on top of each line.

    Rotated text labels are drawn only where they won't collide with the previous label;
    crowded milestones remain identifiable via hover.
    """
    t = chart_tokens()
    m = milestones[milestones["date"].between(x_min, x_max)]
    min_gap = (x_max - x_min) * 0.028
    last_labeled = None
    for _, row in m.iterrows():
        fig.add_shape(
            type="line", x0=row["date"], x1=row["date"], y0=0, y1=1, xref="x", yref="paper",
            line=dict(color=t["rule"], width=1, dash="dash"), layer="below",
        )
        roomy = last_labeled is None or row["date"] - last_labeled >= min_gap
        fig.add_annotation(
            x=row["date"], y=1, xref="x", yref="paper", yanchor="top",
            text=row["title"] if roomy else "◆", textangle=-90 if roomy else 0,
            showarrow=False, xshift=-7 if roomy else 0,
            font=dict(size=10, color=t["ink"] if roomy else t["muted"]),
            bgcolor=t["label_bg"] if roomy else None,
            hovertext=f"<b>{row['title']}</b> ({row['date']:%b %Y})<br>{row['description']}",
        )
        if roomy:
            last_labeled = row["date"]


def over_time_scatter(
    df: pd.DataFrame,
    metric: str,
    fit: GrowthFit | None,
    domains: list[str],
    milestones: pd.DataFrame | None = None,
) -> go.Figure:
    """Log-scale scatter of the chosen metric over time, colored by primary domain, with era shading.

    `domains` is the fixed ordered list of domains that get their own color; others fold into "Other".
    Pass `milestones` (date/title/description) to overlay them as dashed vertical lines.
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
                    line=dict(width=1, color=chart_tokens()["surface"]),
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
                line=dict(color=chart_tokens()["ink"], width=2.5, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_yaxes(type="log", title=f"{metric} ({cfg['unit']}, log scale)", exponentformat="power")
    fig.update_xaxes(title="Publication date")
    pad = pd.Timedelta(days=365)
    x_min, x_max = d[DATE_COL].min() - pad, d[DATE_COL].max() + pad
    add_eras(fig, x_min, x_max)
    if milestones is not None:
        add_milestones(fig, milestones, x_min, x_max)
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_layout(legend=dict(itemsizing="constant"))
    return _base_layout(fig, height=600)


# --- Breakdown charts ------------------------------------------------------------------------

def open_colors() -> dict[str, str]:
    return {"Open": palette()[0], "Closed": palette()[1], "Unknown": OTHER_COLOR}


def top_orgs_bar(df: pd.DataFrame, n: int = 10) -> go.Figure:
    counts = df.loc[df["primary_org"] != "Unknown", "primary_org"].value_counts().head(n)[::-1]
    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=counts.index,
            orientation="h",
            marker=dict(color=palette()[0], cornerradius=4),
            text=counts.values,
            textposition="outside",
            textfont=dict(color=chart_tokens()["muted"]),
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
            marker=dict(colors=[open_colors()[k] for k in counts.index], line=dict(color=chart_tokens()["surface"], width=2)),
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
            line=dict(color=palette()[0], width=2),
            marker=dict(size=8, line=dict(width=2, color=chart_tokens()["surface"])),
            customdata=np.stack([by_year["open"], by_year["total"]], axis=-1),
            hovertemplate="%{x}: %{y:.0%} open (%{customdata[0]} of %{customdata[1]})<extra></extra>",
        )
    )
    fig.update_yaxes(title="Open-weight share", tickformat=".0%", range=[0, 1.05])
    fig.update_xaxes(title=None)
    return _base_layout(fig, "Open-weight share over time", height=380)


# --- Frontier leaderboard --------------------------------------------------------------------


def leaderboard_step(top: pd.DataFrame) -> go.Figure:
    """Step chart of the highest training compute per year (log scale)."""
    fig = go.Figure(
        go.Scatter(
            x=top["year"],
            y=top[COMPUTE_COL],
            mode="lines+markers",
            line=dict(color=palette()[0], width=2, shape="hv"),
            marker=dict(size=8, line=dict(width=2, color=chart_tokens()["surface"])),
            customdata=np.stack([top[MODEL_COL], top["primary_org"]], axis=-1),
            hovertemplate="<b>%{x}: %{customdata[0]}</b><br>%{customdata[1]}<br>%{y:.2e} FLOP<extra></extra>",
        )
    )
    fig.update_yaxes(type="log", title="Training compute (FLOP, log scale)", exponentformat="power")
    fig.update_xaxes(title=None)
    return _base_layout(fig, "Highest training compute per year", height=440)
