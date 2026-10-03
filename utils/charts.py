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
    """Running record of training compute (step line) with each year's top model as a marker.

    Filled markers set a new record; hollow ones were the year's biggest but smaller than an earlier record.
    """
    t = chart_tokens()
    color = palette()[0]
    fig = go.Figure(go.Scatter(
        x=top["year"], y=top["record_to_date"], mode="lines", name="Record to date",
        line=dict(color=color, width=2, shape="hv"), hoverinfo="skip",
    ))
    for is_record, name in ((True, "New record"), (False, "Year's top, not a new record")):
        s = top[top["new_record"] == is_record]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["year"], y=s[COMPUTE_COL], mode="markers", name=name,
            marker=dict(size=9 if is_record else 8, color=color if is_record else t["surface"],
                        line=dict(width=2, color=color if is_record else t["muted"])),
            customdata=np.stack([s[MODEL_COL], s["primary_org"]], axis=-1),
            hovertemplate="<b>%{x}: %{customdata[0]}</b><br>%{customdata[1]}<br>%{y:.2e} FLOP"
                          + ("" if is_record else "<br>Not a new record") + "<extra></extra>",
        ))
    fig.update_yaxes(type="log", title="Training compute (FLOP, log scale)", exponentformat="power")
    fig.update_xaxes(title=None)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return _base_layout(fig, height=460)  # the page heading names the chart; a title here collides with the legend


# --- Compute race ----------------------------------------------------------------------------


def race_chart(frames: pd.DataFrame, domains: list[str], top_n: int) -> go.Figure:
    """Animated bar race: the top_n largest training runs so far, one frame per year (log x axis).

    Bars sit on fixed rank rows (1 = largest) so they grow and swap names smoothly between frames.
    """
    t = chart_tokens()
    colors = color_map(domains + ["Other"])
    d = frames.assign(domain=fold_other(frames["primary_domain"], domains))
    lo = np.floor(np.log10(d[COMPUTE_COL].min()))
    hi = np.log10(d[COMPUTE_COL].max()) + 0.6

    def bar(f: pd.DataFrame) -> go.Bar:
        return go.Bar(
            x=f[COMPUTE_COL], y=f["rank"], orientation="h",
            marker=dict(color=[colors[x] for x in f["domain"]], cornerradius=3),
            text=[f"{m} · {o}, {y}" for m, o, y in zip(f[MODEL_COL], f["primary_org"], f["year"])],
            # inside long bars (right-aligned), outside short ones, so labels never run off the edge. Positions are
            # set per bar rather than "auto", which mis-measures text mid-animation and piles labels up.
            textposition=["inside" if np.log10(x) - lo > 0.55 * (hi - lo) else "outside" for x in f[COMPUTE_COL]],
            insidetextanchor="end", constraintext="none", cliponaxis=False,
            insidetextfont=dict(color="#ffffff", size=12), outsidetextfont=dict(color=t["ink"], size=12),
            customdata=np.stack([f[MODEL_COL], f["primary_org"], f["domain"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<br>%{x:.2e} FLOP<extra></extra>",
            showlegend=False,
        )

    def year_label(year: int) -> list[dict]:
        return [dict(x=0.99, y=0.04, xref="paper", yref="paper", xanchor="right", yanchor="bottom", showarrow=False,
                     text=str(year), font=dict(family=theme.FONT_HEADING, size=64, color=t["line"]))]

    years = sorted(d["frame"].unique())
    first = d[d["frame"] == years[0]]
    fig = go.Figure(data=[bar(first)] + [  # legend-only traces, one per domain present
        go.Bar(x=[None], y=[None], name=dom, marker_color=colors[dom], showlegend=True)
        for dom in domains + ["Other"] if dom in set(d["domain"])
    ])
    fig.frames = [go.Frame(data=[bar(d[d["frame"] == y])], traces=[0], name=str(y),
                           layout=go.Layout(annotations=year_label(y))) for y in years]
    step_ms = 700
    # Controls sit on separate rows so nothing overlaps at any width: legend and Play/Pause above the plot,
    # the year slider below the axis title.
    fig.update_layout(
        annotations=year_label(years[0]),
        bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left"),
        updatemenus=[dict(
            type="buttons", direction="left", x=1, y=1.02, xanchor="right", yanchor="bottom", showactive=False,
            pad=dict(r=0, t=0, b=4), bgcolor=t["card"], bordercolor=t["line"], font=dict(color=t["ink"]),
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, dict(
                    frame=dict(duration=step_ms, redraw=True), transition=dict(duration=step_ms * 0.6),
                    fromcurrent=True, mode="immediate")]),
                dict(label="❚❚ Pause", method="animate", args=[[None], dict(
                    frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0, y=0, len=1, xanchor="left", yanchor="top", pad=dict(t=70, b=0),
            currentvalue=dict(visible=False), bgcolor=t["line"], activebgcolor=t["accent"], bordercolor=t["line"],
            font=dict(color=t["muted"]), ticklen=4,
            steps=[dict(label=str(y), method="animate", args=[[str(y)], dict(
                frame=dict(duration=0, redraw=True), transition=dict(duration=0), mode="immediate")])
                for y in years],
        )],
    )
    fig.update_xaxes(type="log", range=[lo, hi], title="Training compute (FLOP, log scale)", exponentformat="power")
    fig.update_yaxes(range=[top_n + 0.6, 0.4], showticklabels=False, showgrid=False, title=None)
    fig = _base_layout(fig, height=170 + 42 * top_n)
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=120))
    return fig


def autoplay_html(fig: go.Figure, height: int) -> str:
    """A standalone page for an animated figure that starts playing on load.

    st.plotly_chart can't autoplay, so the race is rendered with plotly.js in an st.iframe. The iframe
    gets the site fonts and a transparent background matching the current light/dark mode.
    """
    body = fig.to_html(include_plotlyjs="cdn", full_html=False, auto_play=True, default_height=f"{height}px",
                       config={"displayModeBar": False, "responsive": True})
    fonts = ("https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;600&"
             "family=Source+Serif+4:opsz,wght@8..60,600&display=swap")
    return (f"<!doctype html><html><head><meta charset='utf-8'><link rel='stylesheet' href='{fonts}'>"
            f"<style>:root{{color-scheme:{theme.mode()}}} html,body{{margin:0;background:{theme.tokens()['surface']}}}"
            "</style>"
            f"</head><body>{body}</body></html>")


# --- Model spotlight and comparison ----------------------------------------------------------


def model_spotlight(df: pd.DataFrame, name: str, similar: pd.DataFrame, fit: GrowthFit | None) -> go.Figure:
    """Every model with known compute in gray, the chosen model as a star, its similar models ringed."""
    t = chart_tokens()
    known = df[df[COMPUTE_COL] > 0]
    me = known[known[MODEL_COL] == name]
    near = similar[similar[COMPUTE_COL] > 0]
    fig = go.Figure(go.Scatter(
        x=known[DATE_COL], y=known[COMPUTE_COL], mode="markers", name="All models",
        marker=dict(size=6, color=OTHER_COLOR, opacity=0.35), customdata=known[MODEL_COL],
        hovertemplate="%{customdata}<br>%{y:.2e} FLOP<extra></extra>",
    ))
    if fit is not None:
        xs = pd.date_range(pd.Timestamp(year=fit.start_year, month=1, day=1), known[DATE_COL].max(), periods=40)
        fig.add_trace(go.Scatter(
            x=xs, y=10 ** (fit.intercept + fit.slope * decimal_year(pd.Series(xs))), mode="lines",
            name=f"Trend since {fit.start_year}", line=dict(color=t["muted"], width=1.5, dash="dash"), hoverinfo="skip"))
    if not near.empty:
        fig.add_trace(go.Scatter(
            x=near[DATE_COL], y=near[COMPUTE_COL], mode="markers", name="Similar models",
            marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=2, color=palette()[0])),
            customdata=near[MODEL_COL], hovertemplate="<b>%{customdata}</b><br>%{y:.2e} FLOP<extra></extra>"))
    if not me.empty:
        fig.add_trace(go.Scatter(
            x=me[DATE_COL], y=me[COMPUTE_COL], mode="markers+text", name=name, text=[name],
            textposition="top center", textfont=dict(color=t["ink"], size=13),
            marker=dict(size=20, symbol="star", color=t["accent"], line=dict(width=1.5, color=t["surface"])),
            hovertemplate=f"<b>{name}</b><br>%{{y:.2e}} FLOP<extra></extra>"))
    fig.update_yaxes(type="log", title="Training compute (FLOP, log scale)", exponentformat="power")
    fig.update_xaxes(title=None)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return _base_layout(fig, height=420)


def compare_chart(table: pd.DataFrame, name_a: str, name_b: str) -> go.Figure | None:
    """Horizontal bars of B ÷ A per metric on a log axis centred on 1× (left = B smaller, right = B bigger).

    `table` has metric and ratio columns, plus an optional label column with the bar text.
    """
    d = table.dropna(subset=["ratio"])
    if d.empty:
        return None
    t = chart_tokens()
    colors = [palette()[0] if r >= 1 else palette()[1] for r in d["ratio"]]
    fig = go.Figure(go.Bar(
        # a bar spans base → base + x, so x = ratio - 1 draws from 1× to the ratio, in either direction
        x=d["ratio"] - 1, y=d["metric"], orientation="h", base=1, marker=dict(color=colors, cornerradius=3),
        text=d["label"] if "label" in d else [f"{r:,.3g}×" for r in d["ratio"]], textposition="outside",
        cliponaxis=False, textfont=dict(color=t["ink"]), customdata=d["ratio"],
        hovertemplate=f"{name_b} ÷ {name_a}<br>%{{y}}: %{{customdata:,.3g}}×<extra></extra>",
    ))
    span = max(1.0, float(np.abs(np.log10(d["ratio"])).max())) + 1.6  # room for the outside labels
    fig.add_vline(x=1, line=dict(color=t["muted"], width=1))
    fig.update_xaxes(type="log", range=[-span, span], title=f"{name_b} ÷ {name_a} (log scale)", exponentformat="power")
    fig.update_yaxes(title=None, autorange="reversed")
    return _base_layout(fig, height=110 + 70 * len(d))


# --- Training cost ---------------------------------------------------------------------------


def cost_scatter(df: pd.DataFrame, domains: list[str], fit: GrowthFit | None,
                 references: list[tuple[float, str]]) -> go.Figure:
    """Training cost over time (log scale), coloured by domain, with familiar price tags as reference lines."""
    t = chart_tokens()
    d = df[df[COST_COL] > 0].assign(domain=lambda x: fold_other(x["primary_domain"], domains))
    colors = color_map(domains + ["Other"])
    fig = go.Figure()
    for dom in domains + ["Other"]:
        s = d[d["domain"] == dom]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s[DATE_COL], y=s[COST_COL], mode="markers", name=dom,
            marker=dict(size=9, color=colors[dom], opacity=0.85, line=dict(width=1, color=t["surface"])),
            customdata=np.stack([s[MODEL_COL], s["primary_org"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        ))
    if fit is not None:
        xs = pd.date_range(pd.Timestamp(year=fit.start_year, month=1, day=1), d[DATE_COL].max(), periods=40)
        fig.add_trace(go.Scatter(
            x=xs, y=10 ** (fit.intercept + fit.slope * decimal_year(pd.Series(xs))), mode="lines",
            name=f"Trend (~{fit.factor_per_year:.1f}×/yr)", line=dict(color=t["ink"], width=2, dash="dash"),
            hoverinfo="skip"))
    for value, label in references:
        fig.add_hline(y=value, line=dict(color=t["rule"], width=1, dash="dot"), layer="below")
        fig.add_annotation(x=0, xref="paper", y=np.log10(value), yref="y", text=f"${value:,.0f} · {label}",
                           showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=11, color=t["muted"]),
                           bgcolor=t["label_bg"])
    fig.update_yaxes(type="log", title="Training compute cost (2023 USD, log scale)", tickprefix="$",
                     exponentformat="SI")
    fig.update_xaxes(title=None)
    fig.update_layout(legend=dict(itemsizing="constant"))
    return _base_layout(fig, height=540)


# --- Environmental footprint -----------------------------------------------------------------


def log_over_time(df: pd.DataFrame, col: str, y_title: str, domains: list[str], fit: GrowthFit | None,
                  references: list[tuple[float, str]] = (), hover_value: str = "%{y:,.3s}",
                  height: int = 460) -> go.Figure:
    """Generic log-scale scatter of one column over time, coloured by domain, with an optional trend line
    and labelled horizontal reference lines (e.g. 'one US home for a year')."""
    t = chart_tokens()
    d = df[df[col] > 0].assign(domain=lambda x: fold_other(x["primary_domain"], domains))
    colors = color_map(domains + ["Other"])
    fig = go.Figure()
    for dom in domains + ["Other"]:
        s = d[d["domain"] == dom]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s[DATE_COL], y=s[col], mode="markers", name=dom,
            marker=dict(size=9, color=colors[dom], opacity=0.85, line=dict(width=1, color=t["surface"])),
            customdata=np.stack([s[MODEL_COL], s["primary_org"]], axis=-1),
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>%{{customdata[1]}} · %{{x|%b %Y}}<br>{hover_value}<extra></extra>",
        ))
    if fit is not None and not d.empty:
        xs = pd.date_range(pd.Timestamp(year=fit.start_year, month=1, day=1), d[DATE_COL].max(), periods=40)
        fig.add_trace(go.Scatter(
            x=xs, y=10 ** (fit.intercept + fit.slope * decimal_year(pd.Series(xs))), mode="lines",
            name=f"Trend (~{fit.factor_per_year:.1f}×/yr)", line=dict(color=t["ink"], width=2, dash="dash"),
            hoverinfo="skip"))
    for value, label in references:
        fig.add_hline(y=value, line=dict(color=t["rule"], width=1, dash="dot"), layer="below")
        fig.add_annotation(x=0, xref="paper", y=np.log10(value), yref="y", text=label, showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11, color=t["muted"]), bgcolor=t["label_bg"])
    fig.update_yaxes(type="log", title=y_title, exponentformat="power")
    fig.update_xaxes(title=None)
    fig.update_layout(legend=dict(itemsizing="constant"))
    return _base_layout(fig, height=height)
