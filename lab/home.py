"""'Start here': the one-picture idea of the whole app, then where to go next."""

import html
import random
import time

import numpy as np
import pandas as pd
import streamlit as st

from lab.nav import PAGES, go_button
from ui import chrome
from ui.filters import base, sci, sci_compact
from ui.theme import FONT_BODY, FONT_HEADING, mode, tokens
from utils.charts import fit_growth
from utils.data_loader import COMPUTE_COL, DATE_COL, MODEL_COL
from utils.insights import growth_per_second, growth_since, load_changes, recent_models

ALEXNET_FLOP = 4.7e17  # fallback if AlexNet is ever missing from the data

ERAS = [  # (name, first year, last year, one line on what changed)
    ("Pre-deep learning", None, 2009, "Perceptrons, expert systems, early neural nets. Compute roughly followed Moore's law."),
    ("Deep learning", 2010, 2017, "GPUs and ImageNet. AlexNet, AlphaGo and the Transformer; compute started doubling every few months."),
    ("Large-scale / LLM era", 2018, None, "Foundation models trained by a few labs on enormous clusters, with budgets in the hundreds of millions."),
]


def ladder_html(df) -> str:
    rungs = []
    for i, (name, start, end, line) in enumerate(ERAS, start=1):
        era = df[df["year"].between(start or 0, end or 9999)]
        known = era[era[COMPUTE_COL] > 0]
        years = f"{start or int(era['year'].min())}–{end or int(era['year'].max())}"
        top = known.loc[known[COMPUTE_COL].idxmax()] if not known.empty else None
        peak = f"peak {sci(top[COMPUTE_COL])} FLOP ({top[MODEL_COL]})" if top is not None else "compute not reported"
        this = i == len(ERAS)
        rungs.append(
            f"<div class='rung{' this' if this else ''}'>"
            + ("<span class='here badge data'>today</span>" if this else "")
            + f"<div class='n'>{i}</div><div class='t'>{name}</div>"
            f"<div class='flow'>{years} · {len(era):,} models</div>"
            f"<div class='adds'>{line}</div><div class='flow' style='margin-top:.45rem'>{peak}</div></div>")
    return f"<div class='ladder'>{''.join(rungs)}</div>"


SPARK_W, SPARK_H, PAD = 760, 220, 20
END_ROOM = 185  # space to the right of the last point for the record holder's name


def record_series(df: pd.DataFrame) -> pd.DataFrame:
    """Largest training compute published up to each year (the running record), one row per year with data."""
    known = df[df[COMPUTE_COL] > 0]
    top = known.loc[known.groupby("year")[COMPUTE_COL].idxmax()].sort_values("year")
    return top.assign(record=top[COMPUTE_COL].cummax())


def sparkline_svg(df: pd.DataFrame) -> str:
    """An SVG of the record compute from the first year to the last that draws itself on load (CSS only).

    The latest record holder is highlighted at the end. Colours come from the theme tokens, so it matches
    light and dark mode; animations are skipped for visitors who prefer reduced motion.
    """
    t = tokens()
    rec = record_series(df)
    years = rec["year"].to_numpy(dtype=float)
    logs = np.log10(rec["record"].to_numpy(dtype=float))
    y0, y1 = float(df["year"].min()), float(df["year"].max())
    lo, hi = np.floor(logs.min()), np.ceil(logs.max())
    sx = lambda y: PAD + (y - y0) / (y1 - y0) * (SPARK_W - 2 * PAD - END_ROOM)  # noqa: E731
    sy = lambda v: SPARK_H - PAD - 18 - (v - lo) / (hi - lo) * (SPARK_H - 2 * PAD - 34)  # noqa: E731
    # step line: the record holds until the next year that beats it
    pts = [(sx(years[0]), sy(logs[0]))]
    for i in range(1, len(years)):
        pts += [(sx(years[i]), sy(logs[i - 1])), (sx(years[i]), sy(logs[i]))]
    line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    base_y = SPARK_H - PAD - 18
    area = f"{line} L{pts[-1][0]:.1f},{base_y} L{pts[0][0]:.1f},{base_y} Z"
    ex, ey = pts[-1]
    last, first = rec.iloc[-1], rec.iloc[0]
    ticks = "".join(f"<text x='{sx(y):.1f}' y='{SPARK_H - 4}' class='sp-tick' text-anchor='middle'>{y}</text>"
                    for y in (int(y0), 1970, 1990, 2010, int(y1)) if y0 <= y <= y1)
    accent, data = t["accent"], t["data"]
    css = (f"<style>"
           f".sp-line{{fill:none;stroke:{data};stroke-width:2.6;stroke-linejoin:round;stroke-dasharray:1;"
           f"stroke-dashoffset:1;animation:sp-draw 1.9s cubic-bezier(.5,0,.2,1) forwards}}"
           f".sp-area{{fill:url(#sp-fade);opacity:0;animation:sp-show .9s ease-out 1.3s forwards}}"
           f".sp-end{{opacity:0;animation:sp-show .5s ease-out 1.8s forwards}}"
           f".sp-ring{{fill:none;stroke:{accent};stroke-width:2;transform-origin:{ex:.1f}px {ey:.1f}px;"
           f"animation:sp-pulse 1.8s ease-out 2.2s infinite}}"
           f".sp-tick{{font:13px {FONT_BODY};fill:{t['muted']}}}"
           f".sp-name{{font:600 21px {FONT_HEADING};fill:{t['ink']}}}"
           f".sp-val{{font:14.5px {FONT_BODY};fill:{t['muted']}}}"
           f".sp-start{{font:13.5px {FONT_BODY};fill:{t['muted']}}}"
           f"@keyframes sp-draw{{to{{stroke-dashoffset:0}}}}"
           f"@keyframes sp-show{{to{{opacity:1}}}}"
           f"@keyframes sp-pulse{{0%{{transform:scale(1);opacity:.9}}100%{{transform:scale(3.2);opacity:0}}}}"
           f"@media (prefers-reduced-motion: reduce){{.sp-line,.sp-area,.sp-end,.sp-ring{{animation:none;stroke-dashoffset:0;"
           f"opacity:1}}}}</style>")
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {SPARK_W} {SPARK_H}'>{css}"
        f"<defs><linearGradient id='sp-fade' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='{data}' "
        f"stop-opacity='.28'/><stop offset='1' stop-color='{data}' stop-opacity='0'/></linearGradient></defs>"
        f"<line x1='{PAD}' x2='{SPARK_W - PAD - END_ROOM}' y1='{base_y}' y2='{base_y}' stroke='{t['line']}'/>{ticks}"
        f"<path class='sp-area' d='{area}'/><path class='sp-line' pathLength='1' d='{line}'/>"
        f"<text x='{PAD}' y='{PAD}' class='sp-start'>Started at {sci(first['record'])} FLOP: "
        f"{html.escape(first[MODEL_COL])}, {int(first['year'])}</text>"
        f"<g class='sp-end'><circle class='sp-ring' cx='{ex:.1f}' cy='{ey:.1f}' r='7'/>"
        f"<circle cx='{ex:.1f}' cy='{ey:.1f}' r='7' fill='{accent}' stroke='{t['card']}' stroke-width='2.5'/>"
        f"<text x='{ex + 16:.1f}' y='{ey + 2:.1f}' class='sp-name'>{html.escape(last[MODEL_COL])}</text>"
        f"<text x='{ex + 16:.1f}' y='{ey + 24:.1f}' class='sp-val'>{sci(last['record'])} FLOP · {int(last['year'])}"
        f"</text></g></svg>")


def hero_chart(df: pd.DataFrame, fit) -> None:
    rec = record_series(df)
    first, last = rec.iloc[0], rec.iloc[-1]
    alt = (f"Record training compute from {int(first['year'])} to {int(last['year'])}, rising from "
           f"{sci(first['record'])} FLOP to {sci(last['record'])} FLOP ({last[MODEL_COL]}).")
    chips = [
        (f"{len(df):,} models", f"{len(df):,} notable models, {df[DATE_COL].min():%Y}–{df[DATE_COL].max():%Y}"),
        (f"{df['primary_org'].nunique():,} organizations", "Distinct primary organizations"),
        (f"~{fit.factor_per_year:.1f}× a year", f"Compute growth, log-linear fit over {fit.n:,} models since "
                                               f"{fit.start_year}") if fit else None,
        (f"{sci_compact(last['record'] / first['record'])}× since {int(first['year'])}",
         f"The record grew from {sci(first['record'])} to {sci(last['record'])} FLOP"),
    ]
    chip_html = "".join(f"<span title='{html.escape(full)}'>{html.escape(short)}</span>" for short, full in filter(None, chips))
    # Inline SVG (not an <img>): CSS animations inside SVG images don't run in every browser
    st.markdown(
        f"<div class='hero-chart' data-mode='{mode()}'><div class='hc-label'>Record training compute, "
        f"{int(df['year'].min())} → {int(df['year'].max())} (log scale)</div>"
        f"<div class='hc-svg' role='img' aria-label='{html.escape(alt)}'>{sparkline_svg(df)}</div>"
        f"<div class='hc-chips'>{chip_html}</div></div>", unsafe_allow_html=True)


def random_model_button(df: pd.DataFrame) -> None:
    """Dice: open the profile of a random model with known compute (so its card has a chart)."""
    if "model" in PAGES and st.button("🎲 Pick a random model", width="stretch", key="dice"):
        name = random.choice(df.loc[df[COMPUTE_COL] > 0, MODEL_COL].tolist())
        st.switch_page(PAGES["model"], query_params={"model": name})


@st.fragment(run_every="1s")
def live_counter(fit, alexnet_flop: float, record_flop: float, record_name: str) -> None:
    """Ticks every second: how much the compute trend has grown since this visitor opened the page."""
    opened = st.session_state.setdefault("opened_at", time.time())
    seconds = time.time() - opened
    grown = growth_since(fit, seconds) * 100
    per_second = record_flop * growth_per_second(fit)
    mins, secs = divmod(int(seconds), 60)
    added = record_flop * growth_since(fit, seconds) / alexnet_flop
    st.markdown(
        "<div class='live'>"
        f"<div class='live-row'><div><div class='live-k'><i class='live-dot'></i>Trend growth since you arrived</div>"
        f"<div class='tick'>+{grown:.7f}%</div></div>"
        f"<div><div class='live-k'>AlexNets of compute added at the record's pace</div>"
        f"<div class='tick'>{added:,.0f}</div></div></div>"
        f"<div class='lede'>How much the training-compute trend has grown since you opened this page "
        f"<span class='badge data'>{mins}m {secs:02d}s ago</span>. If the largest run so far ({html.escape(record_name)}) "
        f"kept growing at {fit.factor_per_year:.1f}× per year, it would add about "
        f"<b>{per_second / alexnet_flop:,.0f} AlexNets</b> of training compute every second. "
        "</div></div>", unsafe_allow_html=True)


def whats_new(df) -> None:
    newest = df[DATE_COL].max()
    recent = recent_models(df, days=90)
    st.markdown("#### What's new in the data")
    st.markdown(f"<div class='muted'>Snapshot runs to <b>{newest:%B %-d, %Y}</b>. {len(recent)} models were published "
                "in its last 90 days; the latest are below.</div>", unsafe_allow_html=True)
    cards = []
    for _, r in recent.head(6).iterrows():
        compute = f" · {sci(r[COMPUTE_COL])} FLOP" if pd.notna(r[COMPUTE_COL]) else ""
        cards.append(f"<div class='nugget'><div class='tag'>{r[DATE_COL]:%b %-d, %Y}</div>"
                     f"<b>{html.escape(r[MODEL_COL])}</b><span>{html.escape(r['primary_org'])} · "
                     f"{html.escape(r['primary_domain'])}{compute}</span></div>")
    st.markdown(f"<div class='nuggets three'>{''.join(cards)}</div>", unsafe_allow_html=True)
    changes = load_changes()
    if changes and changes.get("added"):
        added = changes["added"]
        st.markdown(f"<div class='callout'><b>{len(added)} models added</b> in the last data refresh "
                    f"({changes['refreshed']}): " + ", ".join(html.escape(m) for m in added[:12])
                    + (f" and {len(added) - 12} more" if len(added) > 12 else "") + ".</div>",
                    unsafe_allow_html=True)


def home_page() -> None:
    b = base()
    df = b["df_all"]
    fit = fit_growth(df, COMPUTE_COL)

    st.markdown("<div class='eyebrow'>AI Model Evolution Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero'>Seventy years of AI models, and how fast they've grown</div>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div class='lede'>Explore {len(df):,} notable AI models from {b['min_year']} to {b['max_year']}: training "
        "compute, parameters, cost, who builds them, and how open they are. Filter anything, ask questions in plain "
        "English, and see how each piece was built in the Explorer Lab.</div>", unsafe_allow_html=True)
    hero_chart(df, fit)
    biggest = df.loc[df[COMPUTE_COL].idxmax()]
    a, b, c = st.columns(3)
    with a:
        random_model_button(df)
    with b:
        go_button("race", "▶ Watch the compute race", primary=True, button_key="hero_race")
    with c:
        go_button("explorer", "Open the explorer", button_key="hero_explorer")

    if fit:
        st.markdown("#### Happening right now")
        alexnet = df.loc[df[MODEL_COL] == "AlexNet", COMPUTE_COL]
        live_counter(fit, float(alexnet.iloc[0]) if alexnet.notna().any() else ALEXNET_FLOP,
                     float(biggest[COMPUTE_COL]), biggest[MODEL_COL])

    st.markdown("#### Three eras of AI, in one dataset")
    st.markdown(ladder_html(df), unsafe_allow_html=True)
    chrome.credit()

    whats_new(df)
    chrome.credit()

    st.markdown("#### Start exploring")
    a, b, c = st.columns(3)
    with a:
        go_button("race", "▶ Watch the compute race", primary=True)
    with b:
        go_button("model", "Find a model")
    with c:
        go_button("compare", "Compare two models")
    a, b, c = st.columns(3)
    with a:
        go_button("cost", "What training costs")
    with b:
        go_button("surprise", "Surprise me with a fact")
    with c:
        go_button("explorer", "Open the full explorer")

    st.markdown("#### Who does what in this app")
    st.markdown(
        "<div class='steps'>"
        "<div class='data'><b>The data</b><span>Epoch AI's Notable AI Models: one row per model, with publication "
        "date, organization, domain, parameters, training compute and cost.</span></div>"
        "<div class='code'><b>Code guarantees</b><span>Loading, cleaning, filters, the growth-rate fit and nine "
        "predefined query functions. Every number on screen comes from here.</span></div>"
        "<div class='llm'><b>Claude explains</b><span>Picks which query function answers your question, then "
        "summarizes the result. It never writes or runs code.</span></div>"
        "</div>", unsafe_allow_html=True)

    st.markdown("")
    a, c = st.columns(2)
    with a:
        go_button("ask", "Ask the data")
    with c:
        go_button("lab_data", "Tour the Explorer Lab")
