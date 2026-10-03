"""'Start here': the one-picture idea of the whole app, then where to go next."""

import html
import time

import pandas as pd
import streamlit as st

from lab.nav import go_button
from ui import chrome
from ui.filters import base, sci
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


@st.fragment(run_every="1s")
def live_counter(fit, alexnet_flop: float, record_flop: float, record_name: str) -> None:
    """Ticks every second: how much the compute trend has grown since this visitor opened the page."""
    opened = st.session_state.setdefault("opened_at", time.time())
    seconds = time.time() - opened
    grown = growth_since(fit, seconds) * 100
    per_second = record_flop * growth_per_second(fit)
    mins, secs = divmod(int(seconds), 60)
    st.markdown(
        "<div class='live'>"
        f"<div class='tick'>+{grown:.7f}%</div>"
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
    st.markdown("")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Models", f"{len(df):,}", border=True)
    k2.metric("Organizations", f"{df['primary_org'].nunique():,}", border=True)
    if fit:
        k3.metric("Compute growth", f"{fit.factor_per_year:.1f}×/yr", border=True,
                  help=f"Log-linear fit over {fit.n:,} models published since {fit.start_year}")
    biggest = df.loc[df[COMPUTE_COL].idxmax()]
    k4.metric("Largest training run", sci(biggest[COMPUTE_COL]), border=True,
              help=f"FLOP · {biggest[MODEL_COL]} ({biggest['primary_org']}, {biggest[DATE_COL]:%b %Y})")

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
