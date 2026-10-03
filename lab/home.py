"""'Start here': the one-picture idea of the whole app, then where to go next."""

import streamlit as st

from lab.nav import go_button
from ui import chrome
from ui.filters import base, sci
from utils.charts import fit_growth
from utils.data_loader import COMPUTE_COL, DATE_COL, MODEL_COL

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

    st.markdown("#### Three eras of AI, in one dataset")
    st.markdown(ladder_html(df), unsafe_allow_html=True)
    chrome.credit()

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
    a, c, d = st.columns(3)
    with a:
        go_button("explorer", "Open the explorer", primary=True)
    with c:
        go_button("ask", "Ask the data")
    with d:
        go_button("lab_data", "Tour the Explorer Lab")
