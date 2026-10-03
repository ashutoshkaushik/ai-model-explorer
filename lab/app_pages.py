"""App pages: the Explorer (charts) and Ask the Data (Claude tool use)."""

import json

import streamlit as st

from ui import chrome
from ui.filters import base, empty_message, sci, sci_compact, sidebar_filters
from utils.charts import METRICS, domain_area, fit_growth, open_donut, open_share_line, over_time_scatter, top_orgs_bar
from utils.data_loader import COMPUTE_COL, DATE_COL, MODEL_COL
from utils.llm import LLMError, ask, era_stats, era_summary, get_client


def explorer_page() -> None:
    f = sidebar_filters()
    b = base()
    df = f.df
    st.title("Explorer")
    st.markdown("<div class='lede'>How notable AI models have grown in scale, compute and cost. Every chart follows "
                "the filters in the sidebar.</div>", unsafe_allow_html=True)
    if df.empty:
        empty_message()
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Models", f"{len(df):,}", border=True)
    k2.metric("Years", f"{df['year'].min()}–{df['year'].max()}", border=True,
              help=f"Published {df[DATE_COL].min():%B %-d, %Y} to {df[DATE_COL].max():%B %-d, %Y}")
    k3.metric("Organizations", f"{df['primary_org'].nunique():,}", border=True)
    if df[COMPUTE_COL].notna().any():
        biggest = df.loc[df[COMPUTE_COL].idxmax()]
        k4.metric("Top compute", f"{sci_compact(biggest[COMPUTE_COL])} FLOP", border=True,
                  help=f"{sci(biggest[COMPUTE_COL])} FLOP · {biggest[MODEL_COL]} ({biggest['primary_org']}, "
                       f"{biggest[DATE_COL]:%b %Y})")
    else:
        k4.metric("Top compute", "—", border=True)

    st.markdown("### Compute over time")
    ctl1, ctl2 = st.columns([3, 1])
    metric = ctl1.segmented_control("Y-axis metric", list(METRICS), default="Training compute", key="metric") \
        or "Training compute"
    show_milestones = ctl2.toggle("Show milestones", value=True)
    metric_col = METRICS[metric]["col"]
    plotted = df[df[metric_col] > 0]
    excluded = len(df) - len(plotted)
    fit = fit_growth(plotted, metric_col)
    if plotted.empty:
        st.info(f"None of the {len(df):,} filtered models report {metric.lower()}.")
    else:
        if fit:
            st.markdown(
                f"<div class='callout'><b>{METRICS[metric]['noun']} ~{fit.factor_per_year:.1f}× per year</b> "
                f"<span class='muted'>(log-linear fit over {fit.n:,} models published since {fit.start_year})</span>"
                "</div>", unsafe_allow_html=True)
        st.plotly_chart(
            over_time_scatter(plotted, metric, fit, b["color_domains"], b["milestones"] if show_milestones else None),
            width="stretch",
        )
        st.caption(f"{excluded:,} models excluded (missing data). Marker size ∝ log(parameters). Shaded bands mark "
                   "eras; hover ◆ markers for milestones without room for a label.")
    chrome.credit()

    st.markdown("### Who builds what")
    c1, c2 = st.columns(2)
    c1.plotly_chart(top_orgs_bar(df), width="stretch")
    c2.plotly_chart(domain_area(df, b["color_domains"]), width="stretch")
    c3, c4 = st.columns([2, 3])
    c3.plotly_chart(open_donut(df), width="stretch")
    share_fig = open_share_line(df)
    if share_fig:
        c4.plotly_chart(share_fig, width="stretch")
    else:
        c4.info("Not enough models with known accessibility to chart open share over time.")
    chrome.credit()


EXAMPLE_QUESTIONS = [
    "Which organizations have released the most models?",
    "What are the 5 largest models by training compute?",
    "How fast has training compute grown since 2010?",
    "How has the share of open-weight models changed over time?",
]


def set_question(q: str) -> None:
    st.session_state["ask_q"] = q
    st.session_state["ask_submit"] = True


def ask_page() -> None:
    f = sidebar_filters()
    df = f.df
    st.title("Ask the Data")
    st.markdown(
        "<div class='lede'>Ask in plain English. <span class='badge llm'>Claude</span> picks one of nine safe query "
        "functions, <span class='badge code'>code</span> runs it on the filtered data, and Claude summarizes the "
        "result. No generated code is ever executed.</div>", unsafe_allow_html=True)
    if df.empty:
        empty_message()
        return
    client = get_client()
    if client is None:
        st.info("Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` (see `secrets.toml.example`), or to the app's "
                "secrets on Streamlit Community Cloud, to enable questions and era summaries.", icon=":material/key:")
        chrome.credit()
        return

    st.markdown("#### Try an example")
    ex_cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        ex_cols[i % 2].button(q, on_click=set_question, args=(q,), width="stretch")
    with st.form("ask_form", border=False):
        question = st.text_input("Your question", key="ask_q", placeholder="e.g. Which domain grew fastest after 2020?")
        submitted = st.form_submit_button("Ask", type="primary", icon=":material/send:")
    if (submitted or st.session_state.pop("ask_submit", False)) and question.strip():
        try:
            with st.spinner("Asking Claude…"):
                st.session_state["ask_result"] = (question, ask(client, df, question, f.note))
        except LLMError as e:
            st.error(str(e))
    if "ask_result" in st.session_state:
        asked, result = st.session_state["ask_result"]
        with st.container(border=True):
            st.markdown(f"**{asked}**")
            st.markdown(result.answer)
        with st.expander(f"How this was answered ({len(result.calls)} function call(s))", icon=":material/code:"):
            if not result.calls:
                st.caption("Claude answered without calling a query function.")
            for call in result.calls:
                st.code(f"{call.name}({json.dumps(call.args)})", language="python")
                if call.error:
                    st.error(call.error)
                elif call.result is not None:
                    st.dataframe(call.result, hide_index=True, width="stretch")

    st.divider()
    st.markdown("### Era summary")
    st.caption(f"A short narrative of AI in {f.years[0]}–{f.years[1]}, based on aggregated stats for the current filters.")
    if st.button("Summarize this era", icon=":material/history_edu:"):
        try:
            with st.spinner("Writing era summary…"):
                st.session_state["era_result"] = (f.note, era_summary(client, era_stats(df, f.years), f.note))
        except LLMError as e:
            st.error(str(e))
    if "era_result" in st.session_state:
        note, text = st.session_state["era_result"]
        with st.container(border=True):
            st.markdown(text)
            st.caption(f"Filters: {note}")
    chrome.credit()
