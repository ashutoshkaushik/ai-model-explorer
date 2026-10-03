"""Overview pages: the milestone timeline, the frontier leaderboard, the data browser, and how it's built."""

import streamlit as st

from ui import chrome
from ui.filters import base, empty_message, sci, sidebar_filters
from utils.charts import leaderboard_step
from utils.data_loader import (
    ACCESS_COL, COMPUTE_COL, COST_COL, DATE_COL, MODEL_COL, PARAMS_COL, frontier_by_year, models_near,
)
from utils.llm import QUERIES


def timeline_page() -> None:
    f = sidebar_filters()
    st.title("Milestones in AI")
    st.markdown("<div class='lede'>From the Perceptron to today's frontier runs. Each card shows the top three models by "
                "training compute released within ±6 months (current filters apply).</div>", unsafe_allow_html=True)
    milestones = base()["milestones"]
    milestones = milestones[milestones["date"].dt.year.between(*f.years)]
    if milestones.empty:
        st.info("No milestones fall inside the selected year range.")
    cols = st.columns(2)
    for i, (_, m) in enumerate(milestones.iterrows()):
        with cols[i % 2].container(border=True):
            st.markdown(f"#### {m['title']}")
            st.markdown(f"<span class='badge data'>{m['date']:%B %Y}</span>", unsafe_allow_html=True)
            st.caption(m["description"])
            near = models_near(f.df, m["date"])
            if near.empty:
                st.markdown("_No models with known compute within ±6 months._")
            else:
                st.markdown("\n".join(
                    f"{rank}. **{r[MODEL_COL]}** ({r['primary_org']}, {r[DATE_COL]:%b %Y}) · {sci(r[COMPUTE_COL])} FLOP"
                    for rank, (_, r) in enumerate(near.iterrows(), start=1)))
    chrome.credit()


def leaderboard_page() -> None:
    f = sidebar_filters()
    st.title("Frontier leaderboard")
    st.markdown("<div class='lede'>The single highest-compute model of each year, and how many times bigger it was than "
                "the year before.</div>", unsafe_allow_html=True)
    top = frontier_by_year(f.df)
    if top.empty:
        st.info("No models with known training compute match the current filters.")
        chrome.credit()
        return
    st.plotly_chart(leaderboard_step(top), width="stretch")
    st.dataframe(
        top.sort_values("year", ascending=False)
        .assign(**{COMPUTE_COL: lambda t: t[COMPUTE_COL].map("{:.2e}".format)})[
            ["year", MODEL_COL, "primary_org", "primary_domain", COMPUTE_COL, "growth"]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "year": st.column_config.NumberColumn("Year", format="%d"),
            "primary_org": "Organization",
            "primary_domain": "Domain",
            COMPUTE_COL: "Compute (FLOP)",
            "growth": st.column_config.NumberColumn(
                "× previous year's top", format="%.1f×", help="Ratio to the top model of the previous listed year"),
        },
    )
    chrome.credit()


def browse_page() -> None:
    f = sidebar_filters()
    df = f.df
    st.title("Browse the data")
    st.markdown("<div class='lede'>Every model that matches the sidebar filters. Search by name, organization or domain, "
                "and download what you see as CSV.</div>", unsafe_allow_html=True)
    if df.empty:
        empty_message()
        return
    left, right = st.columns([3, 2], vertical_alignment="bottom")
    query = left.text_input("Search", placeholder="Model, organization, or domain…")
    right.download_button(
        f"Download filtered data ({len(df):,} rows, CSV)",
        df.drop(columns=["year"]).to_csv(index=False).encode("utf-8"),
        file_name="notable_ai_models_filtered.csv", mime="text/csv", icon=":material/download:", width="stretch",
    )
    table = df[[MODEL_COL, "primary_org", "primary_domain", DATE_COL, PARAMS_COL, COMPUTE_COL, COST_COL, ACCESS_COL]]
    if query:
        haystack = table[[MODEL_COL, "primary_org", "primary_domain"]].astype(str).agg(" ".join, axis=1)
        table = table[haystack.str.contains(query, case=False, regex=False)]
    st.caption(f"{len(table):,} models")
    st.dataframe(
        table, hide_index=True, width="stretch", height=560,
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
    chrome.credit()


FILES = [
    ("app.py", "code", "Registers the pages and the App / Overview / Lab navigation. No logic."),
    ("utils/data_loader.py", "data", "Loads and cleans the CSV (cached), derives primary org/domain and open status, applies filters."),
    ("utils/charts.py", "code", "Every Plotly chart: takes a DataFrame, returns a figure. Colours follow the site theme."),
    ("utils/llm.py", "llm", "The nine safe query functions, their JSON schemas, and the Claude tool-use loop."),
    ("ui/filters.py", "code", "Sidebar filters shared by every data page; values persist across pages."),
    ("ui/theme.py · ui/chrome.py", "code", "Design tokens and CSS (same as the RAG and Travel Agent projects), author card, footer."),
    ("utils/insights.py", "code", "Model profiles, similar models, comparisons, the race frames and the live growth counter."),
    ("lab/", "code", "One module per page group: home, app and exploration pages, overview, the Explorer Lab, Surprise me."),
    ("scripts/download_data.py", "data", "Downloads the Epoch AI CSV (with a mirror fallback)."),
    ("tests/", "code", "Offline tests: the growth fit and the tool-use loop with a fake Claude client."),
]


def about_page() -> None:
    df = base()["df_all"]
    st.title("How it's built")
    st.markdown("<div class='lede'>A small, layered Streamlit app: data in, cached and cleaned once; code computes every "
                "number; Claude only chooses among predefined queries and writes the summary.</div>",
                unsafe_allow_html=True)

    st.markdown("### The flow of one question")
    st.markdown(
        "<div class='steps'>"
        "<div class='data'><b>1 · Filtered data</b><span>The sidebar narrows the cached DataFrame. The same filtered "
        "frame feeds every chart, table and Claude call.</span></div>"
        "<div class='llm'><b>2 · Claude picks a tool</b><span>It sees nine function names, descriptions and JSON "
        "schemas, and replies with one call such as <code>top_models_by_compute(n=5)</code>.</span></div>"
        "<div class='code'><b>3 · Code runs it</b><span>The app checks the name against a registry, drops unknown "
        "arguments, runs the pandas function and sends the rows back for a 2-3 sentence answer.</span></div>"
        "</div>", unsafe_allow_html=True)

    st.markdown("### Files")
    rows = "".join(f"<tr><td><code>{path}</code></td><td><span class='badge {kind}'>{kind}</span></td><td>{what}</td></tr>"
                   for path, kind, what in FILES)
    st.markdown(f"<div class='tablewrap'><table class='ltable'><thead><tr><th>File</th><th>Layer</th><th>What it does"
                f"</th></tr></thead><tbody>{rows}</tbody></table></div>", unsafe_allow_html=True)

    st.markdown("### The dataset")
    null = lambda col: f"{df[col].isna().mean():.0%}"  # noqa: E731
    st.markdown(
        f"Epoch AI's [Notable AI Models](https://epoch.ai/data/notable-ai-models): **{len(df):,} models**, "
        f"{df['year'].min()}–{df['year'].max()}. Parameters are missing for {null(PARAMS_COL)} of models, training "
        f"compute for {null(COMPUTE_COL)} and cost for {null(COST_COL)}, so each chart says how many models it had to "
        f"leave out. Claude can call **{len(QUERIES)} query functions**: "
        + ", ".join(f"`{name}`" for name in QUERIES) + ".")
    chrome.credit()
