import json
import math

import streamlit as st

from utils.charts import (
    METRICS,
    domain_area,
    fit_growth,
    leaderboard_step,
    open_donut,
    open_share_line,
    over_time_scatter,
    top_categories,
    top_orgs_bar,
)
from utils.data_loader import (
    ACCESS_COL,
    COMPUTE_COL,
    COST_COL,
    DATE_COL,
    MODEL_COL,
    PARAMS_COL,
    apply_filters,
    frontier_by_year,
    load_data,
    load_milestones,
    models_near,
    top_orgs,
)
from utils.llm import LLMError, ask, era_stats, era_summary, get_client

CREDIT = "Data: Epoch AI (CC-BY 4.0)"
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float) -> str:
    """1.0001e27 -> '1.0 × 10²⁷'."""
    exp = math.floor(math.log10(x))
    return f"{x / 10**exp:.1f} × 10{str(exp).translate(SUPERSCRIPT)}"


st.set_page_config(page_title="AI Model Evolution Explorer", page_icon="📈", layout="wide")

df_all = load_data()
milestones_all = load_milestones()
min_year, max_year = int(df_all["year"].min()), int(df_all["year"].max())
domain_options = df_all["primary_domain"].value_counts().index.tolist()
org_options = top_orgs(df_all, 15)
access_options = df_all[ACCESS_COL].fillna("Unknown").value_counts().index.tolist()
# Colors are assigned from the unfiltered data so a filter never repaints a domain.
color_domains = top_categories(df_all["primary_domain"])

DEFAULTS = {
    "f_years": (min_year, max_year),
    "f_domains": [],
    "f_orgs": [],
    "f_access": [],
    "f_frontier": False,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_filters() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


# --- Sidebar filters ---
with st.sidebar:
    header = st.empty()
    years = st.slider("Year range", min_year, max_year, key="f_years")
    domains = st.multiselect("Domain", domain_options, key="f_domains", placeholder="All domains")
    orgs = st.multiselect(
        "Organization", org_options + ["Other"], key="f_orgs", placeholder="All organizations",
        help="Top 15 organizations by model count; 'Other' covers the rest.",
    )
    access = st.multiselect("Accessibility", access_options, key="f_access", placeholder="Any accessibility")
    frontier_only = st.checkbox("Frontier models only", key="f_frontier")
    st.button("Reset filters", on_click=reset_filters, width="stretch")
    active = sum(st.session_state[k] != v for k, v in DEFAULTS.items())
    header.header(f"Filters · {active} active" if active else "Filters")

df = apply_filters(df_all, years, domains, orgs, org_options, access, frontier_only)
filter_note = "; ".join(
    part
    for part in [
        f"years {years[0]}-{years[1]}",
        domains and f"domains: {', '.join(domains)}",
        orgs and f"organizations: {', '.join(orgs)}",
        access and f"accessibility: {', '.join(access)}",
        frontier_only and "frontier models only",
    ]
    if part
)

st.title("AI Model Evolution Explorer")
st.markdown("How notable AI models have grown in scale, compute, and cost from 1950 to today.")

if df.empty:
    st.warning("No models match the current filters. Try widening the year range or clearing a filter.")
    st.caption(CREDIT)
    st.stop()

# --- KPIs ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Models", f"{len(df):,}", border=True)
k2.metric("Year range", f"{df['year'].min()}–{df['year'].max()}", border=True)
k3.metric("Organizations", f"{df['primary_org'].nunique():,}", border=True)
if df[COMPUTE_COL].notna().any():
    biggest = df.loc[df[COMPUTE_COL].idxmax()]
    k4.metric(
        "Largest training compute",
        sci(biggest[COMPUTE_COL]),
        help=f"FLOP · {biggest[MODEL_COL]} ({biggest['primary_org']}, {biggest[DATE_COL]:%b %Y})",
        border=True,
    )
else:
    k4.metric("Largest training compute", "—", border=True)

tab_overview, tab_timeline, tab_leaderboard, tab_ask = st.tabs(
    ["Overview", "Timeline", "Frontier Leaderboard", "Ask the Data"]
)

with tab_overview:
    # --- Compute over time ---
    st.subheader("Compute Over Time")
    ctl1, ctl2 = st.columns([3, 1])
    metric = ctl1.radio("Y-axis metric", list(METRICS), horizontal=True)
    show_milestones = ctl2.toggle("Show milestones", value=True)
    metric_col = METRICS[metric]["col"]
    plotted = df[df[metric_col] > 0]
    excluded = len(df) - len(plotted)
    fit = fit_growth(plotted, metric_col)
    if plotted.empty:
        st.info(f"None of the {len(df):,} filtered models report {metric.lower()}.")
    else:
        if fit:
            st.info(
                f"**{METRICS[metric]['noun']} ~{fit.factor_per_year:.1f}× per year** "
                f"(log-linear fit over {fit.n:,} models published since {fit.start_year})."
            )
        st.plotly_chart(
            over_time_scatter(
                plotted, metric, fit, color_domains, milestones_all if show_milestones else None
            ),
            width="stretch",
        )
        st.caption(
            f"{excluded:,} models excluded (missing data). Marker size ∝ log(parameters). "
            "Shaded bands mark eras; hover ◆ markers for milestones without room for a label."
        )

    # --- Breakdowns ---
    st.subheader("Who builds what")
    c1, c2 = st.columns(2)
    c1.plotly_chart(top_orgs_bar(df), width="stretch")
    c2.plotly_chart(domain_area(df, color_domains), width="stretch")
    c3, c4 = st.columns([2, 3])
    c3.plotly_chart(open_donut(df), width="stretch")
    share_fig = open_share_line(df)
    if share_fig:
        c4.plotly_chart(share_fig, width="stretch")
    else:
        c4.info("Not enough models with known accessibility to chart open share over time.")

    # --- Searchable table ---
    st.subheader("Browse models")
    st.download_button(
        f"Download filtered data ({len(df):,} rows, CSV)",
        df.drop(columns=["year"]).to_csv(index=False).encode("utf-8"),
        file_name="notable_ai_models_filtered.csv",
        mime="text/csv",
        icon=":material/download:",
    )
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
            PARAMS_COL: st.column_config.NumberColumn("Parameters", format="scientific"),
            COMPUTE_COL: st.column_config.NumberColumn("Compute (FLOP)", format="scientific"),
            COST_COL: st.column_config.NumberColumn("Cost (2023 USD)", format="dollar"),
            ACCESS_COL: "Accessibility",
        },
    )


# --- Timeline ---
with tab_timeline:
    st.subheader("Milestones in AI")
    st.caption("Each card shows the top 3 models by training compute released within ±6 months (current filters apply).")
    milestones = milestones_all[milestones_all["date"].dt.year.between(*years)]
    if milestones.empty:
        st.info("No milestones fall inside the selected year range.")
    cols = st.columns(2)
    for i, (_, m) in enumerate(milestones.iterrows()):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{m['title']}** · {m['date']:%B %Y}")
            st.caption(m["description"])
            near = models_near(df, m["date"])
            if near.empty:
                st.markdown("_No models with known compute within ±6 months._")
            else:
                st.markdown(
                    "\n".join(
                        f"{rank}. **{r[MODEL_COL]}** ({r['primary_org']}, {r[DATE_COL]:%b %Y}) · {sci(r[COMPUTE_COL])} FLOP"
                        for rank, (_, r) in enumerate(near.iterrows(), start=1)
                    )
                )

# --- Frontier leaderboard ---
with tab_leaderboard:
    st.subheader("Frontier Leaderboard")
    top = frontier_by_year(df)
    if top.empty:
        st.info("No models with known training compute match the current filters.")
    else:
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
                    "× previous year's top", format="%.1f×", help="Ratio to the top model of the previous listed year"
                ),
            },
        )
# --- Ask the Data ---
EXAMPLE_QUESTIONS = [
    "Which organizations have released the most models?",
    "What are the 5 largest models by training compute?",
    "How fast has training compute grown since 2010?",
    "How has the share of open-weight models changed over time?",
]


def set_question(q: str) -> None:
    st.session_state["ask_q"] = q
    st.session_state["ask_submit"] = True


with tab_ask:
    st.subheader("Ask the Data")
    st.caption(
        "Claude picks one of a fixed set of safe query functions, it runs on the filtered data, "
        "and Claude summarizes the result. No generated code is executed."
    )
    client = get_client()
    if client is None:
        st.info(
            "Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` (see `secrets.toml.example`) "
            "to enable questions and era summaries."
        )
    else:
        st.caption("Try an example:")
        ex_cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            ex_cols[i % 2].button(q, on_click=set_question, args=(q,), width="stretch")
        with st.form("ask_form", border=False):
            question = st.text_input("Your question", key="ask_q", placeholder="e.g. Which domain grew fastest after 2020?")
            submitted = st.form_submit_button("Ask", type="primary")
        if (submitted or st.session_state.pop("ask_submit", False)) and question.strip():
            try:
                with st.spinner("Asking Claude…"):
                    st.session_state["ask_result"] = (question, ask(client, df, question, filter_note))
            except LLMError as e:
                st.error(str(e))
        if "ask_result" in st.session_state:
            asked, result = st.session_state["ask_result"]
            st.markdown(f"**Q: {asked}**")
            st.markdown(result.answer)
            with st.expander(f"How this was answered ({len(result.calls)} function call(s))"):
                if not result.calls:
                    st.caption("Claude answered without calling a query function.")
                for call in result.calls:
                    st.code(f"{call.name}({json.dumps(call.args)})", language="python")
                    if call.error:
                        st.error(call.error)
                    elif call.result is not None:
                        st.dataframe(call.result, hide_index=True, width="stretch")

        st.divider()
        st.subheader("Era Summary")
        st.caption(f"A short narrative of AI in {years[0]}–{years[1]}, based on aggregated stats for the current filters.")
        if st.button("Summarize this era"):
            try:
                with st.spinner("Writing era summary…"):
                    st.session_state["era_result"] = (filter_note, era_summary(client, era_stats(df, years), filter_note))
            except LLMError as e:
                st.error(str(e))
        if "era_result" in st.session_state:
            note, text = st.session_state["era_result"]
            with st.container(border=True):
                st.markdown(text)
                st.caption(f"Filters: {note}")

st.divider()
st.caption(CREDIT)
