"""The Explorer Lab: one page per idea, in the order the app was built. Each page has what the step adds,
a small live experiment on the real data, the key code (read from the source), and what was learned."""

import ast
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab.nav import tour_footer
from ui import chrome
from ui.filters import base
from utils.charts import _base_layout, color_map, fit_growth, fold_other, palette
from utils.data_loader import (
    ACCESS_COL, COMPUTE_COL, DATE_COL, DOMAIN_COL, MODEL_COL, ORG_COL, PARAMS_COL,
)
from utils.llm import QUERIES, TOOLS, run_query

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_KEY_LINES = 24

# The few functions that carry each step, read from the source at render time so they can never go stale
KEY_CODE = {
    "lab_data": [("utils/data_loader.py", "_first_value"), ("utils/data_loader.py", "load_data")],
    "lab_log": [("utils/charts.py", "over_time_scatter")],
    "lab_growth": [("utils/charts.py", "fit_growth")],
    "lab_filters": [("utils/charts.py", "color_map"), ("utils/data_loader.py", "apply_filters")],
    "lab_milestones": [("utils/charts.py", "add_milestones"), ("utils/data_loader.py", "frontier_by_year")],
    "lab_tools": [("utils/llm.py", "run_query"), ("utils/llm.py", "ask")],
}


def key_code_snippet(rel_path: str, name: str) -> tuple[str, int, int]:
    """(code, first line, last line) of the function `name` in the file."""
    source = (PROJECT_ROOT / rel_path).read_text()
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    lines = source.splitlines()[first - 1:node.end_lineno]
    shown = lines[:MAX_KEY_LINES] + (["    # … (see the full function in the file)"] if len(lines) > MAX_KEY_LINES else [])
    return textwrap.dedent("\n".join(shown)), first, node.end_lineno


def key_code(key: str) -> None:
    with st.expander("The key code", icon=":material/code:"):
        for rel_path, name in KEY_CODE[key]:
            code, first, last = key_code_snippet(rel_path, name)
            st.code(code, language="python")
            st.caption(f"{rel_path} · lines {first}–{last}")


def header(step: int, title: str, lede: str) -> None:
    st.markdown(f"<div class='eyebrow'>Explorer Lab · step {step} of 6</div>", unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<div class='lede'>{lede}</div>", unsafe_allow_html=True)


def learn(goals: list[str], insights: list[str], qa: list[tuple[str, str]]) -> None:
    st.markdown("### Learn from this step")
    st.markdown("<div class='learnbox'><b>You'll see</b><ul>" + "".join(f"<li>{g}</li>" for g in goals)
                + "</ul></div>", unsafe_allow_html=True)
    for insight in insights:
        st.markdown(f"- {insight}")
    for q, a in qa:
        with st.expander(q, icon=":material/help:"):
            st.markdown(a)


def nuggets(key: str) -> None:
    """'Knowledge nuggets': short, general facts around this step, as a grid of cards at the bottom of the page."""
    st.markdown("### Knowledge nuggets")
    cards = "".join(f"<div class='nugget'><div class='tag'>{tag}</div><b>{title}</b><span>{body}</span></div>"
                    for tag, title, body in NUGGETS[key])
    st.markdown(f"<div class='nuggets'>{cards}</div>", unsafe_allow_html=True)


NUGGETS = {  # key: [(tag, headline, body)], 4 per Lab page
    "lab_data": [
        ("Data", "Tidy data has one rule per cell",
         "Hadley Wickham's 2014 \"Tidy Data\" paper: each variable is a column, each observation a row, each value a "
         "cell. A field like \"Meta AI,NYU\" breaks that rule, which is why it needs a decision before grouping."),
        ("pandas", "explode() is the other option",
         "<code>df.assign(org=df.Organization.str.split(\",\")).explode(\"org\")</code> gives one row per partner. Use it "
         "to credit every co-author lab; use the first value when totals must add up to the model count."),
        ("Bias", "Missing is not random",
         "Big, famous models are far more likely to report compute and cost. So an average over the known values skews "
         "high: it describes the documented models, not all models."),
        ("Streamlit", "Caching survives reruns, not restarts",
         "<code>@st.cache_data</code> keys the result on the function's code and arguments, and hands each caller a "
         "copy, so one page can't mutate another page's DataFrame."),
    ],
    "lab_log": [
        ("Scale", "10²⁵ FLOP, made tangible",
         "A GPU sustaining 10¹⁵ operations per second (one petaFLOP/s) would need about 10¹⁰ seconds, roughly 300 "
         "years, to do 10²⁵ operations alone. Frontier labs do it in months on tens of thousands of chips."),
        ("Reading", "Each gridline is ×10",
         "On a log axis, equal distances mean equal ratios. Going from 10²⁰ to 10²⁴ is not \"4 more\" but 10,000 "
         "times more."),
        ("Pitfall", "log(0) doesn't exist",
         "Zeros and negatives can't sit on a log axis; plotting libraries silently drop them. That's why the charts "
         "filter to values above zero and say how many models were left out."),
        ("Design", "Tufte's lie factor",
         "Edward Tufte: the size of an effect in the graphic should match its size in the data. A marker whose radius "
         "grew with raw parameters would make one trillion-parameter model swallow the chart."),
    ],
    "lab_growth": [
        ("Math", "Doubling time from growth rate",
         "Doubling time = ln 2 ÷ ln(growth per year). At 4.3× per year that's about 5.7 months; Moore's law "
         "(2× every ~2 years) is about 1.4× per year."),
        ("Context", "Epoch AI's own estimate",
         "Epoch AI estimates training compute for notable models has grown roughly 4–5× per year since 2010, in line "
         "with the fit on this page."),
        ("Statistics", "A log fit is a geometric fit",
         "Least squares on log10(value) assumes errors are multiplicative (\"off by 2×\"), not additive (\"off by 10¹⁸ "
         "FLOP\"). For quantities that span orders of magnitude, that's the right assumption."),
        ("Caution", "Trends are not forecasts",
         "Extrapolating 4× per year for another decade means 10⁶× more compute. Power, chips, data and money all "
         "push back; a fit describes the past, not a law of nature."),
    ],
    "lab_filters": [
        ("Accessibility", "1 in 12 men see colour differently",
         "About 8% of men and 0.5% of women of Northern European descent have red-green colour vision deficiency. "
         "Validated palettes keep neighbouring series distinguishable for them."),
        ("Perception", "Position beats colour",
         "Cleveland and McGill (1984) found position on a common scale is the most accurately read encoding; colour "
         "hue is best for telling categories apart, not for comparing amounts."),
        ("Design", "Seven is plenty",
         "Beyond about seven categorical colours, readers start confusing series. Folding the long tail into a gray "
         "\"Other\" keeps the important ones legible."),
        ("Streamlit", "The whole script reruns on every click",
         "Every widget change reruns the page from the top. <code>st.session_state</code> is what survives, which is "
         "how these filters follow you from page to page."),
    ],
    "lab_milestones": [
        ("History", "Three eras, from a 2022 paper",
         "Sevilla et al., \"Compute Trends Across Three Eras of Machine Learning\", split history into pre-deep "
         "learning, deep learning (from 2010) and a large-scale era from late 2015. This app starts that band at 2018 "
         "for a cleaner picture."),
        ("Design", "Label directly when you can",
         "A label next to the thing it names saves a trip to the legend. When labels would collide, keep the "
         "important ones and move the rest to hover."),
        ("Charts", "Step lines for records",
         "A step chart (<code>shape=\"hv\"</code>) says \"this value held until the next change\". A sloped line "
         "between yearly records would imply models in between that never existed."),
        ("pandas", "idxmax() picks the first tie",
         "If two models share a year's top compute, <code>idxmax</code> returns the first one it meets. The loader "
         "sorts newest first, so ties go to the later model."),
    ],
    "lab_tools": [
        ("Concept", "Tool use is structured output",
         "The model never runs anything. It emits a block naming a tool and JSON arguments; your code decides "
         "whether and how to run it, and sends the result back."),
        ("Security", "Prompt injection is the top LLM risk",
         "OWASP's Top 10 for LLM applications lists prompt injection first. A fixed registry of read-only functions "
         "limits what any injected instruction could make the app do."),
        ("JSON Schema", "additionalProperties: false",
         "Every tool schema here forbids unknown keys, and <code>run_query</code> drops them anyway. Two layers, "
         "because a schema is a request to the model, not a guarantee."),
        ("Prompting", "Descriptions are prompts",
         "Claude picks tools by reading their names and descriptions. Short, specific descriptions with an example "
         "(\"e.g. 'GPT-4'\") choose better than long ones."),
    ],
}


# --- 1 · Load and clean --------------------------------------------------------------------------


def lab_data() -> None:
    df = base()["df_all"]
    header(1, "Load and clean the data",
           "One CSV, 1,000+ rows, 47 columns. Before any chart, the loader parses dates, coerces numbers, and turns "
           "comma-separated fields into one value you can group by.")

    st.markdown("### Try it: raw field → derived column")
    multi = df[df[ORG_COL].str.contains(",", na=False) | df[DOMAIN_COL].str.contains(",", na=False)]
    st.caption(f"{len(multi):,} models list more than one organization or domain. Here are a few:")
    st.dataframe(
        multi[[MODEL_COL, ORG_COL, "primary_org", DOMAIN_COL, "primary_domain", ACCESS_COL, "open_status"]].head(8),
        hide_index=True, width="stretch",
        column_config={"primary_org": "→ primary_org", "primary_domain": "→ primary_domain", "open_status": "→ open_status"})

    st.markdown("### How much is missing?")
    cols = [PARAMS_COL, COMPUTE_COL, "Training compute cost (2023 USD)", ACCESS_COL]
    missing = pd.DataFrame({"Column": cols, "Missing": [df[c].isna().mean() for c in cols]})
    st.dataframe(missing, hide_index=True, column_config={
        "Missing": st.column_config.ProgressColumn("Missing", format="percent", min_value=0, max_value=1)})
    chrome.credit()
    key_code("lab_data")
    learn(
        ["Why multi-value fields need a rule before grouping", "How sparse the numeric columns really are",
         "Why loading is cached"],
        ["**Pick a rule and say it.** `primary_org` is the first listed organization; a split-and-explode would "
         "count a joint model once per partner. The charts use the first value so totals add up to the model count.",
         "**Missing data is the norm.** Half the models report no training compute and most report no cost, so every "
         "chart states how many models it had to leave out.",
         "**`@st.cache_data` runs the loader once per process,** not on every widget click; Streamlit reruns the "
         "whole script on each interaction."],
        [("Why not drop rows with missing compute up front?",
          "Different pages need different columns: a model without compute still counts on the organization and "
          "domain charts. Each chart filters for what it needs.")],
    )
    nuggets("lab_data")
    tour_footer("lab_data")


# --- 2 · Log scale -------------------------------------------------------------------------------


def lab_log() -> None:
    df = base()["df_all"]
    header(2, "Plot on a log scale",
           "Training compute spans more than 20 orders of magnitude. On a linear axis, everything before 2020 is a "
           "flat line at zero.")
    d = df[df[COMPUTE_COL] > 0]
    log = st.toggle("Log scale", value=False, key="lab_log_toggle")
    fig = go.Figure(go.Scatter(x=d[DATE_COL], y=d[COMPUTE_COL], mode="markers", marker=dict(color=palette()[0], size=6,
                    opacity=.7), customdata=d[MODEL_COL], hovertemplate="<b>%{customdata}</b><br>%{y:.2e} FLOP<extra></extra>"))
    fig.update_yaxes(type="log" if log else "linear", title="Training compute (FLOP)", exponentformat="power")
    st.plotly_chart(_base_layout(fig, height=420), width="stretch")
    span = np.log10(d[COMPUTE_COL].max()) - np.log10(d[COMPUTE_COL].min())
    st.caption(f"{len(d):,} models with known compute, spanning {span:.0f} orders of magnitude.")
    chrome.credit()
    key_code("lab_log")
    learn(
        ["Why exponential growth needs a log axis", "How marker size encodes a third variable"],
        ["**On a log axis, constant growth is a straight line.** That's what makes the trend line on the Explorer "
         "readable, and what lets a simple linear fit measure it (next step).",
         "**Marker size is log(parameters), rescaled to 5–22 px,** so a 1-trillion-parameter model is visibly bigger "
         "without hiding everything else."],
        [("Why `exponentformat=\"power\"`?", "Ticks read 10²⁴ instead of 1e+24 or 1,000,000,000,000,000,000,000,000.")],
    )
    nuggets("lab_log")
    tour_footer("lab_log")


# --- 3 · Growth fit ------------------------------------------------------------------------------


def lab_growth() -> None:
    df = base()["df_all"]
    header(3, "Fit the growth rate",
           "A straight line through log10(compute) against time. The slope is the growth rate: 10^slope × per year.")
    start = st.slider("Fit models published from", 1990, 2022, 2010, key="lab_growth_start")
    fit = fit_growth(df, COMPUTE_COL, start_year=start)
    if fit is None:
        st.info("Too few models with known compute after that year.")
    else:
        doubling = 12 * np.log(2) / np.log(fit.factor_per_year)
        a, b, c = st.columns(3)
        a.metric("Growth per year", f"{fit.factor_per_year:.1f}×", border=True)
        b.metric("Doubling time", f"{doubling:.1f} months", border=True)
        c.metric("Models in the fit", f"{fit.n:,}", border=True)
        st.caption("Try 1990 vs 2010: the deep-learning era grew far faster than the decades before it.")
    chrome.credit()
    key_code("lab_growth")
    learn(
        ["How a log-linear least-squares fit turns into '×N per year'", "How the fit was tested without the real data"],
        ["**`np.polyfit(year, log10(value), 1)`** is the whole fit. Growth per year is `10 ** slope`; doubling time is "
         "`log 2 / log(growth)`.",
         "**Test it on data where you know the answer.** `tests/test_growth.py` builds synthetic series growing exactly "
         "2×, 4× and 10× per year and checks the fit recovers them, then adds noise and pre-2010 outliers."],
        [("Why is the default start 2010?",
          "Before deep learning, compute grew far more slowly; one line over both eras fits neither. The Explorer's "
          "shaded eras show the break.")],
    )
    nuggets("lab_growth")
    tour_footer("lab_growth")


# --- 4 · Filters ---------------------------------------------------------------------------------


def lab_filters() -> None:
    b = base()
    df = b["df_all"]
    header(4, "Filters that never repaint",
           "When you filter to just Vision models, Vision should stay the same colour. Colours are assigned from the "
           "unfiltered data, in a fixed order, then kept.")
    pick = st.multiselect("Show only these domains", b["color_domains"], default=b["color_domains"][2:4],
                          key="lab_filter_domains")
    stable = color_map(b["color_domains"] + ["Other"])
    naive = color_map(pick + ["Other"])
    rows = "".join(
        f"<tr><td>{d}</td><td><span style='color:{stable[d]}'>●</span> <code>{stable[d]}</code></td>"
        f"<td><span style='color:{naive[d]}'>●</span> <code>{naive[d]}</code></td></tr>" for d in pick)
    st.markdown("<div class='tablewrap'><table class='ltable'><thead><tr><th>Domain</th><th>Assigned from all data "
                "(what the app does)</th><th>Assigned from the filtered data</th></tr></thead><tbody>"
                f"{rows}</tbody></table></div>", unsafe_allow_html=True)
    other = fold_other(df["primary_domain"], b["color_domains"]).eq("Other").sum()
    st.caption(f"Beyond the top {len(b['color_domains'])} domains, {other:,} models fold into a gray 'Other'.")
    chrome.credit()
    key_code("lab_filters")
    learn(
        ["Why colour must be a property of the category, not of its rank in the current view",
         "How empty filters mean 'no filter'"],
        ["**Colour is identity.** If filtering repaints Vision from green to blue, every chart the reader already "
         "looked at becomes misleading.",
         "**More than 7 colours stop being distinguishable,** so the long tail folds into gray 'Other'.",
         "**Filters live in session state,** so they carry over between the Explorer, Timeline, Leaderboard and Ask "
         "the Data pages."],
        [("Why is 'Other' an option in the organization filter?",
          "The filter lists the top 15 organizations; 'Other' matches everyone else, so you can still select the "
          "long tail.")],
    )
    nuggets("lab_filters")
    tour_footer("lab_filters")


# --- 5 · Milestones and eras ---------------------------------------------------------------------


def lab_milestones() -> None:
    b = base()
    header(5, "Milestones, eras and the leaderboard",
           "Context turns a scatter plot into a story: shaded eras, labelled milestones, and the single biggest model "
           "of each year.")
    m = b["milestones"]
    st.markdown(f"### {len(m)} hand-curated milestones")
    st.dataframe(m.assign(date=m["date"].dt.strftime("%b %Y")), hide_index=True, width="stretch",
                 column_config={"date": "Date", "title": "Milestone", "description": "Why it matters"})
    st.caption("From data/milestones.csv. Labels that would overlap on the chart collapse to a ◆ you can hover.")
    chrome.credit()
    key_code("lab_milestones")
    learn(
        ["How to label a crowded time axis without overlaps", "How 'top model per year' is computed"],
        ["**Label only where there's room.** A milestone gets a rotated label only if it's at least 2.8% of the visible "
         "range from the last labelled one; the rest become hoverable ◆ markers.",
         "**`groupby('year')[compute].idxmax()`** finds each year's frontier model in one line; dividing by the "
         "previous row gives the × growth column."],
        [("Why are milestones a CSV and not code?",
          "They're editorial content. Anyone can add a row without touching the chart code.")],
    )
    nuggets("lab_milestones")
    tour_footer("lab_milestones")


# --- 6 · Safe tool use ---------------------------------------------------------------------------


def lab_tools() -> None:
    df = base()["df_all"]
    header(6, "Ask the Data with safe tool use",
           "Claude never writes code here. It chooses one of nine registered functions and fills in arguments that "
           "match a JSON schema; the app runs the function.")
    st.markdown("### Run a tool yourself (no API call)")
    name = st.selectbox("Function", list(QUERIES), key="lab_tool_name")
    _, description, schema = QUERIES[name]
    st.caption(description)
    args = {}
    props = schema["properties"]
    cols = st.columns(max(1, min(3, len(props))))
    for i, (arg, spec) in enumerate(props.items()):
        with cols[i % len(cols)]:
            if "enum" in spec:
                args[arg] = st.selectbox(arg, spec["enum"], key=f"lab_arg_{name}_{arg}")
            elif spec["type"] == "integer":
                v = st.number_input(arg, value=None, step=1, key=f"lab_arg_{name}_{arg}", placeholder="default")
                args[arg] = int(v) if v is not None else None
            elif spec["type"] == "boolean":
                args[arg] = st.checkbox(arg, key=f"lab_arg_{name}_{arg}")
            else:
                args[arg] = st.text_input(arg, key=f"lab_arg_{name}_{arg}",
                                          value="GPT" if arg == "name" else "") or None
    call = {k: v for k, v in args.items() if v is not None}
    st.code(f"{name}({', '.join(f'{k}={v!r}' for k, v in call.items())})", language="python")
    try:
        st.dataframe(run_query(df, name, call), hide_index=True, width="stretch")
    except Exception as e:  # same as the real loop: report, don't crash
        st.error(f"Error: {e}")
    with st.expander("What Claude sees for this tool", icon=":material/data_object:"):
        st.json(next(t for t in TOOLS if t["name"] == name))
    chrome.credit()
    key_code("lab_tools")
    learn(
        ["The tool-use loop: Claude asks, code runs, Claude answers", "Why a registry beats generated code"],
        ["**The registry is the security boundary.** `run_query` rejects unknown names and drops any argument not in "
         "the schema, so the worst a confused model can do is ask for the wrong table.",
         "**Errors go back to Claude as data** (`is_error: true`) instead of crashing, so it can retry with "
         "different arguments.",
         "**Capped at three rounds,** and results are trimmed to 60 rows before they're sent, which keeps cost and "
         "latency predictable.",
         "**Tested without the API.** `tests/test_llm.py` drives the loop with a fake client that replays scripted "
         "tool calls."],
        [("Why not let Claude write pandas code? It would answer more questions.",
          "Executing model-written code means anything in a prompt can become code on your server. Nine functions "
          "cover the questions people actually ask; new capabilities are added to `QUERIES` deliberately.")],
    )
    nuggets("lab_tools")
    tour_footer("lab_tools")


LAB_PAGES = {  # key: (title, icon, page function), in TOUR order
    "lab_data": ("1 · Load and clean", ":material/dataset:", lab_data),
    "lab_log": ("2 · Log scales", ":material/show_chart:", lab_log),
    "lab_growth": ("3 · The growth fit", ":material/trending_up:", lab_growth),
    "lab_filters": ("4 · Stable filters", ":material/filter_alt:", lab_filters),
    "lab_milestones": ("5 · Milestones and eras", ":material/flag:", lab_milestones),
    "lab_tools": ("6 · Safe tool use", ":material/build:", lab_tools),
}
