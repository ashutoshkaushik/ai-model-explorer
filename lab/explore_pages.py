"""Exploration pages: the compute race, a single model's profile, two models side by side, and training cost."""

import pandas as pd
import streamlit as st

from ui import chrome
from ui.filters import base, empty_message, sci, sci_compact, sidebar_filters
from utils.charts import autoplay_html, compare_chart, cost_scatter, fit_growth, model_spotlight, race_chart
from utils.data_loader import ACCESS_COL, COMPUTE_COL, COST_COL, DATE_COL, MODEL_COL, PARAMS_COL
from utils.insights import REFERENCE_COSTS, compare, model_profile, race_frames, similar_models, times

FAMOUS = ["GPT-3 175B (davinci)", "AlexNet", "AlphaGo Lee", "Transformer", "BERT-Large"]


def fmt(value: float, unit: str, compact: bool = False) -> str:
    if pd.isna(value):
        return "—" if compact else "not reported"
    if unit == "2023 USD":
        return f"${value:,.0f}" if value < 1e6 else f"${value / 1e6:,.1f}M"
    return sci_compact(value) if compact else sci(value)


def model_options(df: pd.DataFrame) -> list[str]:
    """Model names, largest training compute first (unknown compute last), so the famous ones are near the top."""
    return df.sort_values(COMPUTE_COL, ascending=False, na_position="last")[MODEL_COL].tolist()


def model_label(df: pd.DataFrame):
    info = df.set_index(MODEL_COL)
    return lambda name: f"{name} · {info.at[name, 'primary_org']}, {info.at[name, 'year']}"


# --- The compute race ----------------------------------------------------------------------------


def race_page() -> None:
    f = sidebar_filters()
    st.title("The compute race")
    st.markdown("<div class='lede'>The ten largest training runs ever, year by year. It plays on its own; drag the "
                "slider to any year, or replay it, and watch the scale leap by orders of magnitude.</div>",
                unsafe_allow_html=True)
    known = f.df[f.df[COMPUTE_COL] > 0]
    if known.empty:
        empty_message()
        return
    lo, hi = int(known["year"].min()), int(known["year"].max())
    a, b = st.columns(2)
    start = a.slider("Start the race in", lo, max(lo, hi - 1), min(max(lo, 2012), max(lo, hi - 1)), key="race_start")
    top_n = b.slider("Bars", 5, 15, 10, key="race_n")
    frames = race_frames(f.df, start, top_n)
    fig = race_chart(frames, base()["color_domains"], top_n)
    st.iframe(autoplay_html(fig, fig.layout.height), height=fig.layout.height + 10)  # our own HTML, no user input
    first, last = frames[frames["frame"] == frames["frame"].min()], frames[frames["frame"] == frames["frame"].max()]
    leap = last[COMPUTE_COL].max() / first[COMPUTE_COL].max()
    st.markdown(
        f"<div class='callout'>From {int(first['frame'].iloc[0])} to {int(last['frame'].iloc[0])} the record grew "
        f"<b>{times(leap)}</b>: from {first[MODEL_COL].iloc[0]} ({sci(first[COMPUTE_COL].max())} FLOP) to "
        f"{last[MODEL_COL].iloc[0]} ({sci(last[COMPUTE_COL].max())} FLOP).</div>", unsafe_allow_html=True)
    st.caption("Each frame counts models published up to the end of that year. Colours are domains; bars use a log "
               "scale, so one extra bar length of ×10 is ten times the compute.")
    chrome.credit()


# --- Find a model --------------------------------------------------------------------------------


def model_page() -> None:
    f = sidebar_filters()
    df = f.df
    st.title("Find a model")
    st.markdown("<div class='lede'>Pick any model to see where it sits in history: its rank that year, how it compares "
                "with the trend, and the models most like it.</div>", unsafe_allow_html=True)
    if df.empty:
        empty_message()
        return
    options = model_options(df)
    wanted = st.query_params.get("model")
    default = wanted if wanted in options else next((m for m in FAMOUS if m in options), options[0])
    name = st.selectbox("Model", options, index=options.index(default), format_func=model_label(df),
                        placeholder="Type to search…")
    st.query_params["model"] = name  # the URL now links straight to this model

    fit = fit_growth(df, COMPUTE_COL)
    p = model_profile(df, name, fit)
    r = p.row
    st.markdown(f"<div class='eyebrow'>{r['primary_domain']} · {r[ACCESS_COL] if pd.notna(r[ACCESS_COL]) else 'access unknown'}"
                f"</div><div class='hero' style='font-size:2rem'>{name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='lede'>{r['Organization'] if pd.notna(r['Organization']) else 'Unknown organization'} · "
                f"published {r[DATE_COL]:%B %-d, %Y}</div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Compute (FLOP)", fmt(r[COMPUTE_COL], "FLOP", compact=True), border=True,
              help=f"{fmt(r[COMPUTE_COL], 'FLOP')} FLOP")
    k2.metric("Parameters", fmt(r[PARAMS_COL], "parameters", compact=True), border=True,
              help=fmt(r[PARAMS_COL], "parameters"))
    k3.metric("Training cost", fmt(r[COST_COL], "2023 USD", compact=True), border=True,
              help=f"{fmt(r[COST_COL], '2023 USD')} (2023 USD)" if pd.notna(r[COST_COL]) else None)
    k4.metric(f"Rank in {r['year']}", f"#{p.rank_in_year} of {p.known_in_year}" if p.rank_in_year else "—",
              border=True, help="By training compute, among that year's models with known compute")

    facts = []
    if p.percentile is not None:
        facts.append(f"Used more compute than <b>{p.percentile:.0%}</b> of all models with known compute.")
    if p.vs_trend is not None:
        side = "above" if p.vs_trend >= 1 else "below"
        facts.append(f"<b>{times(p.vs_trend if p.vs_trend >= 1 else 1 / p.vs_trend)} {side}</b> the {fit.start_year}+ "
                     f"trend line for its publication date.")
    if p.rank_in_year == 1:
        facts.append(f"The <b>largest training run of {r['year']}</b> in this data.")
    if facts:
        st.markdown("<div class='callout'>" + "<br>".join(facts) + "</div>", unsafe_allow_html=True)

    similar = similar_models(df, name)
    st.markdown("### Where it sits")
    if pd.notna(r[COMPUTE_COL]):
        st.plotly_chart(model_spotlight(df, name, similar, fit), width="stretch")
    else:
        st.info(f"{name} doesn't report training compute, so it can't be placed on the chart.")
    st.markdown("### Similar models")
    st.caption("Closest in training compute (or parameters) and publication date, preferring the same domain.")
    st.dataframe(similar[[MODEL_COL, "primary_org", DATE_COL, COMPUTE_COL, PARAMS_COL]], hide_index=True,
                 width="stretch", column_config={
                     "primary_org": "Organization",
                     DATE_COL: st.column_config.DateColumn("Published", format="YYYY-MM-DD"),
                     COMPUTE_COL: st.column_config.NumberColumn("Compute (FLOP)", format="%.2e"),
                     PARAMS_COL: st.column_config.NumberColumn("Parameters", format="%.2e")})
    if pd.notna(r.get("Notability criteria")):
        st.caption(f"Why it's notable (Epoch AI): {r['Notability criteria']}")
    chrome.credit()


# --- Compare two models --------------------------------------------------------------------------


def compare_page() -> None:
    f = sidebar_filters()
    df = f.df
    st.title("Compare two models")
    st.markdown("<div class='lede'>How much bigger is one model than another? Pick two and see the gap in compute, "
                "parameters and cost.</div>", unsafe_allow_html=True)
    if len(df) < 2:
        empty_message()
        return
    options = model_options(df)
    label = model_label(df)
    default_a = "AlexNet" if "AlexNet" in options else options[-1]
    default_b = next((m for m in FAMOUS if m in options and m != default_a), options[0])
    a, swap, b = st.columns([5, 1, 5], vertical_alignment="bottom")
    name_a = a.selectbox("Model A", options, index=options.index(default_a), format_func=label, key="cmp_a")
    name_b = b.selectbox("Model B", options, index=options.index(default_b), format_func=label, key="cmp_b")

    def do_swap() -> None:
        st.session_state.cmp_a, st.session_state.cmp_b = st.session_state.cmp_b, st.session_state.cmp_a

    swap.button("⇄", on_click=do_swap, help="Swap A and B", width="stretch")
    if name_a == name_b:
        st.info("Pick two different models.")
        return

    ra, rb = (df.loc[df[MODEL_COL] == n].iloc[0] for n in (name_a, name_b))
    table = compare(ra, rb)
    compute = table.iloc[0]
    years = (rb[DATE_COL] - ra[DATE_COL]).days / 365.25
    when = (f"published {abs(years):.1f} years {'later' if years >= 0 else 'earlier'}" if abs(years) >= 0.1
            else "published the same month")
    if pd.notna(compute["ratio"]):
        st.markdown(f"<div class='hero' style='font-size:1.9rem'>{name_b} used {times(compute['ratio'])} the training "
                    f"compute of {name_a}</div><div class='lede'>…and was {when}.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='lede'>{name_b} was {when} than {name_a}. Training compute isn't reported for both, "
                    "so the headline comparison isn't available.</div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    for col, row, name in ((left, ra, name_a), (right, rb, name_b)):
        with col.container(border=True):
            st.markdown(f"#### {name}")
            st.caption(f"{row['primary_org']} · {row['primary_domain']} · {row[DATE_COL]:%b %Y}")
            for _, m in table.iterrows():
                value = row[{'Training compute': COMPUTE_COL, 'Parameters': PARAMS_COL, 'Training cost': COST_COL}[m['metric']]]
                st.markdown(f"**{m['metric']}:** {fmt(value, m['unit'])}")

    labels = table["ratio"].map(lambda r: times(r) if pd.notna(r) else None)
    fig = compare_chart(table.assign(label=labels), name_a, name_b)
    if fig:
        st.plotly_chart(fig, width="stretch")
        st.caption("Bars to the right: B is bigger. To the left: B is smaller. Metrics missing for either model are "
                   "left out.")
    chrome.credit()


# --- Training cost -------------------------------------------------------------------------------


def cost_page() -> None:
    f = sidebar_filters()
    df = f.df
    st.title("What training costs")
    st.markdown("<div class='lede'>The compute bill for a single training run, in 2023 dollars, set against price tags "
                "you already know.</div>", unsafe_allow_html=True)
    costed = df[df[COST_COL] > 0]
    if costed.empty:
        st.info("None of the filtered models report a training cost. Try widening the filters.")
        chrome.credit()
        return
    top = costed.loc[costed[COST_COL].idxmax()]
    fit = fit_growth(costed, COST_COL)
    home, film = REFERENCE_COSTS[1][0], REFERENCE_COSTS[2][0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Priciest run", fmt(top[COST_COL], "2023 USD"), border=True,
              help=f"{top[MODEL_COL]} ({top['primary_org']}, {top['year']})")
    k2.metric("= US homes", f"{top[COST_COL] / home:,.0f}", border=True, help="At about $400,000 per home")
    k3.metric("= Blockbuster films", f"{top[COST_COL] / film:,.1f}", border=True, help="At about $200M per film")
    k4.metric("Cost growth", f"{fit.factor_per_year:.1f}×/yr" if fit else "—", border=True,
              help=f"Log-linear fit over {fit.n} models since {fit.start_year}" if fit else None)

    st.plotly_chart(cost_scatter(df, base()["color_domains"], fit, REFERENCE_COSTS), width="stretch")
    st.caption(f"Only {len(costed):,} of {len(df):,} filtered models ({len(costed) / len(df):.0%}) report a cost. "
               "Costs are Epoch AI's estimates of the compute bill only (2023 USD): no salaries, data or failed runs. "
               "Reference prices are rough and rounded.")

    st.markdown("### The ten most expensive training runs")
    priciest = costed.nlargest(10, COST_COL).assign(homes=lambda t: t[COST_COL] / home)
    st.dataframe(priciest[[MODEL_COL, "primary_org", "year", COST_COL, "homes"]], hide_index=True, width="stretch",
                 column_config={
                     "primary_org": "Organization",
                     "year": st.column_config.NumberColumn("Year", format="%d"),
                     COST_COL: st.column_config.NumberColumn("Cost (2023 USD)", format="dollar"),
                     "homes": st.column_config.NumberColumn("≈ US homes", format="%,.0f")})
    chrome.credit()
