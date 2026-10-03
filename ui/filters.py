"""Sidebar filters shared by every data page (App and Overview groups).

Data pages call sidebar_filters(); it draws the filters under the navigation and returns the
filtered DataFrame. Filter values live in st.session_state, so they carry over between pages.
"""

import math
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from utils.charts import top_categories
from utils.data_loader import ACCESS_COL, apply_filters, load_data, load_milestones, top_orgs

SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float) -> str:
    """1.0001e27 -> '1.0 × 10²⁷'."""
    exp = math.floor(math.log10(x))
    return f"{x / 10**exp:.1f} × 10{str(exp).translate(SUPERSCRIPT)}"


def sci_compact(x: float) -> str:
    """Short form for stat cards: 1.0001e27 -> '10²⁷', 3.14e23 -> '3.1×10²³'."""
    exp = math.floor(math.log10(x))
    mantissa = round(x / 10**exp, 1)
    if mantissa >= 10:
        mantissa, exp = mantissa / 10, exp + 1
    power = f"10{str(exp).translate(SUPERSCRIPT)}"
    return power if mantissa == 1 else f"{mantissa:.1f}×{power}"


@dataclass
class Filtered:
    df: pd.DataFrame          # sidebar-filtered models
    years: tuple[int, int]
    note: str                 # plain-English filter summary, sent to Claude with every question


def base() -> dict:
    """Unfiltered data plus the option lists the filters offer (cached loaders, so this is cheap)."""
    df_all = load_data()
    return {
        "df_all": df_all,
        "milestones": load_milestones(),
        "min_year": int(df_all["year"].min()),
        "max_year": int(df_all["year"].max()),
        "domain_options": df_all["primary_domain"].value_counts().index.tolist(),
        "org_options": top_orgs(df_all, 15),
        "access_options": df_all[ACCESS_COL].fillna("Unknown").value_counts().index.tolist(),
        # Colors are assigned from the unfiltered data so a filter never repaints a domain.
        "color_domains": top_categories(df_all["primary_domain"]),
    }


def defaults() -> dict:
    b = base()
    return {"f_years": (b["min_year"], b["max_year"]), "f_domains": [], "f_orgs": [], "f_access": [],
            "f_frontier": False}


def keep_state() -> None:
    """Call once per run, before any page: re-assigning widget keys stops Streamlit from discarding
    filter values on pages that don't draw the filters (Start here, the Lab)."""
    for key, value in defaults().items():
        st.session_state[key] = st.session_state.get(key, value)


def reset() -> None:
    for key, value in defaults().items():
        st.session_state[key] = value


def sidebar_filters() -> Filtered:
    b = base()
    with st.sidebar:
        header = st.empty()
        years = st.slider("Year range", b["min_year"], b["max_year"], key="f_years")
        domains = st.multiselect("Domain", b["domain_options"], key="f_domains", placeholder="All domains")
        orgs = st.multiselect(
            "Organization", b["org_options"] + ["Other"], key="f_orgs", placeholder="All organizations",
            help="Top 15 organizations by model count; 'Other' covers the rest.",
        )
        access = st.multiselect("Accessibility", b["access_options"], key="f_access", placeholder="Any accessibility")
        frontier_only = st.checkbox("Frontier models only", key="f_frontier")
        st.button("Reset filters", on_click=reset, width="stretch", icon=":material/restart_alt:")
        active = sum(st.session_state[k] != v for k, v in defaults().items())
        header.markdown(f"**Filters · {active} active**" if active else "**Filters**")

    df = apply_filters(b["df_all"], years, domains, orgs, b["org_options"], access, frontier_only)
    note = "; ".join(
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
    return Filtered(df, years, note)


def empty_message() -> None:
    st.warning("No models match the current filters. Try widening the year range or clearing a filter.",
               icon=":material/filter_alt_off:")
