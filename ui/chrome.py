"""Site chrome shown on every page: the author line in the sidebar and the page footer."""

import streamlit as st

AUTHOR = "Ashutosh Kaushik"
LINKEDIN_URL = "https://www.linkedin.com/in/ashutosh-kaushik/"
CREDIT = "Data: Epoch AI (CC-BY 4.0)"


def credit() -> None:
    """The dataset credit, shown under every chart area."""
    st.markdown(f"<div class='credit'>{CREDIT}</div>", unsafe_allow_html=True)


def author_card() -> None:
    with st.sidebar:
        st.divider()
        st.markdown(f"<div class='author-name'>{AUTHOR}</div>", unsafe_allow_html=True)
        st.link_button("Connect on LinkedIn", LINKEDIN_URL, icon=":material/person_add:", width="stretch")


def footer() -> None:
    st.markdown(
        "<div class='site-footer'>Built with Streamlit, Plotly and Claude, using Claude Code and other AI tools, "
        f"with a human in the loop. {CREDIT}: "
        "<a href='https://epoch.ai/data/notable-ai-models'>Notable AI Models</a>.</div>",
        unsafe_allow_html=True)
