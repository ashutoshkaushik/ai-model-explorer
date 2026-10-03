"""AI Model Evolution Explorer: Streamlit app.

    streamlit run app.py

Structure: pages are registered with st.navigation below, in three groups (App, Overview, Explorer Lab),
the same layout as the AI History RAG and Travel Agent Lab projects. Data logic lives in utils/; the
page modules in lab/ only lay out UI.
"""

import streamlit as st

from lab.app_pages import ask_page, explorer_page
from lab.home import home_page
from lab.lab_pages import LAB_PAGES
from lab.nav import PAGES, TOUR
from lab.overview import about_page, browse_page, leaderboard_page, timeline_page
from ui import chrome, filters, theme


def main() -> None:
    st.set_page_config(page_title="AI Model Evolution Explorer", page_icon=":material/insights:", layout="wide")
    theme.apply_theme()
    filters.keep_state()
    pages = {
        "home": st.Page(home_page, title="Start here", icon=":material/home:", default=True),
        "explorer": st.Page(explorer_page, title="Explorer", icon=":material/insights:", url_path="explorer"),
        "ask": st.Page(ask_page, title="Ask the Data", icon=":material/forum:", url_path="ask"),
        "timeline": st.Page(timeline_page, title="Milestones timeline", icon=":material/timeline:", url_path="timeline"),
        "leaderboard": st.Page(leaderboard_page, title="Frontier leaderboard", icon=":material/leaderboard:",
                               url_path="leaderboard"),
        "browse": st.Page(browse_page, title="Browse the data", icon=":material/table_view:", url_path="browse"),
        "about": st.Page(about_page, title="How it's built", icon=":material/account_tree:", url_path="about"),
    }
    for key, (title, icon, fn) in LAB_PAGES.items():
        pages[key] = st.Page(fn, title=title, icon=icon, url_path=key.replace("_", "-"))
    PAGES.update(pages)
    page = st.navigation({
        "App": [pages["home"], pages["explorer"], pages["ask"]],
        "Overview": [pages["timeline"], pages["leaderboard"], pages["browse"], pages["about"]],
        "Explorer Lab · built step by step": [pages[k] for k in TOUR],
    }, expanded=True)  # always show every page; no "View more" in the sidebar
    page.run()
    chrome.author_card()
    chrome.footer()


if __name__ == "__main__":
    main()
