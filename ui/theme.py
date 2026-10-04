"""The site's single source of truth for colour and type (same look as the RAG and Travel Agent Lab apps).

Every page gets these tokens through apply_theme(), which app.main() calls once per run. Page CSS
refers to them as var(--token). Charts read the same tokens through chart_tokens().

Colour tokens have fixed meanings:
  --accent    terracotta: buttons, links and the active nav item ONLY
  --data      the dataset / measured facts
  --llm       Claude decides (Ask the Data, era summaries)
  --code      code guarantees (query functions, filters, the growth fit)
"""

import streamlit as st

FONT_HEADING = '"Source Serif 4", Georgia, "Times New Roman", serif'
FONT_BODY = '"Hanken Grotesk", system-ui, -apple-system, "Segoe UI", sans-serif'
FONT_MONO = '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace'

LIGHT = {
    "accent": "#C2603E", "on-accent": "#ffffff",
    "data": "#2a78d6", "data-soft": "rgba(42,120,214,.10)",
    "llm": "#a86a12", "llm-soft": "rgba(183,121,31,.12)",
    "code": "#4a3aa7", "code-soft": "rgba(74,58,167,.10)",
    "success": "#1f8a4c", "success-soft": "rgba(31,138,76,.13)",
    "error": "#c4521f", "error-soft": "rgba(196,82,31,.12)",
    "muted": "#6f6b62",
    "line": "rgba(31,30,29,.14)",
    "card": "#ffffff",
    "surface": "#FAF9F5",
    "sidebar": "#F3F1EA",
    "ink": "#1F1E1D",
}
DARK = {
    "accent": "#D97757", "on-accent": "#1F1E1D",
    "data": "#3987e5", "data-soft": "rgba(57,135,229,.16)",
    "llm": "#d9a441", "llm-soft": "rgba(217,164,65,.16)",
    "code": "#9085e9", "code-soft": "rgba(144,133,233,.16)",
    "success": "#3fb67a", "success-soft": "rgba(63,182,122,.18)",
    "error": "#e66a3d", "error-soft": "rgba(230,106,61,.18)",
    "muted": "#a8a397",
    "line": "rgba(242,240,232,.16)",
    "card": "#30302E",
    "surface": "#262624",
    "sidebar": "#1F1E1D",
    "ink": "#F2F0E8",
}


def mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def tokens() -> dict[str, str]:
    return DARK if mode() == "dark" else LIGHT


def apply_theme() -> None:
    """Inject the tokens and the shared typography / component styles. Call once per run."""
    t = tokens()
    variables = "\n".join(f"    --{k}: {v};" for k, v in t.items())
    st.markdown(f"""<style>
  :root {{
{variables}
    --font-heading: {FONT_HEADING};
    --font-body: {FONT_BODY};
    --font-mono: {FONT_MONO};
  }}

  /* ---- Typography: one serif for headings, one sans for everything else */
  html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{ font-family: var(--font-body); }}
  h1, h2, h3, h4, [data-testid="stHeading"] * {{ font-family: var(--font-heading) !important; }}
  h1 {{ font-size: 2.1rem !important; font-weight: 600 !important; line-height: 1.2 !important; }}
  h2 {{ font-size: 1.55rem !important; font-weight: 600 !important; }}
  h3 {{ font-size: 1.22rem !important; font-weight: 600 !important; }}
  h4 {{ font-size: 1.05rem !important; font-weight: 600 !important; }}
  code, pre {{ font-family: var(--font-mono); }}
  [data-testid="stMainBlockContainer"] {{ max-width: 1240px; padding-left: 1.5rem; padding-right: 1.5rem; }}
  [data-testid^="stBaseButton"] * , [data-testid="stWidgetLabel"] * {{ white-space: normal !important; }}
  [data-testid^="stBaseButton"] {{ height: auto; min-height: 2.5rem; }}
  /* Stat cards: values scale with the viewport instead of truncating; long labels wrap */
  [data-testid="stMetricValue"] {{ font-family: var(--font-heading); font-size: clamp(1.35rem, 0.9rem + 1vw, 2rem); }}
  [data-testid="stMetricValue"] > div {{ overflow: visible; text-overflow: clip; }}
  [data-testid="stMetricLabel"] p {{ white-space: normal !important; }}

  /* ---- Shared components */
  .muted {{ color: var(--muted); }}
  .lede {{ font-size: 1.1rem; line-height: 1.6; color: var(--muted); max-width: 68ch; }}
  .eyebrow {{ font: 600 .74rem/1 var(--font-body); letter-spacing: .12em; text-transform: uppercase;
             color: var(--muted); margin-bottom: .6rem; }}
  .hero {{ font-family: var(--font-heading); font-size: clamp(2rem, 4vw, 2.8rem); line-height: 1.12;
          font-weight: 600; margin: 0 0 .8rem; text-wrap: balance; }}
  .badge {{ display: inline-block; font-size: .76rem; font-weight: 600; padding: 1px 9px; border-radius: 99px;
            white-space: nowrap; }}
  .badge.data {{ background: var(--data-soft); color: var(--data); }}
  .badge.llm {{ background: var(--llm-soft); color: var(--llm); }}
  .badge.code {{ background: var(--code-soft); color: var(--code); }}
  .badge + .badge {{ margin-left: .3rem; }}
  .callout {{ border-left: 3px solid var(--accent); background: var(--card); border-radius: 6px;
              padding: .6rem .9rem; margin: .4rem 0 .8rem; }}
  .credit {{ font-size: .78rem; color: var(--muted); margin-top: .2rem; }}
  .fp-story {{ font-family: var(--font-heading); font-size: 1.12rem; line-height: 1.7; background: var(--card);
               border: 1px solid var(--line); border-radius: .6rem; padding: .8rem 1rem; }}
  .fp-story .gen {{ background: var(--llm-soft); color: var(--ink); border-radius: 3px; }}

  /* ---- Site chrome (ui/chrome.py) */
  .author-name {{ font-family: var(--font-heading); font-size: 1.05rem; font-weight: 600; margin-bottom: .45rem;
                  text-align: center; }}
  /* Author card pinned to the bottom of the sidebar at any window height. The sidebar's scroll area is
     Streamlit's own, so the card's block is position: fixed, and `contain: layout` makes the sidebar (not the
     window) its containing block: it spans the sidebar's width and stays put while the menu scrolls behind it. */
  [data-testid="stSidebar"] {{ contain: layout; }}
  [data-testid="stSidebarContent"] {{ padding-bottom: 7.5rem; }}
  [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.author-card) {{
      position: fixed; left: 0; right: 0; bottom: 0; z-index: 5; margin: 0; width: auto !important;
      background: var(--sidebar); border-top: 1px solid var(--line); padding: .8rem 1.5rem 1rem; }}
  .author-card {{ max-width: 22rem; margin: 0 auto; }}
  /* LinkedIn's own button style: LinkedIn blue, white "in" logo, pill shape, darker blue on hover.
     Fixed brand colours (not theme tokens), so it reads as LinkedIn in both light and dark mode. */
  a.li-btn {{ display: flex; align-items: center; justify-content: center; gap: .55rem; width: 100%;
             box-sizing: border-box; padding: .55rem 1rem; border-radius: 999px; background: #0A66C2;
             color: #ffffff !important; text-decoration: none !important; font: 600 .95rem/1.2 var(--font-body);
             transition: background-color .15s ease, box-shadow .15s ease; }}
  a.li-btn:hover {{ background: #004182; box-shadow: 0 2px 8px rgba(10,102,194,.35); }}
  a.li-btn:focus-visible {{ outline: 2px solid #70B5F9; outline-offset: 2px; }}
  a.li-btn .li-logo {{ width: 1.15rem; height: 1.15rem; flex: none; }}
  /* Footer: the page column fills at least the window, the footer is pushed to its end, and it sticks to
     the bottom of the window while scrolling. Streamlit's large default bottom padding is removed so the
     footer sits flush with the bottom edge. */
  [data-testid="stMainBlockContainer"] {{ padding-bottom: 0 !important; min-height: 100dvh; display: flex;
                                         flex-direction: column; }}
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {{ flex: 1 0 auto; }}
  [data-testid="stMain"] [data-testid="stElementContainer"]:has(.site-footer) {{
      margin-top: auto; position: sticky; bottom: 0; z-index: 4; }}
  .site-footer {{ padding: .55rem 1rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .78rem;
                  text-align: center; line-height: 1.4;
                  background: color-mix(in srgb, var(--surface) 88%, transparent); backdrop-filter: blur(6px); }}
  .site-footer a {{ color: inherit; text-decoration: underline; }}
  /* Streamlit gives markdown blocks a -1rem bottom margin; without this the pinned card and footer overhang
     the bottom edge */
  [data-testid="stElementContainer"]:has(.site-footer) [data-testid="stMarkdownContainer"],
  [data-testid="stElementContainer"]:has(.author-card) [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
  @media (max-width: 640px) {{ .site-footer .sf-long {{ display: none; }} }}

  /* ---- Growth ladder (landing page): scale through the eras */
  .ladder {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: .6rem 0 1.2rem; }}
  @media (max-width: 900px) {{ .ladder {{ grid-template-columns: 1fr; }} }}
  .rung {{ border: 1px solid var(--line); border-radius: .6rem; padding: .8rem .85rem; position: relative; }}
  .rung .n {{ font-family: var(--font-heading); font-size: 1.5rem; font-weight: 600; color: var(--muted); line-height: 1; }}
  .rung .t {{ font-family: var(--font-heading); font-size: 1.08rem; font-weight: 600; margin: .25rem 0 .35rem; }}
  .rung .flow {{ font-family: var(--font-mono); font-size: .72rem; color: var(--muted); margin-bottom: .45rem; }}
  .rung .adds {{ font-size: .84rem; line-height: 1.4; }}
  .rung.this {{ border: 2px solid var(--accent); }}
  .rung.this .n {{ color: var(--accent); }}
  .rung .here {{ position: absolute; top: .55rem; right: .6rem; }}

  /* ---- Who does what: data / code / Claude */
  .steps {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
  @media (max-width: 1000px) {{ .steps {{ grid-template-columns: 1fr; }} }}
  .steps > div {{ padding-top: .5rem; }}
  .steps .data {{ border-top: 3px solid var(--data); }}
  .steps .code {{ border-top: 3px solid var(--code); }}
  .steps .llm {{ border-top: 3px solid var(--llm); }}
  .steps b {{ display: block; font-size: .95rem; margin-bottom: .2rem; }}
  .steps span {{ font-size: .86rem; color: var(--muted); }}

  /* ---- Lab "learn" boxes + tables */
  .tablewrap {{ overflow-x: auto; }}
  .learnbox {{ border: 1px solid var(--line); border-left: 3px solid var(--data); border-radius: 8px;
               background: var(--card); padding: .7rem 1rem .4rem; margin: .8rem 0 .6rem; max-width: 72ch; }}
  .learnbox ul {{ margin: .35rem 0 .3rem 1.1rem; padding: 0; }}
  .learnbox li {{ margin: .15rem 0; }}
  .nuggets {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin: .4rem 0 1rem; }}
  @media (max-width: 900px) {{ .nuggets {{ grid-template-columns: 1fr; }} }}
  .nugget {{ border: 1px solid var(--line); border-radius: .6rem; background: var(--card); padding: .75rem .95rem; }}
  .nugget .tag {{ font: 600 .68rem/1 var(--font-body); letter-spacing: .1em; text-transform: uppercase;
                  color: var(--accent); margin-bottom: .45rem; }}
  .nugget b {{ display: block; font-family: var(--font-heading); font-size: 1.02rem; margin-bottom: .25rem; }}
  .nugget span {{ font-size: .87rem; line-height: 1.45; color: var(--muted); }}

  .nuggets.three {{ grid-template-columns: repeat(3, 1fr); }}
  @media (max-width: 900px) {{ .nuggets.three {{ grid-template-columns: 1fr; }} }}

  /* ---- ELI5 architecture (lab/architecture.py) */
  .diagram {{ border: 1px solid var(--line); border-radius: .8rem; background: var(--card); padding: .8rem;
              overflow-x: auto; margin: .3rem 0 1.2rem; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; font-size: .85rem; color: var(--muted); margin: .2rem 0 .5rem; }}
  .legend i {{ display: inline-block; width: .75rem; height: .75rem; border-radius: 3px; margin-right: .35rem;
               vertical-align: -1px; }}
  .journey {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: .4rem 0 1rem; }}
  @media (max-width: 1000px) {{ .journey {{ grid-template-columns: 1fr; }} }}
  .journey > div {{ border: 1px solid var(--line); border-top-width: 3px; border-radius: .6rem; background: var(--card);
                    padding: .7rem .8rem; }}
  .journey .you {{ border-top-color: var(--accent); }}
  .journey .llm {{ border-top-color: var(--llm); }}
  .journey .code {{ border-top-color: var(--code); }}
  .journey .n {{ font-family: var(--font-heading); font-size: 1.4rem; font-weight: 600; color: var(--muted);
                 display: block; line-height: 1; }}
  .journey b {{ display: block; font-family: var(--font-heading); font-size: 1rem; margin: .3rem 0 .2rem; }}
  .journey span:last-child {{ font-size: .85rem; color: var(--muted); line-height: 1.4; }}

  /* ---- Hero sparkline (Start here) */
  .hero-chart {{ border: 1px solid var(--line); border-radius: .8rem; background: var(--card); padding: .8rem 1rem .7rem;
                 margin: .6rem 0 .8rem; }}
  .hc-svg svg {{ width: 100%; height: auto; display: block; }}
  .hc-label {{ font: 600 .7rem/1 var(--font-body); letter-spacing: .1em; text-transform: uppercase; color: var(--muted);
               margin-bottom: .2rem; }}
  .hc-chips {{ display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .4rem; }}
  .hc-chips span {{ font-size: .82rem; padding: .15rem .6rem; border-radius: 99px; border: 1px solid var(--line);
                    color: var(--ink); cursor: help; }}

  /* ---- Live counter (Start here) */
  .live {{ border: 1px solid var(--line); border-left: 4px solid var(--accent); border-radius: .6rem;
           background: var(--card); padding: .9rem 1.2rem; margin: .3rem 0 1rem; }}
  .live-row {{ display: flex; flex-wrap: wrap; gap: .4rem 2.4rem; }}
  .live-k {{ font: 600 .7rem/1.2 var(--font-body); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
  .live-dot {{ display: inline-block; width: .55rem; height: .55rem; border-radius: 99px; background: var(--success);
               margin-right: .45rem; vertical-align: 1px; animation: live-pulse 1s ease-in-out infinite; }}
  @keyframes live-pulse {{ 50% {{ opacity: .25; }} }}
  .live-dot.stopped {{ animation: none; background: var(--muted); }}
  @media (prefers-reduced-motion: reduce) {{ .live-dot {{ animation: none; }} }}
  .live .tick {{ font-family: var(--font-heading); font-size: clamp(1.8rem, 4vw, 2.6rem); font-weight: 600;
                 color: var(--accent); font-variant-numeric: tabular-nums; line-height: 1.1; }}
  .live .lede {{ font-size: 1rem; margin-top: .3rem; }}

  /* ---- Surprise me (lab/surprise.py) */
  .fact {{ border: 1px solid var(--line); border-top: 4px solid var(--accent); border-radius: .8rem;
           background: var(--card); padding: 1.6rem 1.8rem 1.3rem; margin: .4rem 0 1rem; }}
  .fact .meta {{ display: flex; justify-content: space-between; font: 600 .72rem/1 var(--font-body);
                 letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-bottom: 1rem; }}
  .fact .q {{ font-family: var(--font-heading); font-size: clamp(1.5rem, 3vw, 2.1rem); line-height: 1.2;
              font-weight: 600; text-wrap: balance; }}
  .fact .a {{ font-size: 1.08rem; line-height: 1.6; margin-top: 1rem; max-width: 70ch; }}
  .fact .why {{ margin-top: .9rem; padding-top: .7rem; border-top: 1px dashed var(--line); color: var(--muted);
                font-size: .92rem; }}
  .fact .why b {{ color: var(--accent); }}
  .dots {{ display: flex; flex-wrap: wrap; gap: 5px; margin: .2rem 0 .8rem; }}
  .dots i {{ width: 9px; height: 9px; border-radius: 99px; background: var(--line); }}
  .dots i.on {{ background: var(--accent); }}
  .dots i.seen {{ background: var(--accent); opacity: .35; }}

  .ltable {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  .ltable th, .ltable td {{ text-align: left; vertical-align: top; padding: .45rem .6rem; border-bottom: 1px solid var(--line); }}
  .ltable thead th {{ border-bottom: 2px solid var(--line); }}
  .ltable code {{ font-size: .8rem; }}
</style>""", unsafe_allow_html=True)
