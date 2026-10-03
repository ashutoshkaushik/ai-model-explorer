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
  [data-testid="stMetricValue"] {{ font-family: var(--font-heading); }}

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

  /* ---- Site chrome (ui/chrome.py) */
  .author-name {{ font-family: var(--font-heading); font-size: 1.05rem; font-weight: 600; margin-bottom: .35rem; }}
  .site-footer {{ margin-top: 3rem; padding-top: .9rem; border-top: 1px solid var(--line); color: var(--muted);
                  font-size: .8rem; text-align: center; }}

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
  .ltable {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  .ltable th, .ltable td {{ text-align: left; vertical-align: top; padding: .45rem .6rem; border-bottom: 1px solid var(--line); }}
  .ltable thead th {{ border-bottom: 2px solid var(--line); }}
  .ltable code {{ font-size: .8rem; }}
</style>""", unsafe_allow_html=True)
