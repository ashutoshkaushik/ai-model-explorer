"""'How it works (ELI5)': the whole app as one simple picture, then the journey of one question.

The diagram is an SVG drawn with the site's colour tokens (ui/theme.py: data = blue, code = purple,
Claude = amber, you = terracotta), so it matches light and dark mode. Streamlit strips inline <svg>,
so it is embedded as an <img> with a data URI, the same approach as the Travel Agent Lab's diagram.
"""

import base64
from html import escape

import streamlit as st

from lab.nav import go_button
from ui import chrome
from ui.filters import base
from ui.theme import FONT_BODY, FONT_HEADING, tokens
from utils.llm import QUERIES

W, H = 1000, 430
NODE_W, NODE_H = 172, 92


def _node(x: int, y: int, kind: str, num: str, title: str, lines: list[str], t: dict) -> str:
    """A rounded card: numbered badge, title, and up to two short lines of plain English."""
    stroke, fill = t[kind], t[f"{kind}-soft"]
    sub = "".join(f"<tspan x='{x + 14}' dy='{0 if i == 0 else 15}'>{escape(line)}</tspan>"
                  for i, line in enumerate(lines))
    return (f"<rect x='{x}' y='{y}' width='{NODE_W}' height='{NODE_H}' rx='12' fill='{t['card']}' "
            f"stroke='{stroke}' stroke-width='1.6'/>"
            f"<rect x='{x}' y='{y}' width='{NODE_W}' height='{NODE_H}' rx='12' fill='{fill}'/>"
            f"<circle cx='{x + 22}' cy='{y + 24}' r='12' fill='{stroke}'/>"
            f"<text x='{x + 22}' y='{y + 28.5}' text-anchor='middle' class='num'>{num}</text>"
            f"<text x='{x + 42}' y='{y + 29.5}' class='title'>{escape(title)}</text>"
            f"<text x='{x + 14}' y='{y + 57}' class='sub'>{sub}</text>")


def _edge(d: str, label: str = "", lx: int = 0, ly: int = 0, dashed: bool = False) -> str:
    dash = " stroke-dasharray='5 4'" if dashed else ""
    text = f"<text x='{lx}' y='{ly}' class='edge-label' text-anchor='middle'>{escape(label)}</text>" if label else ""
    return f"<path d='{d}' class='edge'{dash} marker-end='url(#arrow)'/>{text}"


def diagram_svg(n_models: int) -> str:
    t = {**tokens(), "you": tokens()["accent"], "you-soft": "rgba(217,119,87,.14)"}
    style = (f"<style>"
             f".title{{font:600 15px {FONT_HEADING};fill:{t['ink']}}}"
             f".sub{{font:12px {FONT_BODY};fill:{t['muted']}}}"
             f".num{{font:700 12px {FONT_BODY};fill:{t['card']}}}"
             f".edge{{fill:none;stroke:{t['muted']};stroke-width:1.6}}"
             f".edge-label{{font:italic 11.5px {FONT_BODY};fill:{t['muted']}}}"
             f".band{{font:600 11px {FONT_BODY};letter-spacing:.12em;fill:{t['muted']}}}"
             f"</style>")
    top, mid, low = 70, 190, 330
    x1, x2, x3, x4, x5 = 18, 214, 410, 610, 810
    parts = [
        # column captions: where each step happens
        f"<text x='{x1}' y='34' class='band'>THE DATA</text>",
        f"<text x='{x2}' y='34' class='band'>PLAIN CODE</text>",
        f"<text x='{x3}' y='34' class='band'>YOU CHOOSE</text>",
        f"<text x='{x4}' y='34' class='band'>CODE OR CLAUDE</text>",
        f"<text x='{x5}' y='34' class='band'>YOU SEE</text>",
        _node(x1, mid, "data", "1", "A big list", [f"{n_models:,} AI models,", "from Epoch AI"], t),
        _node(x2, mid, "code", "2", "Tidy it up", ["fix typos, pick the", "main lab and topic"], t),
        _node(x3, mid, "you", "3", "Your filters", ["keep only the years", "and labs you pick"], t),
        _node(x4, top, "code", "4", "Draw pictures", ["charts, the race,", "costs, profiles"], t),
        _node(x4, low - 20, "llm", "5", "Ask Claude", ["Claude picks one", "labelled drawer"], t),
        _node(x5, mid, "you", "6", "Your answer", ["a chart, a table or", "a short answer"], t),
        # the 9 drawers: the only things Claude can open
        f"<rect x='{x2 + 6}' y='{low + 4}' width='{NODE_W + 152}' height='76' rx='12' fill='{t['code-soft']}' "
        f"stroke='{t['code']}' stroke-width='1.3' stroke-dasharray='5 4'/>",
        f"<text x='{x2 + 22}' y='{low + 30}' class='title'>{len(QUERIES)} labelled drawers</text>",
        f"<text x='{x2 + 22}' y='{low + 52}' class='sub'><tspan x='{x2 + 22}' dy='0'>Ready-made questions, like “top models”.</tspan><tspan x='{x2 + 22}' dy='15'>Claude can only open these; it never writes code.</tspan></text>",
        # arrows
        _edge(f"M{x1 + NODE_W},{mid + 46} L{x2 - 6},{mid + 46}"),
        _edge(f"M{x2 + NODE_W},{mid + 46} L{x3 - 6},{mid + 46}"),
        _edge(f"M{x3 + NODE_W},{mid + 30} C{x3 + NODE_W + 20},{mid + 30} {x4 - 26},{top + 46} {x4 - 6},{top + 46}"),
        _edge(f"M{x3 + NODE_W},{mid + 62} C{x3 + NODE_W + 20},{mid + 62} {x4 - 26},{low + 26} {x4 - 6},{low + 26}",
              "a question", x4 - 34, low - 4),
        _edge(f"M{x4 + NODE_W},{top + 46} C{x4 + NODE_W + 14},{top + 46} {x5 - 22},{mid + 30} {x5 - 6},{mid + 30}"),
        _edge(f"M{x4 + NODE_W},{low + 26} C{x4 + NODE_W + 14},{low + 26} {x5 - 22},{mid + 62} {x5 - 6},{mid + 62}"),
        _edge(f"M{x4 - 2},{low + 48} L{x2 + 6 + NODE_W + 152 + 4},{low + 48}", "opens one", x4 - 42, low + 66,
              dashed=True),
    ]
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' role='img'>{style}"
           f"<defs><marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
           f"orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{t['muted']}'/></marker></defs>"
           + "".join(parts) + "</svg>")
    return svg


def diagram_img(n_models: int) -> str:
    data = base64.b64encode(diagram_svg(n_models).encode("utf-8")).decode("ascii")
    alt = ("How the app works: 1, a big list of AI models from Epoch AI is downloaded; 2, plain code tidies it up; "
           "3, your filters keep what you pick; then either 4, code draws charts, or 5, Claude picks one of nine "
           "labelled query drawers, which code opens; 6, you see a chart, table or short answer.")
    return (f"<div class='diagram'><img src='data:image/svg+xml;base64,{data}' alt='{escape(alt)}' "
            f"style='width:100%;min-width:640px;height:auto;display:block'></div>")


JOURNEY = [  # (who, what happens), the life of one "Ask the Data" question
    ("you", "You ask", "“Which labs built the most models since 2020?”"),
    ("llm", "Claude reads the labels", "It sees the names of the 9 drawers and what each one is for. Not the data."),
    ("llm", "Claude points at a drawer", "“Open count_by_org, with year_from = 2020.” That's all it can do."),
    ("code", "The app opens it", "Plain pandas code runs on your filtered list and gets a small table."),
    ("llm", "Claude explains", "It turns the table into 2–3 sentences with the real numbers."),
]

ROOMS = [  # (sidebar group, ELI5 line)
    ("App", "The main rooms: the big picture, the race, any single model, two models side by side, and asking "
            "questions in plain English."),
    ("Overview", "The reference shelf: milestones, the record holders, what training costs, the raw list, and how "
                 "it's built."),
    ("Explorer Lab", "The workshop: seven steps showing how the app was built (and what it costs the planet), each with something to play with."),
    ("Surprise me", "The fun drawer: 25 AI facts, one at a time."),
]


def eli5_page() -> None:
    df = base()["df_all"]
    st.markdown("<div class='eyebrow'>How it works · explained like you're five</div>", unsafe_allow_html=True)
    st.title("How this app works")
    st.markdown("<div class='lede'>Think of a library. There's one big list of AI models. A helper tidies it, you pick "
                "what you want, and then either a painter draws it for you or a librarian answers your question, "
                "using only the labelled drawers she's allowed to open.</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='legend'><span><i style='background:var(--data)'></i>the data</span>"
        "<span><i style='background:var(--code)'></i>plain code: same input, same answer, every time</span>"
        "<span><i style='background:var(--llm)'></i>Claude, the AI</span>"
        "<span><i style='background:var(--accent)'></i>you</span></div>", unsafe_allow_html=True)
    st.markdown(diagram_img(len(df)), unsafe_allow_html=True)

    st.markdown("### The journey of one question")
    steps = "".join(f"<div class='{who}'><span class='n'>{i}</span><b>{title}</b><span>{escape(what)}</span></div>"
                    for i, (who, title, what) in enumerate(JOURNEY, start=1))
    st.markdown(f"<div class='journey'>{steps}</div>", unsafe_allow_html=True)
    st.markdown("<div class='callout'><b>Why the drawers?</b> If Claude could write its own code, a cleverly worded "
                "question could make the app do something nobody planned. With drawers, the worst it can do is open "
                "the wrong one, and you can see which one it opened under every answer.</div>",
                unsafe_allow_html=True)

    st.markdown("### The rooms in the sidebar")
    cards = "".join(f"<div class='nugget'><div class='tag'>{group}</div><span>{line}</span></div>"
                    for group, line in ROOMS)
    st.markdown(f"<div class='nuggets'>{cards}</div>", unsafe_allow_html=True)

    st.markdown("### Three things that make it trustworthy")
    st.markdown(
        "<div class='steps'>"
        "<div class='data'><b>Every number comes from the list</b><span>Charts and answers are computed from Epoch "
        "AI's data, never made up. Each page says where the data came from.</span></div>"
        "<div class='code'><b>Code does the maths</b><span>Growth rates, rankings and costs are calculated by plain "
        "code, and tests check it gets known answers right.</span></div>"
        "<div class='llm'><b>Claude only explains</b><span>It chooses a drawer and puts the result into words. You "
        "can always open “How this was answered” to check.</span></div>"
        "</div>", unsafe_allow_html=True)
    st.markdown("")
    a, b = st.columns(2)
    with a:
        go_button("about", "The grown-up version: files and layers")
    with b:
        go_button("lab_data", "See it built, step by step", primary=True)
    chrome.credit()
