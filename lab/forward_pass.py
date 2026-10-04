"""'Inside a forward pass': a real Llama 2 (Karpathy's stories15M) runs on your prompt, and an animation steps
through run.c's forward() for every word: embedding, six layers of attention and SwiGLU, logits, sampling.

utils/llama.py does the maths (a numpy port of run.c, checked against ./run in tests/test_llama.py) and records
every attention weight and the vector x after each step. This page packs that trace into JSON and plays it in
an st.iframe (lab/forward_pass.html), so the animation runs smoothly in the browser while Python stays idle.
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st

from lab.lab_pages import key_code_snippet
from lab.nav import go_button
from ui import theme
from utils import llama

TEMPLATE = Path(__file__).with_name("forward_pass.html")
RUN_C = llama.LLAMA2C_DIR / "run.c"
MAX_PROMPT_CHARS = 120
CREDIT = ("Model: stories15M, trained on TinyStories, from Andrej Karpathy's "
          "<a href='https://github.com/karpathy/llama2.c' target='_blank' rel='noopener'>llama2.c</a> (MIT)")

# Each animation stage, and the comment in run.c's forward() where its code starts. Line numbers are found
# at render time, so the highlight follows run.c if it changes.
STAGE_ANCHORS = [
    ("embed", "// copy the token embedding into x"),
    ("attn_norm", "// attention rmsnorm"),
    ("qkv", "// qkv matmuls for this position"),
    ("rope", "// RoPE relative positional encoding"),
    ("attention", "// multihead attention"),
    ("attn_out", "// final matmul to get the output of the attention"),
    ("ffn_norm", "// ffn rmsnorm"),
    ("swiglu", "// Now for FFN in PyTorch"),
    ("ffn_residual", "// residual connection"),  # the second one: the first is attention's, before "ffn rmsnorm"
    ("final_norm", "// final rmsnorm"),
    ("logits", "// classifier into logits"),
]


@st.cache_resource(show_spinner="Loading the 15M-parameter model…")
def load_model():
    return llama.load()


def run_c_code() -> dict | None:
    """forward()'s source lines from run.c, and the line range of each stage."""
    if not RUN_C.exists():
        return None
    lines = RUN_C.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("float* forward("))
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    found, i = [], start
    for stage, anchor in STAGE_ANCHORS:
        i = next(j for j in range(i + 1, end) if lines[j].strip().startswith(anchor))
        found.append((stage, i))
    spans = {}
    for (stage, first), nxt in zip(found, [f for _, f in found[1:]] + [end]):
        last = nxt - 1
        while last > first and lines[last].strip() in ("", "}"):  # don't highlight blank lines and closing braces
            last -= 1
        spans[stage] = [first + 1, last + 1]  # 1-based, like an editor
    spans["sample"] = spans["logits"]
    return {"first": start + 1, "lines": lines[start:end + 1], "spans": spans}


def quantize(x: np.ndarray) -> dict:
    """A vector as whole numbers -99..99 plus one scale, which keeps the JSON small."""
    scale = float(np.abs(x).max()) or 1.0
    return {"s": round(scale, 4), "q": np.round(x / scale * 99).astype(int).tolist()}


def payload(steps: list[dict], tok: llama.Tokenizer, model: llama.Model) -> dict:
    c = model.config
    out = []
    for s in steps:
        t = s["trace"]
        out.append({
            "pos": s["pos"], "token": s["token"], "forced": s["forced"], "next": s["next"],
            "nextProb": round(s["next_prob"], 4),
            "top": [[tok.label(i), round(p, 4)] for i, p in s["top"]],
            "embed": quantize(t["embed"]), "final": quantize(t["final"]),
            "layers": [{
                "att": [np.round(a, 3).tolist() for a in layer["att"]],
                "afterAttn": quantize(layer["after_attn"]), "afterFfn": quantize(layer["after_ffn"]),
                "hidden": quantize(layer["hidden"]),
                "attnNorm": round(layer["attn_norm"], 3), "ffnNorm": round(layer["ffn_norm"], 3),
            } for layer in t["layers"]],
        })
    labels = {s["token"]: tok.label(s["token"]) for s in steps} | {s["next"]: tok.label(s["next"]) for s in steps}
    return {
        "config": {"dim": c.dim, "hidden": c.hidden_dim, "layers": c.n_layers, "heads": c.n_heads,
                   "headSize": c.head_size, "vocab": c.vocab_size, "seqLen": c.seq_len, "params": model.n_params,
                   "hiddenShown": llama.HIDDEN_SHOWN},
        "labels": {str(k): v for k, v in labels.items()},
        "steps": out,
        "start": next((s["pos"] for s in steps if not s["forced"]), 0),  # open on the first word the model picks
        "code": run_c_code(),
    }


def animation_html(data: dict) -> str:
    t = theme.tokens()
    css_vars = "".join(f"--{k}:{v};" for k, v in t.items())
    # json.dumps output can't close the <script> tag early once "</" is escaped
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return (TEMPLATE.read_text().replace("/*__VARS__*/", css_vars).replace("__SCHEME__", theme.mode())
            .replace("__FONT_BODY__", theme.FONT_BODY).replace("__FONT_HEADING__", theme.FONT_HEADING)
            .replace("__FONT_MONO__", theme.FONT_MONO).replace("__DATA__", blob))


def story_html(steps: list[dict], tok: llama.Tokenizer) -> str:
    """The finished text: your prompt in plain ink, the model's words highlighted."""
    parts = []
    for s in steps:
        if s["next"] == llama.BOS:
            break
        piece = tok.decode(s["token"], s["next"]).replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br>")
        parts.append(piece if s["forced"] else f"<span class='gen'>{piece}</span>")
    return "<div class='fp-story'>" + "".join(parts) + "</div>"


def forward_page() -> None:
    st.markdown("<div class='eyebrow'>Overview · how a language model thinks</div>", unsafe_allow_html=True)
    st.title("Inside a forward pass")
    st.markdown("<div class='lede'>A real Llama 2 model writes the next words of your sentence, one word at a time. "
                "Watch each word travel through the network: the same six layers of attention and feed-forward maths "
                "that run inside Llama 2 7B, only narrower.</div>", unsafe_allow_html=True)

    if not llama.files_present():
        st.warning(f"The model files aren't on this computer yet. This page reads `{llama.CHECKPOINT}` and "
                   f"`{llama.TOKENIZER}` from `{llama.LLAMA2C_DIR}` (set `LLAMA2C_DIR` to use another folder).",
                   icon=":material/download:")
        st.code("git clone https://github.com/karpathy/llama2.c.git ~/Projects/llama2.c\ncd ~/Projects/llama2.c\n"
                "curl -L -O https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin", language="bash")
        return

    model, tok = load_model()
    with st.form("fp_form", border=False):
        a, b, c, d = st.columns([5, 2, 2, 1.4], vertical_alignment="bottom")
        prompt = a.text_input("Start the story", "Once upon a time, a robot", max_chars=MAX_PROMPT_CHARS,
                              key="fp_prompt")
        new_words = b.slider("Tokens to write", 1, 40, 12, key="fp_new",
                             help="Each token is a word or a piece of one. The model writes them one at a time.")
        temperature = c.slider("Temperature", 0.0, 1.5, 0.8, 0.1, key="fp_temp",
                               help="0 always picks the most likely token. Higher values take more chances.")
        seed = d.number_input("Seed", 0, 9999, 42, key="fp_seed",
                              help="The same seed and settings give the same story, the same as ./run -s.")
        st.form_submit_button("Write and animate", type="primary", icon=":material/play_arrow:")

    prompt_tokens = tok.encode(prompt)
    steps = llama.generate(model, tok, prompt, len(prompt_tokens) - 1 + new_words, temperature, 0.9, int(seed))
    data = payload(steps, tok, model) | {"temperature": temperature}
    st.iframe(animation_html(data), height=720)  # our own HTML; the prompt only reaches it as JSON data

    st.markdown("#### The finished text")
    st.markdown(story_html(steps, tok), unsafe_allow_html=True)
    st.caption("Plain text is your prompt; highlighted text is what the model wrote. Your prompt became "
               f"{len(prompt_tokens)} tokens, including the start-of-text token <s>.")

    st.markdown("### What you're watching")
    st.markdown(
        "<div class='steps'>"
        "<div class='data'><b>1 · A word becomes numbers</b><span>Each of the 32,000 tokens has a learned row of 288 "
        "numbers. That row is the vector x, drawn as the grid of coloured cells.</span></div>"
        "<div class='llm'><b>2 · Attention looks back</b><span>In every layer, 6 heads each score the earlier tokens "
        "and blend in information from the ones that matter. The tinted chips show where they look.</span></div>"
        "<div class='code'><b>3 · Feed-forward thinks</b><span>SwiGLU widens x to 768 numbers, lets some through, "
        "and folds it back. Both steps add to x rather than replace it.</span></div>"
        "<div class='data'><b>4 · Scores for every token</b><span>After 6 layers, x is compared with all 32,000 "
        "tokens. Softmax turns the scores into probabilities, and one token is drawn.</span></div>"
        "</div>", unsafe_allow_html=True)
    with st.expander("The same network in Python (utils/llama.py)", icon=":material/code:"):
        code, first, last = key_code_snippet("utils/llama.py", "forward")
        st.code(code, language="python")
        st.caption(f"utils/llama.py · lines {first}–{last}. It produces the same stories as run.c, token for "
                   "token (tests/test_llama.py checks it against ./run).")
    with st.expander("How big is this model next to Llama 2 7B?", icon=":material/straighten:"):
        cfg = model.config
        st.markdown(
            "| | This model (stories15M) | Llama 2 7B |\n|---|---|---|\n"
            f"| Parameters | {model.n_params / 1e6:.1f} million | 6.7 billion |\n"
            f"| Width of x (`dim`) | {cfg.dim} | 4,096 |\n"
            f"| Layers | {cfg.n_layers} | 32 |\n"
            f"| Attention heads | {cfg.n_heads} | 32 |\n"
            f"| Feed-forward width | {cfg.hidden_dim} | 11,008 |\n"
            f"| Context length | {cfg.seq_len} tokens | 4,096 tokens |\n"
            f"| Vocabulary | {cfg.vocab_size:,} | 32,000 |\n\n"
            "Same code, same steps, same tokenizer. The 7B model is about 450 times bigger, which is why it knows "
            "far more than children's stories.")
    st.markdown("")
    a, b = st.columns(2)
    with a:
        go_button("eli5", "How this app works (ELI5)")
    with b:
        go_button("explorer", "Explore 1,000+ real AI models", primary=True)
    st.markdown(f"<div class='credit'>{CREDIT}</div>", unsafe_allow_html=True)

