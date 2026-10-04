"""Checks the numpy port of llama2.c against run.c itself. Run: python tests/test_llama.py

Needs stories15M.bin and tokenizer.bin in LLAMA2C_DIR (default ~/Projects/llama2.c); skips otherwise.
The expected stories were printed by run.c with the same settings, e.g.
    ./run stories15M.bin -t 1.0 -s 7 -n 80 -i "The little dragon"
"""

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import llama  # noqa: E402

# (prompt, steps, temperature, seed) → what ./run printed
RUN_C = [
    (("Once upon a time", 60, 0.0, 0),
     "Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she "
     "saw a big, red ball in the sky. It was the sun! She thought it was so pretty.\nLily wanted to play with the "
     "ball"),
    (("The little dragon", 80, 1.0, 7),
     "The little dragon had no name. He was called the Dary- Monster. He was harmless. He liked to explore and play "
     "in the woods.\nOne day, the dragon woke up and looked around. He saw some of his friends, his feathered "
     "friends, his wings and his mustache. They were very excited to be in the woods.\nThe"),
    (("", 80, 0.8, 123),
     "Once upon a time, there was a little girl named Lily. She loved to play outside in the garden. One day, she "
     "saw a bee buzzing around some flowers. She wanted to touch the bee, but her mom said, \"Be careful, Lily. Bees "
     "can sting you.\"\nLily didn't listen to her mom and tried to touch the"),
]


def test_matches_run_c():
    model, tok = llama.load()
    for (prompt, steps, temperature, seed), expected in RUN_C:
        got = llama.text_of(tok, llama.generate(model, tok, prompt, steps, temperature, 0.9, seed))
        assert got == expected, f"{prompt!r}: {got!r}"


def test_config_and_tokenizer():
    model, tok = llama.load()
    c = model.config
    assert (c.dim, c.n_layers, c.n_heads, c.vocab_size, c.seq_len) == (288, 6, 6, 32000, 256)
    assert 15e6 < model.n_params < 15.5e6
    ids = tok.encode("Once upon a time")
    assert ids[0] == llama.BOS and "".join(tok.decode(a, b) for a, b in zip(ids, ids[1:])) == "Once upon a time"
    assert tok.encode("") == [llama.BOS]
    assert tok.label(llama.BOS) == "<s>"


def test_trace_is_causal_and_normalised():
    model, tok = llama.load()
    steps = llama.generate(model, tok, "A cat", 8, 0.0)
    for s in steps:
        assert len(s["trace"]["layers"]) == model.config.n_layers
        for layer in s["trace"]["layers"]:
            assert len(layer["att"]) == model.config.n_heads
            for a in layer["att"]:
                assert len(a) == s["pos"] + 1, "attention covers positions 0..pos, never the future"
                assert abs(a.sum() - 1) < 1e-4
        assert s["forced"] == (s["pos"] < len(tok.encode("A cat")) - 1)
        assert s["top"][0][1] >= s["top"][-1][1]


def test_page_payload():
    from lab import forward_pass  # imports Streamlit, but only uses it inside page functions
    model, tok = llama.load()
    steps = llama.generate(model, tok, "Once upon a time, a robot", 40, 0.8)
    data = forward_pass.payload(steps, tok, model)
    assert data["start"] == len(tok.encode("Once upon a time, a robot")) - 1
    assert len(json.dumps(data)) < 1_500_000, "the iframe gets the whole trace; keep it small"
    code = forward_pass.run_c_code()
    if code:
        spans = [code["spans"][stage] for stage, _ in forward_pass.STAGE_ANCHORS]
        assert all(a <= b for a, b in spans) and spans == sorted(spans), "stages appear in run.c's order"
        assert "rmsnorm" in code["lines"][spans[1][0] - code["first"] + 1]


if __name__ == "__main__":
    if not llama.files_present():
        print(f"skipped: no {llama.CHECKPOINT} / {llama.TOKENIZER} in {llama.LLAMA2C_DIR}")
        sys.exit(0)
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
