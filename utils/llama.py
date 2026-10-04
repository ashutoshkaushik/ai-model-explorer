"""A numpy port of Karpathy's llama2.c (run.c), so the 'Inside a forward pass' page can show a real Llama 2 at work.

It reads the same two files run.c reads, the checkpoint (stories15M.bin) and tokenizer.bin, from LLAMA2C_DIR
(default ~/Projects/llama2.c). forward() mirrors run.c's forward() line for line and also records what happens
inside (every attention weight, the vector x after each step), which the page animates. generate() mirrors
run.c's generate(), including its xorshift random numbers, so the same seed gives the same story as ./run.

Pure numpy, no Streamlit; tested in tests/test_llama.py.
"""

import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LLAMA2C_DIR = Path(os.environ.get("LLAMA2C_DIR", Path.home() / "Projects" / "llama2.c")).expanduser()
CHECKPOINT = "stories15M.bin"
TOKENIZER = "tokenizer.bin"
BOS, EOS = 1, 2
HIDDEN_SHOWN = 96  # how many of the feed-forward's hidden units the trace keeps (all of x is kept)


@dataclass
class Config:
    dim: int  # width of the vector x that flows through the network
    hidden_dim: int  # width of the feed-forward layer
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int  # longest sequence the model was trained on

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads


class Model:
    """The weights, laid out as run.c's memory_map_weights() reads them. Matrices are (out, in), as in run.c."""

    def __init__(self, path: Path):
        raw = path.read_bytes()
        c = Config(*struct.unpack("7i", raw[:28]))
        shared = c.vocab_size > 0  # run.c: a negative vocab_size means the classifier has its own weights
        c.vocab_size = abs(c.vocab_size)
        self.config = c
        w = np.frombuffer(raw, dtype=np.float32, offset=28)
        kv_dim = c.n_kv_heads * c.head_size
        L = c.n_layers
        pos = 0

        def take(*shape):
            nonlocal pos
            n = int(np.prod(shape))
            out = w[pos:pos + n].reshape(shape)
            pos += n
            return out

        self.token_embedding = take(c.vocab_size, c.dim)
        self.rms_att = take(L, c.dim)
        self.wq = take(L, c.dim, c.dim)
        self.wk = take(L, kv_dim, c.dim)
        self.wv = take(L, kv_dim, c.dim)
        self.wo = take(L, c.dim, c.dim)
        self.rms_ffn = take(L, c.dim)
        self.w1 = take(L, c.hidden_dim, c.dim)
        self.w2 = take(L, c.dim, c.hidden_dim)
        self.w3 = take(L, c.hidden_dim, c.dim)
        self.rms_final = take(c.dim)
        take(c.seq_len * c.head_size)  # skip the old precomputed RoPE tables (real and imaginary halves)
        self.wcls = self.token_embedding if shared else take(c.vocab_size, c.dim)
        self.n_params = pos - c.seq_len * c.head_size


class Tokenizer:
    """tokenizer.bin: the 32,000 SentencePiece pieces and their merge scores, read as run.c's build_tokenizer()."""

    def __init__(self, path: Path, vocab_size: int):
        raw = path.read_bytes()
        self.max_token_length = struct.unpack_from("i", raw, 0)[0]
        self.vocab: list[bytes] = []
        self.scores = np.empty(vocab_size, dtype=np.float32)
        off = 4
        for i in range(vocab_size):
            score, length = struct.unpack_from("fi", raw, off)
            off += 8
            self.vocab.append(raw[off:off + length])
            self.scores[i] = score
            off += length
        self.lookup = {piece: i for i, piece in enumerate(self.vocab)}

    def encode(self, text: str) -> list[int]:
        """BOS + byte-pair merges, as run.c's encode(): start from single characters, then keep merging the
        adjacent pair whose merged piece has the highest score."""
        tokens = [BOS]
        if text:
            tokens.append(self.lookup[b" "])  # the dummy-prefix space SentencePiece adds
        for ch in text:
            piece = ch.encode("utf-8")
            if piece in self.lookup:
                tokens.append(self.lookup[piece])
            else:
                tokens.extend(b + 3 for b in piece)  # byte fallback: <0x00> is token 3
        while True:
            best = (-1e10, -1, -1)  # (score, id, index)
            for i in range(len(tokens) - 1):
                merged = self.lookup.get(self.vocab[tokens[i]] + self.vocab[tokens[i + 1]])
                if merged is not None and self.scores[merged] > best[0]:
                    best = (self.scores[merged], merged, i)
            if best[2] == -1:
                return tokens
            _, merged, i = best
            tokens[i:i + 2] = [merged]

    def decode(self, prev_token: int, token: int) -> str:
        piece = self.vocab[token]
        if prev_token == BOS and piece.startswith(b" "):
            piece = piece[1:]
        if piece.startswith(b"<0x") and piece.endswith(b">") and len(piece) == 6:
            piece = bytes([int(piece[3:5], 16)])
        return piece.decode("utf-8", errors="replace")

    def label(self, token: int) -> str:
        """How a token reads on screen: <s> for BOS, ↵ for a newline, ▁ marks a leading space."""
        if token == BOS:
            return "<s>"
        if token == EOS:
            return "</s>"
        piece = self.decode(0, token)
        return piece.replace("\n", "↵").replace(" ", "▁")


def load(directory: Path = LLAMA2C_DIR) -> tuple[Model, Tokenizer]:
    model = Model(directory / CHECKPOINT)
    return model, Tokenizer(directory / TOKENIZER, model.config.vocab_size)


def files_present(directory: Path = LLAMA2C_DIR) -> bool:
    return (directory / CHECKPOINT).exists() and (directory / TOKENIZER).exists()


# --- The network: run.c's forward(), with a trace ------------------------------------------------


def rmsnorm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return weight * (x / np.sqrt(np.mean(x * x) + 1e-5))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class KVCache:
    def __init__(self, c: Config):
        kv_dim = c.n_kv_heads * c.head_size
        self.k = np.zeros((c.n_layers, c.seq_len, kv_dim), dtype=np.float32)
        self.v = np.zeros((c.n_layers, c.seq_len, kv_dim), dtype=np.float32)


def rope(vec: np.ndarray, pos: int, head_size: int) -> np.ndarray:
    """Rotate each (even, odd) pair by pos × freq, where freq falls with the pair's place inside its head."""
    i = np.arange(0, len(vec), 2)
    angle = pos / np.power(10000.0, (i % head_size) / head_size, dtype=np.float32)
    cos, sin = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    v0, v1 = vec[0::2], vec[1::2]
    out = np.empty_like(vec)
    out[0::2] = v0 * cos - v1 * sin
    out[1::2] = v0 * sin + v1 * cos
    return out


def forward(model: Model, cache: KVCache, token: int, pos: int) -> tuple[np.ndarray, dict]:
    """One token through the whole network. Returns (logits, trace); the trace is what the page animates."""
    c = model.config
    hs, kv_mul = c.head_size, c.n_heads // c.n_kv_heads
    x = model.token_embedding[token].copy()  # 1. the token's learned embedding row
    trace = {"embed": x.copy(), "layers": []}
    for l in range(c.n_layers):
        # 2a. attention: RMSNorm, then q, k, v; k and v go into the cache at this position
        xb = rmsnorm(x, model.rms_att[l])
        q = model.wq[l] @ xb
        k = model.wk[l] @ xb
        v = model.wv[l] @ xb
        q, k = rope(q, pos, hs), rope(k, pos, hs)  # RoPE: the angle depends on pos, which encodes word order
        cache.k[l, pos], cache.v[l, pos] = k, v
        heads, att = [], []
        for h in range(c.n_heads):
            kv = (h // kv_mul) * hs
            scores = cache.k[l, :pos + 1, kv:kv + hs] @ q[h * hs:(h + 1) * hs] / np.sqrt(hs)  # causal: 0..pos
            a = softmax(scores)
            att.append(a)
            heads.append(a @ cache.v[l, :pos + 1, kv:kv + hs])  # weighted sum of past values
        attn_out = model.wo[l] @ np.concatenate(heads)
        x = x + attn_out  # residual
        after_attn = x.copy()
        # 2b. feed-forward: RMSNorm, then SwiGLU: w2( silu(w1 x) * w3 x )
        xb = rmsnorm(x, model.rms_ffn[l])
        h1, h3 = model.w1[l] @ xb, model.w3[l] @ xb
        hb = h1 / (1.0 + np.exp(-h1)) * h3
        ffn_out = model.w2[l] @ hb
        x = x + ffn_out  # residual
        trace["layers"].append({
            "att": att, "attn_norm": float(np.linalg.norm(attn_out)), "after_attn": after_attn,
            "hidden": hb[:HIDDEN_SHOWN].copy(), "ffn_norm": float(np.linalg.norm(ffn_out)), "after_ffn": x.copy(),
        })
    x = rmsnorm(x, model.rms_final)  # 3. final norm, then one score per vocabulary token
    trace["final"] = x.copy()
    return model.wcls @ x, trace


# --- Sampling and the generate loop: run.c's sample() and generate() ------------------------------


class Rng:
    """run.c's xorshift* generator, so a seed here gives the same numbers as ./run -s seed."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        self.state = seed & self.MASK

    def u32(self) -> int:
        s = self.state
        s ^= s >> 12
        s ^= (s << 25) & self.MASK
        s ^= s >> 27
        self.state = s
        return ((s * 0x2545F4914F6CDD1D) & self.MASK) >> 32

    def f32(self) -> float:
        return np.float32((self.u32() >> 8) / 16777216.0)


def probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits / np.float32(temperature)) if temperature > 0 else softmax(logits)


def sample(logits: np.ndarray, temperature: float, topp: float, rng: Rng) -> int:
    if temperature == 0:
        return int(np.argmax(logits))
    probs = probabilities(logits, temperature)
    coin = rng.f32()
    if topp <= 0 or topp >= 1:
        return int(min(np.searchsorted(np.cumsum(probs), coin, side="right"), len(probs) - 1))
    # top-p: keep the most likely tokens until their probabilities add up to topp, then pick among them
    cutoff = (1.0 - topp) / (len(probs) - 1)
    idx = np.flatnonzero(probs >= cutoff)
    idx = idx[np.argsort(-probs[idx], kind="stable")]
    cum = np.cumsum(probs[idx], dtype=np.float32)
    last = int(np.argmax(cum > topp)) if (cum > topp).any() else len(idx) - 1
    r = coin * cum[last]
    pick = int(np.searchsorted(cum[:last + 1], r, side="right"))
    return int(idx[min(pick, last)])


def generate(model: Model, tok: Tokenizer, prompt: str, steps: int, temperature: float = 1.0, topp: float = 0.9,
             seed: int = 42, top_k: int = 10) -> list[dict]:
    """run.c's generate(): feed the prompt, then sample until `steps` positions or BOS. One dict per position:
    the input token, the trace, the top_k next-token probabilities, and the token that comes next."""
    rng = Rng(seed)
    prompt_tokens = tok.encode(prompt)
    steps = min(steps, model.config.seq_len)
    cache = KVCache(model.config)
    token, out = prompt_tokens[0], []
    for pos in range(steps):
        logits, trace = forward(model, cache, token, pos)
        forced = pos < len(prompt_tokens) - 1
        nxt = prompt_tokens[pos + 1] if forced else sample(logits, temperature, topp, rng)
        probs = probabilities(logits, temperature)
        top = np.argsort(-probs)[:top_k]
        out.append({"pos": pos, "token": token, "trace": trace, "forced": forced, "next": nxt,
                    "top": [(int(t), float(probs[t])) for t in top], "next_prob": float(probs[nxt])})
        if nxt == BOS:
            break
        token = nxt
    return out


def text_of(tok: Tokenizer, steps: list[dict]) -> str:
    """The text ./run prints for the same run."""
    return "".join(tok.decode(s["token"], s["next"]) for s in steps if s["next"] != BOS)
