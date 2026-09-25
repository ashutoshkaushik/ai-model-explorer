"""Claude-powered "Ask the Data" and era summaries.

Claude never writes or runs code here. It picks one of a fixed set of safe query functions via
tool use; we run that function on the (already filtered) DataFrame and hand the result back for
Claude to summarize. The API key comes from st.secrets (or the environment) and is never hardcoded.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic
import pandas as pd
import streamlit as st

from utils.charts import METRICS, fit_growth
from utils.data_loader import (
    ACCESS_COL,
    COMPUTE_COL,
    COST_COL,
    DATE_COL,
    FRONTIER_COL,
    MODEL_COL,
    ORG_COL,
    PARAMS_COL,
)

MODEL = "claude-opus-5"
# Server-side fallback: if a request is declined by a safety classifier, the API retries it on
# Anthropic's recommended fallback model instead of returning a refusal.
BETAS = ["server-side-fallback-2026-07-01"]
MAX_TOOL_ROUNDS = 3
MAX_RESULT_ROWS = 60


class LLMError(Exception):
    """An error with a user-facing message."""


def _api_key() -> str | None:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.getenv("ANTHROPIC_API_KEY")


def get_client() -> anthropic.Anthropic | None:
    key = _api_key()
    return anthropic.Anthropic(api_key=key) if key else None


# --- Safe query functions --------------------------------------------------------------------
# Each takes the filtered DataFrame plus validated keyword args and returns a small DataFrame.

DISPLAY_COLS = [MODEL_COL, "primary_org", "primary_domain", DATE_COL, PARAMS_COL, COMPUTE_COL, COST_COL, ACCESS_COL]


def _subset(df, year_from=None, year_to=None, domain=None, organization=None) -> pd.DataFrame:
    if year_from is not None:
        df = df[df["year"] >= int(year_from)]
    if year_to is not None:
        df = df[df["year"] <= int(year_to)]
    if domain:
        df = df[df["primary_domain"].str.contains(domain, case=False, regex=False)]
    if organization:
        # Match against the full org list so co-developed models count for every partner.
        df = df[df[ORG_COL].fillna("").str.contains(organization, case=False, regex=False)]
    return df


def _clamp(n, lo=1, hi=50, default=10) -> int:
    try:
        return max(lo, min(hi, int(n)))
    except (TypeError, ValueError):
        return default


def _models_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[DISPLAY_COLS].copy()
    out[DATE_COL] = out[DATE_COL].dt.strftime("%Y-%m-%d")
    return out.rename(columns={"primary_org": "Organization", "primary_domain": "Domain"})


def count_by_org(df, top_n=10, year_from=None, year_to=None):
    d = _subset(df, year_from, year_to)
    counts = d.loc[d["primary_org"] != "Unknown", "primary_org"].value_counts().head(_clamp(top_n))
    return counts.rename_axis("Organization").reset_index(name="Models")


def count_by_domain(df, top_n=10, year_from=None, year_to=None):
    d = _subset(df, year_from, year_to)
    return d["primary_domain"].value_counts().head(_clamp(top_n)).rename_axis("Domain").reset_index(name="Models")


def top_models_by_compute(df, n=10, year_from=None, year_to=None, domain=None, organization=None):
    d = _subset(df, year_from, year_to, domain, organization)
    return _models_table(d.nlargest(_clamp(n), COMPUTE_COL))


def top_models_by_parameters(df, n=10, year_from=None, year_to=None, domain=None, organization=None):
    d = _subset(df, year_from, year_to, domain, organization)
    return _models_table(d.nlargest(_clamp(n), PARAMS_COL))


def models_in_range(df, year_from=None, year_to=None, domain=None, organization=None, limit=25):
    d = _subset(df, year_from, year_to, domain, organization)
    table = _models_table(d.sort_values(DATE_COL).head(_clamp(limit)))
    return table.assign(total_matching=len(d))


def domain_trend(df, domain=None, year_from=None, year_to=None):
    d = _subset(df, year_from, year_to, domain)
    return d.groupby("year").size().rename_axis("Year").reset_index(name="Models")


def open_vs_closed(df, by_year=False, year_from=None, year_to=None):
    d = _subset(df, year_from, year_to)
    if by_year:
        t = d.groupby(["year", "open_status"]).size().unstack(fill_value=0).rename_axis(columns=None)
        known = t.get("Open", 0) + t.get("Closed", 0)
        t["open_share_of_known"] = (t.get("Open", 0) / known.where(known > 0)).round(3)
        return t.rename_axis("Year").reset_index()
    counts = d["open_status"].value_counts()
    return counts.rename_axis("Status").reset_index(name="Models").assign(share=lambda x: (x.Models / x.Models.sum()).round(3))


def growth_rate(df, metric="Training compute", start_year=2010):
    if metric not in METRICS:
        metric = "Training compute"
    fit = fit_growth(df, METRICS[metric]["col"], start_year=int(start_year))
    if fit is None:
        return pd.DataFrame({"note": [f"Not enough {metric.lower()} data since {start_year} to fit a trend."]})
    return pd.DataFrame(
        {
            "metric": [metric],
            "start_year": [fit.start_year],
            "growth_per_year": [round(fit.factor_per_year, 2)],
            "doubling_time_months": [round(12 * 0.30103 / fit.slope, 1) if fit.slope > 0 else None],
            "models_in_fit": [fit.n],
        }
    )


def find_model(df, name):
    d = df[df[MODEL_COL].str.contains(str(name), case=False, regex=False, na=False)]
    return _models_table(d.head(10))


_YEARS = {
    "year_from": {"type": "integer", "description": "First publication year to include (inclusive)."},
    "year_to": {"type": "integer", "description": "Last publication year to include (inclusive)."},
}
_SCOPE = {
    "domain": {"type": "string", "description": "Domain substring, e.g. 'Language', 'Vision', 'Biology'."},
    "organization": {"type": "string", "description": "Organization substring, e.g. 'OpenAI', 'Google'."},
}


def _schema(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


QUERIES: dict[str, tuple[Callable[..., pd.DataFrame], str, dict]] = {
    "count_by_org": (
        count_by_org,
        "Count models per organization (primary org), largest first.",
        _schema({"top_n": {"type": "integer", "description": "How many orgs (1-50)."}, **_YEARS}),
    ),
    "count_by_domain": (
        count_by_domain,
        "Count models per domain (Language, Vision, ...), largest first.",
        _schema({"top_n": {"type": "integer", "description": "How many domains (1-50)."}, **_YEARS}),
    ),
    "top_models_by_compute": (
        top_models_by_compute,
        "Models with the highest training compute (FLOP), optionally scoped by years/domain/org.",
        _schema({"n": {"type": "integer", "description": "How many models (1-50)."}, **_YEARS, **_SCOPE}),
    ),
    "top_models_by_parameters": (
        top_models_by_parameters,
        "Models with the most parameters, optionally scoped by years/domain/org.",
        _schema({"n": {"type": "integer", "description": "How many models (1-50)."}, **_YEARS, **_SCOPE}),
    ),
    "models_in_range": (
        models_in_range,
        "List models (oldest first) matching years/domain/org; includes total_matching count.",
        _schema({**_YEARS, **_SCOPE, "limit": {"type": "integer", "description": "Max rows (1-50)."}}),
    ),
    "domain_trend": (
        domain_trend,
        "Number of models published per year, optionally for one domain.",
        _schema({"domain": _SCOPE["domain"], **_YEARS}),
    ),
    "open_vs_closed": (
        open_vs_closed,
        "Open-weight vs closed vs unknown model counts; by_year=true gives a per-year table with open share.",
        _schema({"by_year": {"type": "boolean"}, **_YEARS}),
    ),
    "growth_rate": (
        growth_rate,
        "Log-linear growth rate per year (and doubling time) of a metric since start_year.",
        _schema(
            {
                "metric": {"type": "string", "enum": list(METRICS)},
                "start_year": {"type": "integer", "description": "Fit models from this year on (default 2010)."},
            }
        ),
    ),
    "find_model": (
        find_model,
        "Look up specific models by (partial) name, e.g. 'GPT-4' or 'AlphaGo'.",
        _schema({"name": {"type": "string"}}, ["name"]),
    ),
}

TOOLS = [{"name": name, "description": desc, "input_schema": schema} for name, (_, desc, schema) in QUERIES.items()]


def run_query(df: pd.DataFrame, name: str, args: dict[str, Any]) -> pd.DataFrame:
    """Dispatch a tool call to its query function, dropping any argument not in its schema."""
    if name not in QUERIES:
        raise ValueError(f"Unknown function {name!r}")
    func, _, schema = QUERIES[name]
    allowed = schema["properties"].keys()
    return func(df, **{k: v for k, v in args.items() if k in allowed and v is not None})


# --- Claude calls ----------------------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result: pd.DataFrame | None = None
    error: str | None = None


@dataclass
class AskResult:
    answer: str
    calls: list[ToolCall] = field(default_factory=list)


ASK_SYSTEM = """You answer questions about Epoch AI's Notable AI Models dataset.
The user has already filtered the dataset in the app; every tool runs on that filtered data.
Answer by calling the single most relevant tool (call more only if the question truly needs it).
Then answer in 2-3 sentences using specific numbers from the tool results. Use only the data in the
tool results; if the data can't answer the question, say so briefly. No preamble, no markdown headers."""


def _create(client: anthropic.Anthropic, **kwargs):
    try:
        response = client.beta.messages.create(
            model=MODEL,
            betas=BETAS,
            fallbacks="default",
            output_config={"effort": "low"},
            **kwargs,
        )
    except anthropic.AuthenticationError:
        raise LLMError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in .streamlit/secrets.toml.")
    except anthropic.PermissionDeniedError:
        raise LLMError("This API key doesn't have access to the model.")
    except anthropic.RateLimitError:
        raise LLMError("Rate limited by the Anthropic API. Wait a moment and try again.")
    except anthropic.APIConnectionError:
        raise LLMError("Couldn't reach the Anthropic API. Check your internet connection.")
    except anthropic.APIStatusError as e:
        raise LLMError(f"The Anthropic API returned an error ({e.status_code}). Please try again.")
    if response.stop_reason == "refusal":
        raise LLMError("Claude declined to answer this request.")
    return response


def _text(response) -> str:
    return "\n".join(b.text for b in response.content if b.type == "text").strip()


def _result_payload(table: pd.DataFrame) -> str:
    csv = table.head(MAX_RESULT_ROWS).to_csv(index=False)
    note = f"\n(showing {MAX_RESULT_ROWS} of {len(table)} rows)" if len(table) > MAX_RESULT_ROWS else ""
    return f"{len(table)} rows\n{csv}{note}"


def ask(client: anthropic.Anthropic, df: pd.DataFrame, question: str, filter_note: str) -> AskResult:
    messages: list[dict] = [{"role": "user", "content": f"Current filters: {filter_note}\n\nQuestion: {question}"}]
    calls: list[ToolCall] = []
    for _ in range(MAX_TOOL_ROUNDS + 1):
        response = _create(client, max_tokens=4000, system=ASK_SYSTEM, tools=TOOLS, messages=messages)
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses or len(calls) >= MAX_TOOL_ROUNDS:
            return AskResult(_text(response) or "Claude didn't return an answer.", calls)
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in tool_uses:
            call = ToolCall(block.name, dict(block.input))
            try:
                call.result = run_query(df, block.name, call.args)
                content, is_error = _result_payload(call.result), False
            except Exception as e:  # bad args from the model: report back instead of crashing
                call.error = str(e)
                content, is_error = f"Error: {e}", True
            calls.append(call)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    return AskResult("Claude didn't return an answer.", calls)


def era_stats(df: pd.DataFrame, years: tuple[int, int]) -> dict[str, Any]:
    """Aggregate stats for the era summary prompt (JSON-serializable)."""
    top_compute = df.nlargest(5, COMPUTE_COL)
    fit = fit_growth(df, COMPUTE_COL, start_year=years[0])
    known = df[df["open_status"] != "Unknown"]
    return {
        "years": f"{years[0]}-{years[1]}",
        "models": len(df),
        "organizations": int(df["primary_org"].nunique()),
        "top_organizations": df.loc[df["primary_org"] != "Unknown", "primary_org"].value_counts().head(5).to_dict(),
        "top_domains": df["primary_domain"].value_counts().head(5).to_dict(),
        "models_per_year": df.groupby("year").size().to_dict(),
        "open_weight_share_of_known": round((known["open_status"] == "Open").mean(), 2) if len(known) else None,
        "compute_growth_per_year": round(fit.factor_per_year, 2) if fit else None,
        "largest_models_by_compute": [
            f"{r[MODEL_COL]} ({r['primary_org']}, {r[DATE_COL]:%Y}): {r[COMPUTE_COL]:.2e} FLOP"
            for _, r in top_compute.iterrows()
        ],
        "frontier_models": int(df[FRONTIER_COL].sum()),
    }


ERA_SYSTEM = """You are a concise historian of artificial intelligence. Given aggregate statistics from
Epoch AI's Notable AI Models dataset for a period, write a short narrative (one paragraph, 90-140 words)
of what characterized that period in AI. Weave in the specific numbers and model names from the stats.
You may add widely known historical context, but never invent statistics. Plain prose, no headers or lists."""


def era_summary(client: anthropic.Anthropic, stats: dict[str, Any], filter_note: str) -> str:
    response = _create(
        client,
        max_tokens=2000,
        system=ERA_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Filters: {filter_note}\n\nStats:\n{json.dumps(stats, indent=1, default=str)}",
            }
        ],
    )
    return _text(response) or "Claude didn't return a summary."
