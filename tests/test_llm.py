"""Offline checks for Ask the Data: query functions + the tool-use loop (fake client, no API calls).

Run: python tests/test_llm.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from utils import llm  # noqa: E402
from utils.data_loader import load_data  # noqa: E402

DF = load_data()


def test_every_query_runs_with_defaults_and_scopes():
    args = {
        "count_by_org": {"top_n": 5, "year_from": 2020},
        "count_by_domain": {"top_n": 3},
        "top_models_by_compute": {"n": 3, "domain": "language", "organization": "openai"},
        "top_models_by_parameters": {"n": 3, "year_from": 2015, "year_to": 2020},
        "models_in_range": {"year_from": 1990, "year_to": 1995, "limit": 5},
        "domain_trend": {"domain": "Vision"},
        "open_vs_closed": {"by_year": True, "year_from": 2018},
        "growth_rate": {"metric": "Parameters", "start_year": 2012},
        "find_model": {"name": "alphago"},
    }
    assert set(args) == set(llm.QUERIES)
    for name, a in args.items():
        out = llm.run_query(DF, name, a)
        assert isinstance(out, pd.DataFrame) and not out.empty, name
        empty = llm.run_query(DF, name, {}) if name != "find_model" else None
        assert empty is None or isinstance(empty, pd.DataFrame)


def test_query_results_correct():
    top = llm.run_query(DF, "top_models_by_compute", {"n": 1})
    assert top.iloc[0]["Model"] == DF.loc[DF["Training compute (FLOP)"].idxmax(), "Model"]
    orgs = llm.run_query(DF, "count_by_org", {"top_n": 3})
    assert orgs["Models"].is_monotonic_decreasing and len(orgs) == 3
    oc = llm.run_query(DF, "open_vs_closed", {})
    assert oc["Models"].sum() == len(DF)


def test_run_query_rejects_unknown_and_ignores_extra_args():
    try:
        llm.run_query(DF, "drop_table", {})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    out = llm.run_query(DF, "count_by_org", {"top_n": 500, "__import__": "os"})
    assert len(out) == 50  # clamped, unknown arg dropped


def test_growth_rate_insufficient_data():
    out = llm.run_query(DF, "growth_rate", {"start_year": 2026, "metric": "Training cost"})
    assert "note" in out.columns


class FakeClient:
    """Returns scripted responses from client.beta.messages.create and records requests."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def test_ask_tool_loop():
    tool_turn = NS(
        stop_reason="tool_use",
        content=[
            NS(type="thinking", thinking=""),
            NS(type="tool_use", id="tu_1", name="top_models_by_compute", input={"n": 2}),
        ],
    )
    final = NS(stop_reason="end_turn", content=[NS(type="text", text="GPT-6 Astra leads with 1e27 FLOP.")])
    client = FakeClient([tool_turn, final])
    result = llm.ask(client, DF, "largest?", "years 1950-2026")

    assert result.answer.startswith("GPT-6 Astra")
    assert [c.name for c in result.calls] == ["top_models_by_compute"]
    assert len(result.calls[0].result) == 2
    first, second = client.requests
    assert first["model"] == llm.MODEL and first["fallbacks"] == "default"
    assert {t["name"] for t in first["tools"]} == set(llm.QUERIES)
    # Assistant turn echoed unchanged (incl. thinking), then one user turn with the tool result.
    assert second["messages"][1]["content"] is tool_turn.content
    tr = second["messages"][2]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "tu_1" and not tr["is_error"]


def test_ask_bad_tool_args_reported_not_raised():
    bad = NS(stop_reason="tool_use", content=[NS(type="tool_use", id="tu_1", name="nope", input={})])
    final = NS(stop_reason="end_turn", content=[NS(type="text", text="Sorry.")])
    client = FakeClient([bad, final])
    result = llm.ask(client, DF, "q", "")
    assert result.calls[0].error and client.requests[1]["messages"][2]["content"][0]["is_error"]


def test_refusal_is_friendly_error():
    client = FakeClient([NS(stop_reason="refusal", content=[])])
    try:
        llm.ask(client, DF, "q", "")
        raise AssertionError("expected LLMError")
    except llm.LLMError:
        pass


def test_era_stats_serializable():
    import json

    stats = llm.era_stats(DF[DF["year"].between(2010, 2017)], (2010, 2017))
    json.dumps(stats, default=str)
    assert stats["models"] > 0 and len(stats["largest_models_by_compute"]) == 5


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
