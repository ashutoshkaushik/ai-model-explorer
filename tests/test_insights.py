"""Offline checks for utils/insights.py and the downloader's change log. Run: python tests/test_insights.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from download_data import changes  # noqa: E402
from utils import insights  # noqa: E402
from utils.charts import fit_growth  # noqa: E402
from utils.data_loader import COMPUTE_COL, MODEL_COL, load_data  # noqa: E402

DF = load_data()
FIT = fit_growth(DF, COMPUTE_COL)


def test_race_frames_are_cumulative_top_n():
    frames = insights.race_frames(DF, 2010, top_n=10)
    for year, f in frames.groupby("frame"):
        assert len(f) <= 10 and list(f["rank"]) == list(range(1, len(f) + 1))
        assert (f["year"] <= year).all()
        assert f[COMPUTE_COL].is_monotonic_decreasing
    tops = frames[frames["rank"] == 1].sort_values("frame")[COMPUTE_COL]
    assert tops.is_monotonic_increasing, "the record can only grow"


def test_profile_rank_and_percentile():
    top = DF.loc[DF[COMPUTE_COL].idxmax()]
    p = insights.model_profile(DF, top[MODEL_COL], FIT)
    assert p.rank_in_year == 1 and p.percentile > 0.99 and p.vs_trend > 1


def test_profile_without_compute():
    name = DF.loc[DF[COMPUTE_COL].isna(), MODEL_COL].iloc[0]
    p = insights.model_profile(DF, name, FIT)
    assert p.rank_in_year is None and p.percentile is None and p.vs_trend is None


def test_similar_models_excludes_self():
    sim = insights.similar_models(DF, "AlexNet", n=5)
    assert len(sim) == 5 and "AlexNet" not in set(sim[MODEL_COL])


def test_compare_ratios():
    a, b = (DF.loc[DF[MODEL_COL] == n].iloc[0] for n in ("AlexNet", "GPT-3 175B (davinci)"))
    t = insights.compare(a, b).set_index("metric")
    assert np.isclose(t.at["Training compute", "ratio"], b[COMPUTE_COL] / a[COMPUTE_COL])


def test_times_formatting():
    assert insights.times(4.3) == "4.3×"
    assert insights.times(64000) == "64,000×"
    assert insights.times(0.25) == "1/4.0"
    assert insights.times(6.9e8) == "690 million×"


def test_mojibake_repaired():
    from utils.data_loader import _fix_mojibake
    assert _fix_mojibake("UniversitÃ© de MontrÃ©al") == "Université de Montréal"
    assert _fix_mojibake("224Ã\x97224") == "224×224"
    assert _fix_mojibake("plain text") == "plain text" and _fix_mojibake(None) is None
    text = DF.select_dtypes(include=["object", "string"]).astype(str)
    assert not text.apply(lambda c: c.str.contains("Ã")).any().any()


def test_growth_since_matches_annual_factor():
    assert np.isclose(insights.growth_since(FIT, insights.SECONDS_PER_YEAR), FIT.factor_per_year - 1)
    assert insights.growth_since(FIT, 0) == 0


def test_footprint_units_and_plausibility():
    f = insights.footprint(DF, g_co2_per_kwh=500, water_l_per_kwh=2.0)
    r = f.iloc[0]
    kwh = r["Training power draw (W)"] * r["Training time (hours)"] / 1000
    assert np.isclose(r["energy_mwh"], kwh / 1000) and np.isclose(r["co2_t"], kwh * 500 / 1e6)
    assert np.isclose(r["water_l"], kwh * 2.0)
    assert (f["flop_per_joule"].dropna() <= insights.MAX_FLOP_PER_JOULE).all()
    zero = insights.footprint(DF, 0, 0)
    assert (zero["co2_t"] == 0).all() and (zero["water_l"] == 0).all()


def test_leaderboard_running_record():
    from utils.data_loader import frontier_by_year
    top = frontier_by_year(DF)
    assert top["record_to_date"].is_monotonic_increasing, "the record never goes down"
    assert (top.loc[~top["new_record"], COMPUTE_COL] <= top["record_to_date"].shift()[~top["new_record"]]).all()
    assert (top.loc[top["new_record"], "growth"].dropna() > 1).all()


def test_sci_compact():
    from ui.filters import sci_compact
    assert sci_compact(1.0001e27) == "10²⁷" and sci_compact(3.14e23) == "3.1×10²³" and sci_compact(9.97e24) == "10²⁵"


def test_downloader_changes():
    old = "Model,Publication date\nA,2020-01-01\nB,2021-01-01\n"
    new = "Model,Publication date\nB,2021-01-01\nC,2026-01-01\n"
    d = changes(old, new)
    assert d["added"] == ["C"] and d["removed"] == ["A"] and d["previous_rows"] == 2


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
