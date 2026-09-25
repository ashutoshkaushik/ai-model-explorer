"""Sanity checks for the log-linear growth-rate fit. Run: python tests/test_growth.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.charts import fit_growth  # noqa: E402
from utils.data_loader import COMPUTE_COL, DATE_COL  # noqa: E402


def synthetic(factor: float, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-01", "2025-12-31", freq="30D")
    years = dates.year + (dates.dayofyear - 1) / 365.25
    values = 1e18 * factor ** (years - 2010) * 10 ** rng.normal(0, noise, len(dates))
    return pd.DataFrame({DATE_COL: dates, COMPUTE_COL: values})


def test_exact_growth_recovered():
    for factor in (2.0, 4.0, 10.0):
        fit = fit_growth(synthetic(factor), COMPUTE_COL)
        assert abs(fit.factor_per_year - factor) / factor < 1e-6, (factor, fit.factor_per_year)


def test_noisy_growth_close():
    fit = fit_growth(synthetic(4.0, noise=0.5), COMPUTE_COL)
    assert 3.5 < fit.factor_per_year < 4.6, fit.factor_per_year


def test_pre_start_year_ignored():
    df = synthetic(4.0)
    # Wild pre-2010 values must not affect the post-2010 fit.
    df.loc[df[DATE_COL].dt.year < 2010, COMPUTE_COL] = 1e40
    fit = fit_growth(df, COMPUTE_COL)
    assert abs(fit.factor_per_year - 4.0) < 1e-6
    assert fit.n == (df[DATE_COL].dt.year >= 2010).sum()


def test_nonpositive_and_missing_dropped():
    df = synthetic(4.0)
    df.loc[::5, COMPUTE_COL] = np.nan
    df.loc[1::7, COMPUTE_COL] = 0
    assert abs(fit_growth(df, COMPUTE_COL).factor_per_year - 4.0) < 1e-6


def test_too_few_points_returns_none():
    assert fit_growth(synthetic(4.0).head(2).assign(**{DATE_COL: pd.Timestamp("2020-01-01")}), COMPUTE_COL) is None


def test_real_data_plausible():
    df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "notable_ai_models.csv")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    fit = fit_growth(df, COMPUTE_COL)
    print(f"Real data: compute grows ~{fit.factor_per_year:.2f}x/yr since 2010 (n={fit.n})")
    # Epoch AI reports roughly 4-5x/yr for notable models since 2010.
    assert 2.5 < fit.factor_per_year < 7, fit.factor_per_year


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
