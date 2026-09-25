"""Download Epoch AI's Notable AI Models dataset to data/notable_ai_models.csv.

Usage:
    python scripts/download_data.py [--force]

Data: Epoch AI (CC-BY 4.0) — https://epoch.ai/data/notable-ai-models
"""

import argparse
import csv
import io
import sys
from pathlib import Path

import requests

SOURCES = [
    ("Epoch AI", "https://epoch.ai/data/notable_ai_models.csv"),
    (
        "DataHub mirror",
        "https://datahub.io/ai/epoch-data-on-ai-models/_r/-/data/notable_ai_models.csv",
    ),
]

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "notable_ai_models.csv"
TIMEOUT = 60


def looks_like_csv(text: str) -> bool:
    """Reject HTML pages / error bodies masquerading as CSV."""
    head = text.lstrip()[:500].lower()
    if head.startswith("<!doctype") or head.startswith("<html") or "<head" in head:
        return False
    first_line = text.splitlines()[0] if text else ""
    return first_line.count(",") >= 5


def summarize(text: str) -> None:
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[1:]
    print(f"Rows: {len(body)}")
    print(f"Columns ({len(header)}):")
    for col in header:
        print(f"  - {col}")


def download(url: str) -> str:
    resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "ai-model-explorer/0.1"})
    resp.raise_for_status()
    resp.encoding = resp.encoding or "utf-8"
    return resp.text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()

    if DATA_PATH.exists() and not args.force:
        print(f"{DATA_PATH} already exists; skipping download (use --force to re-download).")
        summarize(DATA_PATH.read_text(encoding="utf-8"))
        return 0

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES:
        print(f"Trying {name}: {url}")
        try:
            text = download(url)
        except requests.RequestException as exc:
            print(f"  failed: {exc}")
            continue
        if not looks_like_csv(text):
            print("  failed: response does not look like a CSV (HTML or empty body)")
            continue
        DATA_PATH.write_text(text, encoding="utf-8")
        print(f"  saved to {DATA_PATH}")
        summarize(text)
        return 0

    print(
        "\nAll sources failed. Download the CSV manually from "
        "https://epoch.ai/data/notable-ai-models and save it to data/notable_ai_models.csv",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
