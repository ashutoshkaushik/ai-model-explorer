"""Checks for the static 'Surprise me' facts. Run: python tests/test_surprise.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.surprise import FACTS  # noqa: E402


def test_at_most_25_facts():
    assert 0 < len(FACTS) <= 25, len(FACTS)


def test_facts_complete_and_unique():
    for year, topic, question, answer, why in FACTS:
        assert 1940 < year < 2030 and topic and question.endswith("?") and answer and why, question
    assert len({f[2] for f in FACTS}) == len(FACTS), "duplicate question"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
