import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generate import generate  # noqa: E402
from settings import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides={"llm": {"enabled": False}, "research": {"mode": "cache"}})


@pytest.fixture(scope="session")
def store(cfg, tmp_path_factory):
    """A generated store in a temp out/ dir, shared by the whole session (treat as read-only)."""
    out = tmp_path_factory.mktemp("out")
    generate(cfg, out)
    return out


@pytest.fixture()
def fresh_store(cfg, tmp_path):
    generate(cfg, tmp_path)
    return tmp_path
