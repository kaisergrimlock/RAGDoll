from pathlib import Path

import pytest

from pi_trec.config import CACHE_DIR_ENV


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the default prompt cache at a per-test temp dir, never the repo."""
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "cache"))
