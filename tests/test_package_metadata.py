from __future__ import annotations

import re
from pathlib import Path

from alpha_foundry import __version__


def test_package_version_matches_pyproject() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', pyproject, flags=re.MULTILINE)
    assert match is not None
    assert __version__ == match.group(1)
