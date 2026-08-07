from __future__ import annotations

import json
import re
from pathlib import Path


def test_third_party_lock_has_immutable_revisions() -> None:
    repository_root = Path(__file__).parents[1]
    lock = json.loads((repository_root / "third_party" / "LOCK.json").read_text(encoding="utf-8"))
    repositories = lock["repositories"]

    assert len(repositories) == 19
    assert len({item["name"] for item in repositories}) == len(repositories)
    assert all(item["url"].startswith("https://github.com/") for item in repositories)
    assert all(re.fullmatch(r"[0-9a-f]{40}", item["commit"]) for item in repositories)
