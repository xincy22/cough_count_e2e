from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    env_root = os.environ.get("COUGHCOUNT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    p = Path(__file__).resolve()
    for d in [p, *p.parents]:
        if (d / "pyproject.toml").exists():
            return d
    return p.parents[2]


class ProjectPaths:
    def __new__(cls, *args, **kwargs):
        raise TypeError("ProjectPaths cannot be instantiated")

    root = _find_project_root()

    configs = root / "configs"
    data = root / "data"
    raw = data / "raw"
    processed = data / "processed"

    edgeai_raw = raw / "edgeai"
    edgeai_processed = processed / "edgeai"
    edgeai_manifest_csv = edgeai_processed / "manifest.csv"
    edgeai_splits_json = edgeai_processed / "splits.json"
    edgeai_npy = edgeai_processed / "npy"

    runs = root / "runs"
