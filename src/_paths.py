"""Central path resolution.

Every module imports paths from here instead of hard-coding "." or absolute
paths, so the repository can be cloned anywhere.

Environment overrides (optional):
    SPATIAL_DATA     directory holding clean_spatial_ext.csv  (default: ../data)
    SPATIAL_RESULTS  directory for generated tables / reports (default: ../results)
    SPATIAL_FIGURES  directory for figures                    (default: ../figures)
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = Path(os.environ.get("SPATIAL_DATA", ROOT / "data")).resolve()
RESULTS = Path(os.environ.get("SPATIAL_RESULTS", ROOT / "results")).resolve()
FIGURES = Path(os.environ.get("SPATIAL_FIGURES", ROOT / "figures")).resolve()


def load_json(name: str):
    """Load a derived JSON asset from data/ or data/derived/<name>.gz."""
    plain = DATA / name
    if plain.exists():
        with open(plain, encoding="utf-8") as f:
            return json.load(f)
    packed = DATA / "derived" / (name + ".gz")
    if packed.exists():
        with gzip.open(packed, "rt", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(
        f"{name} not found. Expected {plain} or {packed}. "
        "See data/README.md for how to obtain the derived assets."
    )
