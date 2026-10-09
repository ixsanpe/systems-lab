"""Canonical project paths. Everything is relative to the project root, not the CWD."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DOWNLOADS = RAW / "_downloads"
PILAR = RAW / "pilar"
FLORES = RAW / "flores"
PROCESSED = DATA / "processed"
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
