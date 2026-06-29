"""Lightweight versioned feature store.

Each ETL run writes a timestamped + content-hashed snapshot alongside the
canonical feature_matrix.parquet that train.py and predict.py already read.
The registry JSON maps every feature to its source, cadence, and PIT safety.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

FEATURES_DIR = Path("data/features")
CANONICAL_PATH = FEATURES_DIR / "feature_matrix.parquet"
REGISTRY_PATH = FEATURES_DIR / "feature_registry.json"
_VERSION_PREFIX = "feature_matrix_"

logger = logging.getLogger(__name__)


def _sha8(df: pd.DataFrame) -> str:
    """8-char SHA-256 of the serialised DataFrame bytes (uncompressed for stability)."""
    buf = df.to_parquet(None, index=False, compression=None)
    return hashlib.sha256(buf).hexdigest()[:8]


def write_versioned_matrix(df: pd.DataFrame) -> Path:
    """Write a versioned snapshot and update the canonical pointer.

    Versioned path: data/features/feature_matrix_{YYYYMMDD}_{sha8}.parquet
    Canonical path: data/features/feature_matrix.parquet  (always latest)

    Skips the versioned write if an identical snapshot already exists today
    (content hash matches), so repeated intra-day runs are idempotent.

    Returns the versioned path.
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    sha = _sha8(df)
    today_str = date.today().strftime("%Y%m%d")
    versioned = FEATURES_DIR / f"{_VERSION_PREFIX}{today_str}_{sha}.parquet"

    if not versioned.exists():
        df.to_parquet(versioned, index=False, compression="snappy")
        logger.info(f"Feature store: wrote {versioned.name} ({df.shape})")
    else:
        logger.info(f"Feature store: snapshot unchanged ({versioned.name})")

    df.to_parquet(CANONICAL_PATH, index=False, compression="snappy")
    return versioned


def get_version_id(path: str | Path) -> str:
    """Return the version ID string (YYYYMMDD_sha8) for a given feature matrix path.

    For versioned filenames the ID is parsed from the filename directly.
    For the canonical path (or any other path) the hash is computed from the
    file contents and returned as 'live_{sha8}'.
    """
    p = Path(path)
    stem = p.stem  # e.g. "feature_matrix_20260627_a3f1bc9e"
    if stem.startswith(_VERSION_PREFIX):
        return stem[len(_VERSION_PREFIX):]  # "20260627_a3f1bc9e"
    df = pd.read_parquet(p)
    return f"live_{_sha8(df)}"


def list_versions() -> list[dict]:
    """Return a sorted list of all versioned snapshots with metadata."""
    result = []
    for f in sorted(FEATURES_DIR.glob(f"{_VERSION_PREFIX}????????_????????.parquet")):
        suffix = f.stem[len(_VERSION_PREFIX):]
        parts = suffix.split("_", 1)
        result.append({
            "version_id": suffix,
            "date":       parts[0] if len(parts) == 2 else "unknown",
            "sha8":       parts[1] if len(parts) == 2 else suffix,
            "path":       str(f),
            "size_mb":    round(f.stat().st_size / 1e6, 2),
        })
    return result


def load_registry() -> dict:
    """Load the feature registry JSON. Returns {} if not found."""
    if not REGISTRY_PATH.exists():
        logger.warning(f"Feature registry not found: {REGISTRY_PATH}")
        return {}
    with open(REGISTRY_PATH) as fh:
        return json.load(fh)
