"""Maps unstructured WARN company names to canonical stock tickers via fuzzy matching."""

from __future__ import annotations

import logging
import re

import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# Strip common corporate suffixes before comparing
_CORP_SUFFIX = re.compile(
    r"\b(inc\.?|llc\.?|corp\.?|ltd\.?|co\.?|plc\.?|group|holdings|"
    r"international|services|technologies|solutions|enterprises|"
    r"systems|partners|capital|management)\b",
    flags=re.IGNORECASE,
)
_LOCATION = re.compile(r",.*$")  # drop ", Mountain View" style suffixes


def _normalise(name: str) -> str:
    name = _LOCATION.sub("", str(name))
    name = _CORP_SUFFIX.sub("", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def build_ticker_universe(ticker_to_name: dict[str, str]) -> tuple[list[str], list[str]]:
    """Return (normalised_names, tickers) aligned lists for fuzzy lookup."""
    tickers = list(ticker_to_name.keys())
    names = [_normalise(n) for n in ticker_to_name.values()]
    return names, tickers


def resolve_entities(
    warn_df: pd.DataFrame,
    ticker_to_name: dict[str, str],
    score_cutoff: float = 80.0,
    company_col: str = "company",
) -> pd.DataFrame:
    """Fuzzy-match WARN company names to tickers.

    ticker_to_name: {ticker_symbol: 'Canonical Company Name'}
    Returns warn_df with an added 'ticker' column (NaN where unmatched).

    Matching uses token_sort_ratio to handle re-ordered tokens
    (e.g. 'Google LLC Mountain View' → 'Alphabet Inc').
    """
    names, tickers = build_ticker_universe(ticker_to_name)
    resolved: list[str | None] = []

    for raw in warn_df[company_col]:
        query = _normalise(raw)
        match = process.extractOne(
            query, names, scorer=fuzz.token_sort_ratio, score_cutoff=score_cutoff
        )
        if match:
            idx = names.index(match[0])
            resolved.append(tickers[idx])
        else:
            resolved.append(None)

    result = warn_df.copy()
    result["ticker"] = resolved
    n_matched = result["ticker"].notna().sum()
    logger.info(
        f"Entity resolution: {n_matched}/{len(result)} WARN entries matched "
        f"({len(result) - n_matched} unresolved at cutoff={score_cutoff})"
    )
    return result
