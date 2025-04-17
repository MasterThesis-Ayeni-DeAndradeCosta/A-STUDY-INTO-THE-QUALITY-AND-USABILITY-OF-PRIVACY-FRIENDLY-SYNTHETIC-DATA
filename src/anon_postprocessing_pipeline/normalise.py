"""
anon_postprocessing_pipeline.normalise
=====================================

Utility functions that turn ARX‑generated values into something the shared
BinaryEncoder can understand – without refitting, and without any disk I/O.

Typical workflow
----------------
    from anon_postprocessing_pipeline.normalise import (
        build_mode_maps, normalise_dataframe
    )

    # after you have the **raw** train split
    cat_modes, num_modes = build_mode_maps(train_raw_df)

    # normalise all three: train, test, anonymised
    train_raw_df  = normalise_dataframe(train_raw_df,  cat_modes, num_modes)
    test_raw_df   = normalise_dataframe(test_raw_df,   cat_modes, num_modes)
    anonym_df     = normalise_dataframe(anonym_df,     cat_modes, num_modes)

    # then apply the SAME fitted BinaryEncoder everywhere
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regular expressions for ARX artefacts
# ---------------------------------------------------------------------------

_RE_RANGE  = re.compile(r"\[?(\d+)[-–](\d+)\]?")   #  "20-30" or "[20‑30]"
_RE_TOP    = re.compile(r">=?\s*(\d+)")            #  ">=90" or ">90"
_RE_BOTTOM = re.compile(r"<\s*(\d+)")              #  "<5"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mid(a: str, b: str) -> float:
    """Return the midpoint of two numeric strings."""
    return (float(a) + float(b)) / 2.0


def _clean_scalar(
    x: Any,
    *,
    col: str,
    cat_modes: Dict[str, str],
    num_modes: Dict[str, float],
) -> Any:
    """Convert a single cell value to something the encoder accepts."""
    if pd.isna(x) or x == "":
        return np.nan

    s = str(x)

    # ----- closed numeric interval (20‑30) ---------------------------------
    m = _RE_RANGE.fullmatch(s)
    if m:
        a, b = m.groups()
        return _mid(a, b)

    # ----- open interval >=90 ---------------------------------------------
    m = _RE_TOP.fullmatch(s)
    if m:
        return float(m.group(1))

    # ----- open interval <5 ------------------------------------------------
    m = _RE_BOTTOM.fullmatch(s)
    if m:
        return float(m.group(1))

    # ----- partially masked numeric 12.3* ----------------------------------
    if s.endswith("*") and s[:-1].replace(".", "", 1).isdigit():
        return float(s[:-1])

    # ----- suppression token "*" ------------------------------------------
    if s == "*":
        # categorical column → use its mode
        if col in cat_modes:
            return cat_modes[col]
        # numeric column → use its median
        if col in num_modes:
            return num_modes[col]
        return np.nan

    # ----- everything else: leave string (new category) --------------------
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_mode_maps(
    df: pd.DataFrame,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Compute per‑column modes / medians from a *clean* training DataFrame.

    Returns
    -------
    cat_modes : dict {column -> most frequent categorical value}
    num_modes : dict {column -> median numeric value}
    """
    cat_modes: Dict[str, str] = {
        c: df[c].mode(dropna=True).iloc[0]
        for c in df.select_dtypes(include="object").columns
        if not df[c].mode(dropna=True).empty
    }

    num_modes: Dict[str, float] = {
        c: float(df[c].median())
        for c in df.select_dtypes(exclude="object").columns
    }

    return cat_modes, num_modes


def normalise_dataframe(
    df: pd.DataFrame,
    cat_modes: Dict[str, str],
    num_modes: Dict[str, float],
) -> pd.DataFrame:
    """
    Apply `_clean_scalar` column‑wise; returns a **new** normalised DataFrame.
    """
    out = df.copy()

    for col in out.columns:
        out[col] = out[col].apply(
            lambda v: _clean_scalar(v,
                                    col=col,
                                    cat_modes=cat_modes,
                                    num_modes=num_modes)
        )

    return out
