"""
anon_postprocessing_pipeline.normalise
--------------------------------------

Cleans ARX‑generated artefacts *before* they reach the shared BinaryEncoder.

Main ideas
~~~~~~~~~~
* Convert ranges like ``"[20‑30]"`` or ``"20-30"`` → midpoint (= 25.0).
* Convert top / bottom coding (``">=90"``, ``"<5"``) → numeric bound.
* Convert partially masked numbers (``"12.3*"``) → 12.3.
* Convert suppression token ``"*"``
  → column mode (categorical) *or* column median (numeric).
* **NEW:** any *brand‑new* categorical label that never occurred in the raw
  training data is also replaced by the column’s mode.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------#
# 1) Regular‑expression patterns recognised in ARX output                    #
# ---------------------------------------------------------------------------#
_RE_RANGE  = re.compile(r"\[?(\d+)[-–](\d+)\]?")       # "[20‑30]" or "20-30"
_RE_TOP    = re.compile(r">=?\s*(\d+)")                # ">=90" or ">90"
_RE_BOTTOM = re.compile(r"<\s*(\d+)")                  # "<5"


# ---------------------------------------------------------------------------#
# 2) Convenience helpers                                                     #
# ---------------------------------------------------------------------------#
def _mid(a: str, b: str) -> float:
    """Mid-point between *a* and *b* (numeric strings)."""
    return (float(a) + float(b)) / 2.0


# ---------------------------------------------------------------------------#
# 3) Core scalar cleaner                                                     #
# ---------------------------------------------------------------------------#
def _clean_scalar(
    x: Any,
    *,
    col: str,
    cat_modes: Dict[str, str],
    cat_uniques: Dict[str, set],
    num_modes: Dict[str, float],
) -> Any:
    """
    Convert a single cell value to something the encoder has already seen.

    Parameters
    ----------
    x            : the cell value (any type).
    col          : column name the value belongs to.
    cat_modes    : per-column categorical mode.
    cat_uniques  : per-column set of *all* categories that appeared in raw
                   **training** data.
    num_modes    : per-column numeric median (used as fallback).

    Returns
    -------
    Cleaned value (numeric, np.nan or *an existing* string category).
    """
    # ------------------------------------------------------------------#
    # a) Missing or empty string → NaN                                   #
    # ------------------------------------------------------------------#
    if pd.isna(x) or x == "":
        return np.nan

    s = str(x)  # Work with a string representation

    # ------------------------------------------------------------------#
    # b) Closed numeric interval: "[20-30]" or "20-30"                  #
    # ------------------------------------------------------------------#
    m = _RE_RANGE.fullmatch(s)
    if m:
        a, b = m.groups()
        return _mid(a, b)

    # ------------------------------------------------------------------#
    # c) Top‑coded (">=90")                                             #
    # ------------------------------------------------------------------#
    m = _RE_TOP.fullmatch(s)
    if m:
        return float(m.group(1))   # keeps upper bound

    # ------------------------------------------------------------------#
    # d) Bottom‑coded ("<5")                                            #
    # ------------------------------------------------------------------#
    m = _RE_BOTTOM.fullmatch(s)
    if m:
        return float(m.group(1))   # keeps lower bound

    # ------------------------------------------------------------------#
    # e) Partially masked numeric ("12.3*")                             #
    # ------------------------------------------------------------------#
    if s.endswith("*") and s[:-1].replace(".", "", 1).isdigit():
        return float(s[:-1])

    # ------------------------------------------------------------------#
    # f) Suppression token "*"                                          #
    # ------------------------------------------------------------------#
    if s == "*":
        # Prefer categorical mode; fall back to numeric median; last resort NaN
        return cat_modes.get(col, num_modes.get(col, np.nan))
    
    # g) NEW: Fallback for stringified float lists like "[30.9, 34.3]"
    if s.startswith("[") and s.endswith("]"):
        try:
            nums = [float(x.strip()) for x in s[1:-1].split(",")]
            return sum(nums) / len(nums) if nums else np.nan
        except Exception:
            return np.nan

    # ------------------------------------------------------------------#
    # h) Any *other* categorical string                                #
    #    -> if we've never seen it in raw training data,                #
    #       replace by the column's mode.                               #
    # ------------------------------------------------------------------#
    if s not in cat_uniques.get(col, set()):
        return cat_modes.get(col, s)   # use mode; if col not categorical, keep
    
   

    # String is an *existing* category → keep as‑is
    return s


# ---------------------------------------------------------------------------#
# 4) Build lookup dictionaries from the *raw TRAIN* split                    #
# ---------------------------------------------------------------------------#
def build_mode_maps(
    df: pd.DataFrame,
) -> Tuple[Dict[str, str], Dict[str, set], Dict[str, float]]:
    """
    Scan raw training data and compute:

    * cat_modes   : most frequent label per categorical column
    * cat_uniques : *all* labels observed per categorical column
    * num_modes   : median per numeric column

    These three dicts travel together through the pipeline.
    """
    # ---- categorical columns -------------------------------------------
    cat_cols = df.select_dtypes(include="object").columns
    cat_modes: Dict[str, str] = {
        col: df[col].mode(dropna=True).iloc[0]  # first mode if multimodal
        for col in cat_cols
        if not df[col].mode(dropna=True).empty
    }
    cat_uniques: Dict[str, set] = {
        col: set(df[col].dropna().unique())
        for col in cat_cols
    }

    # ---- numeric columns -----------------------------------------------
    num_cols = df.select_dtypes(exclude="object").columns
    num_modes: Dict[str, float] = {
        col: float(df[col].median())
        for col in num_cols
    }

    return cat_modes, cat_uniques, num_modes


# ---------------------------------------------------------------------------#
# 5) Column‑wise normalisation                                               #
# ---------------------------------------------------------------------------#
def normalise_dataframe(
    df: pd.DataFrame,
    cat_modes: Dict[str, str],
    cat_uniques: Dict[str, set],
    num_modes: Dict[str, float],
) -> pd.DataFrame:
    """
    Apply `_clean_scalar` to every cell in *df*; returns a **new** DataFrame.

    All heavy work happens inside `_clean_scalar`.
    """
    def _apply(col: pd.Series) -> pd.Series:
        col_name = str(col.name) if col.name is not None else ""
        return col.apply(
            lambda v: _clean_scalar(v,
                                    col=col_name,
                                    cat_modes=cat_modes,
                                    cat_uniques=cat_uniques,
                                    num_modes=num_modes)
        )

    return df.apply(_apply)
