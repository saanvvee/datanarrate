"""
Module 0 - Missing-data detector (rule R0)
Converts hidden placeholders like "?" into real missing values and
builds an accurate missing-data report.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

# Converted to NaN automatically (after trimming spaces, ignoring case)
PLACEHOLDER_TOKENS = {"?", "", "na", "n/a", "#n/a", "nan", "null", "-", "--", "missing"}

# Could be a real category, so only reported, never converted
AMBIGUOUS_TOKENS = {"none", "unknown", "not available", "not applicable", "undefined"}

# Numeric codes often used for "no data". Reported only.
SENTINEL_NUMBERS = {-9999, -999, 999, 9999, 99999}
SENTINEL_MIN_SHARE = 0.005

SEVERITY_BANDS = [(0.0, "None"), (5.0, "Low"), (15.0, "Moderate"), (40.0, "High")]


def severity(pct: float) -> str:
    if pct <= 0:
        return "None"
    for upper, label in SEVERITY_BANDS[1:]:
        if pct < upper:
            return label
    return "Severe"


@dataclass
class MissingReport:
    table: pd.DataFrame
    n_cells: int
    n_empty: int
    n_hidden: int
    rows_with_missing: int
    n_rows: int
    converted_to_numeric: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_missing(self) -> int:
        return self.n_empty + self.n_hidden

    @property
    def pct_missing(self) -> float:
        return round(self.n_missing / self.n_cells * 100, 2) if self.n_cells else 0.0

    @property
    def pct_rows_with_missing(self) -> float:
        return round(self.rows_with_missing / self.n_rows * 100, 2) if self.n_rows else 0.0

    def summary_lines(self) -> List[str]:
        lines = [
            f"{self.n_missing:,} of {self.n_cells:,} cells are missing ({self.pct_missing}%).",
            f"{self.n_empty:,} were empty cells and {self.n_hidden:,} were hidden placeholders "
            f"(such as '?') that R0 converted to real missing values.",
            f"{self.rows_with_missing:,} of {self.n_rows:,} rows ({self.pct_rows_with_missing}%) "
            f"have at least one missing value.",
        ]
        for col in self.converted_to_numeric:
            lines.append(f"'{col}' was stored as text only because of placeholders; it is now numeric.")
        return lines


def _is_text_column(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s))


def normalize_missing(df: pd.DataFrame):
    """Returns (clean_df, MissingReport). The input DataFrame is not modified."""
    clean = df.copy()
    rows = []
    converted_numeric: List[str] = []
    warnings: List[str] = []
    n_rows = len(clean)

    for col in clean.columns:
        s = clean[col]
        empty = int(s.isna().sum())
        hidden = 0
        found = {}

        if _is_text_column(s):
            text = s.astype("string").str.strip().str.lower()
            mask = (text.isin(list(PLACEHOLDER_TOKENS)) & s.notna()).fillna(False).to_numpy(dtype=bool)
            if mask.any():
                raw_text = s.astype("string").str.strip()
                found = raw_text[mask].value_counts().to_dict()
                hidden = int(mask.sum())
                clean[col] = s.mask(mask)
                as_num = pd.to_numeric(clean[col], errors="coerce")
                if as_num.notna().sum() == clean[col].notna().sum() and as_num.notna().sum() > 0:
                    clean[col] = as_num
                    converted_numeric.append(col)

            amb = text[text.isin(list(AMBIGUOUS_TOKENS)) & s.notna()]
            for tok, n in amb.value_counts().items():
                warnings.append(
                    f"'{col}' has {int(n)} value(s) equal to '{tok}'. This may mean missing, but it "
                    f"could also be a real category, so it was left as it is."
                )
        elif pd.api.types.is_numeric_dtype(s) and n_rows:
            vals = s.dropna()
            for code in SENTINEL_NUMBERS:
                n = int((vals == code).sum())
                if n / n_rows >= SENTINEL_MIN_SHARE and code in (vals.min(), vals.max()):
                    warnings.append(
                        f"'{col}' has {n} value(s) equal to {code}, an extreme value that is often "
                        f"used for 'no data'. It was left as it is; check whether it is real."
                    )

        total = empty + hidden
        pct = round(total / n_rows * 100, 2) if n_rows else 0.0
        rows.append({
            "Column": col,
            "Type": "text" if _is_text_column(clean[col]) else "numeric",
            "Empty cells": empty,
            "Hidden placeholders": hidden,
            "Total missing": total,
            "Missing %": pct,
            "Severity": severity(pct),
            "Placeholders found": ", ".join(f"'{k}' x{v}" for k, v in found.items()),
        })

    table = pd.DataFrame(rows)
    report = MissingReport(
        table=table,
        n_cells=int(clean.size),
        n_empty=int(table["Empty cells"].sum()),
        n_hidden=int(table["Hidden placeholders"].sum()),
        rows_with_missing=int(clean.isna().any(axis=1).sum()),
        n_rows=n_rows,
        converted_to_numeric=converted_numeric,
        warnings=warnings,
    )
    return clean, report