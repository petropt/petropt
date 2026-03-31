"""Production CSV reader with auto-detection of columns.

Reads production CSV files with flexible column name matching,
handling common variations in naming conventions.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


# Column name mappings — maps normalized names to possible CSV header values
_DATE_NAMES = {"date", "production_date", "prod_date", "month", "period"}
_WELL_NAMES = {"well", "well_name", "wellname", "well name", "api", "well_id"}
_OIL_NAMES = {"oil", "oil_rate", "oil rate", "oil_bopd", "bopd", "oil_production"}
_GAS_NAMES = {"gas", "gas_rate", "gas rate", "gas_mcfd", "mcfd", "gas_production"}
_WATER_NAMES = {"water", "water_rate", "water rate", "water_bwpd", "bwpd", "water_production"}


def _match_column(columns: list[str], candidates: set[str]) -> str | None:
    """Find a column name matching one of the candidate names."""
    col_lower = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in col_lower:
            return col_lower[candidate]
    # Partial match: check if any candidate is a substring
    for candidate in candidates:
        for col_l, col_orig in col_lower.items():
            if candidate in col_l:
                return col_orig
    return None


def read_production_csv(
    path: str | Path,
    date_col: str | None = None,
    well_col: str | None = None,
    oil_col: str | None = None,
    gas_col: str | None = None,
    water_col: str | None = None,
) -> pd.DataFrame:
    """Read a production CSV file with auto-detection of columns.

    Reads a CSV file and auto-detects date, well name, oil, gas, and water
    columns based on common naming conventions. Column overrides can be
    provided explicitly.

    Args:
        path: Path to the CSV file.
        date_col: Override for date column name.
        well_col: Override for well name column name.
        oil_col: Override for oil rate column name.
        gas_col: Override for gas rate column name.
        water_col: Override for water rate column name.

    Returns:
        DataFrame with standardized column names:
        'date', 'well_name' (if present), 'oil', 'gas', 'water'.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no date column can be detected.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Production file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV file: {path}")

    columns = list(df.columns)

    # Build column mapping
    mapping = {}

    # Date column
    detected_date = date_col or _match_column(columns, _DATE_NAMES)
    if detected_date is None:
        raise ValueError(
            f"No date column found in {path}. Columns: {columns}. "
            "Pass date_col= explicitly."
        )
    mapping[detected_date] = "date"

    # Well column
    detected_well = well_col or _match_column(columns, _WELL_NAMES)
    if detected_well:
        mapping[detected_well] = "well_name"

    # Rate columns
    for override, names, target in [
        (oil_col, _OIL_NAMES, "oil"),
        (gas_col, _GAS_NAMES, "gas"),
        (water_col, _WATER_NAMES, "water"),
    ]:
        detected = override or _match_column(columns, names)
        if detected:
            mapping[detected] = target

    # Select and rename
    result = df[list(mapping.keys())].rename(columns=mapping)

    # Parse dates
    result["date"] = pd.to_datetime(result["date"], format="mixed", dayfirst=False)

    # Ensure numeric columns
    for col in ["oil", "gas", "water"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)

    return result.sort_values("date").reset_index(drop=True)
