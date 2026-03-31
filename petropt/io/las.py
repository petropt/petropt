"""LAS file reader — wraps lasio into clean DataFrames.

Provides a simple interface for reading LAS 2.0 well log files
and returning tidy pandas DataFrames.
"""

from __future__ import annotations

from pathlib import Path

import lasio
import numpy as np
import pandas as pd


def read_las(path: str | Path) -> pd.DataFrame:
    """Read a LAS file and return a clean DataFrame.

    Reads a LAS 2.0 file using lasio, converts to a pandas DataFrame
    with the depth/index as a column, and cleans up null values.

    Args:
        path: Path to the LAS file.

    Returns:
        DataFrame with depth as first column and all log curves.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid LAS file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LAS file not found: {path}")
    if path.suffix.lower() not in (".las",):
        raise ValueError(f"Not a LAS file: {path}")

    try:
        las = lasio.read(str(path))
    except Exception as e:
        raise ValueError(f"Failed to parse LAS file: {e}") from e

    # Build DataFrame from curves
    df = las.df().reset_index()

    # Replace lasio null values with NaN
    null_val = las.well.NULL.value if hasattr(las.well, "NULL") else -999.25
    if null_val is not None:
        df = df.replace(null_val, np.nan)

    return df


def read_las_header(path: str | Path) -> dict:
    """Read just the header metadata from a LAS file.

    Args:
        path: Path to the LAS file.

    Returns:
        Dict with well header fields (mnemonic -> {value, unit, description}).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LAS file not found: {path}")

    las = lasio.read(str(path))

    header = {}
    for item in las.well:
        header[item.mnemonic] = {
            "value": item.value,
            "unit": item.unit,
            "description": item.descr,
        }
    return header


def list_curves(path: str | Path) -> list[dict]:
    """List all curves in a LAS file.

    Args:
        path: Path to the LAS file.

    Returns:
        List of dicts with 'mnemonic', 'unit', 'description' for each curve.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LAS file not found: {path}")

    las = lasio.read(str(path))
    return [
        {
            "mnemonic": c.mnemonic,
            "unit": c.unit,
            "description": c.descr,
        }
        for c in las.curves
    ]
