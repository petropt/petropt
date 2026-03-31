"""Dataset loaders for public petroleum engineering datasets.

Provides easy access to:
    - Bundled sample production data
    - Petrobras 3W well event dataset (CC BY 4.0)
    - Norwegian Petroleum Directorate wellbore data (NLOD)
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd


_SAMPLES_DIR = Path(__file__).parent / "samples"


def load_sample_production() -> pd.DataFrame:
    """Load the bundled sample production dataset.

    Returns a small production dataset with two wells (Well-A, Well-B)
    and 12 months of oil, gas, and water rates. Useful for quick
    demonstrations and testing.

    Returns:
        DataFrame with columns: date, well_name, oil, gas, water.

    Example:
        >>> df = load_sample_production()
        >>> df.head()
    """
    csv_path = _SAMPLES_DIR / "sample_production.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Sample data not found at {csv_path}. "
            "Please reinstall petropt."
        )
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


def load_3w(
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load the Petrobras 3W dataset metadata.

    The 3W dataset contains labeled multivariate time series from
    offshore oil wells, used for fault detection and classification.
    Licensed under CC BY 4.0.

    This function returns the metadata/labels. The full time series
    data is large (~2 GB) and should be downloaded separately.

    Args:
        cache_dir: Directory to cache downloaded data. Defaults to
            ~/.petropt/datasets/3w/

    Returns:
        DataFrame with well event metadata.

    References:
        Vargas, R.E.V. et al., "A Realistic and Public Dataset with Rare
        Undesirable Real Events in Oil Wells," Journal of Petroleum Science
        and Engineering, 2019.

    Note:
        First call downloads metadata from GitHub. Subsequent calls
        use the cached version.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".petropt" / "datasets" / "3w"
    else:
        cache_dir = Path(cache_dir)

    cache_file = cache_dir / "3w_metadata.csv"

    if cache_file.exists():
        return pd.read_csv(cache_file)

    # Download from the 3W dataset repository
    try:
        import urllib.request

        url = (
            "https://raw.githubusercontent.com/petrobras/3W/main/"
            "dataset/metadata.csv"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "petropt-python-library")
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
        with open(cache_file, "wb") as f:
            f.write(data)
        return pd.read_csv(cache_file)
    except Exception as e:
        raise ConnectionError(
            f"Failed to download 3W dataset: {e}. "
            "Check your internet connection or download manually from "
            "https://github.com/petrobras/3W"
        ) from e


def load_npd_wellbore(
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load Norwegian Petroleum Directorate wellbore data.

    Provides well metadata from the Norwegian Continental Shelf (NCS),
    including well name, location, operator, status, and dates.
    Licensed under NLOD (Norwegian Licence for Open Government Data).

    Args:
        cache_dir: Directory to cache downloaded data. Defaults to
            ~/.petropt/datasets/npd/

    Returns:
        DataFrame with wellbore metadata.

    Note:
        First call downloads from SODIR (formerly NPD). Subsequent
        calls use the cached version.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".petropt" / "datasets" / "npd"
    else:
        cache_dir = Path(cache_dir)

    cache_file = cache_dir / "npd_wellbore.csv"

    if cache_file.exists():
        return pd.read_csv(cache_file)

    try:
        import urllib.request

        url = (
            "https://factpages.sodir.no/public?/Factpages/external/"
            "tableview/wellbore_exploration_all&rs:Command=Render"
            "&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_telefonkatalog"
            "&CultureCode=en&rs:Format=CSV&Top100=false"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        req = urllib.request.Request(url)
        req.add_header("User-Agent", "petropt-python-library")
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode("utf-8-sig")

        # Save to cache
        with open(cache_file, "w") as f:
            f.write(data)

        return pd.read_csv(io.StringIO(data))
    except Exception as e:
        raise ConnectionError(
            f"Failed to download NPD wellbore data: {e}. "
            "Check your internet connection or download manually from "
            "https://factpages.sodir.no"
        ) from e
