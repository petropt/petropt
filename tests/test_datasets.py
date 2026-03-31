"""Tests for dataset loaders."""

import pandas as pd
import pytest
from petropt.datasets import load_sample_production


class TestLoadSampleProduction:
    def test_returns_dataframe(self):
        df = load_sample_production()
        assert isinstance(df, pd.DataFrame)

    def test_columns(self):
        df = load_sample_production()
        assert "date" in df.columns
        assert "well_name" in df.columns
        assert "oil" in df.columns
        assert "gas" in df.columns
        assert "water" in df.columns

    def test_has_data(self):
        df = load_sample_production()
        assert len(df) == 24  # 12 months x 2 wells

    def test_well_names(self):
        df = load_sample_production()
        wells = df["well_name"].unique()
        assert "Well-A" in wells
        assert "Well-B" in wells

    def test_positive_rates(self):
        df = load_sample_production()
        assert (df["oil"] > 0).all()
        assert (df["gas"] > 0).all()
        assert (df["water"] > 0).all()
