"""Tests for I/O modules."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from petropt.io.csv import read_production_csv


class TestReadProductionCSV:
    def test_auto_detect_columns(self):
        """Test auto-detection of standard column names."""
        csv_content = (
            "date,well_name,oil,gas,water\n"
            "2023-01-01,Well-A,500,1200,150\n"
            "2023-02-01,Well-A,480,1150,160\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            df = read_production_csv(f.name)

        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "oil" in df.columns
        assert len(df) == 2

    def test_alternate_column_names(self):
        """Test detection of alternate column names."""
        csv_content = (
            "Production_Date,BOPD,MCFD,BWPD\n"
            "2023-01-01,500,1200,150\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            df = read_production_csv(f.name)

        assert "date" in df.columns
        assert "oil" in df.columns
        assert "gas" in df.columns
        assert "water" in df.columns

    def test_explicit_column_override(self):
        """Test explicit column name specification."""
        csv_content = (
            "timestamp,q_oil,q_gas,q_water\n"
            "2023-01-01,500,1200,150\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            df = read_production_csv(
                f.name, date_col="timestamp",
                oil_col="q_oil", gas_col="q_gas", water_col="q_water"
            )

        assert "date" in df.columns
        assert len(df) == 1

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_production_csv("/nonexistent/path.csv")

    def test_no_date_column(self):
        csv_content = "col_a,col_b\n1,2\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            with pytest.raises(ValueError, match="No date column"):
                read_production_csv(f.name)

    def test_sample_production_file(self):
        """Test reading the bundled sample production CSV."""
        sample_path = (
            Path(__file__).parent.parent
            / "petropt" / "datasets" / "samples" / "sample_production.csv"
        )
        df = read_production_csv(sample_path)
        assert len(df) == 24
        assert "date" in df.columns
