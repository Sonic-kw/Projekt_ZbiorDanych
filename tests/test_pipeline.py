from unittest.mock import Mock

import pandas as pd
import pytest

from src.main import get_data_source, prepare_dataset, read_existing_data


def _base_config(raw_path: str, processed_path: str, data_source: str) -> dict:
    return {
        "pipeline": {"data_source": data_source},
        "cleaning": {"outlier_threshold": 1.5},
        "paths": {
            "raw_data": raw_path,
            "processed_data": processed_path,
        },
    }


def test_get_data_source_defaults_to_scrape():
    assert get_data_source({}) == "scrape"


def test_get_data_source_rejects_unknown_value():
    with pytest.raises(ValueError):
        get_data_source({"pipeline": {"data_source": "cache"}})


def test_read_existing_data_raises_for_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        read_existing_data(str(missing_path), "raw")


def test_prepare_dataset_uses_existing_processed_data(tmp_path):
    processed_path = tmp_path / "processed.csv"
    raw_path = tmp_path / "raw.csv"
    expected = pd.DataFrame({
        "brand": ["Honda"],
        "model": ["CB500"],
        "year": [2020],
        "mileage": [10000],
        "price": [20000],
    })
    expected.to_csv(processed_path, index=False)

    result = prepare_dataset(_base_config(str(raw_path), str(processed_path), "processed"), Mock())

    pd.testing.assert_frame_equal(result, expected)


def test_prepare_dataset_uses_existing_raw_data_and_saves_processed(tmp_path):
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed" / "cleaned.csv"
    raw_df = pd.DataFrame({
        "brand": ["Honda Motor", "Honda Motor", "Bmw"],
        "model": ["CB500", "CB500", "R 1250 GS"],
        "year": [2020, 2020, 2021],
        "mileage": [10000, 10000, 8000],
        "price": [20000, 20000, 65000],
    })
    raw_df.to_csv(raw_path, index=False)

    result = prepare_dataset(_base_config(str(raw_path), str(processed_path), "raw"), Mock())

    assert processed_path.exists()
    assert result["brand"].tolist() == ["Honda", "BMW"]
    assert len(result) == 2
