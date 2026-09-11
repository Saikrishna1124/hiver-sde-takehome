import pandas as pd
import pytest

from src.data_loader import iter_chunks, load_rows


@pytest.fixture()
def tiny_csv(tmp_path):
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"tweet_id": list(range(10)), "text": [f"msg {i}" for i in range(10)]}).to_csv(
        path, index=False
    )
    return path


def test_iter_chunks_yields_expected_chunks(tiny_csv):
    chunks = list(iter_chunks(tiny_csv, chunk_size=4))
    assert [len(c) for c in chunks] == [4, 4, 2]


def test_load_rows_respects_max_rows(tiny_csv):
    assert len(load_rows(tiny_csv, max_rows=7, chunk_size=4)) == 7


def test_load_rows_all(tiny_csv):
    assert len(load_rows(tiny_csv, chunk_size=4)) == 10


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        list(iter_chunks("data/raw/definitely_missing.csv"))
