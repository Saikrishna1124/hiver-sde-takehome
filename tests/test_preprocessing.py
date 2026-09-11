import pandas as pd
import pytest

from src.preprocessing import add_clean_column, clean_text, drop_unusable


def test_clean_text_removes_urls_and_mentions():
    assert clean_text("@AmazonHelp my order https://t.co/abc123 is late") == "my order is late"


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""


def test_clean_text_collapses_whitespace():
    assert clean_text("hello   \n world ") == "hello world"


def test_add_clean_column_keeps_raw():
    df = pd.DataFrame({"text": ["@brand help"]})
    out = add_clean_column(df, "text")
    assert out["text"].iloc[0] == "@brand help"
    assert out["clean_text"].iloc[0] == "help"


def test_add_clean_column_missing_col():
    with pytest.raises(KeyError):
        add_clean_column(pd.DataFrame({"a": [1]}), "text")


def test_drop_unusable_removes_empty():
    df = pd.DataFrame({"clean_text": ["hi", "", "there"]})
    assert len(drop_unusable(df)) == 2
