from __future__ import annotations

import pandas as pd
from src.data.processing import transform_akshare_hist


def test_transform_akshare_hist_matches_expected_schema(
    mock_akshare_data: pd.DataFrame,
    processed_df,
) -> None:
    out = transform_akshare_hist(mock_akshare_data)

    assert list(out.columns) == list(processed_df.columns)

    for col in processed_df.columns:
        assert out[col].to_list() == processed_df[col].to_list()
