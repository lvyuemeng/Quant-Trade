class TestGetStockData:
    def test_column_names(self, processed_df):
        """Test that processed DataFrame has English column names."""
        expected_columns = [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "price_change_pct",
            "color",
        ]
        assert list(processed_df.columns) == expected_columns

    def test_date_parsing(self, processed_df):
        """Test that date column is parsed correctly."""
        dates = processed_df["date"].to_list()
        assert all(d is not None for d in dates)

    def test_volume_conversion(self, processed_df):
        """Test that volume is converted to 万股."""
        volumes = processed_df["volume"].to_list()
        assert volumes[0] == 10000.0

    def test_color_calculation_up(self, processed_df):
        """Test that color is red when close > open."""
        colors = processed_df["color"].to_list()
        assert colors[0] == "#FF0000"

    def test_color_calculation_down(self, processed_df):
        """Test that color is green when close < open."""
        colors = processed_df["color"].to_list()
        assert colors[1] == "#00B800"

    def test_data_sorted_by_date(self, processed_df):
        """Test that data is sorted by date ascending."""
        dates = processed_df["date"].to_list()
        assert dates == sorted(dates)
