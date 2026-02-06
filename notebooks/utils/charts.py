import narwhals as nw
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_kline(
    df: nw.DataFrame,
    title: str = "K-line",
    show_volume: bool = True,
    show_range_selector: bool = True,
) -> go.Figure:
    fig = make_subplots(
        rows=2 if show_volume else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("K-line", "Volume") if show_volume else (title,),
        row_heights=[0.7, 0.3] if show_volume else [1.0],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#FF0000",
            decreasing_line_color="#00B800",
            increasing_fillcolor="#FF0000",
            decreasing_fillcolor="#00B800",
            name="OHLC",
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Open: %{open:.2f}<br>"
                "High: %{high:.2f}<br>"
                "Low: %{low:.2f}<br>"
                "Close: %{close:.2f}<br>"
                "Change: %{customdata:.2f}%"
                "<extra></extra>"
            ),
            customdata=df["price_change_pct"],
        ),
        row=1,
        col=1,
    )

    if show_volume:
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                marker_color=df["color"],
                name="Volume",
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Volume: %{y:.2f}万股<br><extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=800 if show_volume else 600,
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
    )

    if show_range_selector:
        fig.update_xaxes(
            rangebreaks=[
                {"bounds": ["sat", "mon"]},
            ],
            rangeselector={
                "buttons": [
                    {
                        "count": 1,
                        "label": "1M",
                        "step": "month",
                        "stepmode": "backward",
                    },
                    {
                        "count": 3,
                        "label": "3M",
                        "step": "month",
                        "stepmode": "backward",
                    },
                    {
                        "count": 6,
                        "label": "6M",
                        "step": "month",
                        "stepmode": "backward",
                    },
                    {"count": 1, "label": "YTD", "step": "year", "stepmode": "todate"},
                    {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ],
                "xanchor": "left",
                "x": 0.01,
            },
            row=1,
            col=1,
        )

    fig.update_yaxes(title_text="Price (CNY)", row=1, col=1)
    fig.update_yaxes(title_text="Volume (万股)", row=2, col=1)

    return fig
