import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from utils import get_stock_data, plot_kline

    return get_stock_data, mo, plot_kline


@app.cell
def _(mo):
    mo.md("""
    # K-line Chart Demo
    """)
    return


@app.cell
def _(mo):
    stock_code = mo.ui.text(placeholder="000001", label="Stock Code")
    return (stock_code,)


@app.cell
def _(get_stock_data, stock_code):
    df = get_stock_data(stock_code.value or "000001", adjust="qfq")
    return (df,)


@app.cell
def _(df, plot_kline, stock_code):
    if not df.is_empty():
        fig = plot_kline(df, title=f"{stock_code.value or '000001'} K-line")
        fig.show()
    return


@app.cell
def _(df):
    df.tail(10)
    return


if __name__ == "__main__":
    app.run()
