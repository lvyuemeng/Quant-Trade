"""Smoke test for acquisition functions using provider interface.

Run:
  uv run scripts/smoke_acquisition.py # ensure you already load by uv pip install . for root package.
  python scripts/smoke_acquisition.py

This script is intentionally verbose and prints explicit checkpoints so we can
debug issues quickly (quoting issues, network failures, schema mismatches, etc.).
"""

from __future__ import annotations

import sys
import traceback


def _run_step(name: str, fn) -> None:
    print("\n" + "=" * 80)
    print(f"STEP: {name}")
    print("=" * 80)
    try:
        fn()
        print(f"STEP OK: {name}")
    except Exception as e:  # noqa: BLE001 - intentional in a smoke script
        print(f"STEP FAIL: {name}")
        print(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()


def main() -> None:
    print("smoke_acquisition: starting")
    print("python:", sys.version)

    from quant_trade.data.acquisition import DataAcquisition

    # Initialize with default AkShare provider
    acq = DataAcquisition()

    _run_step(
        "CSI300 index valuation (000300)",
        lambda: (
            (
                lambda df: (
                    print("rows=", df.height),
                    print("cols=", df.columns),
                    print(df.head(5)),
                )
            )(acq.fetch_index_valuation_csindex(symbol="000300"))
        ),
    )

    _run_step(
        "Macro indicators (20240101-20240110)",
        lambda: (
            (
                lambda df: (
                    print("rows=", df.height),
                    print("cols=", df.columns),
                    print(df.head(5)),
                )
            )(acq.fetch_macro_indicators(start_date="20240101", end_date="20240110"))
        ),
    )

    _run_step(
        "Industry classification (capped; max_sw3=2)",
        lambda: (
            (
                lambda df: (
                    print("rows=", df.height),
                    print("cols=", df.columns),
                    print(df.head(10)),
                )
            )(acq.fetch_industry_classification(trade_date="20240131", max_sw3=2))
        ),
    )

    _run_step(
        "Stock code and name mapping (first 5)",
        lambda: (
            (
                lambda df: (
                    print("rows=", df.height),
                    print("cols=", df.columns),
                    print(df.head(5)),
                )
            )(acq.fetch_stock_name_code_map())
        ),
    )

    _run_step(
        "Stock info for 000001 (Ping An Bank)",
        lambda: (
            (
                lambda df: (
                    print("rows=", df.height),
                    print("cols=", df.columns),
                    print(df.head(10)),
                )
            )(acq.fetch_stock_info("000001"))
        ),
    )

    print("\nsmoke_acquisition: done")


if __name__ == "__main__":
    main()
