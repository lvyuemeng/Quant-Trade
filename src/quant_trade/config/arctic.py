from pathlib import Path
from typing import Any

import arcticdb as adb
import narwhals as nw
import pandas as pd
import polars as pl
import pyarrow as pa
import yaml

from quant_trade.config.logger import log

type DB = adb.Arctic
type Lib = adb.library.Library


class ArcticDB:
    """ArcticDB connection & library manager (lazy, cache-safe)."""

    def __init__(
        self,
        uri: str,
        *,
        output_format=adb.OutputFormat.POLARS,
    ):
        self._uri = uri
        self._output_format = output_format
        self._conn: adb.Arctic | None = None
        self._libs: dict[str, adb.library.Library] = {}

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "ArcticDB":
        with open(config_path) as f:
            conf = yaml.safe_load(f)["database"]

        uri = cls._build_uri(conf)
        return cls(uri)

    @staticmethod
    def _build_uri(conf: dict) -> str:
        base_path = conf.get("base_path")
        arctic = conf["arctic"]

        if uri := arctic.get("uri"):
            base_uri = uri.strip()
        else:
            raw_path = arctic.get("path", "db/")
            path = raw_path.strip() if raw_path and raw_path.strip() else "db/"
            root = Path(base_path) if base_path else Path.cwd()
            base_uri = f"lmdb://{(root / path).resolve().as_posix()}"
        log.info(f"Loading from {base_uri}")

        if map_size := arctic.get("map_size"):
            log.info(f"Loading with {map_size}")
            return f"{base_uri}?map_size={map_size}"
        return base_uri

    @property
    def conn(self) -> adb.Arctic:
        if self._conn is None:
            self._conn = adb.Arctic(
                self._uri,
                output_format=self._output_format,
            )
        return self._conn

    def get_lib(self, name: str, *, create: bool = True):
        if name in self._libs:
            return self._libs[name]

        if name in self.conn.list_libraries():
            lib = self.conn[name]
        elif create:
            lib = self.conn.create_library(name)
        else:
            raise KeyError(f"Library '{name}' does not exist")

        self._libs[name] = lib
        return lib


class ArcticAdapter:
    """Unified, lightweight adapter for ArcticDB read/write.

    Usage:
        # Write (accepts pandas, polars, arrow, narwhals, dict)
        arctic.write(symbol, ArcticAdapter.input(df))     # df can be any supported type

        # Read (always returns Narwhals DataFrame)
        result = arctic.read(symbol)
        df = ArcticAdapter.output(result)                 # always nw.DataFrame
    """

    @staticmethod
    def to_write(data: Any) -> pd.DataFrame:
        """Convert input → ArcticDB-friendly format (pandas DataFrame).
        Priority: Polars > Arrow > Pandas > dict
        """
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()

        if isinstance(data, pa.Table):
            return data.to_pandas()

        if isinstance(data, nw.DataFrame):
            return data.to_pandas()

        if isinstance(data, pd.DataFrame):
            return data

        if isinstance(data, dict):
            return pd.DataFrame(data)

        raise TypeError(f"Unsupported write type: {type(data)}")

    @staticmethod
    def from_read(data: Any) -> pl.DataFrame:
        """Normalize ArcticDB output → Narwhals DataFrame"""
        if hasattr(data, "collect"):
            data = data.collect()

        if hasattr(data, "data"):
            data = data.data

        if isinstance(data, pl.DataFrame):
            return data

        if isinstance(data, (pa.Table, pd.DataFrame)):
            return nw.from_native(data).to_polars()

        if isinstance(data, dict):
            return nw.from_native(pl.DataFrame(data)).to_polars()

        raise TypeError(f"Unsupported read type: {type(data)}")
