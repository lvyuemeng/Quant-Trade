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
    """ArcticDB state management"""

    def __init__(self, config_path: str = "config.yaml"):
        log.info(f"loading from config: {config_path}")
        self._load_config(config_path)
        self._base_path: str | None = None
        self._conn: adb.Arctic | None = None
        self._libs: dict[str, adb.library.Library] = {}

    def _load_config(self, config_path: str):
        with open(config_path) as f:
            db_conf = yaml.safe_load(f)["database"]
            self._base_path: str | None = db_conf["base_path"]
            self.conf: dict[str, Any] = db_conf["arctic"]

    def _load_uri(self) -> str:
        """Convert relative path to absolute path for local storage."""
        if (uri := self.conf.get("uri")) is not None and str(uri).strip():
            base_uri = str(uri).strip()
        else:
            path: str | None = self.conf.get("path")
            if not path:
                path = "db/"
            path = str(path).strip()
            if Path(path).is_absolute():
                return path

            if self._base_path:
                root = Path(self._base_path)
            else:
                root = Path.cwd()
            base_uri = f"lmdb://{(root / path).resolve().as_posix()}"

        log.info(f"loading base uri: {base_uri}")
        if (map_size := self.conf.get("map_size")) is not None:
            log.info(f"loading map size: {map_size}")
            return f"{base_uri}?map_size={map_size}"
        else:
            return f"{base_uri}"

    @property
    def conn(self) -> DB:
        if self._conn is None:
            self._conn = adb.Arctic(
                self._load_uri(), output_format=adb.OutputFormat.POLARS
            )
        return self._conn

    def get_lib(self, lib: str) -> Lib:
        if lib in self._libs:
            return self._libs[lib]
        elif lib in self.conn.list_libraries():
            self._libs[lib] = self.conn[lib]
            return self._libs[lib]
        else:
            self._libs[lib] = self.conn.create_library(lib)
            return self._libs[lib]


class ArcticAdapter:
    """
    Unified, lightweight adapter for ArcticDB read/write.

    Usage:
        # Write (accepts pandas, polars, arrow, narwhals, dict)
        arctic.write(symbol, ArcticAdapter.input(df))     # df can be any supported type

        # Read (always returns Narwhals DataFrame)
        result = arctic.read(symbol)
        df = ArcticAdapter.output(result)                 # always nw.DataFrame
    """

    @staticmethod
    def input(data: Any) -> pd.DataFrame:
        """Convert any input type → pandas DataFrame (ready for ArcticDB.write)"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, (pa.Table, pl.DataFrame)):
            return nw.from_native(data).to_pandas()
        elif isinstance(data, nw.DataFrame):
            return data.to_pandas()
        raise ValueError(
            f"ArcticAdapter.input cannot handle type '{type(data).__name__}'.\n"
            "Supported: pd.DataFrame, dict, pa.Table, pl.DataFrame, nw.DataFrame"
        )

    @staticmethod
    def output(data: Any) -> nw.DataFrame:
        """Normalize ANY ArcticDB output → Narwhals DataFrame"""
        # Handle lazy evaluation
        if hasattr(data, "collect"):  # LazyDataFrame
            data = data.collect()

        # Handle VersionedItem wrapper
        if hasattr(data, "data"):  # VersionedItem
            data = data.data

        if isinstance(data, dict):
            return nw.from_native(pl.DataFrame(data))
        if isinstance(data, (pa.Table, pd.DataFrame, pl.DataFrame)):
            return nw.from_native(data)

        raise ValueError(
            f"ArcticAdapter.output cannot handle type '{type(data).__name__}'.\n"
            "Common ArcticDB outputs: VersionedItem, LazyDataFrame, pa.Table, "
            "pd.DataFrame, pl.DataFrame, dict"
        )
