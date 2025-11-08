"""
storage_agent.py

Medallion Storage Agent (LangGraph Node)
----------------------------------------
Responsibilities:
- Persist tables from a SQLite database into Medallion layers:
  data/medallion/{bronze|silver|gold}/{table_name}/<version>/*.parquet
- Maintain metadata (json + sqlite) with versions, rowcounts, checksum (sha1)
- Support optional partitioning by a column
- Return a summary of persisted artifacts

Usage (example):
    from storage_agent import persist_medallion_tool, create_storage_agent
    persist_medallion_tool(database_path="data/olist_transformed.db", layer="silver")

Author: Aniket Deepak Malpure
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
import json
from datetime import datetime
import hashlib
import os
import sqlite3 as pysql
import traceback

MEDALLION_ROOT = Path("data/medallion")
METADATA_JSON = MEDALLION_ROOT / "metadata.json"
METADATA_DB = MEDALLION_ROOT / "metadata.db"
PARQUET_ENGINE = "pyarrow"  # ensure pyarrow is installed; fallback handled in code


# ----------------------------
# Utility helpers
# ----------------------------
def _ensure_dirs():
    MEDALLION_ROOT.mkdir(parents=True, exist_ok=True)


def _ts_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _content_hash(df: pd.DataFrame) -> str:
    """Simple deterministic hash of dataframe contents (rows + cols)."""
    try:
        # convert to bytes deterministically
        b = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha1(b).hexdigest()
    except Exception:
        # fallback: hash of CSV bytes
        csv = df.to_csv(index=False).encode("utf-8")
        return hashlib.sha1(csv).hexdigest()


def _load_metadata_json() -> Dict[str, Any]:
    if not METADATA_JSON.exists():
        return {"records": []}
    try:
        return json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {"records": []}


def _save_metadata_json(data: Dict[str, Any]):
    METADATA_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _init_metadata_db():
    """Initialize a small sqlite metadata DB for quick queries."""
    METALL_DIR = MEDALLION_ROOT
    METADATA_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = pysql.connect(str(METADATA_DB))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT,
            layer TEXT,
            version_tag TEXT,
            ts_utc TEXT,
            rows INTEGER,
            cols INTEGER,
            checksum TEXT,
            path TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def _insert_metadata_db(row: Dict[str, Any]):
    conn = pysql.connect(str(METADATA_DB))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO versions (table_name, layer, version_tag, ts_utc, rows, cols, checksum, path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["table_name"],
            row["layer"],
            row["version_tag"],
            row["ts_utc"],
            row["rows"],
            row["cols"],
            row["checksum"],
            row["path"],
        ),
    )
    conn.commit()
    conn.close()


# ----------------------------
# Core persistence logic
# ----------------------------
def _write_table_to_medallion(
    df: pd.DataFrame,
    table_name: str,
    layer: str,
    partition_by: Optional[str],
    version_tag: str,
) -> Dict[str, Any]:
    """
    Write `df` to medallion storage under layer/table_name/version_tag/.
    Returns metadata dict for this write.
    """
    layer_dir = MEDALLION_ROOT / layer / table_name / version_tag
    layer_dir.mkdir(parents=True, exist_ok=True)

    # filename: table_part_000.parquet (we don't chunk here; single file)
    filename = layer_dir / f"{table_name}.parquet"

    try:
        # to_parquet with engine fallback
        try:
            df.to_parquet(filename, index=False, engine=PARQUET_ENGINE)
        except Exception:
            # fallback to default engine if pyarrow not present
            df.to_parquet(filename, index=False)

        rows, cols = len(df), len(df.columns)
        checksum = _content_hash(df)

        meta = {
            "table_name": table_name,
            "layer": layer,
            "version_tag": version_tag,
            "ts_utc": _ts_now_iso(),
            "rows": rows,
            "cols": cols,
            "checksum": checksum,
            "path": str(filename),
        }
        return {"status": "success", "meta": meta}
    except Exception as e:
        return {"status": "error", "error": str(e), "trace": traceback.format_exc()}


def _next_version_tag(table_name: str, layer: str) -> str:
    """
    Determine a new version tag for a table in a layer.
    We use: v{N}_{YYYYMMDDTHHMMSSZ}
    """
    records = _load_metadata_json().get("records", [])
    same = [r for r in records if r.get("table_name") == table_name and r.get("layer") == layer]
    next_n = len(same) + 1
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"v{next_n}_{ts}"


# ----------------------------
# Public API (tool)
# ----------------------------
def persist_medallion(database_path: str, layer: str = "bronze", tables: Optional[List[str]] = None,
                      partition_by: Optional[str] = None, version_tag: Optional[str] = None) -> Dict[str, Any]:
    """
    Persist tables from `database_path` into medallion `layer`.
    - database_path: path to SQLite DB
    - layer: one of 'bronze', 'silver', 'gold'
    - tables: optional list of table names to persist (defaults to all)
    - partition_by: optional column name to partition parquet by (not chunking, just folder structure)
    - version_tag: optional explicit version tag; if not provided auto-generated
    """
    _ensure_dirs()
    _init_metadata_db()
    db = Path(database_path)
    if not db.exists():
        return {"status": "error", "message": f"Database not found at {database_path}"}

    conn = sqlite3.connect(str(db))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [r[0] for r in cursor.fetchall()]
    tables_to_write = tables or all_tables

    results = []
    metadata = _load_metadata_json()

    for table in tables_to_write:
        if table not in all_tables:
            results.append({"table": table, "status": "error", "message": "Table not found in DB"})
            continue

        df = pd.read_sql(f"SELECT * FROM {table}", conn)

        # Partitioning: if requested, create subfolders per partition value (not highly performant but simple)
        vtag = version_tag or _next_version_tag(table, layer)

        if partition_by and partition_by in df.columns:
            # write each partition as its own parquet file in version folder
            for part_val, df_part in df.groupby(partition_by):
                # sanitize partition value for folder name
                safe_part = str(part_val).replace("/", "_").replace(" ", "_")
                part_dir = MEDALLION_ROOT / layer / table / vtag / f"{partition_by}={safe_part}"
                part_dir.mkdir(parents=True, exist_ok=True)
                out_path = part_dir / f"{table}_{safe_part}.parquet"
                try:
                    try:
                        df_part.to_parquet(out_path, index=False, engine=PARQUET_ENGINE)
                    except Exception:
                        df_part.to_parquet(out_path, index=False)
                    checksum = _content_hash(df_part)
                    meta = {
                        "table_name": table,
                        "layer": layer,
                        "version_tag": vtag,
                        "ts_utc": _ts_now_iso(),
                        "rows": len(df_part),
                        "cols": len(df_part.columns),
                        "checksum": checksum,
                        "path": str(out_path),
                        "partition_by": partition_by,
                        "partition_value": part_val,
                    }
                    metadata["records"].append(meta)
                    _insert_metadata_db(meta)
                except Exception as e:
                    results.append({"table": table, "status": "error", "message": str(e)})
            results.append({"table": table, "status": "success", "version": vtag, "mode": "partitioned"})
        else:
            # single-file write for the table
            write_res = _write_table_to_medallion(df, table, layer, partition_by, vtag)
            if write_res["status"] == "success":
                meta = write_res["meta"]
                metadata["records"].append(meta)
                _insert_metadata_db(meta)
                results.append({"table": table, "status": "success", "version": vtag, "rows": meta["rows"]})
            else:
                results.append({"table": table, "status": "error", "message": write_res.get("error")})

    conn.close()
    _save_metadata_json(metadata)

    return {"status": "success", "database": database_path, "layer": layer, "results": results, "message": "Persisted medallion layer"}


# ----------------------------
# LangChain Tool wrapper
# ----------------------------
@tool("persist_medallion")
def persist_medallion_tool(database_path: str, layer: str = "bronze", tables: Optional[List[str]] = None,
                           partition_by: Optional[str] = None, version_tag: Optional[str] = None) -> dict:
    """
    Tool wrapper for LangGraph.
    Example call:
      persist_medallion_tool(database_path="data/db/olist_transformed.db", layer="silver")
    """
    return persist_medallion(database_path=database_path, layer=layer, tables=tables,
                             partition_by=partition_by, version_tag=version_tag)


# ----------------------------
# Node factory for LangGraph
# ----------------------------
def create_storage_agent():
    return ToolNode(tools=[persist_medallion_tool])


# ----------------------------
# Manual test
# ----------------------------
if __name__ == "__main__":
    print("🚀 Running storage agent local test...")
    test_db = "data/db/olist_transformed.db"
    out = persist_medallion(test_db, layer="silver")
    print(json.dumps(out, indent=2))
