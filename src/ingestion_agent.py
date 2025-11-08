"""
Ingestion Agent (LangGraph Node)
--------------------------------
This agent ingests multiple CSV files from a GitHub repository and stores
them as tables inside a single SQLite database for unified downstream access.

Author: Aniket Deepak Malpure
"""

from langgraph.prebuilt import ToolNode
from langchain.tools import tool
import pandas as pd
import requests
from io import StringIO
import sqlite3
import os
from pathlib import Path
from typing import List, Dict


# --------------------------------------------------------
# --- Configuration
# --------------------------------------------------------
GITHUB_USER = "aniket-malpure"
REPO_NAME = "E-Commerce-Big-Data-Engineering"
BRANCH = "main"
DATA_PATH = "Data"

DEFAULT_CSV_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
]

DB_PATH = Path("data/db/olist_database.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------
# --- Core Helper Functions
# --------------------------------------------------------
def load_csv_from_github(user: str, repo: str, branch: str, data_path: str, filename: str) -> pd.DataFrame:
    """Load a CSV file directly from GitHub (raw link)."""
    url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{data_path}/{filename}"
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text))
        print(f"✅ Loaded {filename} ({len(df)} rows, {len(df.columns)} columns)")
        return df
    else:
        print(f"❌ Failed to load {filename} — HTTP {response.status_code}")
        return None


def store_to_sqlite(df: pd.DataFrame, table_name: str, db_path: Path):
    """Store a DataFrame into SQLite database."""
    conn = sqlite3.connect(str(db_path))
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()


def ingest_all_to_sqlite(csv_files: List[str] = None) -> Dict:
    """Load all CSVs from GitHub and persist to SQLite database."""
    csv_files = csv_files or DEFAULT_CSV_FILES
    conn = sqlite3.connect(str(DB_PATH))
    metadata = []

    for file in csv_files:
        df = load_csv_from_github(GITHUB_USER, REPO_NAME, BRANCH, DATA_PATH, file)
        if df is not None:
            table = file.replace(".csv", "")
            df.to_sql(table, conn, if_exists="replace", index=False)
            metadata.append({"table": table, "rows": len(df), "cols": len(df.columns)})
    conn.close()

    return {
        "status": "success",
        "database": str(DB_PATH),
        "tables": metadata,
        "message": f"Ingested {len(metadata)} tables into {DB_PATH}"
    }


# --------------------------------------------------------
# --- LangChain Tool (for LangGraph)
# --------------------------------------------------------
@tool("github_to_sqlite")
def github_to_sqlite_tool(csv_files: List[str] = None) -> dict:
    """
    Ingest all specified CSVs from a GitHub repository into one SQLite database.
    If csv_files is None, defaults to the Olist dataset files.
    """
    return ingest_all_to_sqlite(csv_files)


# --------------------------------------------------------
# --- LangGraph Node Factory
# --------------------------------------------------------
def create_ingestion_agent():
    """Return LangGraph ToolNode for ingestion."""
    return ToolNode(tools=[github_to_sqlite_tool])


# --------------------------------------------------------
# --- Manual Test / Debug Run
# --------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Running ingestion agent locally...\n")
    result = ingest_all_to_sqlite()
    print("\n📘 Ingestion Summary:")
    for t in result["tables"]:
        print(f" - {t['table']}: {t['rows']} rows, {t['cols']} cols")
    print(f"\n✅ All data stored in SQLite at: {result['database']}")
