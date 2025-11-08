"""
Validation Agent (LangGraph Node)
---------------------------------
This agent validates the ingested data stored in the SQLite database.
It checks for schema consistency, missing values, and data types.

Author: Aniket Deepak Malpure
"""

import sqlite3
import pandas as pd
from pathlib import Path
from langchain.tools import tool
from langgraph.prebuilt import ToolNode


# --------------------------------------------------------
# --- Core Validation Logic
# --------------------------------------------------------
def validate_sqlite_database(db_path: str) -> dict:
    """Perform schema and data validation for all tables in the database."""
    db = Path(db_path)
    if not db.exists():
        return {"status": "error", "message": f"Database not found at {db_path}"}

    conn = sqlite3.connect(str(db))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    report = []

    for table in tables:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        table_report = {"table": table, "rows": len(df), "columns": len(df.columns)}

        # Missing values summary
        missing = df.isnull().sum()
        missing_summary = {col: int(count) for col, count in missing.items() if count > 0}

        # Data type detection
        dtypes = df.dtypes.apply(lambda x: str(x)).to_dict()

        # Basic health checks
        table_report.update({
            "missing_values": missing_summary,
            "dtypes": dtypes,
            "missing_count": sum(missing_summary.values()),
        })

        # Optional: quick anomaly detection
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if numeric_cols:
            desc = df[numeric_cols].describe().T.reset_index().rename(columns={"index": "column"})
            table_report["numeric_summary"] = desc.to_dict(orient="records")

        report.append(table_report)

    conn.close()

    return {
        "status": "success",
        "database": db_path,
        "table_count": len(report),
        "validation_report": report,
        "message": f"Validated {len(report)} tables successfully"
    }


# --------------------------------------------------------
# --- LangChain Tool for LangGraph
# --------------------------------------------------------
@tool("validate_database")
def validate_database_tool(database_path: str) -> dict:
    """
    Validate all tables inside the provided SQLite database.
    Returns schema summary, missing value report, and basic stats.
    """
    return validate_sqlite_database(database_path)


# --------------------------------------------------------
# --- LangGraph Node Factory
# --------------------------------------------------------
def create_validation_agent():
    """Return LangGraph ToolNode for validation tasks."""
    return ToolNode(tools=[validate_database_tool])


# --------------------------------------------------------
# --- Manual Test / Debug Run
# --------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Running validation agent locally...\n")
    db_path = "data/db/olist_database.db"
    result = validate_sqlite_database(db_path)

    if result["status"] == "success":
        print(f"\n✅ {result['message']}")
        print(f"Tables validated: {result['table_count']}")
        print("Report:")
        for t in result["validation_report"]:
            print(f" - {t['table']}: {t['rows']} rows, {t['columns']} cols, missing={t['missing_count']}")
    else:
        print("❌ Validation failed:", result["message"])
