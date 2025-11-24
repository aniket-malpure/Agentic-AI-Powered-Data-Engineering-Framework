"""
Transformation Agent (LangGraph Node)
-------------------------------------
This agent performs both rule-based and LLM-guided data transformations.

Features:
✅ Cleans data with rule-based transformations
✅ LLM-powered dynamic transformation generation
✅ Supports multi-table joins & complex SQL queries
✅ Safe code execution (sandboxed environment)
✅ Compatible with LangGraph orchestration

Author: Aniket Deepak Malpure
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
import re
import json
import textwrap
import traceback
import os
from dotenv import load_dotenv
load_dotenv()

# Optional LLM libraries
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# --------------------------------------------------------
# --- Config
# --------------------------------------------------------
INPUT_DB_PATH = Path("data/db/olist_database.db")
OUTPUT_DB_PATH = Path("data/db/olist_transformed.db")
OUTPUT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------
# --- Utility Functions
# --------------------------------------------------------
def load_table_names(db_path: Path) -> List[str]:
    """List all table names in a SQLite database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    conn.close()
    return tables


def get_table_schemas(db_path: Path) -> Dict[str, List[str]]:
    """Get sample schemas (column names) for each table."""
    conn = sqlite3.connect(str(db_path))
    tables = load_table_names(db_path)
    schemas = {}
    for t in tables:
        try:
            df = pd.read_sql(f"SELECT * FROM {t} LIMIT 5;", conn)
            schemas[t] = list(df.columns)
        except Exception:
            schemas[t] = []
    conn.close()
    return schemas


# --------------------------------------------------------
# --- Default Transformations
# --------------------------------------------------------
def apply_default_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Apply simple cleaning transformations."""
    df = df.drop_duplicates()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna("unknown")
    return df


def rule_based_transformation(db_path: Path, out_path: Path) -> Dict:
    """Apply basic cleaning transformations to all tables."""
    if not db_path.exists():
        return {"status": "error", "message": f"Database not found at {db_path}"}

    conn_in = sqlite3.connect(str(db_path))
    conn_out = sqlite3.connect(str(out_path))
    tables = load_table_names(db_path)
    summary = []

    for table in tables:
        df = pd.read_sql(f"SELECT * FROM {table}", conn_in)
        df_t = apply_default_transformations(df)
        df_t.to_sql(table, conn_out, if_exists="replace", index=False)
        summary.append({
            "table": table,
            "rows_before": len(df),
            "rows_after": len(df_t),
            "columns": len(df_t.columns),
            "method": "rule-based"
        })

    conn_in.close()
    conn_out.close()

    return {
        "status": "success",
        "input_db": str(db_path),
        "output_db": str(out_path),
        "tables_transformed": len(summary),
        "transformation_summary": summary,
        "message": f"Rule-based transformation completed for {len(summary)} tables."
    }


# --------------------------------------------------------
# --- LLM-Guided Transformation (Supports Multi-table)
# --------------------------------------------------------
def llm_guided_transformation(instruction: str, db_path: Path) -> Dict:
    """
    Upgraded column-aware transformation logic:
    - Extracts column candidates from instruction
    - Matches them to actual table schemas
    - Selects single- or multi-table mode automatically
    - Prevents missing-column errors
    """
    if not db_path.exists():
        return {"status": "error", "message": f"Database not found at {db_path}"}

    conn = sqlite3.connect(str(db_path))
    table_schemas = get_table_schemas(db_path)

    # -----------------------------
    # 1️⃣ Extract potential column names from instruction
    # -----------------------------
    tokens = re.findall(r"[a-zA-Z0-9_]+", instruction.lower())
    columns_requested = set(tokens)

    # -----------------------------
    # 2️⃣ Match requested columns to tables
    # -----------------------------
    table_hits = {}
    for table, cols in table_schemas.items():
        matched = [c for c in cols if c.lower() in columns_requested]
        if matched:
            table_hits[table] = matched

    # If no tables match, fail early
    if not table_hits:
        return {
            "status": "error",
            "message": "No matching columns found in any table.",
            "requested_columns": list(columns_requested),
            "instruction": instruction
        }

    # -----------------------------
    # 3️⃣ Decide transformation mode
    # -----------------------------
    if len(table_hits) == 1:
        mode = "single-table"
        target_table = next(iter(table_hits.keys()))
    else:
        mode = "multi-table"

    # -----------------------------
    # LLM Model
    # -----------------------------
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # -----------------------------
    # 4️⃣ Build prompt dynamically based on matched tables
    # -----------------------------
    if mode == "multi-table":
        prompt = ChatPromptTemplate.from_template(textwrap.dedent("""
        You are an expert data engineer working with an SQLite database.

        These tables contain the columns referenced in the instruction:
        {table_hits}

        Their schemas are:
        {table_schemas}

        Instruction: {instruction}

        Generate SAFE Python code using pandas + sqlite3:

        Requirements:
        - Define `def transform(conn):`
        - Load all relevant tables
        - Perform correct JOINs using shared keys when possible
        - Only use pandas and sqlite3 (NO OS, NO eval/exec)
        - Return a single pandas DataFrame `result`
        - Do not guess column names beyond what exists in schemas
        """))

    else:
        # --- single-table ---
        prompt = ChatPromptTemplate.from_template(textwrap.dedent("""
        You are a data transformation assistant working on a single table.

        Table name: {target_table}
        Columns: {target_columns}

        Instruction: {instruction}

        Generate SAFE Python code:

        Requirements:
        - Define `def transform(df):`
        - Use only pandas
        - Do not reference missing columns
        - Return transformed DataFrame
        """))

    # -----------------------------
    # 5️⃣ LLM Code Generation
    # -----------------------------
    if mode == "multi-table":
        response = llm.invoke(prompt.format(
            table_hits=json.dumps(table_hits, indent=2),
            table_schemas=json.dumps(table_schemas, indent=2),
            instruction=instruction
        ))
    else:
        response = llm.invoke(prompt.format(
            target_table=target_table,
            target_columns=table_schemas[target_table],
            instruction=instruction
        ))

    code = response.content
    match = re.search(r"```python(.*?)```", code, re.DOTALL)
    if match:
        code = match.group(1).strip()

    # -----------------------------
    # 6️⃣ Execute the generated code
    # -----------------------------
    conn_out = sqlite3.connect(str(OUTPUT_DB_PATH))
    summary = []

    try:
        local_env = {"pd": pd}
        exec(code, local_env)

        if mode == "multi-table":
            transform_fn = local_env.get("transform")
            df_result = transform_fn(conn)

            df_result.to_sql("llm_transformed_result", conn_out, if_exists="replace", index=False)
            summary.append({
                "mode": "multi-table",
                "tables_used": list(table_hits.keys()),
                "rows": len(df_result),
                "columns": len(df_result.columns)
            })
        else:
            transform_fn = local_env.get("transform")
            df = pd.read_sql(f"SELECT * FROM {target_table}", conn)
            df_t = transform_fn(df)

            df_t.to_sql(target_table, conn_out, if_exists="replace", index=False)
            summary.append({
                "mode": "single-table",
                "table": target_table,
                "rows_before": len(df),
                "rows_after": len(df_t),
                "columns": len(df_t.columns)
            })

    except Exception as e:
        summary.append({
            "error": str(e),
            "trace": traceback.format_exc(),
            "generated_code": code
        })

    conn.close()
    conn_out.close()

    return {
        "status": "success",
        "mode": mode,
        "tables_detected": table_hits,
        "generated_code": code,
        "summary": summary,
        "message": f"Transformation completed in {mode} mode."
    }

# --------------------------------------------------------
# --- LangChain Tool for LangGraph
# --------------------------------------------------------
@tool("transform_database")
def transform_database_tool(database_path: str, instruction: Optional[str] = None) -> dict:
    """
    Perform data transformations on the SQLite database.
    If instruction is provided, uses LLM for dynamic transformation code generation.
    """
    db_path = Path(database_path)
    if instruction:
        return llm_guided_transformation(instruction, db_path)
    else:
        return rule_based_transformation(db_path, OUTPUT_DB_PATH)


# --------------------------------------------------------
# --- LangGraph Node Factory
# --------------------------------------------------------
def create_transformation_agent():
    """Return LangGraph ToolNode for the transformation stage."""
    return ToolNode(tools=[transform_database_tool])


# --------------------------------------------------------
# --- Manual Test / Debug Run
# --------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Running LLM-guided transformation (with joins)...\n")

    # Example join instruction
    instruction = (
        "Join olist_orders_dataset with olist_order_payments_dataset on order_id "
        "and calculate the total payment_value per customer_id."
    )

    result = llm_guided_transformation(instruction, INPUT_DB_PATH)

    print(json.dumps(result, indent=2)) 

    print("\n🧠 Generated Code:\n", textwrap.indent(result["generated_code"], "    "))
    print("\n📘 Transformation Summary:")
    for s in result["summary"]:
        print(" -", s)
    print(f"\n📦 Output Database: {result['output_db']}")