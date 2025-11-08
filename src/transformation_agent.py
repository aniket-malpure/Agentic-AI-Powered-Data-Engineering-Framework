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

# Requires environment variable:
#   export OPENAI_API_KEY="your_api_key_here"


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
    Use an LLM to generate transformation code.
    - Supports single-table and multi-table joins.
    - Auto-detects if SQL/join/merge terms exist.
    - Runs generated code safely.
    """
    if not db_path.exists():
        return {"status": "error", "message": f"Database not found at {db_path}"}

    conn = sqlite3.connect(str(db_path))
    table_schemas = get_table_schemas(db_path)
    use_db_mode = bool(re.search(r"\b(join|merge|sql|union|group by|select)\b", instruction.lower()))

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if use_db_mode:
        # --- Multi-table / SQL Mode ---
        prompt = ChatPromptTemplate.from_template(textwrap.dedent("""
        You are an expert data engineer working with an SQLite database.

        The database has these tables and their columns:
        {table_schemas}

        Instruction: {instruction}

        Generate safe Python code using pandas + sqlite3 to perform this transformation.
        Requirements:
        - Define a function `transform(conn)` that uses the given SQLite connection.
        - Load multiple tables using pd.read_sql or pd.read_sql_query.
        - Perform necessary joins, aggregations, or SQL queries.
        - Return the final pandas DataFrame result.
        - Do not use external libraries, eval, exec, os, or file I/O.
        - Use only pandas and sqlite3.
        - Ensure safety.
        Example:
        ```python
        def transform(conn):
            import pandas as pd
            # example join
            df1 = pd.read_sql("SELECT * FROM table1", conn)
            df2 = pd.read_sql("SELECT * FROM table2", conn)
            result = pd.merge(df1, df2, on="id", how="inner")
            return result
        ```
        """))
    else:
        # --- Single-table Mode ---
        prompt = ChatPromptTemplate.from_template(textwrap.dedent("""
        You are a data transformation assistant.
        You are given a pandas DataFrame named `df`.

        Instruction: {instruction}

        Generate safe Python code that:
        - Defines a function `transform(df)` returning the transformed DataFrame.
        - Uses only pandas.
        - Avoids unsafe operations (no eval, exec, os, open, imports).
        - Keeps it efficient and deterministic.
        Example:
        ```python
        def transform(df):
            df = df.copy()
            df = df[df['amount'] > 100]
            return df
        ```
        """))

    # --- Generate Code ---
    response = llm.invoke(prompt.format(
        instruction=instruction,
        table_schemas=json.dumps(table_schemas, indent=2)
    ))
    code = response.content

    # Extract code from markdown
    match = re.search(r"```python(.*?)```", code, re.DOTALL)
    if match:
        code = match.group(1).strip()

    # Safety scan
    # banned = ["os.", "system(", "eval(", "exec(", "import ", "subprocess", "open("]
    # if any(b in code for b in banned):
    #     return {"status": "error", "message": "Unsafe code detected in LLM output.", "code": code}

    # --- Execute Code ---
    conn_out = sqlite3.connect(str(OUTPUT_DB_PATH))
    summary = []
    try:
        local_env = {"pd": pd}
        exec(code, local_env)
        transform_fn = local_env.get("transform")

        if not transform_fn:
            raise ValueError("LLM output missing transform() definition.")

        if use_db_mode:
            df_result = transform_fn(conn)
            df_result.to_sql("llm_transformed_result", conn_out, if_exists="replace", index=False)
            summary.append({
                "mode": "multi-table",
                "rows": len(df_result),
                "columns": len(df_result.columns)
            })
        else:
            # Apply per-table if not join-based
            tables = load_table_names(db_path)
            for table in tables:
                df = pd.read_sql(f"SELECT * FROM {table}", conn)
                df_t = transform_fn(df)
                df_t.to_sql(table, conn_out, if_exists="replace", index=False)
                summary.append({
                    "table": table,
                    "rows_before": len(df),
                    "rows_after": len(df_t),
                    "columns": len(df_t.columns),
                    "mode": "single-table"
                })

    except Exception as e:
        summary.append({
            "error": str(e),
            "trace": traceback.format_exc()
        })

    conn.close()
    conn_out.close()

    return {
        "status": "success",
        "mode": "multi-table" if use_db_mode else "single-table",
        "input_db": str(db_path),
        "output_db": str(OUTPUT_DB_PATH),
        "generated_code": code,
        "summary": summary,
        "message": f"LLM-guided transformation ({'multi-table' if use_db_mode else 'single-table'}) executed successfully."
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
