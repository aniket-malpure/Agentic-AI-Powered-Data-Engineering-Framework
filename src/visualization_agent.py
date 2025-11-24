"""
visualization_agent.py

LLM-Driven Visualization Agent (LangGraph Node)
-----------------------------------------------
Responsibilities:
- Load data from medallion layers (Bronze/Silver/Gold)
- Accept a natural-language instruction
- Generate visualization code using an LLM (matplotlib/seaborn)
- Execute safely and save the output to `data/visualizations/`
- Return both file path and generated code

Author: Aniket Deepak Malpure
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import textwrap
import traceback
from typing import Dict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
import json
import os
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Config
# -----------------------------
VIS_DIR = Path("data/visualizations")
MEDALLION_DIR = Path("data/medallion")
VIS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helper: Load latest table version
# -----------------------------
def _load_table_from_layer(layer: str, table_name: str) -> pd.DataFrame:
    """Load the latest version of a table from medallion storage."""
    table_path = MEDALLION_DIR / layer / table_name
    if not table_path.exists():
        raise FileNotFoundError(f"No table '{table_name}' found in layer '{layer}'")

    versions = sorted(table_path.glob("v*"), reverse=True)
    if not versions:
        raise FileNotFoundError(f"No versions found for '{table_name}'")

    latest_version = versions[0]
    files = list(latest_version.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {latest_version}")

    df = pd.read_parquet(files[0])
    return df


# -----------------------------
# Core: LLM Visualization
# -----------------------------
def llm_visualization(instruction: str, df: pd.DataFrame) -> Dict:
    """
    Upgraded column-aware visualization:
    - Extracts candidate columns from instruction
    - Matches them to actual df columns
    - Ensures LLM only uses existing columns
    - Prevents KeyErrors
    """
    available_columns = [c.lower() for c in df.columns]

    # Extract column-like tokens from instruction
    tokens = re.findall(r"[a-zA-Z0-9_]+", instruction.lower())
    requested_cols = set(tokens)

    matched_cols = [c for c in available_columns if c in requested_cols]

    # If none matched, fallback to safe visualization
    if not matched_cols:
        fallback_prompt = f"""
        You are a data visualization assistant.

        The DataFrame has the following columns:
        {available_columns}

        The user instruction cannot be satisfied because the requested
        columns are not present: {list(requested_cols)}.

        Generate Python code for a SIMPLE, SAFE fallback visualization.

        Requirements:
        - Define a function `visualize(df)` that:
            - Creates a plot based on the instruction.
            - Saves the plot under `data/visualizations/` as a `.png` file.
            - Returns the file path as a string.
        - Use only matplotlib and seaborn (no Plotly, os, open, system, eval, exec, or file I/O outside saving the image).
        - Example:
        ```python
        def visualize(df):
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df['payment_value'], bins=30, kde=True, ax=ax)
            out_path = 'data/visualizations/payment_value_hist.png'
            plt.savefig(out_path)
            plt.close()
            return out_path
        ```
        Output only the Python code between triple backticks.
        """

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke(fallback_prompt)
        code = extract_python_code(response.content)

    else:
        # Build column-aware visualization prompt for LLM
        prompt = f"""
        You are a data visualization expert.

        User instruction: {instruction}

        Available columns in the DataFrame:
        {available_columns}

        Columns that match user request:
        {matched_cols}

        Generate Python code:

        Requirements:
        - Define a function `visualize(df)` that:
            - Creates a plot based on the instruction.
            - Saves the plot under `data/visualizations/` as a `.png` file.
            - Returns the file path as a string.
        - Use only matplotlib and seaborn (no Plotly, os, open, system, eval, exec, or file I/O outside saving the image).
        - Example:
        ```python
        def visualize(df):
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df['payment_value'], bins=30, kde=True, ax=ax)
            out_path = 'data/visualizations/payment_value_hist.png'
            plt.savefig(out_path)
            plt.close()
            return out_path
        ```
        Output only the Python code between triple backticks.
        """

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke(prompt)
        code = extract_python_code(response.content)

    # Execute visualization code safely
    try:
        local_env = {}
        exec(code, local_env)
        vis_fn = local_env["visualize"]
        out_path = vis_fn(df)
        out_path = str(Path(out_path).resolve())

        return {
            "status": "success",
            "path": out_path,
            "used_columns": matched_cols,
            "generated_code": code
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc(),
            "code": code
        }


def extract_python_code(text: str) -> str:
    """Extract ```python ... ``` blocks safely."""
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# -----------------------------
# LangChain Tool
# -----------------------------
@tool("visualize_table")
def visualize_table_tool(layer: str, table_name: str, instruction: str) -> dict:
    """
    LLM-Driven visualization of a medallion layer table.
    Parameters:
        - layer: bronze | silver | gold
        - table_name: name of the table
        - instruction: natural language description of desired visualization
    """
    try:
        df = _load_table_from_layer(layer, table_name)
        return llm_visualization(instruction, df)
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }


# -----------------------------
# LangGraph Node Factory
# -----------------------------
def create_visualization_agent():
    """Return LangGraph ToolNode for visualization stage."""
    return ToolNode(tools=[visualize_table_tool])


# -----------------------------
# Manual test
# -----------------------------
if __name__ == "__main__":
    print("🚀 Running LLM Visualization Agent test...\n")
    test_instruction = (
        "Plot a bar chart showing average payment_value per payment_type "
        "from the olist_order_payments_dataset table."
    )
    result = visualize_table_tool(
        layer="silver",
        table_name="olist_order_payments_dataset",
        instruction=test_instruction
    )
    print(json.dumps(result, indent=2))