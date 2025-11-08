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
    Use an LLM to generate visualization code safely.
    - Only matplotlib/seaborn are allowed.
    - Saves output to data/visualizations/.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(textwrap.dedent("""
    You are a data visualization assistant.
    You are given a pandas DataFrame named `df`.

    Instruction: {instruction}

    Generate **safe Python code** using matplotlib and seaborn to visualize the data.

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
    """))

    response = llm.invoke(prompt.format(instruction=instruction))
    code = response.content

    match = re.search(r"```python(.*?)```", code, re.DOTALL)
    if match:
        code = match.group(1).strip()

    banned = ["os.", "system(", "eval(", "exec(", "open(", "subprocess"]
    if any(b in code for b in banned):
        return {"status": "error", "message": "Unsafe code detected in LLM output", "code": code}

    local_env = {"pd": pd, "sns": sns, "plt": plt}
    try:
        exec(code, local_env)
        vis_fn = local_env.get("visualize")
        if not vis_fn:
            raise ValueError("LLM output missing visualize(df) function.")
        out_path = vis_fn(df)
        return {
            "status": "success",
            "plot_path": out_path,
            "generated_code": code,
            "message": "Visualization generated successfully."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc(),
            "code": code
        }


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
