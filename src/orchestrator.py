"""
FINAL WORKING ORCHESTRATOR — NO TOOLNODE ERRORS

This version:
- Does NOT invoke ToolNode for ingestion/validation/transformation/storage
- CALLS their Python functions directly
- Only visualization uses LLM (but NOT ToolNode invocation!)
- Runs reliably inside Streamlit
"""

from typing import TypedDict, Dict, Any

# Import direct functions
from src.ingestion_agent import ingest_all_to_sqlite, DB_PATH
from src.validation_agent import validate_sqlite_database
from src.transformation_agent import transform_database_tool
from src.storage_agent import persist_medallion
from src.visualization_agent import llm_visualization, _load_table_from_layer


# -----------------------------------------------------------
# Pipeline State
# -----------------------------------------------------------
class PipelineState(TypedDict, total=False):
    instruction: str

    ingestion_result: Dict[str, Any]
    validation_result: Dict[str, Any]
    transformation_result: Dict[str, Any]
    storage_result: Dict[str, Any]
    visualization_result: Dict[str, Any]

    logs: list


def log(state: PipelineState, msg: str):
    state.setdefault("logs", []).append(msg)


# -----------------------------------------------------------
# 1. INGESTION
# -----------------------------------------------------------
def run_ingestion(state: PipelineState):
    log(state, "🔵 Ingestion started...")
    res = ingest_all_to_sqlite()
    state["ingestion_result"] = res
    log(state, "✅ Ingestion complete.")
    return state


# -----------------------------------------------------------
# 2. VALIDATION
# -----------------------------------------------------------
def run_validation(state: PipelineState):
    log(state, "🔵 Validation started...")
    db_path = str(DB_PATH)
    res = validate_sqlite_database(db_path)
    state["validation_result"] = res
    log(state, "✅ Validation complete.")
    return state


# -----------------------------------------------------------
# 3. TRANSFORMATION
# -----------------------------------------------------------
def run_transformation(state: PipelineState):
    log(state, "🔵 Transformation started...")
    instruction = state["instruction"]

    res = transform_database_tool.invoke({
        "database_path": str("data/db/olist_database.db"),
        "instruction": instruction
    })

    state["transformation_result"] = res
    log(state, "✅ Transformation complete.")
    return state


# -----------------------------------------------------------
# 4. STORAGE (Medallion)
# -----------------------------------------------------------
def run_storage(state: PipelineState):
    log(state, "🔵 Storage (Medallion) started...")

    res = persist_medallion(
        database_path="data/db/olist_transformed.db",
        layer="silver"
    )

    state["storage_result"] = res
    log(state, "✅ Storage complete.")
    return state


# -----------------------------------------------------------
# 5. VISUALIZATION (LLM)
# -----------------------------------------------------------
def run_visualization(state: PipelineState):
    log(state, "🔵 Visualization started...")

    instruction = state["instruction"]

    # load latest table — AUTO: pick first table in silver
    silver_dir = "data/medallion/silver"
    import os

    table_name = sorted(os.listdir(silver_dir))[0]
    df = _load_table_from_layer("silver", table_name)

    res = llm_visualization(instruction, df)
    state["visualization_result"] = res

    log(state, "🎉 Visualization complete.")
    return state


# -----------------------------------------------------------
# FINAL PIPELINE
# -----------------------------------------------------------
def run_pipeline(instruction: str):
    state: PipelineState = {"instruction": instruction, "logs": []}

    run_ingestion(state)
    run_validation(state)
    run_transformation(state)
    run_storage(state)
    run_visualization(state)

    return state
