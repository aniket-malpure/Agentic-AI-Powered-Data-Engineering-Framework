"""
orchestrator.py

LangGraph Orchestrator for Agentic AI Data Engineering Framework
---------------------------------------------------------------
This script builds and executes the full agent pipeline:
Ingestion → Validation → Transformation → Storage → Visualization

Agents involved:
- ingestion_agent.py
- validation_agent.py
- transformation_agent.py
- storage_agent.py
- visualization_agent.py

Author: Aniket Deepak Malpure
"""

from langgraph.graph import StateGraph, END
from ingestion_agent import create_ingestion_agent
from validation_agent import create_validation_agent
from transformation_agent import create_transformation_agent
from storage_agent import create_storage_agent
from visualization_agent import create_visualization_agent
import json
import time


# ---------------------------------------------------
# Shared state schema
# ---------------------------------------------------
def initial_state(instruction: str) -> dict:
    """
    Shared memory that flows between agents.
    Each agent reads/writes to this dictionary.
    """
    return {
        "instruction": instruction,
        "logs": [],
        "data_paths": {},
        "results": {},
    }


def log(state, message: str):
    """Append a log message and print for visibility."""
    state["logs"].append(f"{time.strftime('%H:%M:%S')} | {message}")
    print("🧩", message)
    return state


# ---------------------------------------------------
# Agentic Orchestration Logic
# ---------------------------------------------------
def build_pipeline():
    """Build the complete LangGraph pipeline for the data engineering process."""
    graph = StateGraph()

    # Add agents
    graph.add_node("ingestion", create_ingestion_agent())
    graph.add_node("validation", create_validation_agent())
    graph.add_node("transformation", create_transformation_agent())
    graph.add_node("storage", create_storage_agent())
    graph.add_node("visualization", create_visualization_agent())

    # Define edges (pipeline flow)
    graph.add_edge("ingestion", "validation")
    graph.add_edge("validation", "transformation")
    graph.add_edge("transformation", "storage")
    graph.add_edge("storage", "visualization")
    graph.add_edge("visualization", END)

    # Set entrypoint
    graph.set_entry_point("ingestion")

    return graph.compile()


# ---------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------
def run_pipeline(instruction: str):
    """
    Execute the pipeline sequentially using LangGraph.
    Returns final state and logs.
    """
    pipeline = build_pipeline()
    state = initial_state(instruction)

    # ---- Stage 1: Ingestion ----
    state = log(state, "📥 Starting data ingestion...")
    ingestion_result = pipeline.step("ingestion", {"instruction": instruction})
    state["results"]["ingestion"] = ingestion_result
    state = log(state, "✅ Ingestion completed.")

    # ---- Stage 2: Validation ----
    state = log(state, "🔍 Running validation checks...")
    validation_result = pipeline.step("validation", {"instruction": instruction})
    state["results"]["validation"] = validation_result
    state = log(state, "✅ Validation completed.")

    # ---- Stage 3: Transformation ----
    state = log(state, "⚙️ Performing LLM-guided transformation...")
    transform_result = pipeline.step("transformation", {"instruction": instruction})
    state["results"]["transformation"] = transform_result
    state = log(state, "✅ Transformation completed.")

    # ---- Stage 4: Storage ----
    state = log(state, "💾 Persisting transformed data to Medallion storage...")
    storage_result = pipeline.step("storage", {"instruction": instruction})
    state["results"]["storage"] = storage_result
    state = log(state, "✅ Storage completed.")

    # ---- Stage 5: Visualization ----
    state = log(state, "📊 Generating visualization via LLM...")
    visualization_result = pipeline.step("visualization", {"instruction": instruction})
    state["results"]["visualization"] = visualization_result
    state = log(state, "✅ Visualization generated successfully.")

    state = log(state, "🎉 Pipeline execution complete.")
    return state


# ---------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Running Full Agentic Data Engineering Pipeline\n")
    user_instruction = input("Enter your instruction: ").strip()
    final_state = run_pipeline(user_instruction)

    print("\n🧾 Execution Logs:")
    for line in final_state["logs"]:
        print(line)

    print("\n📈 Final Results Summary:")
    print(json.dumps(final_state["results"], indent=2))
