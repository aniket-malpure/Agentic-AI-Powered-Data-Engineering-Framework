"""
streamlit_app.py

Agentic AI-Powered Data Engineering Framework Dashboard
--------------------------------------------------------
This Streamlit dashboard orchestrates the full LangGraph pipeline:
1. Ingestion
2. Validation
3. Transformation (LLM-driven)
4. Storage
5. Visualization (LLM-driven)

Users provide a single natural language instruction that is interpreted
and passed to the transformation & visualization agents.

Author: Aniket Deepak Malpure
"""

import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your agents
from src.ingestion_agent import github_to_sqlite_tool
from src.validation_agent import validate_database_tool
from src.transformation_agent import transform_database_tool
from src.storage_agent import persist_medallion_tool
from src.visualization_agent import visualize_table_tool

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="Agentic AI Data Engineering Framework", layout="wide")

st.title("🧠 Agentic AI-Powered Data Engineering Framework")
st.markdown("""
This interactive dashboard demonstrates an **LLM-driven data engineering pipeline**  
that goes from **ingestion → validation → transformation → storage → visualization**.  
Simply provide an instruction (like *"Generate a pie chart showing yearly sales by category"*)  
and watch the entire system orchestrate itself.
""")

st.divider()

# ----------------------------
# Instruction Input
# ----------------------------
st.subheader("💡 Enter your instruction")
instruction = st.text_area(
    "Describe the task you want to perform:",
    placeholder="e.g. Generate a pie chart showing yearly sales by category for Electronics products",
    height=100
)

run_button = st.button("🚀 Run Agentic Pipeline")

# ----------------------------
# Pipeline Execution
# ----------------------------
if run_button:
    if not instruction.strip():
        st.error("Please provide an instruction before running the pipeline.")
        st.stop()

    st.divider()
    st.subheader("📜 Execution Logs")

    log_box = st.empty()
    logs = []

    def log(msg: str):
        """Helper to update logs in real-time."""
        logs.append(msg)
        log_box.markdown("\n".join([f"- {l}" for l in logs]))
        time.sleep(0.7)

    # ----------------------------
    # Step 1: Ingestion
    # ----------------------------
    log("📥 Starting ingestion agent...")
    try:
        ingestion_result = github_to_sqlite_tool()
        log("✅ Ingestion completed successfully.")
    except Exception as e:
        log(f"❌ Ingestion failed: {e}")
        st.stop()

    # ----------------------------
    # Step 2: Validation
    # ----------------------------
    log("🔍 Running data validation checks...")
    try:
        validation_result = validate_database_tool(database_path="data/olist_database.db")
        log("✅ Validation completed successfully.")
    except Exception as e:
        log(f"❌ Validation failed: {e}")
        st.stop()

    # ----------------------------
    # Step 3: Transformation (LLM)
    # ----------------------------
    log("⚙️ Performing LLM-guided transformation...")
    try:
        transform_result = transform_database_tool(
            database_path="data/olist_database.db",
            instruction=instruction
        )
        log(f"✅ Transformation completed in {transform_result['mode']} mode.")
    except Exception as e:
        log(f"❌ Transformation failed: {e}")
        st.stop()

    # ----------------------------
    # Step 4: Storage
    # ----------------------------
    log("💾 Persisting transformed data to Medallion storage (Silver layer)...")
    try:
        storage_result = persist_medallion_tool(database_path="data/olist_transformed.db", layer="silver")
        log("✅ Storage completed successfully. Data saved in `data/medallion/silver/`.")
    except Exception as e:
        log(f"❌ Storage failed: {e}")
        st.stop()

    # ----------------------------
    # Step 5: Visualization (LLM)
    # ----------------------------
    log("📊 Generating visualization via LLM agent...")
    try:
        # For simplicity, visualize one of the main tables (you can make this dynamic)
        vis_result = visualize_table_tool(
            layer="silver",
            table_name="olist_order_payments_dataset",
            instruction=instruction
        )

        if vis_result["status"] == "success":
            log("✅ Visualization generated successfully.")
            plot_path = vis_result["plot_path"]
        else:
            log(f"❌ Visualization failed: {vis_result['message']}")
            st.stop()
    except Exception as e:
        log(f"❌ Visualization failed: {e}")
        st.stop()

    st.divider()

    # ----------------------------
    # Display Visualization
    # ----------------------------
    st.subheader("📈 Generated Visualization")
    st.image(plot_path, caption="AI-Generated Visualization", use_container_width=True)

    # Show generated code for transparency
    with st.expander("🧠 Show Generated LLM Code for Visualization"):
        st.code(vis_result["generated_code"], language="python")

    # ----------------------------
    # Display Summary Results
    # ----------------------------
    st.divider()
    st.subheader("📊 Pipeline Summary")

    summary_data = {
        "Ingestion": ingestion_result.get("status", "completed"),
        "Validation": validation_result.get("status", "completed"),
        "Transformation Mode": transform_result.get("mode", "N/A"),
        "Storage Layer": "silver",
        "Visualization": "completed"
    }

    st.table(pd.DataFrame(summary_data.items(), columns=["Stage", "Status"]))

    st.success("🎉 Agentic pipeline executed successfully!")
