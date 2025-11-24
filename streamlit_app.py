import streamlit as st
from src.orchestrator import run_pipeline
from pathlib import Path
import json

st.set_page_config(page_title="Agentic AI Data Engineering Framework", layout="wide")
st.title("🤖 Agentic AI-Powered Data Engineering Framework")
st.markdown(
    "This app runs the full pipeline: **Ingestion → Validation → Transformation → Storage → Visualization**"
)

# ----------------------------
# USER INPUT
# ----------------------------
instruction = st.text_area(
    "Enter your data transformation or visualization instruction:",
    height=120,
    placeholder="Example: Join orders and payments tables and plot the average payment_value per payment_type."
)

if st.button("🚀 Run Pipeline"):
    if not instruction.strip():
        st.error("Please enter an instruction.")
        st.stop()

    with st.spinner("Running pipeline..."):
        state = run_pipeline(instruction)

    st.success("Pipeline finished successfully!")

    # ----------------------------
    # Show logs
    # ----------------------------
    st.subheader("📜 Pipeline Logs")
    for log in state["logs"]:
        st.write(log)

    # ----------------------------
    # Show ingestion details
    # ----------------------------
    st.subheader("📥 Ingestion Result")
    st.json(state["ingestion_result"])

    # ----------------------------
    # Show validation details
    # ----------------------------
    st.subheader("🔍 Validation Result")
    st.json(state["validation_result"])

    # ----------------------------
    # Show transformation details
    # ----------------------------
    st.subheader("🛠 Transformation Result")
    st.json(state["transformation_result"])

    # ----------------------------
    # Show storage details
    # ----------------------------
    st.subheader("📦 Storage Result")
    st.json(state["storage_result"])

    # ----------------------------
    # Show visualization
    # ----------------------------
    vis_result = state.get("visualization_result", {})
    st.subheader("📊 Visualization Result")

    if vis_result.get("status") == "success":
        plot_path = vis_result.get("path")
        code = vis_result.get("generated_code", "")

        if plot_path:
            abs_path = Path(plot_path).resolve()
            if abs_path.exists():
                st.image(str(abs_path), caption="Generated Visualization", use_column_width=True)
            else:
                st.error(f"Plot file not found at: {abs_path}")
        else:
            st.warning("No plot generated.")

        with st.expander("Show generated matplotlib code"):
            st.code(code, language="python")
    else:
        st.error("Visualization failed")
        st.json(vis_result)
