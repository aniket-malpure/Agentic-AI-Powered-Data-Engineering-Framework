# 🤖 Agentic AI–Powered Data Engineering Framework

### 🧩 Project Overview
This project builds an **Agentic AI framework** that automates the end-to-end data engineering lifecycle, from data ingestion to visualization, using **multi-agent collaboration**.

The system executes the complete pipeline:

```Ingestion → Validation → Transformation → Storage (Medallion) → Visualization```

The framework:
- Ingests raw data from GitHub and a SQLite database
- Validates and transforms data using Python + PySpark
- Organizes data into **Medallion Architecture** (Bronze → Silver → Gold)
- Generates automated visualizations based on human instructions
- Provides an interactive **Streamlit dashboard** for execution and monitoring

---

### 📂 Repository Structure
| Folder | Description |
|---------|--------------|
| `data/` | Raw GitHub CSV & SQLite database and visualizations |
| `notebooks/` | Initial setup notebook with environment validation |
| `src/` | Source code for each agent and orchestrator logic |
| `streamlit_app.py` | Streamlit interface for user interaction |
| `results/` | UI screenshots for the report |
| `docs/` | Architecture diagram and design docs |

---

### ⚙️ Tech Stack
- **Python 3.10+**
- **PySpark**
- **LangChain / LangGraph**
- **SQLite3**
- **Matplotlib / Seaborn**
- **Streamlit**

---

### 🧠 Key Capabilities

- Multi-Agent Workflow built on LangGraph
- LLM-guided transformations (safe, column-aware, sandboxed execution)
- Automatic visualization generation via GPT-4o-mini
- Full operational transparency with logs and system messages
- Medallion Architecture implemented with versioned Parquet stores

---

### 🚀 Setup Instructions

1️⃣ Clone the Repository
```
git clone https://github.com/aniket-malpure/Agentic_AI_Data_Engineering_Framework.git
cd Agentic_AI_Data_Engineering_Framework
```

2️⃣ Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

4️⃣ Set Your OpenAI API Key
```
export OPENAI_API_KEY="your_api_key"   # macOS/Linux
setx OPENAI_API_KEY "your_api_key"     # Windows
```

5️⃣ Run the Full Pipeline (Manual Mode)
```
python src/orchestrator.py
```

6️⃣ Launch the Streamlit Interface
```
streamlit run streamlit_app.py
```

---

### 📊 Current Results

| Component            | Status    | Notes                                   |
| -------------------- | --------- | --------------------------------------- |
| Ingestion Agent      | ✅ Working | Downloads + loads all Olist tables      |
| Validation Agent     | ✅ Working | Summaries, checks, profiling            |
| Transformation Agent | ✅ Working | Column-aware LLM transformations        |
| Storage Agent        | ✅ Working | Bronze → Silver → Gold Parquet creation |
| Visualization Agent  | ✅ Working | Saves plots & code for UI               |
| Orchestrator         | ✅ Working | Full sequential pipeline                |
| Streamlit UI         | ✅ Working | Visualizes all outputs                  |

---

### 🧩 Known Issues
- Some transformations may require additional guardrails for multi-table inference
- Large datasets may cause slow parquet writing on Windows
- Occasional LLM latency depending on API load

---

### 🤖 Author

Aniket Deepak Malpure  
M.S. in Applied Data Science  
University of Florida (2024–2026)  
📧 Email: aniketmalpure@ufl.edu  
🔗 GitHub: aniket-malpure  
💼 LinkedIn: linkedin.com/in/aniketmalpure  