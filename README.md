# 🤖 Agentic AI–Powered Data Engineering Framework

### 🧩 Project Overview
This project builds an **Agentic AI framework** that automates the end-to-end data engineering lifecycle, from data ingestion to visualization, using **multi-agent collaboration** with a human-in-the-loop.

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
| `data/` | Raw GitHub CSV and SQLite sample database |
| `notebooks/` | Initial setup notebook with environment validation |
| `src/` | Source code for each agent and orchestrator logic |
| `ui/` | Streamlit interface for user interaction |
| `results/` | Sample visualizations or processed outputs |
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

### 🎯 Current Progress (Deliverable 2)
✅ **Implemented & Verified:**
- **Ingestion Agent:** Pulls data from GitHub and databases into a unified SQLite warehouse.  
- **Validation Agent:** Performs schema validation, missing value detection, and data profiling.  
- **Transformation Agent:** Executes both rule-based and LLM-guided data transformations, including multi-table joins and aggregations.  
- **LangGraph Orchestrator:** Connects agents into an end-to-end pipeline (Ingestion → Validation → Transformation).  
- **Streamlit UI:** Enables natural-language instructions and displays logs interactively.

🚧 **Upcoming:**
- Storage Agent (Medallion architecture)
- Visualization Agent (LLM-driven chart generation)

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
streamlit run ui/streamlit_app.py
```

---

### 📊 Current Results

| Component            | Status     | Output                                                    |
| -------------------- | ---------- | --------------------------------------------------------- |
| Ingestion Agent      | ✅ Working  | Created `olist_database.db` from 8 GitHub CSVs            |
| Validation Agent     | ✅ Working  | Detected schema consistency issues, nulls, and duplicates |
| Transformation Agent | ✅ Working  | Generated dynamic joins and aggregations via GPT-4o-mini  |
| Streamlit UI         | ✅ Working  | Displays logs and accepts instructions                    |
| Storage Agent        | 🚧 Planned | To implement Medallion-tier storage                       |
| Visualization Agent  | 🚧 Planned | To enable LLM-driven visual plotting                      |

---

### 🧩 Known Issues
- Module Path Imports: Requires adding the project root to sys.path in ui/streamlit_app.py.
- LLM Timeout Handling: Occasionally, API latency causes transformation retries.
- Visualization Agent Pending: No plots are generated yet (placeholder integrated).
- Environment Variables: OPENAI_API_KEY must be set before running.

---

### 🤖 Author

Aniket Deepak Malpure  
M.S. in Applied Data Science  
University of Florida (2024–2026)  
📧 Email: aniketmalpure@ufl.edu  
🔗 GitHub: aniket-malpure  
💼 LinkedIn: linkedin.com/in/aniketmalpure  