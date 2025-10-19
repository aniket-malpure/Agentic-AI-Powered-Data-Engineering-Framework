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

### 🚀 How to Run
```bash
# 1️⃣ Create a new virtual environment
python -m venv venv
source venv/bin/activate   # on macOS/Linux
venv\Scripts\activate      # on Windows

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run setup notebook
jupyter notebook notebooks/setup.ipynb

# 4️⃣ Launch Streamlit UI (later stage)
streamlit run ui/streamlit_app.py

### 🤖 Author
- Name: Aniket Malpure
- Email: aniketmalpure@ufl.edu