# 🧬 CellAtria 2.0: Autonomous scRNA-seq Drug Repurposing Swarm

CellAtria 2.0 is an enterprise-grade, multi-agent artificial intelligence platform designed to autonomously ingest single-cell RNA sequencing (scRNA-seq) literature, perform cloud-based bioinformatics quality control, and output therapeutic drug candidates.

## 🚀 System Architecture
The platform leverages a LangGraph-orchestrated AI swarm powered by Llama-3.3-70B, tightly integrated with an E2B secure cloud sandbox for heavy computational biology tasks.

1. **Literature Ingestion Engine:** Extracts target GEO Accession IDs (e.g., GSE117570) from uploaded scientific PDFs.
2. **Bioinformatician Agent:** Bypasses slow FTPs to autonomously fetch raw transcriptomic data, deploying `Scanpy` within an E2B cloud sandbox for clustering and Quality Control.
3. **Subpopulation Strategist:** Computes Differential Expression (DEG) and cross-references results with epigenetic drug databases to identify high-confidence therapeutic reversal agents.

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Custom Dark-Mode UI)
* **LLM Engine:** Groq (Llama-3.3-70B-versatile)
* **Agent Orchestration:** LangGraph
* **Cloud Compute Sandbox:** E2B Code Interpreter
* **Bioinformatics:** Scanpy, AnnData, Pandas

## ⚙️ Installation & Usage
1. Clone the repository.
2. Install requirements: `pip install -r requirements.txt`
3. Create a `.env` file with your `GROQ_API_KEY`, `E2B_API_KEY`, and `E2B_TEMPLATE_ID`.
4. Run the medical terminal: `streamlit run app.py`

*Note: Includes a bypass "Demo Mode" for instant UI rendering and presentation without exhausting API rate limits.*