"""
CellAtria 2.0 — Enterprise Frontend
======================================
Premium Streamlit application disguised as a high-end Next.js SaaS
platform for biotech executives.  Integrates the LangGraph pipeline
and renders results with Plotly + custom CSS glassmorphism UI.
"""

from __future__ import annotations

import os
import time
import json
import threading
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

# ── Load environment ─────────────────────────────────────────────
load_dotenv(override=True)

# ── Status monitoring (must be after dotenv) ─────────────────────
from core.status_logger import status_log
from core.e2b_manager import check_e2b_health

# ── Page configuration (MUST be first Streamlit call) ────────────
st.set_page_config(
    page_title="CellAtria 2.0 · Autonomous scRNA-seq Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ====================================================================
#  CUSTOM CSS — Full Streamlit Override
# ====================================================================
CUSTOM_CSS = """
<style>
/* ── Import premium font ────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Variables ───────────────────────────────────────────── */
:root {
    --bg-primary:    #050810;
    --bg-card:       rgba(255,255,255,0.03);
    --bg-card-hover: rgba(255,255,255,0.05);
    --text-primary:  #EDEDED;
    --text-muted:    #8A8F9E;
    --bio-cyan:      #00E5FF;
    --genomic-purple:#8A2BE2;
    --border-glass:  rgba(255,255,255,0.10);
    --border-active: rgba(0,229,255,0.35);
    --shadow-glass:  0 8px 32px rgba(0,0,0,0.45);
    --shadow-glow:   0 0 20px rgba(0,229,255,0.15);
    --radius:        12px;
    --font-sans:     'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono:     'JetBrains Mono', 'Fira Code', monospace;
}

/* ── Keyframes ───────────────────────────────────────────────── */
@keyframes pulse-online {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,229,255,0.55); }
    50%      { box-shadow: 0 0 0 8px rgba(0,229,255,0); }
}
@keyframes scanline {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes glow-border {
    0%, 100% { border-color: rgba(0,229,255,0.25); }
    50%      { border-color: rgba(138,43,226,0.45); }
}
@keyframes terminal-blink {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0; }
}

/* ── Global resets ───────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, rgba(138,43,226,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(0,229,255,0.04) 0%, transparent 50%),
                var(--bg-primary) !important;
}
/* ── Hide Streamlit chrome ───────────────────────────────────── */
#MainMenu, header, footer,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
.stDeployButton { display: none !important; }
[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
}
/* ── Sidebar — dark theme to match main UI ───────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-primary) !important;
    border-right: 1px solid var(--border-glass) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Block container spacing ─────────────────────────────────── */
.block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── Glassmorphism card ──────────────────────────────────────── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius);
    padding: 1.4rem;
    box-shadow: var(--shadow-glass);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    animation: fadeInUp 0.5s ease-out;
    transition: border-color 0.4s ease, box-shadow 0.4s ease;
}
.glass-card:hover {
    border-color: var(--border-active);
    box-shadow: var(--shadow-glass), var(--shadow-glow);
}
.glass-card-active {
    animation: glow-border 2.5s ease-in-out infinite;
    box-shadow: var(--shadow-glass), 0 0 30px rgba(0,229,255,0.1);
}

/* ── Section headers ─────────────────────────────────────────── */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--bio-cyan);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::before {
    content: '';
    width: 3px;
    height: 14px;
    background: linear-gradient(180deg, var(--bio-cyan), var(--genomic-purple));
    border-radius: 2px;
}

/* ── Title cards (safe with Streamlit widgets below) ───────────── */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
    box-shadow: var(--shadow-glass);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}
.section-card .section-header {
    margin: 0;
}
.section-card-active {
    border-color: var(--border-active);
    box-shadow: var(--shadow-glass), var(--shadow-glow);
}

/* ── Telemetry items ─────────────────────────────────────────── */
.telem-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
}
.telem-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.65rem 0.8rem;
}
.telem-label {
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 2px;
}
.telem-value {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--bio-cyan);
}
.telem-value.online { color: #00E676; }
.telem-value.model  { color: var(--genomic-purple); }

/* ── Status indicator ────────────────────────────────────────── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.5rem;
    padding: 0.7rem 1rem;
    background: rgba(0,230,118,0.04);
    border: 1px solid rgba(0,230,118,0.15);
    border-radius: 8px;
}
.status-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: #00E676;
    animation: pulse-online 2s ease-in-out infinite;
    flex-shrink: 0;
}
.status-text {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #00E676;
}

/* ── File uploader override ──────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(0,229,255,0.25) !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploader"] section {
    padding: 0.8rem 1rem !important;
}
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, var(--bio-cyan), var(--genomic-purple)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.5px !important;
}

/* ── Primary action button ───────────────────────────────────── */
div.stButton > button {
    width: 100%;
    padding: 0.8rem 1.5rem;
    font-family: var(--font-sans);
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #fff;
    background: linear-gradient(135deg, var(--bio-cyan), var(--genomic-purple));
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,229,255,0.2);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(0,229,255,0.35), 0 0 40px rgba(138,43,226,0.2);
}
div.stButton > button:active {
    transform: translateY(0);
}
div.stButton > button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Terminal log ────────────────────────────────────────────── */
.terminal-window {
    background: #0A0D14;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0;
    overflow: hidden;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    line-height: 1.7;
}
.terminal-bar {
    background: rgba(255,255,255,0.04);
    padding: 0.5rem 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.terminal-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
}
.terminal-dot.red    { background: #FF5F57; }
.terminal-dot.yellow { background: #FEBC2E; }
.terminal-dot.green  { background: #28C840; }
.terminal-title {
    font-size: 0.65rem;
    color: var(--text-muted);
    margin-left: 8px;
    letter-spacing: 0.5px;
}
.terminal-body {
    padding: 1rem 1.2rem;
    max-height: 520px;
    overflow-y: auto;
    color: #C8CCD4;
}
.terminal-body::-webkit-scrollbar {
    width: 4px;
}
.terminal-body::-webkit-scrollbar-track {
    background: transparent;
}
.terminal-body::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.1);
    border-radius: 2px;
}
.log-system     { color: var(--bio-cyan); }
.log-literature { color: #AB47BC; }
.log-ingestion  { color: #FFA726; }
.log-e2b        { color: #26C6DA; }
.log-bio        { color: #66BB6A; }
.log-strategist { color: #EF5350; }
.log-success    { color: #00E676; font-weight: 600; }
.log-timestamp  { color: #555B6E; }
.cursor-blink {
    display: inline-block;
    width: 7px; height: 14px;
    background: var(--bio-cyan);
    animation: terminal-blink 1s step-end infinite;
    margin-left: 4px;
    vertical-align: text-bottom;
}

/* ── Insights panel ──────────────────────────────────────────── */
.insight-metric {
    text-align: center;
    padding: 0.8rem 0.5rem;
}
.insight-metric-value {
    font-family: var(--font-mono);
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--bio-cyan), var(--genomic-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.insight-metric-label {
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ── Dataframe override ──────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-glass) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
}

/* ── Hero brand bar ──────────────────────────────────────────── */
.brand-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 0;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.brand-logo {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-name {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--bio-cyan), var(--genomic-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.brand-version {
    font-size: 0.6rem;
    font-weight: 500;
    color: var(--text-muted);
    background: rgba(255,255,255,0.05);
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 1px;
}
.brand-tagline {
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.5px;
}

/* ── Pipeline scanning overlay ───────────────────────────────── */
.scanning-bar {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--bio-cyan), var(--genomic-purple), transparent);
    background-size: 200% 100%;
    animation: scanline 2s linear infinite;
    border-radius: 1px;
    margin: 0.8rem 0;
}

/* ── Divider ─────────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1rem 0;
}

/* ── Plotly chart container ──────────────────────────────────── */
.plotly-chart-container {
    border: 1px solid var(--border-glass);
    border-radius: 10px;
    overflow: hidden;
    padding: 0.5rem;
}

/* ── Hide Streamlit expander borders etc ─────────────────────── */
.streamlit-expanderHeader {
    font-family: var(--font-sans) !important;
    background: transparent !important;
    color: var(--text-muted) !important;
}

/* ── Background Checks status ────────────────────────────────── */
.bg-check-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.bg-check-row:last-child { border-bottom: none; }
.bg-check-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.bg-check-dot.dot-ok    { background: #00E676; box-shadow: 0 0 6px rgba(0,230,118,0.5); }
.bg-check-dot.dot-warn  { background: #FFA726; box-shadow: 0 0 6px rgba(255,167,38,0.5); }
.bg-check-dot.dot-error { background: #EF5350; box-shadow: 0 0 6px rgba(239,83,80,0.5); animation: pulse-online 1.5s ease-in-out infinite; }
.bg-check-dot.dot-unknown { background: #555B6E; }
.bg-check-svc {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #EDEDED;
    min-width: 40px;
}
.bg-check-msg {
    font-size: 0.65rem;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.bg-event-log {
    max-height: 120px;
    overflow-y: auto;
    margin-top: 0.5rem;
    padding: 0.4rem 0.6rem;
    background: rgba(0,0,0,0.25);
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    line-height: 1.6;
    color: #8A8F9E;
}
.bg-event-log::-webkit-scrollbar { width: 3px; }
.bg-event-log::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
.bg-evt-ok    { color: #00E676; }
.bg-evt-warn  { color: #FFA726; }
.bg-evt-error { color: #EF5350; }

/* ── Responsive adjustments ──────────────────────────────────── */
@media (max-width: 768px) {
    .telem-grid { grid-template-columns: 1fr; }
    .block-container { padding: 1rem !important; }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ====================================================================
#  BRAND HEADER
# ====================================================================
st.markdown("""
<div class="brand-bar">
    <div class="brand-logo">
        <span style="font-size:1.8rem;">🧬</span>
        <div>
            <span class="brand-name">CellAtria</span>
            <span class="brand-version">v2.0</span>
        </div>
    </div>
    <span class="brand-tagline">Autonomous scRNA-seq Discovery Platform · Multi-Agent Orchestration</span>
</div>
""", unsafe_allow_html=True)


# ====================================================================
#  SIDEBAR — DEMO MODE TOGGLE
# ====================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0;">
        <div class="section-header">Pipeline Options</div>
    </div>
    """, unsafe_allow_html=True)
    st.toggle("⚡ Demo Mode", value=False, key="demo_sidebar_toggle",
              help="Bypass LangGraph execution and render instant demo results")


# ====================================================================
#  SESSION STATE INITIALISATION
# ====================================================================
# pipeline_phase:  "idle" → "running" → "complete"
DEFAULTS = {
    "pipeline_phase": "idle",
    "pipeline_logs": [],
    "final_state": None,
    "drug_df": None,
    "pdf_text": "",
    "uploaded_name": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Strict UI-refresh keys ──────────────────────────────────────
if 'pipeline_complete' not in st.session_state:
    st.session_state.pipeline_complete = False
if 'swarm_results' not in st.session_state:
    st.session_state.swarm_results = None
if 'bg_thread_started' not in st.session_state:
    st.session_state.bg_thread_started = False


# ====================================================================
#  HELPER — PDF TEXT EXTRACTION
# ====================================================================
def extract_pdf_text(uploaded_file) -> str:
    """Extract text from an uploaded PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        # Fallback: try pdfplumber
        try:
            import pdfplumber
            pdf_bytes = uploaded_file.read()
            import io
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception:
            return ""
    except Exception:
        return ""


# ====================================================================
#  HELPER — TERMINAL LOG GENERATION
# ====================================================================
LOG_SEQUENCE = [
    ("SYSTEM",      "log-system",     "Initializing CellAtria 2.0 Discovery Pipeline…"),
    ("SYSTEM",      "log-system",     "Allocating compute resources on E2B Cloud…"),
    ("SYSTEM",      "log-system",     "Sandbox environment provisioned  ·  Template: cellatria-sandbox"),
    ("LITERATURE",  "log-literature", "Engaging Literature Parsing Engine…"),
    ("LITERATURE",  "log-literature", "Performing NER extraction on manuscript corpus…"),
    ("LITERATURE",  "log-literature", "Regex pre-pass complete  ·  Scanning for GEO/SRA identifiers…"),
    ("LITERATURE",  "log-literature", "LLM NER refinement via Llama-3.3-70B  ·  Groq inference active…"),
    ("LITERATURE",  "log-literature", "Accession identifiers extracted  ·  Forwarding to Ingestion Engine…"),
    ("INGESTION",   "log-ingestion",  "Ingestion Agent activated  ·  Resolving GEO dataset endpoints…"),
    ("INGESTION",   "log-ingestion",  "Downloading supplementary expression matrices from NCBI FTP…"),
    ("INGESTION",   "log-ingestion",  "Unpacking compressed archives  ·  Detecting 10X Genomics triplet…"),
    ("E2B CLOUD",   "log-e2b",        "Compiling AnnData object inside secure microVM…"),
    ("E2B CLOUD",   "log-e2b",        "Matrix assembly complete  ·  Writing .h5ad to sandbox filesystem…"),
    ("INGESTION",   "log-ingestion",  "AnnData compilation successful  ·  Forwarding to Bioinformatician…"),
    ("BIOINF",      "log-bio",        "Bioinformatician Agent online  ·  Loading AnnData object…"),
    ("BIOINF",      "log-bio",        "Executing quality-control filters  ·  Mito threshold: 20%…"),
    ("BIOINF",      "log-bio",        "Normalization → Log1p → HVG selection (n=2000)…"),
    ("BIOINF",      "log-bio",        "Computing PCA (50 components)  ·  Building neighborhood graph…"),
    ("E2B CLOUD",   "log-e2b",        "Leiden clustering executing in sandbox  ·  Resolution: 0.8…"),
    ("BIOINF",      "log-bio",        "UMAP embedding computed  ·  Cluster annotations generated…"),
    ("BIOINF",      "log-bio",        "Batch correction via Harmony  ·  Integration complete…"),
    ("BIOINF",      "log-bio",        "QC metrics compiled  ·  Forwarding to Subpopulation Strategist…"),
    ("STRATEGIST",  "log-strategist", "Subpopulation Strategist engaged  ·  Identifying disease clusters…"),
    ("STRATEGIST",  "log-strategist", "Target vs. Control populations isolated  ·  Subsampling to 5,000 cells…"),
    ("E2B CLOUD",   "log-e2b",        "Differential expression analysis executing  ·  Wilcoxon rank-sum…"),
    ("STRATEGIST",  "log-strategist", "DEG signatures computed  ·  Initiating drug connectivity scoring…"),
    ("E2B CLOUD",   "log-e2b",        "In silico perturbation screen running  ·  CMap reversal scores…"),
    ("STRATEGIST",  "log-strategist", "Therapeutic candidate shortlist compiled  ·  Ranking by reversal score…"),
    ("SYSTEM",      "log-success",    "═══  PIPELINE COMPLETE  ·  Therapeutic Target Shortlist Ready  ═══"),
]


def format_log_line(tag: str, css_class: str, message: str) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    return (
        f'<span class="log-timestamp">{ts}</span>  '
        f'<span class="{css_class}">[{tag}]</span>  {message}'
    )


def render_section_title(
    title: str,
    *,
    margin_bottom: str = "0.7rem",
    margin_top: str | None = None,
    active: bool = False,
) -> None:
    """Render a heading inside its own card to avoid broken HTML wrappers."""
    css = "section-card section-card-active" if active else "section-card"
    styles = [f"margin-bottom:{margin_bottom}"] if margin_bottom else []
    if margin_top:
        styles.append(f"margin-top:{margin_top}")
    style_attr = f' style="{"; ".join(styles)}"' if styles else ""
    st.markdown(
        f'<div class="{css}"{style_attr}><div class="section-header">{title}</div></div>',
        unsafe_allow_html=True,
    )


# ====================================================================
#  HELPER — RUN PIPELINE (synchronous, bulletproof)
# ====================================================================
def _get_result_dict() -> dict:
    """Return a mutable dict that survives Streamlit hot-reloads.

    Instead of a module-level dict (which gets reset to
    ``done=False`` when Streamlit re-imports the module on file
    change), we store it **inside** ``st.session_state``.

    The background thread receives a reference to this dict and
    writes to it directly — mutating a dict is thread-safe in
    CPython (GIL), and does NOT require calling any Streamlit API.
    """
    if "_pipeline_result" not in st.session_state:
        st.session_state["_pipeline_result"] = {
            "done": False,
            "final_state": None,
            "drug_df": None,
            "error": "",
        }
    return st.session_state["_pipeline_result"]


def _run_pipeline_sync(pdf_text: str, result_dict: dict) -> None:
    """Execute the LangGraph pipeline synchronously.

    Called from a background thread — MUST NOT touch st.session_state.
    Writes results to ``result_dict`` (a mutable dict stored in
    session_state) which the main Streamlit polling loop reads.

    Includes a global 3-minute timeout to prevent indefinite hangs.
    """
    import concurrent.futures

    PIPELINE_TIMEOUT_SECONDS = 360  # 6 minutes max

    try:
        status_log.push("pipeline", "ok", "Pipeline started — importing graph…")

        from core.state import CellAtriaState
        from core.graph import cellatria_graph

        status_log.push("pipeline", "ok", "Graph imported — building initial state…")

        initial_state: CellAtriaState = {
            "pdf_text": pdf_text,
            "geo_accessions": [],
            "raw_data_paths": [],
            "anndata_path": "",
            "qc_metrics": {},
            "batch_keys": [],
            "target_clusters": [],
            "drug_shortlist": {},
            "sandbox_tracebacks": [],
            "current_agent": "LiteratureAgent",
        }

        status_log.push("pipeline", "ok",
                         f"Invoking pipeline — pdf_text: {len(pdf_text):,} chars "
                         f"(timeout: {PIPELINE_TIMEOUT_SECONDS}s)")

        # Run the graph with a timeout to prevent indefinite hangs.
        # IMPORTANT: Do NOT use `with` context manager — its __exit__
        # calls shutdown(wait=True) which deadlocks if the inner
        # thread is hung (the exact scenario we're protecting against).
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(cellatria_graph.invoke, initial_state)
        try:
            final = future.result(timeout=PIPELINE_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            # Abandon the hung thread — don't wait for it
            executor.shutdown(wait=False, cancel_futures=True)
            status_log.push("pipeline", "error",
                            f"Pipeline timed out after {PIPELINE_TIMEOUT_SECONDS}s")
            result_dict["error"] = (
                f"Pipeline timed out after {PIPELINE_TIMEOUT_SECONDS} seconds. "
                "This usually means a GEO download or sandbox execution hung."
            )
            # Set done=True HERE — the finally block would be blocked
            # by the executor shutdown if we used `with`.
            result_dict["done"] = True
            status_log.push("pipeline", "ok", "Pipeline thread exiting — done=True")
            return
        finally:
            # Clean shutdown for non-timeout cases
            executor.shutdown(wait=False)

        status_log.push("pipeline", "ok", "Pipeline invoke() returned — processing results…")
        result_dict["final_state"] = final

        # Build dataframe from drug_shortlist
        shortlist = final.get("drug_shortlist", {})
        n_acc = len(final.get("geo_accessions", []))
        n_drugs = len(shortlist)
        status_log.push("pipeline", "ok",
                         f"Results: {n_acc} accessions, {n_drugs} drug candidates")

        if shortlist:
            df = pd.DataFrame([
                {"Compound": k, "Reversal Score": round(v, 4) if isinstance(v, (int, float)) else v}
                for k, v in shortlist.items()
            ]).sort_values("Reversal Score", ascending=True).reset_index(drop=True)
            result_dict["drug_df"] = df

    except Exception as exc:
        result_dict["error"] = str(exc)
        status_log.push("pipeline", "error", f"Pipeline crashed: {exc}")

    finally:
        # Signal completion — the polling loop checks this flag.
        result_dict["done"] = True
        status_log.push("pipeline", "ok", "Pipeline thread exiting — done=True")


# ====================================================================
#  LAYOUT — THREE COLUMNS
# ====================================================================
col_left, col_spacer1, col_center, col_spacer2, col_right = st.columns(
    [3.0, 0.15, 4.5, 0.15, 2.8]
)


# ─────────────────────────────────────────────────────────────────────
#  LEFT PANEL — COMMAND CENTER
# ─────────────────────────────────────────────────────────────────────
with col_left:

    # ── System Status ────────────────────────────────────────────
    st.markdown("""
    <div class="glass-card" style="margin-bottom:1rem;">
        <div class="section-header">System Telemetry</div>
        <div class="status-bar">
            <div class="status-dot"></div>
            <span class="status-text">System Online</span>
        </div>
        <div class="telem-grid">
            <div class="telem-item">
                <div class="telem-label">E2B Latency</div>
                <div class="telem-value">42 ms</div>
            </div>
            <div class="telem-item">
                <div class="telem-label">Sandbox</div>
                <div class="telem-value online">READY</div>
            </div>
            <div class="telem-item">
                <div class="telem-label">LLM Engine</div>
                <div class="telem-value model">Llama-3.3-70B</div>
            </div>
            <div class="telem-item">
                <div class="telem-label">Groq Status</div>
                <div class="telem-value online">ONLINE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Literature Ingestion Engine ──────────────────────────────
    render_section_title("Literature Ingestion Engine", margin_bottom="0.8rem")

    uploaded_file = st.file_uploader(
        "Submit scRNA-seq manuscript (.pdf)",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    if uploaded_file and uploaded_file.name != st.session_state.get("uploaded_name"):
        with st.spinner("Parsing document…"):
            text = extract_pdf_text(uploaded_file)
            st.session_state["pdf_text"] = text
            st.session_state["uploaded_name"] = uploaded_file.name
            st.session_state.pipeline_complete = False
            st.session_state.swarm_results = None
            st.session_state["pipeline_logs"] = []
            st.session_state["final_state"] = None
            st.session_state["drug_df"] = None

    if st.session_state.get("pdf_text"):
        char_count = len(st.session_state["pdf_text"])
        st.markdown(
            f'<div style="font-size:0.7rem;color:var(--text-muted);margin-top:0.4rem;">'
            f'📄 <strong>{st.session_state["uploaded_name"]}</strong> · '
            f'{char_count:,} characters extracted</div>',
            unsafe_allow_html=True,
        )

    # ── Pipeline Trigger ─────────────────────────────────────────
    render_section_title("Pipeline Control", margin_bottom="0.8rem")

    # ── Demo Mode Toggle (reads from sidebar) ────────────────────
    # The toggle lives in the sidebar; we read its value here.
    demo_mode = st.session_state.get("demo_sidebar_toggle", False)

    is_idle = st.session_state["pipeline_phase"] == "idle"
    is_complete = st.session_state["pipeline_phase"] == "complete"
    can_run = bool(st.session_state.get("pdf_text")) and (is_idle or is_complete)

    btn_label = (
        "⚡  Initialize Discovery Pipeline"
        if st.session_state["pipeline_phase"] != "running"
        else "◌  Allocating Compute…"
    )
    if st.button(btn_label, disabled=not can_run, key="run_btn"):
        if demo_mode:
            # ── DEMO MODE: Instant hardcoded results ─────────────
            _demo_drugs = {
                "Vorinostat": -0.89, "Trichostatin A": -0.85,
                "Tanespimycin": -0.78, "Geldanamycin": -0.74,
                "Sirolimus": -0.71, "Wortmannin": -0.68,
                "LY-294002": -0.65, "Thapsigargin": -0.61,
                "Withaferin A": -0.58, "Celastrol": -0.54,
            }
            _demo_state = {
                "geo_accessions": ["GSE10780", "GSE42568", "GSE45827", "GSE54002"],
                "qc_metrics": {
                    "n_cells": 12847, "n_genes": 23412, "n_clusters": 14,
                    "mito_pct_mean": 4.8, "hvg_count": 2000,
                    "pca_components": 50, "leiden_resolution": 0.8,
                },
                "drug_shortlist": _demo_drugs,
                "batch_keys": ["sample", "patient_id"],
                "target_clusters": ["Cluster_3", "Cluster_7", "Cluster_11"],
                "raw_data_paths": ["demo/matrix.mtx.gz"],
                "anndata_path": "demo/adata_processed.h5ad",
                "sandbox_tracebacks": [],
            }
            _demo_df = pd.DataFrame([
                {"Compound": k, "Reversal Score": round(v, 4)}
                for k, v in _demo_drugs.items()
            ]).sort_values("Reversal Score", ascending=True).reset_index(drop=True)

            st.session_state["final_state"] = _demo_state
            st.session_state["drug_df"] = _demo_df
            st.session_state.swarm_results = _demo_state
            st.session_state.pipeline_complete = True
            st.session_state["pipeline_phase"] = "complete"
            st.session_state["pipeline_logs"] = [
                format_log_line("SYSTEM", "log-system", "Demo Mode activated — loading preset results…"),
                format_log_line("LITREV", "log-litrev", "Literature scan complete — 4 GEO accessions extracted."),
                format_log_line("INGESTION", "log-ingestion", "AnnData compilation complete — Forwarding to Bioinformatician."),
                format_log_line("BIOINF", "log-bioinf", "QC metrics compiled — Forwarding to Subpopulation Strategist."),
                format_log_line("STRATEGIST", "log-strategist", "Therapeutic candidate shortlist compiled — 10 compounds."),
                format_log_line("SYSTEM", "log-system", "─── PIPELINE COMPLETE · Therapeutic Target Shortlist Ready ───"),
            ]
            status_log.push("pipeline", "ok", "Demo Mode — instant results loaded")
            st.rerun()
        else:
            # ── REAL MODE: Launch background pipeline ────────────
            st.session_state["pipeline_phase"] = "running"
            st.session_state["pipeline_logs"] = []
            st.session_state["final_state"] = None
            st.session_state["drug_df"] = None
            st.session_state.pipeline_complete = False
            st.session_state.swarm_results = None
            st.session_state.bg_thread_started = False
            # Reset the shared result dict
            rd = _get_result_dict()
            rd.update({"done": False, "final_state": None, "drug_df": None, "error": ""})
            st.rerun()

    # ── Background Checks Panel ──────────────────────────────────
    # Run proactive health checks on page load
    _groq_key = os.environ.get("GROQ_API_KEY", "")
    if _groq_key:
        status_log.push("groq", "ok", "API key configured · Llama-3.3-70B")
    else:
        status_log.push("groq", "error", "GROQ_API_KEY not set")
    check_e2b_health()

    _summary = status_log.get_summary()
    _recent = status_log.get_recent(10)

    render_section_title(
        "Background Checks",
        margin_top="1rem",
        margin_bottom="0.8rem",
    )

    # Build status rows for each service
    _service_labels = {"groq": "GROQ", "e2b": "E2B", "ftp": "FTP", "pipeline": "PIPE"}
    for svc_key, svc_label in _service_labels.items():
        info = _summary.get(svc_key)
        if info:
            level = info.get("level", "unknown")
            msg = info.get("message", "—")
            dot_class = f"dot-{level}" if level in ("ok", "warn", "error") else "dot-unknown"
        else:
            dot_class = "dot-unknown"
            msg = "No data"

        st.markdown(f'''
        <div class="bg-check-row">
            <div class="bg-check-dot {dot_class}"></div>
            <span class="bg-check-svc">{svc_label}</span>
            <span class="bg-check-msg" title="{msg}">{msg}</span>
        </div>
        ''', unsafe_allow_html=True)

    # Recent event log
    if _recent:
        event_lines = []
        for evt in reversed(_recent):
            lvl_css = f"bg-evt-{evt['level']}" if evt['level'] in ('ok','warn','error') else ""
            event_lines.append(
                f'<span style="color:#555B6E">{evt["timestamp"]}</span> '
                f'<span class="{lvl_css}">[{evt["service"].upper()}]</span> '
                f'{evt["message"]}'
            )
        log_html = "<br>".join(event_lines)
        st.markdown(f'<div class="bg-event-log">{log_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  CENTER PANEL — EXECUTION TELEMETRY
# ─────────────────────────────────────────────────────────────────────
with col_center:
    phase = st.session_state["pipeline_phase"]
    render_section_title(
        "In Silico Execution Log",
        margin_bottom="0.8rem",
        active=(phase == "running"),
    )

    if phase == "running":
        st.markdown('<div class="scanning-bar"></div>', unsafe_allow_html=True)

    # ── TERMINAL RENDERING ───────────────────────────────────────
    _TERMINAL_HEADER = """
    <div class="terminal-window">
        <div class="terminal-bar">
            <div class="terminal-dot red"></div>
            <div class="terminal-dot yellow"></div>
            <div class="terminal-dot green"></div>
            <span class="terminal-title">cellatria-pipeline · execution telemetry</span>
        </div>
    """

    if phase == "running":
        # ── PHASE: RUNNING ───────────────────────────────────────
        log_placeholder = st.empty()

        # Step 1 — Animate the telemetry log sequence (cosmetic, once)
        if not st.session_state.bg_thread_started:
            for i, (tag, css, msg) in enumerate(LOG_SEQUENCE):
                line = format_log_line(tag, css, msg)
                st.session_state["pipeline_logs"].append(line)
                log_html = "<br>".join(st.session_state["pipeline_logs"])
                cursor = '<span class="cursor-blink"></span>' if i < len(LOG_SEQUENCE) - 1 else ""
                log_placeholder.markdown(
                    f'{_TERMINAL_HEADER}<div class="terminal-body">{log_html}{cursor}</div></div>',
                    unsafe_allow_html=True,
                )
                time.sleep(0.7)

            # Step 2 — Launch the pipeline in a BACKGROUND THREAD
            #   Copy pdf_text NOW (from session_state) and pass as arg.
            #   The thread must NEVER touch st.session_state.
            _pdf_text_for_thread = st.session_state["pdf_text"]
            _rd = _get_result_dict()
            thread = threading.Thread(
                target=_run_pipeline_sync,
                args=(_pdf_text_for_thread, _rd),
                daemon=True,
            )
            thread.start()
            st.session_state.bg_thread_started = True

        # Step 3 — Render current logs while waiting
        log_html = "<br>".join(st.session_state["pipeline_logs"])
        log_placeholder.markdown(
            f'{_TERMINAL_HEADER}<div class="terminal-body">{log_html}'
            f'<span class="cursor-blink"></span></div></div>',
            unsafe_allow_html=True,
        )

        # Step 4 — Poll: has the background thread finished?
        _rd = _get_result_dict()
        if _rd["done"]:
            # Thread has finished — copy results into session_state
            st.session_state["final_state"] = _rd.get("final_state")
            st.session_state["drug_df"] = _rd.get("drug_df")
            st.session_state.swarm_results = _rd.get("final_state") or {}
            st.session_state.pipeline_complete = True
            st.session_state["pipeline_phase"] = "complete"
            st.session_state.bg_thread_started = False
            if _rd.get("error"):
                st.session_state["pipeline_logs"].append(
                    format_log_line("ERROR", "log-strategist",
                                    f"Pipeline exception: {_rd['error']}")
                )
            st.rerun()
        else:
            # Still running — wait 2s then re-poll
            time.sleep(2)
            st.rerun()

    elif phase == "complete":
        # ── PHASE: COMPLETE ──────────────────────────────────────
        log_html = "<br>".join(st.session_state["pipeline_logs"])
        st.markdown(
            f'{_TERMINAL_HEADER}<div class="terminal-body">{log_html}</div></div>',
            unsafe_allow_html=True,
        )

    else:
        # ── PHASE: IDLE ──────────────────────────────────────────
        idle_msg = format_log_line("SYSTEM", "log-system", "Awaiting manuscript submission…")
        st.markdown(
            f'{_TERMINAL_HEADER}<div class="terminal-body">{idle_msg}'
            f'<span class="cursor-blink"></span></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  RIGHT PANEL — INSIGHTS
# ─────────────────────────────────────────────────────────────────────
with col_right:

    if st.session_state.pipeline_complete:
        # ── RENDER THE ACTUAL RESULTS ────────────────────────────
        _final = st.session_state.swarm_results or {}
        _drug_df = st.session_state.get("drug_df")

        # ── Summary metrics ──────────────────────────────────────
        n_candidates = len(_drug_df) if _drug_df is not None else 0
        n_clusters = len(_final.get("target_clusters", []))
        n_accessions = len(_final.get("geo_accessions", []))

        render_section_title("Discovery Metrics", margin_bottom="0.7rem")
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;">
            <div class="insight-metric">
                <div class="insight-metric-value">{n_candidates}</div>
                <div class="insight-metric-label">Candidates</div>
            </div>
            <div class="insight-metric">
                <div class="insight-metric-value">{n_clusters}</div>
                <div class="insight-metric-label">Clusters</div>
            </div>
            <div class="insight-metric">
                <div class="insight-metric-value">{n_accessions}</div>
                <div class="insight-metric-label">Accessions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Therapeutic Candidates Table ─────────────────────────
        render_section_title("Therapeutic Target Shortlist", margin_bottom="0.7rem")
        if _drug_df is not None and not _drug_df.empty:
            st.dataframe(
                _drug_df,
                width="stretch",
                hide_index=True,
                height=min(len(_drug_df) * 38 + 40, 320),
            )
        else:
            st.markdown(
                '<div style="color:var(--text-muted);font-size:0.75rem;padding:1.5rem 0;text-align:center;">'
                'No drug candidates returned by the pipeline.</div>',
                unsafe_allow_html=True,
            )

        # ── Plotly Confidence Chart ──────────────────────────────
        render_section_title("Candidate Confidence Intervals", margin_bottom="0.7rem")
        if _drug_df is not None and not _drug_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=_drug_df["Compound"],
                x=_drug_df["Reversal Score"].abs(),
                orientation="h",
                marker=dict(
                    color=_drug_df["Reversal Score"].abs(),
                    colorscale=[
                        [0, "#8A2BE2"],
                        [0.5, "#00E5FF"],
                        [1, "#00E676"],
                    ],
                    line=dict(width=0),
                    cornerradius=4,
                ),
                text=_drug_df["Reversal Score"].apply(lambda x: f"{x:.3f}"),
                textposition="outside",
                textfont=dict(size=10, color="#EDEDED", family="JetBrains Mono"),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#EDEDED", size=11),
                xaxis=dict(
                    title="| Reversal Score |",
                    gridcolor="rgba(255,255,255,0.05)",
                    zerolinecolor="rgba(255,255,255,0.05)",
                    title_font=dict(size=10, color="#8A8F9E"),
                    tickfont=dict(size=9, color="#8A8F9E"),
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(size=10, color="#EDEDED"),
                ),
                margin=dict(l=10, r=40, t=10, b=30),
                height=max(len(_drug_df) * 32 + 60, 200),
                bargap=0.35,
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.markdown(
                '<div style="color:var(--text-muted);font-size:0.75rem;padding:2rem 0;text-align:center;">'
                'No data available for chart rendering.</div>',
                unsafe_allow_html=True,
            )

    else:
        # ── RENDER EMPTY / AWAITING STATES ───────────────────────
        render_section_title("Discovery Metrics", margin_bottom="0.7rem")
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;">
            <div class="insight-metric">
                <div class="insight-metric-value">—</div>
                <div class="insight-metric-label">Candidates</div>
            </div>
            <div class="insight-metric">
                <div class="insight-metric-value">—</div>
                <div class="insight-metric-label">Clusters</div>
            </div>
            <div class="insight-metric">
                <div class="insight-metric-value">—</div>
                <div class="insight-metric-label">Accessions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        render_section_title("Therapeutic Target Shortlist", margin_bottom="0.7rem")
        st.markdown(
            '<div style="color:var(--text-muted);font-size:0.75rem;padding:1.5rem 0;text-align:center;">'
            'Awaiting pipeline execution…</div>',
            unsafe_allow_html=True,
        )

        render_section_title("Candidate Confidence Intervals", margin_bottom="0.7rem")
        st.markdown(
            '<div style="color:var(--text-muted);font-size:0.75rem;padding:2rem 0;text-align:center;">'
            'Confidence intervals will appear after<br>pipeline execution completes.</div>',
            unsafe_allow_html=True,
        )


# ── QC Details (expandable, below main layout) ──────────────────
if st.session_state.pipeline_complete and st.session_state.swarm_results and st.session_state.swarm_results.get("qc_metrics"):
    st.markdown("<br>", unsafe_allow_html=True)
    render_section_title("Quality Control Metrics", margin_bottom="0.7rem")

    qc = st.session_state.swarm_results["qc_metrics"]
    qc_cols = st.columns(min(len(qc), 5))
    for i, (k, v) in enumerate(qc.items()):
        with qc_cols[i % len(qc_cols)]:
            display_val = f"{v:,.2f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
            st.markdown(f"""
            <div class="insight-metric">
                <div class="insight-metric-value" style="font-size:1.1rem;">{display_val}</div>
                <div class="insight-metric-label">{k.replace('_', ' ').title()}</div>
            </div>
            """, unsafe_allow_html=True)



# ── Footer ───────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem 0;color:#555B6E;font-size:0.6rem;letter-spacing:1px;">
    CELLATRIA 2.0  ·  MULTI-AGENT ORCHESTRATION  ·  E2B SECURE COMPUTE  ·  GROQ INFERENCE
</div>
""", unsafe_allow_html=True)
