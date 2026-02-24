"""
CellAtria 2.0 — LangGraph Swarm State Payload
================================================
Canonical state schema for the autonomous single-cell RNA-seq
analysis pipeline.  Every node in the LangGraph graph reads from
and writes to this single TypedDict, ensuring deterministic
message-passing between agents.

Design notes
------------
* All paths are *sandbox-local* (e2b microVM filesystem).
* ``qc_metrics`` and ``drug_shortlist`` are intentionally kept as
  unstructured dicts so downstream agents can evolve their schemas
  without touching this file.
* ``sandbox_tracebacks`` powers the self-correction loop: if a node
  fails inside the E2B VM, the traceback is appended here and a
  retry / fallback agent can inspect it.
"""

from __future__ import annotations

from typing import TypedDict


class CellAtriaState(TypedDict):
    """Full memory payload shared across all LangGraph nodes.

    Attributes
    ----------
    pdf_text : str
        Raw text extracted from the ingested single-cell RNA-seq paper
        (via PyMuPDF / pdfplumber / GROBID).
    geo_accessions : list[str]
        GEO / SRA dataset accession IDs parsed from the paper
        (e.g. ``["GSE123456", "SRP789012"]``).
    raw_data_paths : list[str]
        Sandbox-local paths to downloaded raw expression files
        (``.tar.gz``, ``.tsv``, ``.mtx``, ``barcodes.tsv.gz``, etc.).
    anndata_path : str
        Sandbox-local path to the compiled, unified AnnData object
        stored as ``.h5ad``.
    qc_metrics : dict
        Quality-control metadata produced by the QC agent.
        Expected keys (non-exhaustive):
        ``n_cells_pre_filter``, ``n_cells_post_filter``,
        ``median_genes_per_cell``, ``median_counts_per_cell``,
        ``pct_mito_mean``, ``pct_mito_threshold``.
    batch_keys : list[str]
        Observation-level keys used for batch integration /
        correction (e.g. ``["sample_id", "disease_status"]``).
        Critical for separating disease vs. control populations.
    target_clusters : list[str]
        Cluster labels or Leiden community IDs representing
        disease-critical cell subpopulations flagged for
        downstream drug-repurposing analysis.
    drug_shortlist : dict
        Final output of the DrugReSC / perturbation-analysis agent.
        Maps drug names to reversal / connectivity scores, e.g.
        ``{"Imatinib": -0.87, "Decitabine": -0.72}``.
    sandbox_tracebacks : list[str]
        Ordered execution logs and exception tracebacks captured
        from the E2B microVM.  Used by the self-correction loop
        to diagnose and recover from pipeline failures.
    current_agent : str
        Identifier of the LangGraph node that currently holds
        control (e.g. ``"paper_parser"``, ``"qc_agent"``,
        ``"drug_ranker"``).
    """

    # ── Paper Ingestion ──────────────────────────────────────────
    pdf_text: str
    geo_accessions: list[str]

    # ── Data Acquisition ─────────────────────────────────────────
    raw_data_paths: list[str]

    # ── Preprocessing & QC ───────────────────────────────────────
    anndata_path: str
    qc_metrics: dict

    # ── Batch Integration ────────────────────────────────────────
    batch_keys: list[str]

    # ── Cluster Analysis ─────────────────────────────────────────
    target_clusters: list[str]

    # ── Drug Repurposing ─────────────────────────────────────────
    drug_shortlist: dict

    # ── Observability & Control ──────────────────────────────────
    sandbox_tracebacks: list[str]
    current_agent: str
