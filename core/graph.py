"""
CellAtria 2.0 — LangGraph Assembler
======================================
Wires the four agent nodes into a single LangGraph StateGraph
with conditional routing, self-correction loops, and a clean
terminal state.

Graph topology
--------------
    START
      │
      ▼
  LiteratureAgent ──→ IngestionAgent ──→ BioinformaticianAgent
                                              │          ▲
                                        (success?)       │
                                          │    ╰── (fail: retry)
                                          ▼
                                 SubpopulationStrategist
                                          │
                                          ▼
                                         END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from core.state import CellAtriaState
from agents.literature_agent import literature_agent
from agents.ingestion_agent import ingestion_agent
from agents.bioinformatician_agent import bioinformatician_agent
from agents.subpopulation_strategist import subpopulation_strategist

# ── Logging ──────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ====================================================================
#  Routing Functions
# ====================================================================

def route_after_literature(state: CellAtriaState) -> str:
    """After the Literature Agent, always proceed to ingestion."""
    next_agent = state.get("current_agent", "IngestionAgent")
    logger.info("Routing from LiteratureAgent → %s", next_agent)

    if next_agent == "IngestionAgent":
        return "ingestion_agent"
    # Fallback
    return "ingestion_agent"


def route_after_ingestion(state: CellAtriaState) -> str:
    """After the Ingestion Agent, proceed to the Bioinformatician."""
    next_agent = state.get("current_agent", "BioinformaticianAgent")
    logger.info("Routing from IngestionAgent → %s", next_agent)

    if next_agent == "BioinformaticianAgent":
        return "bioinformatician_agent"
    return "bioinformatician_agent"


def route_after_bioinformatician(state: CellAtriaState) -> str:
    """After the Bioinformatician Agent, check for self-correction
    or proceed to the Subpopulation Strategist.

    Self-correction trigger: if ``current_agent`` is still
    ``"BioinformaticianAgent"`` (meaning it exhausted retries and
    is requesting re-entry) AND ``sandbox_tracebacks`` is populated.
    """
    next_agent = state.get("current_agent", "")
    tracebacks = state.get("sandbox_tracebacks", [])

    if next_agent == "BioinformaticianAgent" and tracebacks:
        logger.info(
            "Self-correction loop: re-routing to BioinformaticianAgent "
            "(%d traceback(s)).",
            len(tracebacks),
        )
        return "bioinformatician_agent"

    if next_agent == "SubpopulationStrategist":
        logger.info("Routing from BioinformaticianAgent → SubpopulationStrategist")
        return "subpopulation_strategist"

    # Default: proceed forward
    logger.info("Routing from BioinformaticianAgent → SubpopulationStrategist (default)")
    return "subpopulation_strategist"


def route_after_strategist(state: CellAtriaState) -> str:
    """After the Subpopulation Strategist, either terminate or
    loop back if there were execution failures."""
    next_agent = state.get("current_agent", "END")
    tracebacks = state.get("sandbox_tracebacks", [])

    if next_agent == "END" or (not tracebacks):
        logger.info("Pipeline complete → END")
        return END

    if next_agent == "SubpopulationStrategist" and tracebacks:
        logger.info(
            "Strategist self-correction: re-routing (%d traceback(s)).",
            len(tracebacks),
        )
        return "subpopulation_strategist"

    # Fallback to Bioinformatician if something deeper is wrong
    if next_agent == "BioinformaticianAgent":
        logger.info("Routing back to BioinformaticianAgent from Strategist")
        return "bioinformatician_agent"

    logger.info("Pipeline complete → END (default)")
    return END


# ====================================================================
#  Graph Assembly
# ====================================================================

def build_graph() -> Any:
    """Construct and compile the CellAtria 2.0 LangGraph.

    Returns
    -------
    CompiledGraph
        The compiled, runnable LangGraph that can be invoked with
        an initial ``CellAtriaState``.
    """
    workflow = StateGraph(CellAtriaState)

    # ── Add nodes ────────────────────────────────────────────────
    workflow.add_node("literature_agent", literature_agent)
    workflow.add_node("ingestion_agent", ingestion_agent)
    workflow.add_node("bioinformatician_agent", bioinformatician_agent)
    workflow.add_node("subpopulation_strategist", subpopulation_strategist)

    # ── Entry point ──────────────────────────────────────────────
    workflow.set_entry_point("literature_agent")

    # ── Conditional edges ────────────────────────────────────────
    workflow.add_conditional_edges(
        "literature_agent",
        route_after_literature,
        {
            "ingestion_agent": "ingestion_agent",
        },
    )

    workflow.add_conditional_edges(
        "ingestion_agent",
        route_after_ingestion,
        {
            "bioinformatician_agent": "bioinformatician_agent",
        },
    )

    workflow.add_conditional_edges(
        "bioinformatician_agent",
        route_after_bioinformatician,
        {
            "bioinformatician_agent": "bioinformatician_agent",  # self-loop
            "subpopulation_strategist": "subpopulation_strategist",
        },
    )

    workflow.add_conditional_edges(
        "subpopulation_strategist",
        route_after_strategist,
        {
            "subpopulation_strategist": "subpopulation_strategist",  # retry
            "bioinformatician_agent": "bioinformatician_agent",      # fallback
            END: END,
        },
    )

    # ── Compile ──────────────────────────────────────────────────
    graph = workflow.compile()
    logger.info("CellAtria 2.0 LangGraph compiled successfully.")

    return graph


# ── Convenience: module-level compiled graph ─────────────────────
cellatria_graph = build_graph()


# ====================================================================
#  CLI Entry Point
# ====================================================================

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    )

    # Minimal initial state — in production, pdf_text would come
    # from a document loader.
    initial_state: CellAtriaState = {
        "pdf_text": "",
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

    # Allow passing PDF text via CLI argument
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        with open(pdf_path, "r", encoding="utf-8", errors="ignore") as f:
            initial_state["pdf_text"] = f.read()
        logger.info("Loaded PDF text from %s (%d chars)",
                     pdf_path, len(initial_state["pdf_text"]))

    print("=" * 60)
    print("  CellAtria 2.0 — Autonomous scRNA-seq Swarm")
    print("=" * 60)

    final_state = cellatria_graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Drug Shortlist:")
    print(json.dumps(final_state.get("drug_shortlist", {}), indent=4))
    print(f"\n  QC Metrics:")
    print(json.dumps(final_state.get("qc_metrics", {}), indent=4))

    if final_state.get("sandbox_tracebacks"):
        print(f"\n  ⚠  Remaining tracebacks:")
        for tb in final_state["sandbox_tracebacks"]:
            print(f"    • {tb[:200]}")
