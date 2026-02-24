"""
CellAtria 2.0 — Literature Agent (LangGraph Node)
====================================================
Performs strict Named Entity Recognition (NER) on the raw PDF text
to extract GEO (GSE*) and SRA (SRP*) accession identifiers.

Strategy
--------
1. **Regex pre-pass** — deterministic sweep for ``GSE\d+`` and
   ``SRP\d+`` patterns.  Fast, zero-cost, catches most accessions.
2. **LLM NER refinement** — the Groq-hosted Llama-3 model receives
   the paper text and a strict extraction prompt.  This catches
   accessions that are mentioned in prose without explicit IDs
   (e.g. "… deposited in GEO under accession GSE 123456 …" where
   the space would fool a naive regex).
3. **Deduplication & merge** — the union of both passes is returned
   as a sorted, unique list.

Environment
-----------
Requires ``GROQ_API_KEY`` in the environment (or ``.env`` file).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from core.state import CellAtriaState
from core.llm_utils import call_llm
from core.status_logger import status_log

# ── Logging ──────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
GEO_PATTERN = re.compile(r"GSE\d{3,9}")
SRP_PATTERN = re.compile(r"SRP\d{3,9}")

EXTRACTION_SYSTEM_PROMPT = """\
You are a biomedical named-entity recognition (NER) specialist.
Your ONLY task is to extract dataset accession identifiers from
the following scientific paper text.

Rules
─────
• Return ONLY accession IDs, one per line.
• Valid formats: GSE followed by digits (e.g. GSE123456)
                 SRP followed by digits (e.g. SRP098765)
• Do NOT return any other text, commentary, or formatting.
• If you find zero accessions, return the single word: NONE
"""


# ── Regex Pre-Pass ───────────────────────────────────────────────
def _regex_extract(text: str) -> set[str]:
    """Deterministic regex sweep for GEO / SRA accession IDs."""
    hits: set[str] = set()
    hits.update(GEO_PATTERN.findall(text))
    hits.update(SRP_PATTERN.findall(text))
    return hits


# ── LLM NER Pass ────────────────────────────────────────────────
def _llm_extract(text: str) -> set[str]:
    """Use Groq / Llama-3 to extract accessions the regex may miss.

    Uses the centralized call_llm() wrapper which handles rate
    limits via exponential backoff.
    """
    try:
        # Truncate to ~12 000 chars to stay well within context
        truncated = text[:12_000]

        raw_output = call_llm(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=truncated,
            temperature=0.0,
            max_tokens=512,
            caller="LiteratureAgent",
        )

        if raw_output is None:
            logger.warning("LLM NER extraction failed — falling back to regex only.")
            return set()

        if raw_output.upper() == "NONE":
            return set()

        # Parse the LLM output line-by-line, keeping only valid IDs
        hits: set[str] = set()
        for line in raw_output.splitlines():
            token = line.strip()
            if GEO_PATTERN.fullmatch(token) or SRP_PATTERN.fullmatch(token):
                hits.add(token)

        return hits

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LLM NER extraction failed — falling back to regex only. "
            "Error: %s",
            exc,
        )
        return set()


# ── LangGraph Node ───────────────────────────────────────────────
def literature_agent(state: CellAtriaState) -> dict[str, Any]:
    """LangGraph node: Literature Agent."""
    pdf_text: str = state.get("pdf_text", "")

    status_log.push("pipeline", "ok", "📚 Literature Agent — extracting accessions…")

    if not pdf_text:
        logger.error("literature_agent received empty pdf_text — nothing to extract.")
        status_log.push("pipeline", "warn", "Literature Agent — empty PDF text")
        return {"geo_accessions": [], "current_agent": "IngestionAgent"}

    # ── Pass 1: Regex ────────────────────────────────────────────
    regex_hits = _regex_extract(pdf_text)
    logger.info("Regex pass found %d accession(s): %s", len(regex_hits), regex_hits)

    # ── Pass 2: LLM NER ─────────────────────────────────────────
    status_log.push("pipeline", "ok", "📚 Literature Agent — calling LLM for NER…")
    llm_hits = _llm_extract(pdf_text)
    logger.info("LLM NER pass found %d accession(s): %s", len(llm_hits), llm_hits)

    # ── Merge & deduplicate ──────────────────────────────────────
    all_accessions = sorted(regex_hits | llm_hits)
    status_log.push("pipeline", "ok",
                     f"📚 Literature Agent done — {len(all_accessions)} accessions found")

    return {
        "geo_accessions": all_accessions,
        "current_agent": "IngestionAgent",
    }
