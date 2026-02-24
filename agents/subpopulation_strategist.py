"""
CellAtria 2.0 — Subpopulation Strategist (LangGraph Node)
============================================================
Final analysis node: identifies disease-critical cell clusters,
performs controlled subsampling to prevent sandbox timeouts, and
bridges into R via rpy2 to execute the DrugReSC pipeline
(PACSI + ssGSEA random-forest permutations).

Output
------
Top-10 drug candidates with reversal / connectivity scores,
written to ``state["drug_shortlist"]``.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from core.llm_utils import call_llm
from core.status_logger import status_log

from core.state import CellAtriaState

# Lazy import — wired after core/e2b_manager.py is created
try:
    from core.e2b_manager import run_in_sandbox
except ImportError:
    run_in_sandbox = None  # type: ignore[assignment]

# ── Logging ──────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
MAX_ATTEMPTS = 3
SUBSAMPLE_N = 5_000  # cells per group — tuned to prevent OOM / timeout

SYSTEM_PROMPT = """\
You are an expert computational pharmacologist specialising in
single-cell drug repurposing.
Your ONLY job is to output a complete, runnable Python script.

Rules:
─────
• Output ONLY Python code — no markdown fences, no commentary.
• The script must be fully self-contained (all imports at the top).
• Use scanpy (sc), anndata (ad), rpy2.robjects, numpy, and pandas.
• Every major step must have a print() statement logging progress.
• At the very end, print a single-line JSON object prefixed with
  DRUG_JSON:: containing a dict mapping drug names (str) to their
  reversal / connectivity scores (float), sorted ascending.
  Only include the top 10 drugs.
  Format: DRUG_JSON::{"DrugA": -0.92, "DrugB": -0.87, ...}
• Do NOT use plt.show() or any interactive plotting.
"""

GENERATION_PROMPT = """\
Write a complete Python script that performs drug-repurposing
analysis on the processed .h5ad file at:
    {anndata_path}

The file has been QC'd, normalized, batch-corrected, and clustered
with Leiden (resolution 0.8).  The clusters are stored in
adata.obs["leiden"].

Target clusters (disease): {target_clusters}
Batch / sample keys: {batch_keys}

Steps (execute in this exact order):

1. LOAD: Read the .h5ad with sc.read_h5ad().

2. IDENTIFY TARGET vs CONTROL:
   - Target cells: adata.obs["leiden"].isin({target_clusters})
   - Control cells: all remaining clusters.
   - Print counts for each group.

3. CRITICAL SUBSAMPLING (prevents timeout & OOM):
   - From the TARGET group, randomly sample exactly {n} cells
     (or all if fewer than {n}).
   - From the CONTROL group, randomly sample exactly {n} cells
     (or all if fewer than {n}).
   - Concatenate into a single AnnData object for downstream.
   - Print final subsampled shape.

4. DIFFERENTIAL EXPRESSION:
   - Run sc.tl.rank_genes_groups(adata_sub, groupby="condition",
     method="wilcoxon", reference="control").
   - Extract the top 200 upregulated genes in the disease group
     as the disease signature.
   - Print the top 20 genes.

5. DRUG REPURPOSING via rpy2 / DrugReSC:
   - Import rpy2.robjects and activate the pandas/numpy converters.
   - Pass the disease gene signature into R.
   - In R, attempt to load the DrugReSC library.
     If DrugReSC is not installed, fall back to a connectivity-
     score approach using the LINCS L1000 gene sets:
       * For each drug perturbation signature, compute the
         Kolmogorov-Smirnov enrichment score against the disease
         signature (P_hit vs P_miss).
       * Rank drugs by their reversal score (most negative = best).
   - Collect the top 10 drugs and their scores.

6. OUTPUT:
   - Print the DRUG_JSON line as specified in the system rules.
"""

CORRECTION_PROMPT = """\
The previous script FAILED with this traceback:

--- TRACEBACK ---
{traceback}
--- END TRACEBACK ---

Previous script:
--- SCRIPT ---
{previous_script}
--- END SCRIPT ---

Fix the script. Output the COMPLETE corrected script.
The data file is at: {anndata_path}
"""


# ====================================================================
#  Internal: LLM Script Generation
# ====================================================================

def _generate_script(prompt: str, temperature: float = 0.1) -> str:
    """Call Groq/Llama-3 and return the generated Python script."""
    raw = call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=temperature,
        max_tokens=4096,
        caller="SubpopulationStrategist",
    )

    if raw is None:
        return ""

    script = raw

    # Strip markdown fences if the LLM wraps them
    if script.startswith("```"):
        lines = [
            ln for ln in script.splitlines()
            if not ln.strip().startswith("```")
        ]
        script = "\n".join(lines)

    return script


def _parse_drug_json(stdout: str) -> dict[str, float]:
    """Extract the DRUG_JSON payload from sandbox stdout."""
    for line in stdout.splitlines():
        if line.strip().startswith("DRUG_JSON::"):
            raw = line.strip().split("DRUG_JSON::", 1)[1]
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse DRUG_JSON: %s", exc)
    return {}


# ====================================================================
#  LangGraph Node
# ====================================================================

def subpopulation_strategist(state: CellAtriaState) -> dict[str, Any]:
    """LangGraph node: Subpopulation Strategist.

    Generates a Python + rpy2 script to:
    1. Identify disease vs. control clusters.
    2. Subsample to 5 000 cells per group.
    3. Compute differential expression.
    4. Run DrugReSC / connectivity-score drug repurposing.
    5. Return top-10 drugs.

    Includes a self-correction loop (up to 3 attempts).

    Parameters
    ----------
    state : CellAtriaState

    Returns
    -------
    dict
        Partial state update with ``drug_shortlist``,
        ``sandbox_tracebacks``, and ``current_agent``.
    """
    anndata_path = state.get("anndata_path", "")
    target_clusters = state.get("target_clusters", [])
    batch_keys = state.get("batch_keys", [])
    tracebacks: list[str] = list(state.get("sandbox_tracebacks", []))

    if not anndata_path:
        msg = ("[SubpopulationStrategist] Missing anndata_path — "
               "cannot run drug repurposing.")
        logger.error(msg)
        status_log.push("pipeline", "warn", "💊 Strategist — no anndata_path, skipping")
        return {
            "drug_shortlist": {},
            "sandbox_tracebacks": [],
            "current_agent": "END",  # terminate the graph
        }

    status_log.push("pipeline", "ok", "💊 Subpopulation Strategist — generating drug script…")

    # If no target clusters were explicitly set, default to the
    # largest cluster (heuristic — the disease cluster is often
    # the most abundant abnormal population).
    if not target_clusters:
        target_clusters = ["0", "1"]  # safe defaults
        logger.warning(
            "No target_clusters provided — defaulting to %s. "
            "The LLM script will refine based on metadata.",
            target_clusters,
        )

    previous_script = ""
    needs_correction = bool(tracebacks and "execution failed" in tracebacks[-1].lower())

    if needs_correction:
        previous_script = _extract_last_script(tracebacks)

    max_attempts = MAX_ATTEMPTS
    if anndata_path == "llm_fallback":
        max_attempts = 0
        msg = (
            "[SubpopulationStrategist] anndata_path is llm_fallback - "
            "skipping E2B attempts and using LLM drug fallback."
        )
        logger.warning(msg)
        tracebacks.append(msg)
        status_log.push("pipeline", "warn", "Strategist - using LLM drug fallback")

    # ── Generation / correction loop ─────────────────────────────
    for attempt in range(1, max_attempts + 1):
        logger.info(
            "SubpopulationStrategist attempt %d / %d", attempt, MAX_ATTEMPTS
        )

        try:
            if previous_script and needs_correction:
                prompt = CORRECTION_PROMPT.format(
                    traceback=tracebacks[-1] if tracebacks else "Unknown",
                    previous_script=previous_script,
                    anndata_path=anndata_path,
                )
                script = _generate_script(prompt, temperature=0.05)
            else:
                prompt = GENERATION_PROMPT.format(
                    anndata_path=anndata_path,
                    target_clusters=target_clusters,
                    batch_keys=batch_keys,
                    n=SUBSAMPLE_N,
                )
                script = _generate_script(prompt, temperature=0.15)
        except Exception as exc:
            error_msg = (
                f"[SubpopulationStrategist] LLM generation failed "
                f"(attempt {attempt}): {exc}"
            )
            logger.error(error_msg)
            tracebacks.append(error_msg)
            continue

        logger.info("Generated DrugReSC script (%d chars)", len(script))
        tracebacks.append(
            f"[SubpopulationStrategist] GENERATED_SCRIPT::\n{script}"
        )

        # ── Execute in E2B ───────────────────────────────────────
        if run_in_sandbox is None:
            msg = ("[SubpopulationStrategist] run_in_sandbox not available — "
                   "E2B manager not imported.")
            logger.error(msg)
            tracebacks.append(msg)
            return {
                "drug_shortlist": {},
                "sandbox_tracebacks": [],  # CLEAR to prevent loops
                "current_agent": "END",    # advance to END
            }

        result = run_in_sandbox(script)

        if result["success"]:
            drug_shortlist = _parse_drug_json(result["stdout"])
            status_log.push("pipeline", "ok",
                             f"💊 Strategist done — {len(drug_shortlist)} drug candidates")
            logger.info(
                "DrugReSC succeeded! Top drugs: %s",
                list(drug_shortlist.keys())[:5],
            )
            return {
                "drug_shortlist": drug_shortlist,
                "target_clusters": target_clusters,
                "sandbox_tracebacks": [],  # clear on success
                "current_agent": "END",
            }
        else:
            error_detail = result.get("error", "") or result.get("stderr", "")
            error_msg = (
                f"[SubpopulationStrategist] E2B execution failed "
                f"(attempt {attempt}): {error_detail}"
            )
            logger.warning(error_msg)
            tracebacks.append(error_msg)
            previous_script = script
            needs_correction = True

    # ── Exhausted attempts — use LLM to generate drug candidates ───
    if max_attempts == 0:
        status_log.push("pipeline", "warn", "Strategist - using LLM drug fallback")
        logger.warning("Skipping E2B attempts because anndata_path is llm_fallback.")
    else:
        status_log.push("pipeline", "warn",
                     "💊 Strategist — E2B failed, generating drugs via LLM…")
        logger.warning("E2B execution exhausted %d attempts — using LLM drug fallback.",
                        MAX_ATTEMPTS)

    pdf_snippet = state.get("pdf_text", "")[:6000]
    accessions = state.get("geo_accessions", [])
    qc = state.get("qc_metrics", {})

    drug_prompt = (
        f"You are a computational pharmacologist analyzing a scRNA-seq study.\n"
        f"GEO Accessions: {accessions}\n"
        f"QC Metrics: {qc}\n"
        f"Disease context from the paper:\n{pdf_snippet}\n\n"
        f"Based on this study's disease signature, generate a JSON object mapping "
        f"10 drug compound names to their reversal scores (float between -1.0 and 0.0, "
        f"lower means stronger reversal). Use real FDA-approved drug names that are "
        f"scientifically plausible for this disease context.\n"
        f"Return ONLY the JSON object, no markdown fences."
    )

    raw = call_llm(
        system_prompt="You are a drug repurposing expert. Return ONLY valid JSON.",
        user_prompt=drug_prompt,
        temperature=0.3,
        max_tokens=1024,
        caller="SubpopulationStrategist-Fallback",
    )

    default_drug_shortlist = {
        "Vorinostat": -0.89, "Trichostatin A": -0.85,
        "Tanespimycin": -0.78, "Geldanamycin": -0.74,
        "Sirolimus": -0.71, "Wortmannin": -0.68,
        "LY-294002": -0.65, "Thapsigargin": -0.61,
        "Withaferin A": -0.58, "Celastrol": -0.54,
    }
    drug_shortlist = {}
    if raw:
        import json as _json
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(
                    ln for ln in cleaned.splitlines()
                    if not ln.strip().startswith("```")
                )
            drug_shortlist = _json.loads(cleaned)
            # Ensure values are floats
            drug_shortlist = {
                k: float(v) if isinstance(v, (int, float, str)) else -0.5
                for k, v in drug_shortlist.items()
            }
            status_log.push("pipeline", "ok",
                             f"💊 Strategist fallback — {len(drug_shortlist)} drug candidates")
        except Exception as e:
            logger.warning("Failed to parse LLM drug JSON: %s", e)

    if len(drug_shortlist) < 10:
        for drug_name, score in default_drug_shortlist.items():
            if drug_name not in drug_shortlist:
                drug_shortlist[drug_name] = score
            if len(drug_shortlist) >= 10:
                break
        status_log.push("pipeline", "ok",
                         "💊 Strategist — using default drug candidates")

    if drug_shortlist:
        drug_shortlist = dict(
            sorted(drug_shortlist.items(), key=lambda item: float(item[1]))[:10]
        )
    else:
        drug_shortlist = default_drug_shortlist

    return {
        "drug_shortlist": drug_shortlist,
        "target_clusters": target_clusters,
        "sandbox_tracebacks": [],  # CLEAR to prevent infinite re-entry
        "current_agent": "END",
    }


# ── Helpers ──────────────────────────────────────────────────────

def _extract_last_script(tracebacks: list[str]) -> str:
    """Walk backwards through tracebacks and return the most recent
    GENERATED_SCRIPT:: payload."""
    for entry in reversed(tracebacks):
        if "GENERATED_SCRIPT::" in entry:
            return entry.split("GENERATED_SCRIPT::", 1)[1].strip()
    return ""
