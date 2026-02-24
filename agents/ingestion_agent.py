"""
CellAtria 2.0 - Ingestion Agent (LangGraph Node)
================================================
Downloads GEO supplementary files and prepares artifacts for downstream
analysis. This module is intentionally defensive around network and
filesystem failures so the pipeline can fall back cleanly.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import logging
import os
import re
import shutil
import tarfile
import textwrap
from ftplib import FTP
from pathlib import Path
from typing import Any

import requests
from langchain_core.tools import tool

from core.state import CellAtriaState
from core.status_logger import status_log

logger = logging.getLogger(__name__)

RUNTIME_ROOT = Path(
    os.environ.get("CELLATRIA_RUNTIME_DIR", str((Path.cwd() / "runtime").resolve()))
)
SANDBOX_DATA_DIR = Path(
    os.environ.get("CELLATRIA_DATA_DIR", str((RUNTIME_ROOT / "cellatria_data").resolve()))
)
SANDBOX_SCRIPTS_DIR = Path(
    os.environ.get(
        "CELLATRIA_SCRIPTS_DIR",
        str((RUNTIME_ROOT / "cellatria_scripts").resolve()),
    )
)

DOWNLOAD_TIMEOUT_SECONDS = int(
    os.environ.get("INGESTION_DOWNLOAD_TIMEOUT_SECONDS", "120")
)
HTTP_TIMEOUT_SECONDS = int(os.environ.get("INGESTION_HTTP_TIMEOUT_SECONDS", "12"))
FTP_TIMEOUT_SECONDS = int(os.environ.get("INGESTION_FTP_TIMEOUT_SECONDS", "30"))

_INVALID_WINDOWS_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
_MATRIX_LINK_PATTERN = re.compile(
    r'href="([^"]+\.(?:h5ad|h5|mtx\.gz|tar\.gz))"',
    re.IGNORECASE,
)


def _clean_windows_dirname(name: str) -> str:
    """Sanitize a path segment for Windows compatibility."""
    cleaned = _INVALID_WINDOWS_PATH_CHARS.sub("_", str(name))
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip(" ._")
    return cleaned or "unnamed"


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _download_file(url: str, local_path: Path, timeout: int) -> bool:
    """Download a URL to `local_path`. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        if resp.status_code != 200:
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Download failed for %s: %s", url, exc)
        return False


@tool
def download_and_compile_matrix(geo_accession: str) -> dict[str, Any]:
    """Download GEO files and generate a compile script.

    Returns keys:
    - raw_paths
    - anndata_path (only if an actual .h5ad already exists)
    - sandbox_script
    - error
    """
    accession_dir = SANDBOX_DATA_DIR / geo_accession
    accession_dir.mkdir(parents=True, exist_ok=True)

    raw_paths: list[str] = []
    try:
        raw_paths = _download_via_geoparse(geo_accession, accession_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "GEOparse download failed for %s (%s) - trying raw FTP.",
            geo_accession,
            exc,
        )
        try:
            raw_paths = _download_via_ftp(geo_accession, accession_dir)
        except Exception as ftp_exc:  # noqa: BLE001
            error_msg = (
                f"Both GEOparse and FTP download failed for {geo_accession}: {ftp_exc}"
            )
            logger.error(error_msg)
            return {
                "raw_paths": [],
                "anndata_path": "",
                "sandbox_script": "",
                "error": error_msg,
            }

    raw_paths = _unpack_archives(accession_dir, raw_paths)

    triplet = _detect_10x_triplet(accession_dir)
    anndata_out = str(accession_dir / f"{geo_accession}.h5ad")

    if triplet:
        sandbox_script = _generate_10x_compile_script(
            mtx_dir=triplet["parent_dir"],
            output_path=anndata_out,
            accession=geo_accession,
        )
    else:
        sandbox_script = _generate_generic_compile_script(
            data_dir=str(accession_dir),
            output_path=anndata_out,
            accession=geo_accession,
        )

    SANDBOX_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    script_path = SANDBOX_SCRIPTS_DIR / f"compile_{geo_accession}.py"
    script_path.write_text(sandbox_script, encoding="utf-8")
    logger.info("Sandbox compilation script written -> %s", script_path)

    existing_h5ad = ""
    for p in raw_paths:
        if p.lower().endswith(".h5ad"):
            existing_h5ad = p
            break

    return {
        "raw_paths": _dedupe_keep_order([str(p) for p in raw_paths]),
        "anndata_path": existing_h5ad,
        "sandbox_script": sandbox_script,
        "error": "",
    }


def _download_via_geoparse(accession: str, accession_dir: Path) -> list[str]:
    """Use GEOparse to pull supplementary files for a GEO accession."""
    import GEOparse  # type: ignore[import-untyped]

    gse = GEOparse.get_GEO(geo=accession, destdir=str(accession_dir), silent=True)
    downloaded: list[str] = []

    def _iter_values(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        as_text = str(value).strip()
        return [as_text] if as_text else []

    for gsm_name, gsm in gse.gsms.items():
        gsm_dir = accession_dir / _clean_windows_dirname(f"Supp_{gsm_name}")
        gsm_dir.mkdir(parents=True, exist_ok=True)

        for file_url in _iter_values(gsm.metadata.get("supplementary_file")):
            lower_url = file_url.lower()
            if not any(
                ext in lower_url
                for ext in (".h5", ".h5ad", ".mtx", ".csv", ".tsv", ".txt", ".tar.gz")
            ):
                continue
            safe_name = _clean_windows_dirname(file_url.split("/")[-1])
            local_path = gsm_dir / safe_name
            if local_path.exists() or _download_file(
                file_url,
                local_path,
                timeout=HTTP_TIMEOUT_SECONDS + 8,
            ):
                downloaded.append(str(local_path))

    for file_url in _iter_values(gse.metadata.get("supplementary_file")):
        safe_name = _clean_windows_dirname(file_url.split("/")[-1])
        local_path = accession_dir / safe_name
        if local_path.exists() or _download_file(
            file_url,
            local_path,
            timeout=HTTP_TIMEOUT_SECONDS + 8,
        ):
            downloaded.append(str(local_path))

    downloaded = _dedupe_keep_order(downloaded)
    logger.info("GEOparse downloaded %d file(s) for %s", len(downloaded), accession)
    return downloaded


def _download_via_ftp(accession: str, dest: Path) -> list[str]:
    """Fallback: download supplementary files from NCBI GEO FTP."""
    nnn = accession[3:]
    prefix = nnn[:-3] if len(nnn) > 3 else ""
    ftp_series = f"GSE{prefix}nnn"
    ftp_path = f"/geo/series/{ftp_series}/{accession}/suppl/"

    dest.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []

    ftp = FTP("ftp.ncbi.nlm.nih.gov", timeout=FTP_TIMEOUT_SECONDS)
    ftp.login()
    ftp.cwd(ftp_path)

    for fname in ftp.nlst():
        safe_name = _clean_windows_dirname(fname)
        local_path = dest / safe_name
        with open(local_path, "wb") as fh:
            ftp.retrbinary(f"RETR {fname}", fh.write)
        downloaded.append(str(local_path))
        logger.info("FTP downloaded -> %s", local_path)

    ftp.quit()
    return downloaded


def _unpack_archives(accession_dir: Path, raw_paths: list[str]) -> list[str]:
    """Unpack .tar.gz / .gz files in-place and return updated paths."""
    updated_paths: list[str] = list(raw_paths)

    for fpath in list(raw_paths):
        p = Path(fpath)
        if not p.exists():
            continue

        if p.suffixes[-2:] == [".tar", ".gz"] or p.suffix == ".tgz":
            try:
                with tarfile.open(p, "r:gz") as tar:
                    try:
                        tar.extractall(path=accession_dir, filter="data")
                    except TypeError:
                        tar.extractall(path=accession_dir)
                    extracted = [
                        str(accession_dir / m.name)
                        for m in tar.getmembers()
                        if m.isfile()
                    ]
                logger.info("Unpacked tarball -> %s", p.name)
                updated_paths.extend(extracted)
                if fpath in updated_paths:
                    updated_paths.remove(fpath)
            except tarfile.TarError as exc:
                logger.warning("Failed to unpack %s: %s", p, exc)

        elif p.suffix == ".gz" and ".tar" not in p.name:
            out_path = accession_dir / p.stem
            try:
                with gzip.open(p, "rb") as gz_in, open(out_path, "wb") as f_out:
                    shutil.copyfileobj(gz_in, f_out)
                logger.info("Decompressed %s -> %s", p.name, out_path.name)
                updated_paths.append(str(out_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to decompress %s: %s", p, exc)

    return _dedupe_keep_order(updated_paths)


def _detect_10x_triplet(data_dir: Path) -> dict[str, str] | None:
    """Find matrix.mtx + barcodes + features/genes in one directory."""
    for mtx_file in data_dir.rglob("matrix.mtx*"):
        parent = mtx_file.parent
        barcodes = _find_one(parent, "barcodes.tsv*")
        features = _find_one(parent, "features.tsv*") or _find_one(parent, "genes.tsv*")
        if barcodes and features:
            logger.info("10X triplet detected in %s", parent)
            return {
                "matrix": str(mtx_file),
                "barcodes": str(barcodes),
                "features": str(features),
                "parent_dir": str(parent),
            }
    return None


def _find_one(directory: Path, pattern: str) -> Path | None:
    matches = list(directory.glob(pattern))
    return matches[0] if matches else None


def _generate_10x_compile_script(mtx_dir: str, output_path: str, accession: str) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import scanpy as sc

        print("Reading 10X matrix from: {mtx_dir}")
        adata = sc.read_10x_mtx(
            "{mtx_dir}",
            var_names="gene_symbols",
            cache=False,
        )
        adata.var_names_make_unique()
        adata.obs["geo_accession"] = "{accession}"

        print(f"AnnData: {{adata.n_obs}} cells x {{adata.n_vars}} genes")
        adata.write_h5ad("{output_path}")
        print("Written -> {output_path}")
        """
    )


def _generate_generic_compile_script(
    data_dir: str,
    output_path: str,
    accession: str,
) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import pathlib
        import anndata as ad
        import pandas as pd

        data_dir = pathlib.Path("{data_dir}")
        candidates = (
            list(data_dir.rglob("*.tsv"))
            + list(data_dir.rglob("*.csv"))
            + list(data_dir.rglob("*.txt"))
        )

        if not candidates:
            raise FileNotFoundError(f"No expression matrix files found in {{data_dir}}")

        matrix_file = max(candidates, key=lambda p: p.stat().st_size)
        print(f"Loading expression matrix: {{matrix_file}}")

        sep = "\\t" if matrix_file.suffix in (".tsv", ".txt") else ","
        df = pd.read_csv(matrix_file, sep=sep, index_col=0)

        if df.shape[0] < df.shape[1]:
            df = df.T

        adata = ad.AnnData(df)
        adata.var_names_make_unique()
        adata.obs["geo_accession"] = "{accession}"

        print(f"AnnData: {{adata.n_obs}} cells x {{adata.n_vars}} genes")
        adata.write_h5ad("{output_path}")
        print("Written -> {output_path}")
        """
    )


def ingestion_agent(state: CellAtriaState) -> dict[str, Any]:
    """LangGraph node: Ingestion Agent."""
    accessions: list[str] = state.get("geo_accessions", [])

    if len(accessions) > 1:
        logger.info("Limiting accessions from %d to 1 for pipeline speed.", len(accessions))
        status_log.push("pipeline", "ok", f"Ingestion - using 1 of {len(accessions)} accessions")
        accessions = accessions[:1]

    if not accessions:
        logger.warning("ingestion_agent received zero accessions - nothing to download.")
        status_log.push("pipeline", "warn", "Ingestion Agent - no accessions, skipping")
        return {
            "raw_data_paths": [],
            "anndata_path": "",
            "current_agent": "BioinformaticianAgent",
        }

    status_log.push("pipeline", "ok", "Ingestion Agent - downloading accession data...")

    SANDBOX_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SANDBOX_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    all_raw_paths: list[str] = []
    final_anndata_path = ""
    tracebacks: list[str] = list(state.get("sandbox_tracebacks", []))

    for accession in accessions:
        logger.info("Processing accession: %s", accession)
        status_log.push("pipeline", "ok", f"Ingestion Agent - downloading {accession}...")

        accession_dir = SANDBOX_DATA_DIR / accession
        accession_dir.mkdir(parents=True, exist_ok=True)

        # Try 1: direct HTTP listing from NCBI.
        http_ok = False
        try:
            http_url = (
                f"https://ftp.ncbi.nlm.nih.gov/geo/series/"
                f"{accession[:-3]}nnn/{accession}/suppl/"
            )
            status_log.push("ftp", "ok", f"Trying HTTP: {accession}")
            resp = requests.get(http_url, timeout=HTTP_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                links = _MATRIX_LINK_PATTERN.findall(resp.text)
                if links:
                    def _priority(name: str) -> int:
                        lname = name.lower()
                        if lname.endswith(".h5ad"):
                            return 0
                        if lname.endswith(".h5"):
                            return 1
                        if lname.endswith(".mtx.gz"):
                            return 2
                        return 3

                    for candidate in sorted(links, key=_priority):
                        file_url = http_url.rstrip("/") + "/" + candidate
                        safe_name = _clean_windows_dirname(candidate.split("/")[-1])
                        local_path = accession_dir / safe_name
                        if _download_file(file_url, local_path, timeout=HTTP_TIMEOUT_SECONDS):
                            all_raw_paths.append(str(local_path))
                            http_ok = True
                            status_log.push("ftp", "ok", f"HTTP download OK: {safe_name}")
                            logger.info("[HTTP] Downloaded %s for %s", safe_name, accession)
                            if safe_name.lower().endswith(".h5ad"):
                                final_anndata_path = str(local_path)
                                break
            else:
                logger.info("[HTTP] %s returned status %d", accession, resp.status_code)
        except Exception as exc:  # noqa: BLE001
            logger.info("[HTTP] %s failed: %s", accession, exc)
            status_log.push("ftp", "warn", f"HTTP failed: {accession} - trying FTP...")

        # Try 2: GEOparse/FTP helper with timeout.
        if not http_ok:
            try:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(
                    download_and_compile_matrix.invoke,
                    {"geo_accession": accession},
                )
                try:
                    result = future.result(timeout=DOWNLOAD_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    executor.shutdown(wait=False, cancel_futures=True)
                    msg = (
                        f"[IngestionAgent] {accession}: download timed out "
                        f"({DOWNLOAD_TIMEOUT_SECONDS}s)"
                    )
                    logger.warning(msg)
                    tracebacks.append(msg)
                    status_log.push(
                        "ftp",
                        "warn",
                        f"Download timeout: {accession} ({DOWNLOAD_TIMEOUT_SECONDS}s)",
                    )
                    continue
                finally:
                    executor.shutdown(wait=False)
            except Exception as exc:  # noqa: BLE001
                msg = f"[IngestionAgent] {accession}: download error - {exc}"
                logger.warning(msg)
                tracebacks.append(msg)
                status_log.push("ftp", "error", f"Download error: {accession}")
                continue

            if result.get("error"):
                tracebacks.append(f"[IngestionAgent] {accession}: {result['error']}")
                status_log.push("ftp", "error", f"Download failed: {accession}")
                continue

            all_raw_paths.extend(result.get("raw_paths", []))
            if result.get("anndata_path"):
                final_anndata_path = str(result["anndata_path"])
            logger.info(
                "Accession %s -> %d raw files, anndata -> %s",
                accession,
                len(result.get("raw_paths", [])),
                result.get("anndata_path", ""),
            )

    all_raw_paths = _dedupe_keep_order(all_raw_paths)
    status_log.push(
        "pipeline",
        "ok",
        f"Ingestion Agent done - {len(all_raw_paths)} files, "
        f"anndata: {'yes' if final_anndata_path else 'no'}",
    )

    return {
        "raw_data_paths": all_raw_paths,
        "anndata_path": final_anndata_path,
        "sandbox_tracebacks": tracebacks,
        "current_agent": "BioinformaticianAgent",
    }
