# ============================================================
# CellAtria 2.0 — E2B Bioinformatics Sandbox
# ============================================================
# Highly-optimized, layered Dockerfile for the custom E2B
# sandbox template.  Separates system libs → R/Bioconductor →
# Python to maximise Docker layer caching and avoid the
# notorious R ↔ Python dependency hell.
#
# Build:  docker build -t cellatria-sandbox .
# ============================================================

# --------------------------------------------------
# Stage 0 · Base image + locale
# --------------------------------------------------
FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Etc/UTC

# --------------------------------------------------
# Layer 1 · System dependencies
# --------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build toolchain
    build-essential \
    gfortran \
    cmake \
    pkg-config \
    # R runtime (CRAN upstream)
    r-base \
    r-base-dev \
    # Curl / XML / SSL (needed by R packages & rpy2)
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    # Graphics & font stack (Seurat plotting, ggplot2)
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libcairo2-dev \
    # HDF5 (anndata / h5ad read/write)
    libhdf5-dev \
    # Python 3.10
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    # Misc utilities
    wget \
    curl \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 / pip point to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# --------------------------------------------------
# Layer 2 · R & Bioconductor
# --------------------------------------------------
# Install Seurat + BiocManager + core Bioconductor deps in a
# single Rscript invocation to keep layer count low.
# NCPUS parallelises compilation of heavy C++/Fortran packages.
RUN Rscript -e "\
    options(repos = c(CRAN = 'https://cloud.r-project.org'), \
    Ncpus  = parallel::detectCores()); \
    \
    # BiocManager first \
    install.packages('BiocManager', quiet = TRUE); \
    \
    # Core Bioconductor infrastructure \
    BiocManager::install(c( \
    'GenomicRanges', \
    'SingleCellExperiment', \
    'scran', \
    'scater', \
    'DESeq2', \
    'limma', \
    'ComplexHeatmap', \
    'clusterProfiler', \
    'org.Hs.eg.db' \
    ), ask = FALSE, update = FALSE); \
    \
    # Seurat (CRAN) \
    install.packages('Seurat', quiet = TRUE); \
    \
    cat('\\n=== R/Bioconductor layer complete ===\\n') \
    "

# --------------------------------------------------
# Layer 3 · Python scientific / ML stack
# --------------------------------------------------
# Pin major versions to lock the dependency tree.
# rpy2 bridges R ↔ Python; installed last to link against
# the R shared library compiled above.
COPY e2b/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Fallback: if no requirements.txt is mounted at build time,
# install the core stack directly.
RUN python3 -m pip install --no-cache-dir \
    "scanpy>=1.10,<2" \
    "anndata>=0.10,<1" \
    "pandas>=2.0,<3" \
    "numpy>=1.24,<2" \
    "torch>=2.1,<3" --extra-index-url https://download.pytorch.org/whl/cpu \
    "harmonypy>=0.0.9" \
    "scvi-tools>=1.1,<2" \
    "rpy2>=3.5,<4" \
    "e2b-code-interpreter>=0.0.11" \
    "leidenalg>=0.10" \
    "session-info" \
    2>/dev/null || true

# --------------------------------------------------
# Smoke-test: verify critical imports
# --------------------------------------------------
RUN python3 -c "import scanpy, anndata, torch, harmonypy, rpy2; print('=== Python Layer OK ===')"

# --------------------------------------------------
# Runtime defaults
# --------------------------------------------------
WORKDIR /home/user
ENV HOME=/home/user \
    R_HOME=/usr/lib/R \
    R_LIBS_USER=/usr/local/lib/R/site-library

CMD ["python3"]
