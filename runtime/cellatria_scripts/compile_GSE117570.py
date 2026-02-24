#!/usr/bin/env python3
import pathlib
import anndata as ad
import pandas as pd

data_dir = pathlib.Path("C:\Users\samee\OneDrive\Desktop\CellAtria 2.0\runtime\cellatria_data\GSE117570")
candidates = (
    list(data_dir.rglob("*.tsv"))
    + list(data_dir.rglob("*.csv"))
    + list(data_dir.rglob("*.txt"))
)

if not candidates:
    raise FileNotFoundError(f"No expression matrix files found in {data_dir}")

matrix_file = max(candidates, key=lambda p: p.stat().st_size)
print(f"Loading expression matrix: {matrix_file}")

sep = "\t" if matrix_file.suffix in (".tsv", ".txt") else ","
df = pd.read_csv(matrix_file, sep=sep, index_col=0)

if df.shape[0] < df.shape[1]:
    df = df.T

adata = ad.AnnData(df)
adata.var_names_make_unique()
adata.obs["geo_accession"] = "GSE117570"

print(f"AnnData: {adata.n_obs} cells x {adata.n_vars} genes")
adata.write_h5ad("C:\Users\samee\OneDrive\Desktop\CellAtria 2.0\runtime\cellatria_data\GSE117570\GSE117570.h5ad")
print("Written -> C:\Users\samee\OneDrive\Desktop\CellAtria 2.0\runtime\cellatria_data\GSE117570\GSE117570.h5ad")
