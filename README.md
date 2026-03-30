# Federated GWAS — VCF Processing Pipeline

## Prerequisites

- Python 3.12 or later
- Required Python packages: `pandas`, `matplotlib`, `tqdm`

Install dependencies with:
```
pip install -r requirements.txt
```

---

## Input requirements

Place all your `.vcf` files in the `data/` folder at the root of the project.

---

## Step 1 — Parse VCF files

From the project root directory, run:
```bash
python src/parsing/parse_vcf.py
```

This scans the `data/` folder, parses all `.vcf` files, and saves the result to:
```
output/variants.csv
```
Each row corresponds to one variant–transcript pair per patient, with columns for standard VCF fields, VEP annotations (consequence, gene, HGVS, gnomAD frequencies), and genotype fields (GT, DP, VAF, etc.).

---

## Step 2 — Run variant distribution analysis

Once the CSV files are produced, run:
```bash
python src/analysis/variant_distribution.py --input-dir output/synthetic_patients --no-show
```

The script computes per-locus, gene-level, and bin-level mutation frequencies across the cohort and saves the results to `output/distribution_analysis/`:

| File | Description |
|------|-------------|
| `variant_level_analysis.csv` | Per-locus frequencies across patients |
| `gene_level_analysis.csv` | Gene-level aggregation |
| `bin_distribution.csv` | Fixed-size genomic bin aggregation |
| `variant_distribution.png` | Manhattan-style variant frequency plot |
| `gene_distribution.png` | Gene-level frequency plot |
| `bin_distribution.png` | Bin-level frequency plot |

---

## Optional parameters

The following parameters can be appended to the command above to customise the analysis:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir PATH` | `output/synthetic_patients` | Folder containing the per-patient CSV files produced in Step 1. Change this if your CSVs are stored elsewhere. |
| `--output-dir PATH` | `output/distribution_analysis` | Destination folder for all output files (CSVs and plots). It is created automatically if it does not exist. |
| `--top N` | `15` | Number of top variants (by frequency) to annotate and highlight in the plots. Increase this if you want more loci labelled, reduce it for cleaner figures. |
| `--min-freq VALUE` | `0.0` | Minimum cohort frequency (0–1) for a variant to appear in the plots. For example, `--min-freq 0.05` restricts the visualisation to variants present in at least 5 % of patients, filtering out very rare singletons. Does not affect the CSV outputs. |
| `--bin-length BP` | `2000000` | Size of each genomic bin in base pairs for the bin-level aggregation (default: 2 Mb). Smaller bins give finer resolution but produce more rows; larger bins give a smoother, higher-level view. |
| `--bin-overlap BP` | `500000` | Overlap between consecutive bins in base pairs (default: 500 kb). Must be strictly less than `--bin-length`. A non-zero overlap creates sliding windows, which can help detect signals near bin boundaries. Set to `0` for non-overlapping bins (faster). |
| `--no-show` | *(off)* | Suppresses the interactive plot window and only saves figures to disk. Use this when running on a server without a graphical interface. |
| `--output-name FILE` | `variant_distribution.png` | Filename for the variant-level Manhattan plot. |
| `--gene-output-name FILE` | `gene_distribution.png` | Filename for the gene-level frequency plot. |
| `--bin-plot-name FILE` | `bin_distribution.png` | Filename for the bin-level frequency plot. |
| `--bin-output-name FILE` | `bin_distribution.csv` | Filename for the bin-level aggregation CSV. |

Example combining several options:
```bash
python src/analysis/variant_distribution.py \
    --input-dir output/synthetic_patients \
    --output-dir output/my_analysis \
    --min-freq 0.05 \
    --top 20 \
    --bin-length 1000000 \
    --bin-overlap 0 \
    --no-show
```

---

## Column reference

A full description of every output column from Step 1 is available in [`src/parsing/README.md`](src/parsing/README.md).