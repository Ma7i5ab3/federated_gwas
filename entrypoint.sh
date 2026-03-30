#!/bin/bash
set -e

# Step 1 — Parse VCF files from /app/data → /app/output/variants.csv
echo "=== Step 1: Parsing VCF files ==="
python src/parsing/parse_vcf.py

# Step 2 — Run variant distribution analysis
# Override ANALYSIS_INPUT_DIR to point to a different folder of per-patient CSVs.
ANALYSIS_INPUT_DIR="${ANALYSIS_INPUT_DIR:-output/parsed}"

echo "=== Step 2: Running first variant distribution analysis (input: $ANALYSIS_INPUT_DIR) ==="
python src/analysis/variant_distribution.py \
    --input-dir "$ANALYSIS_INPUT_DIR" \
    --bin-length 500000 \
    --bin-overlap 0 \
    --no-show \
    ${ANALYSIS_EXTRA_ARGS:-}

echo "=== Step 3: Running second variant distribution analysis (input: $ANALYSIS_INPUT_DIR) ==="
python src/analysis/variant_distribution.py \
    --input-dir "$ANALYSIS_INPUT_DIR" \
    --bin-length 2000000 \
    --bin-overlap 500000 \
    --no-show \
    ${ANALYSIS_EXTRA_ARGS:-}

echo "=== Done. Results are in /app/output/distribution_analysis ==="
