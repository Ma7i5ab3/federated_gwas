"""
variant_distribution.py
-----------------------
GWAS-style variant frequency analysis across a cohort of patients.

Reads per-patient Parquet files and computes:
  - per-locus  variant counts
  - per-gene   variant statistics
  - per-genomic-bin statistics

Memory model
------------
All heavy aggregation is executed inside DuckDB, which reads Parquet files
in columnar streaming batches and automatically spills intermediate state to
disk when working sets exceed available RAM.  Python only receives the final,
compacted result DataFrames — never the raw per-row data.

Usage
-----
    python analysis/variant_distribution.py
    python analysis/variant_distribution.py --input-dir output/parsed
    python analysis/variant_distribution.py --input-dir /path/to/parquets --top 20
    python analysis/variant_distribution.py --min-freq 0.05
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from plots import (
    CHROM_ORDER,
    IMPACT_RANK,
    make_gene_figure,
    make_bin_figure,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ANN_COLS   = ["SYMBOL", "Consequence", "IMPACT", "Gene"]
_PASS1_COLS = ["CHROM", "POS", "REF", "ALT"] + ["GT"] + _ANN_COLS
_PASS2_COLS = ["CHROM", "POS", "REF", "ALT", "GT", "SYMBOL", "IMPACT"]

# SQL IN-list for reference / missing genotype strings
_REF_CALLS_SQL = "('0/0', './.', '0|0', '.|.')"

# SQL CASE expression: maps IMPACT text to integer rank (mirrors IMPACT_RANK)
_IMPACT_RANK_SQL = """
    CASE IMPACT WHEN 'HIGH' THEN 4 WHEN 'MODERATE' THEN 3
                WHEN 'LOW'  THEN 2 ELSE 1 END
"""

_RANK_TO_IMPACT = {v: k for k, v in IMPACT_RANK.items()}


# ---------------------------------------------------------------------------
# DuckDB utilities
# ---------------------------------------------------------------------------

def _open_db() -> duckdb.DuckDBPyConnection:
    """
    Open an in-process DuckDB connection.
    The temp directory allows DuckDB to spill to disk automatically, so
    large cohorts (hundreds of patients) do not exhaust RAM.
    """
    con = duckdb.connect()
    con.execute("SET temp_directory = '/tmp/duckdb_gwas_spill'")
    return con


def _available(parquet_files: list[Path], wanted: list[str]) -> list[str]:
    """Return the subset of *wanted* columns present in the Parquet schema."""
    present = set(pq.read_schema(parquet_files[0]).names)
    return [c for c in wanted if c in present]


def _union_sql(parquet_files: list[Path], cols: list[str]) -> str:
    """
    Build a UNION ALL query that reads all Parquet files in one sweep.
    Each row is tagged with a numeric patient index (_pid) so downstream
    queries can use COUNT(DISTINCT _pid) for per-patient deduplication.
    Columns absent from the schema are emitted as NULL to keep a uniform schema.
    """
    present = set(pq.read_schema(parquet_files[0]).names)
    exprs = []
    for c in cols:
        if c == "POS":
            # Always cast POS to BIGINT for safe arithmetic
            exprs.append("CAST(POS AS BIGINT) AS POS")
        elif c in present:
            exprs.append(c)
        else:
            # Column missing from this cohort — emit NULL with the expected name
            exprs.append(f"NULL AS {c}")
    col_str = ", ".join(exprs)
    parts = [
        f"SELECT {i} AS _pid, {col_str} FROM read_parquet('{f}')"
        for i, f in enumerate(parquet_files)
    ]
    return "\n    UNION ALL\n    ".join(parts)


# ---------------------------------------------------------------------------
# Pass 1 — locus + gene aggregation
# ---------------------------------------------------------------------------

def _gnomad_af_filter_sql(present: list[str], max_af: float) -> str:
    """
    Return a SQL WHERE fragment that keeps only rare variants.

    Variants where gnomADg_AF is NULL (novel / not in gnomAD) are always
    retained — they are the most relevant candidates in rare-disease studies.
    Variants with gnomADg_AF > max_af are dropped.

    If the gnomADg_AF column is absent from the schema the filter is omitted
    and a no-op empty string is returned.
    """
    if "gnomADg_AF" not in present:
        logger.warning("gnomADg_AF column absent — gnomAD AF filter will not be applied")
        return ""
    return (
        f" AND (gnomADg_AF IS NULL"
        f" OR TRY_CAST(gnomADg_AF AS DOUBLE) IS NULL"
        f" OR TRY_CAST(gnomADg_AF AS DOUBLE) <= {max_af})"
    )


def _pass1(
    con: duckdb.DuckDBPyConnection,
    parquet_files: list[Path],
    max_gnomad_af: float,
) -> pd.DataFrame:
    """
    Compute per-gene aggregated stats via DuckDB.

    Strategy:
      1. UNION ALL all Parquet files, tagging every row with its patient index.
      2. GROUP BY (patient, CHROM, POS, REF, ALT) collapses multi-transcript
         VEP duplicates so each variant is counted only once per patient.
      3. A second GROUP BY across patients gives cohort-level counts.

    DuckDB executes this with streaming I/O and disk spill; Python never
    holds more than the final aggregated DataFrames in memory.
    """
    logger.info("Pass1 starting — {} parquet file(s)", len(parquet_files))
    logger.info("Pass1 gnomAD AF filter  : gnomADg_AF <= {}", max_gnomad_af)

    # Include gnomADg_AF in the columns to read so the filter can be applied
    pass1_cols  = _PASS1_COLS + ["gnomADg_AF"]
    present     = _available(parquet_files, pass1_cols)
    ann_present = [c for c in _ANN_COLS if c in present]
    logger.debug("Pass1 columns available : {}", present)
    logger.debug("Pass1 annotation cols   : {}", ann_present)
    missing = [c for c in _PASS1_COLS if c not in present]
    if missing:
        logger.warning("Pass1 missing expected cols: {}", missing)

    union       = _union_sql(parquet_files, present)

    # Filter: drop reference/missing calls, null coordinates, and common variants
    base_where = (
        f"GT IS NOT NULL AND GT NOT IN {_REF_CALLS_SQL}"
        " AND CHROM IS NOT NULL AND POS IS NOT NULL"
        " AND REF IS NOT NULL AND ALT IS NOT NULL"
        + _gnomad_af_filter_sql(present, max_gnomad_af)
    )

    # Dedup CTE: one row per (patient, locus).
    # FIRST() keeps annotations from whichever transcript row DuckDB sees first.
    ann_agg = (
        ",\n               ".join(f"FIRST({c}) AS {c}" for c in ann_present)
    )
    ann_comma = (",\n               " + ann_agg) if ann_present else ""

    deduped_cte = f"""
    deduped AS (
        SELECT _pid, CHROM, POS, REF, ALT{ann_comma}
        FROM ({union})
        WHERE {base_where}
        GROUP BY _pid, CHROM, POS, REF, ALT
    )"""

    # Gene table: aggregate per (patient, gene) first to avoid double-counting
    # variants shared across transcripts, then sum across patients.
    if "SYMBOL" not in present:
        logger.warning("Pass1 SYMBOL column absent — returning empty gene table")
        gene_df = pd.DataFrame(columns=[
            "SYMBOL", "CHROM", "patient_count", "variant_count",
            "gene_start", "gene_end", "worst_impact_rank",
        ])
    else:
        impact_expr = _IMPACT_RANK_SQL if "IMPACT" in present else "1"
        if "IMPACT" not in present:
            logger.warning("Pass1 IMPACT column absent — all impact ranks will be 1")
        logger.info("Pass1 executing gene aggregation SQL …")
        try:
            gene_df = con.execute(f"""
                WITH {deduped_cte},
                -- Per-patient gene stats (locus dedup already applied above)
                per_patient AS (
                    SELECT _pid, SYMBOL, CHROM,
                           COUNT(*)            AS variant_count,
                           MIN(POS)            AS gene_start,
                           MAX(POS)            AS gene_end,
                           MAX({impact_expr})  AS worst_impact_rank
                    FROM deduped
                    WHERE SYMBOL IS NOT NULL AND SYMBOL != ''
                    GROUP BY _pid, SYMBOL, CHROM
                )
                -- Cohort-level aggregation across all patients
                SELECT SYMBOL, CHROM,
                       COUNT(DISTINCT _pid)   AS patient_count,
                       SUM(variant_count)     AS variant_count,
                       MIN(gene_start)        AS gene_start,
                       MAX(gene_end)          AS gene_end,
                       MAX(worst_impact_rank) AS worst_impact_rank
                FROM per_patient
                GROUP BY SYMBOL, CHROM
            """).df()
        except Exception as exc:
            logger.error("Pass1 SQL execution failed: {}", exc)
            raise

        logger.info("Pass1 result: {:,} gene rows, columns={}", len(gene_df), list(gene_df.columns))
        if gene_df.empty:
            logger.warning("Pass1 gene aggregation returned 0 rows after filtering")
        else:
            null_symbol = gene_df["SYMBOL"].isna().sum()
            null_chrom  = gene_df["CHROM"].isna().sum()
            if null_symbol:
                logger.warning("Pass1 {:,} rows with null SYMBOL", null_symbol)
            if null_chrom:
                logger.warning("Pass1 {:,} rows with null CHROM", null_chrom)
            logger.debug(
                "Pass1 variant_count range: [{}, {}]",
                gene_df["variant_count"].min(), gene_df["variant_count"].max(),
            )
            logger.debug(
                "Pass1 patient_count range: [{}, {}]",
                gene_df["patient_count"].min(), gene_df["patient_count"].max(),
            )

    return gene_df


# ---------------------------------------------------------------------------
# Pass 2 — bin aggregation
# ---------------------------------------------------------------------------

def _pass2(
    con: duckdb.DuckDBPyConnection,
    parquet_files: list[Path],
    gene_bounds_df: pd.DataFrame,
    n_patients: int,
    bin_length: int,
    overlap: int,
    max_gnomad_af: float,
) -> pd.DataFrame:
    """
    Assign each unique variant locus to genomic bins and aggregate per-bin.

    Gene boundaries from Pass 1 are registered as a small DuckDB virtual
    table for an efficient JOIN (O(n_genes) memory, not O(n_variants)).

    No-overlap bins: integer division maps each locus to exactly one bin.
    Overlapping bins: UNNEST(range(b_min, b_max+1)) explodes each locus into
    all bins it overlaps before aggregating — runs entirely inside DuckDB.
    """
    logger.info(
        "Pass2 starting — bin_length={:,}, overlap={:,}, n_patients={}, gene_bounds={:,} genes",
        bin_length, overlap, n_patients, len(gene_bounds_df),
    )

    step = bin_length - overlap

    if gene_bounds_df.empty:
        logger.warning("Pass2 gene_bounds_df is empty — no bins can be computed")
        return pd.DataFrame()

    null_bounds = gene_bounds_df[["gene_start", "gene_end"]].isna().any(axis=1).sum()
    if null_bounds:
        logger.warning("Pass2 {:,} gene_bounds rows have null start/end", null_bounds)

    # Register gene boundaries so DuckDB can JOIN without re-reading Parquet
    con.register("_gene_bounds", gene_bounds_df[["SYMBOL", "CHROM", "gene_start", "gene_end"]])

    pass2_cols  = _PASS2_COLS + ["gnomADg_AF"]
    present     = _available(parquet_files, pass2_cols)
    logger.debug("Pass2 columns available : {}", present)
    missing = [c for c in _PASS2_COLS if c not in present]
    if missing:
        logger.warning("Pass2 missing expected cols: {}", missing)

    union       = _union_sql(parquet_files, present)
    impact_expr = _IMPACT_RANK_SQL if "IMPACT" in present else "1"
    if "IMPACT" not in present:
        logger.warning("Pass2 IMPACT column absent — all impact ranks will be 1")

    # Prefix gnomADg_AF with "v." to match the aliased subquery in Pass 2
    gnomad_filter = _gnomad_af_filter_sql(present, max_gnomad_af).replace(
        "gnomADg_AF", "v.gnomADg_AF"
    )
    base_where = (
        f"v.GT IS NOT NULL AND v.GT NOT IN {_REF_CALLS_SQL}"
        " AND v.SYMBOL IS NOT NULL AND v.SYMBOL != ''"
        " AND v.POS IS NOT NULL"
        + gnomad_filter
    )

    # Include REF/ALT in the dedup key when available so that two different
    # alleles at the same POS (multi-allelic site) are counted separately,
    # consistent with how Pass 1 deduplicates on (CHROM, POS, REF, ALT).
    ref_alt_group = ", v.REF, v.ALT" if all(c in present for c in ["REF", "ALT"]) else ""
    if not ref_alt_group:
        logger.warning("Pass2 REF/ALT absent — multi-allelic sites will not be split")

    # Dedup CTE: one row per (patient, SYMBOL, CHROM, POS, REF, ALT) with worst impact.
    # The JOIN with _gene_bounds is inlined here to avoid a second scan later.
    deduped_cte = f"""
    deduped AS (
        SELECT v._pid, v.SYMBOL, v.CHROM, v.POS,
               MAX({impact_expr}) AS impact_rank,
               g.gene_start, g.gene_end
        FROM ({union}) v
        JOIN _gene_bounds g ON v.SYMBOL = g.SYMBOL AND v.CHROM = g.CHROM
        WHERE {base_where}
          AND v.POS BETWEEN g.gene_start AND g.gene_end
        GROUP BY v._pid, v.SYMBOL, v.CHROM, v.POS{ref_alt_group}, g.gene_start, g.gene_end
    )"""

    _BIN_GROUP = "SYMBOL, CHROM, bin_id, bin_start, bin_end"

    if overlap == 0:
        logger.info("Pass2 using no-overlap (fast) bin path")
        # Fast path: integer division assigns each locus to one bin directly
        bin_sql = f"""
        WITH {deduped_cte},
        binned AS (
            SELECT _pid, SYMBOL, CHROM, impact_rank,
                   (POS - gene_start) / {bin_length}                              AS bin_id,
                   gene_start + ((POS - gene_start) / {bin_length}) * {bin_length}  AS bin_start,
                   LEAST(
                       gene_start + ((POS - gene_start) / {bin_length}) * {bin_length} + {bin_length} - 1,
                       gene_end
                   )                                                               AS bin_end
            FROM deduped
        )
        SELECT {_BIN_GROUP},
               COUNT(*)             AS variant_count,
               MAX(impact_rank)     AS worst_impact_rank,
               COUNT(DISTINCT _pid) AS patient_count
        FROM binned
        GROUP BY {_BIN_GROUP}
        """
    else:
        logger.info("Pass2 using overlap bin path (step={:,})", step)
        # Overlap path: compute the range of bin indices [b_min, b_max] that
        # each locus overlaps, then UNNEST(range(...)) explodes it into one
        # row per (locus, bin) before aggregating.
        bin_sql = f"""
        WITH {deduped_cte},
        with_range AS (
            SELECT *,
                   -- Earliest bin index whose window reaches this locus
                   GREATEST(0,
                       CAST(CEIL((POS - gene_start - {bin_length} + 1.0) / {step}) AS BIGINT)
                   )                                                              AS b_min,
                   -- Latest bin index whose window starts at or before this locus
                   CAST((POS - gene_start) / {step} AS BIGINT)                   AS b_max
            FROM deduped
        ),
        -- One row per (locus × bin) — DuckDB unnests the integer range in-engine
        exploded AS (
            SELECT _pid, SYMBOL, CHROM, impact_rank, gene_start, gene_end,
                   UNNEST(range(b_min, b_max + 1)) AS bin_id
            FROM with_range
        )
        SELECT SYMBOL, CHROM, bin_id,
               gene_start + bin_id * {step}                                      AS bin_start,
               LEAST(gene_start + bin_id * {step} + {bin_length} - 1, gene_end) AS bin_end,
               COUNT(*)             AS variant_count,
               MAX(impact_rank)     AS worst_impact_rank,
               COUNT(DISTINCT _pid) AS patient_count
        FROM exploded
        WHERE gene_start + bin_id * {step} <= gene_end
        GROUP BY SYMBOL, CHROM, bin_id, gene_start, gene_end
        """

    logger.info("Pass2 executing bin aggregation SQL …")
    try:
        bin_df = con.execute(bin_sql).df()
    except Exception as exc:
        logger.error("Pass2 SQL execution failed: {}", exc)
        raise

    logger.info("Pass2 raw result: {:,} bin rows", len(bin_df))
    if bin_df.empty:
        logger.warning(
            "Pass2 bin aggregation returned 0 rows — "
            "check that SYMBOL/CHROM values match between variants and gene_bounds"
        )
        return pd.DataFrame()

    # Derive frequency, density, and a human-readable impact label in Python
    bin_df["bin_length_actual"]  = bin_df["bin_end"] - bin_df["bin_start"] + 1
    bin_df["density"]            = bin_df["variant_count"] / (bin_df["bin_length_actual"] * n_patients)
    bin_df["worst_impact"]       = bin_df["worst_impact_rank"].map(_RANK_TO_IMPACT).fillna("MODIFIER")
    bin_df = bin_df.drop(columns=["worst_impact_rank"])

    # Sort and assign sequential bin indices + readable names
    bin_df = bin_df.sort_values(["CHROM", "SYMBOL", "bin_start"]).reset_index(drop=True)
    bin_df["bin_index"] = bin_df.groupby(["SYMBOL", "CHROM"]).cumcount() + 1
    bin_df["bin_name"]  = (
        bin_df["CHROM"] + "_" + bin_df["SYMBOL"]
        + "_bin" + bin_df["bin_index"].astype(str)
    )
    bin_df["POS"] = bin_df["bin_start"].astype(int)

    col_order = [
        "bin_name", "CHROM", "SYMBOL", "bin_index",
        "bin_start", "bin_end", "bin_length_actual",
        "patient_count", "variant_count", "density",
        "worst_impact", "POS",
    ]
    return (
        bin_df[[c for c in col_order if c in bin_df.columns]]
        .sort_values(["CHROM", "bin_start"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stream_aggregate(
    input_dir: Path,
    bin_length: int,
    overlap:    int,
    max_gnomad_af: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Two-pass DuckDB aggregation over all patient Parquet files.

    Pass 1 computes gene-level stats.
    Pass 2 uses the gene boundaries from Pass 1 to compute bin-level stats.

    Returns
    -------
    gene_df    : per-gene   variant frequency DataFrame
    bin_df     : per-bin    variant frequency DataFrame
    n_patients : number of patient files processed
    """
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error("No Parquet files found in: {}", input_dir.resolve())
        sys.exit(1)

    n_patients = len(parquet_files)
    logger.info("Found {} patient Parquet files in {}", n_patients, input_dir.resolve())

    con = _open_db()

    # ---- Pass 1: gene ---------------------------------------------------
    gene_raw = _pass1(con, parquet_files, max_gnomad_af)

    if gene_raw.empty:
        logger.error("No variant data found after filtering.")
        sys.exit(1)

    # Build gene_df: add derived columns (frequency, density, impact label)
    gene_df = gene_raw.copy()
    gene_df["gene_length"]        = (gene_df["gene_end"] - gene_df["gene_start"] + 1).clip(lower=1)
    gene_df["density"]            = gene_df["variant_count"] / (gene_df["gene_length"] * n_patients)
    gene_df["worst_impact"]       = gene_df["worst_impact_rank"].map(_RANK_TO_IMPACT).fillna("MODIFIER")
    gene_df["POS"]                = gene_df["gene_start"].fillna(0).astype(int)
    gene_df = (
        gene_df
        .drop(columns=["worst_impact_rank"])
        .sort_values(["CHROM", "POS"])
        .reset_index(drop=True)
    )
    logger.info("Unique genes        : {:,}", len(gene_df))

    # ---- Pass 2: bins ---------------------------------------------------
    gene_bounds_df = gene_df[["SYMBOL", "CHROM", "gene_start", "gene_end"]]
    bin_df = _pass2(con, parquet_files, gene_bounds_df, n_patients, bin_length, overlap, max_gnomad_af)
    logger.info("Total bins          : {:,}", len(bin_df))

    con.close()
    return gene_df, bin_df, n_patients


# ---------------------------------------------------------------------------
# Chromosome layout helpers
# ---------------------------------------------------------------------------

def build_chrom_offsets(gene_df: pd.DataFrame) -> dict[str, int]:
    """Cumulative x-axis offset per chromosome for Manhattan-style layout."""
    present   = [c for c in CHROM_ORDER if c in gene_df["CHROM"].values]
    chrom_max = gene_df.groupby("CHROM")["gene_end"].max().to_dict()
    gap       = 10_000_000
    offsets: dict[str, int] = {}
    cursor = 0
    for chrom in present:
        offsets[chrom] = cursor
        cursor += chrom_max.get(chrom, 0) + gap
    return offsets


def assign_cumulative_pos(df: pd.DataFrame, offsets: dict[str, int]) -> pd.DataFrame:
    """Vectorised cumulative position assignment (no row-wise apply)."""
    df = df.copy()
    df["cum_pos"] = df["CHROM"].map(offsets).fillna(0).astype(int) + df["POS"]
    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(
    gene_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    n_patients: int,
    top_n: int = 10,
) -> None:
    logger.info("=" * 60)
    logger.info("  VARIANT DISTRIBUTION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Patients in cohort          : {}", n_patients)

    # ------------------------------------------------------------------
    # Gene-level section
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("  GENE LEVEL")
    logger.info("-" * 60)
    logger.info("  Unique genes                : {:,}", len(gene_df))

    if "worst_impact" in gene_df.columns:
        for impact in ["HIGH", "MODERATE", "LOW", "MODIFIER"]:
            n = (gene_df["worst_impact"] == impact).sum()
            logger.info("  {:<10} worst-impact genes : {:,}", impact, n)

    logger.info("  Density statistics (variant_count / (gene_length × n_patients)):")
    logger.info("    Mean   : {:.4e}", gene_df["density"].mean())
    logger.info("    Median : {:.4e}", gene_df["density"].median())
    logger.info("    Max    : {:.4e}", gene_df["density"].max())

    top = gene_df.nlargest(top_n, "density")
    logger.info("  Top {} genes by density:", top_n)
    logger.info("  {:<15} {:<8} {:>9} {:>10} {:>9} {}", "Gene", "CHROM", "Variants", "Density", "Patients", "Worst Impact")
    logger.info("  " + "-" * 65)
    for _, row in top.iterrows():
        gene    = str(row.get("SYMBOL", "—"))[:15]
        chrom   = str(row.get("CHROM", ""))
        n_var   = f"{int(row['variant_count'])}"
        cov     = f"{row['density']:.4e}"
        pts     = f"{int(row['patient_count'])}/{n_patients}"
        impact  = str(row.get("worst_impact", ""))
        logger.info("  {:<15} {:<8} {:>9} {:>10} {:>9} {}", gene, chrom, n_var, cov, pts, impact)

    # ------------------------------------------------------------------
    # Bin-level section
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("  BIN LEVEL")
    logger.info("-" * 60)

    if bin_df is None or bin_df.empty:
        logger.info("  No bin-level data available.")
    else:
        logger.info("  Total bins                  : {:,}", len(bin_df))

        if "worst_impact" in bin_df.columns:
            for impact in ["HIGH", "MODERATE", "LOW", "MODIFIER"]:
                n = (bin_df["worst_impact"] == impact).sum()
                logger.info("  {:<10} worst-impact bins  : {:,}", impact, n)

        logger.info("  Density statistics (variant_count / (bin_length × n_patients)):")
        logger.info("    Mean   : {:.4e}", bin_df["density"].mean())
        logger.info("    Median : {:.4e}", bin_df["density"].median())
        logger.info("    Max    : {:.4e}", bin_df["density"].max())

        top_bins = bin_df.nlargest(top_n, "density")
        logger.info("  Top {} bins by density:", top_n)
        logger.info("  {:<30} {:<8} {:>9} {:>10} {:>9} {}", "Bin", "CHROM", "Variants", "Density", "Patients", "Worst Impact")
        logger.info("  " + "-" * 80)
        for _, row in top_bins.iterrows():
            bin_name = str(row.get("bin_name", "—"))[:30]
            chrom    = str(row.get("CHROM", ""))
            n_var    = f"{int(row['variant_count'])}"
            cov      = f"{row['density']:.4e}"
            pts      = f"{int(row['patient_count'])}/{n_patients}"
            impact   = str(row.get("worst_impact", ""))
            logger.info("  {:<30} {:<8} {:>9} {:>10} {:>9} {}", bin_name, chrom, n_var, cov, pts, impact)

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    project_root   = Path(__file__).resolve().parent.parent.parent
    default_input  = project_root / "output" / "parsed"
    default_output = project_root / "output" / "distribution_analysis"

    p = argparse.ArgumentParser(
        description="GWAS-style variant frequency analysis for a patient cohort.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir",  type=Path, default=default_input,
                   help="Folder containing per-patient Parquet files.")
    p.add_argument("--output-dir", type=Path, default=default_output,
                   help="Directory where output plots and tables will be saved.")
    p.add_argument("--gene-output-name", default="gene_distribution.png",
                   help="Filename for the gene-level plot.")
    p.add_argument("--top",      type=int,   default=15,
                   help="Number of top variants/genes to annotate.")
    p.add_argument("--min-freq", type=float, default=0.0,
                   help="Minimum frequency threshold to display (0–1).")
    p.add_argument("--max-gnomad-af", type=float, default=0.01,
                   help=(
                       "Maximum gnomADg allele frequency (0–1). "
                       "Variants with gnomADg_AF above this value are excluded. "
                       "Variants absent from gnomAD (NULL) are always retained. "
                       "Default 0.01 is appropriate for rare-disease studies such as hypotonia."
                   ))
    p.add_argument("--no-show",  action="store_true",
                   help="Skip the interactive plot window; only save to file.")
    p.add_argument("--bin-length",  type=int, default=1_000_000,
                   help="Bin size in base pairs for bin-level aggregation.")
    p.add_argument("--bin-overlap", type=int, default=0,
                   help="Overlap between consecutive bins in base pairs.")
    p.add_argument("--bin-output-name", default="bin_distribution.csv",
                   help="Filename for the bin-level aggregation CSV.")
    p.add_argument("--bin-plot-name",   default="bin_distribution.png",
                   help="Filename for the bin-level plot.")
    args = p.parse_args()
    if args.bin_overlap >= args.bin_length:
        p.error(f"--bin-overlap ({args.bin_overlap}) must be strictly less than "
                f"--bin-length ({args.bin_length}).")
    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.no_show:
        plt.switch_backend("Agg")

    logger.info(
        "[1/3] Streaming patient Parquet files "
        "(bin_length={:,} bp, overlap={:,} bp, max_gnomad_af={}) …",
        args.bin_length, args.bin_overlap, args.max_gnomad_af,
    )
    gene_df, bin_df, n_patients = stream_aggregate(
        args.input_dir, args.bin_length, args.bin_overlap,
        max_gnomad_af=args.max_gnomad_af,
    )

    logger.info("[2/3] Building genomic layout …")
    offsets = build_chrom_offsets(gene_df)
    gene_df = assign_cumulative_pos(gene_df, offsets)
    if not bin_df.empty:
        bin_df = assign_cumulative_pos(bin_df, offsets)

    print_summary(gene_df, bin_df, n_patients, top_n=args.top)

    logger.info("[3/3] Saving aggregation results …")
    run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    gene_csv = run_dir / "gene_level_analysis.csv"
    bin_csv  = run_dir / args.bin_output_name
    gene_df.to_csv(gene_csv, index=False)
    if not bin_df.empty:
        bin_df.to_csv(bin_csv, index=False)
    logger.info("Gene-level results  : {}", gene_csv.resolve())
    if not bin_df.empty:
        logger.info("Bin-level results   : {}", bin_csv.resolve())

    logger.info("Generating plots …")
    make_gene_figure(
        gene_df, offsets, n_patients,
        top_n=args.top,
        min_density=args.min_freq,
        output_path=run_dir / args.gene_output_name,
    )
    make_bin_figure(
        bin_df, offsets, n_patients,
        bin_length=args.bin_length,
        overlap=args.bin_overlap,
        top_n=args.top,
        min_density=args.min_freq,
        output_path=run_dir / args.bin_plot_name,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
