"""
variant_distribution.py
-----------------------
GWAS-style variant frequency analysis across a cohort of patients.

Streams patient Parquet files one at a time to accumulate:
  - per-locus variant counts    (locus-level)
  - per-gene  variant statistics (gene-level)
  - per-genomic-bin statistics  (bin-level)

Memory model
------------
Pass 1 reads only ~9 columns per file for locus + gene accumulation.
Pass 2 reads only ~7 columns per file for bin accumulation, using gene
boundaries already determined in Pass 1.

The locus accumulator grows with the number of *unique* (CHROM, POS, REF, ALT)
loci across the cohort — not n_patients × rows_per_patient.
Gene stats are O(n_genes × n_patients) — negligible for any realistic cohort.

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

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from plots import (
    CHROM_ORDER,
    IMPACT_RANK,
    make_figure,
    make_gene_figure,
    make_bin_figure,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOCUS_KEY  = ["CHROM", "POS", "REF", "ALT"]
_ANN_COLS   = ["SYMBOL", "Consequence", "IMPACT", "Gene"]
_PASS1_COLS = _LOCUS_KEY + _ANN_COLS + ["GT"]
_PASS2_COLS = ["SYMBOL", "CHROM", "POS", "REF", "ALT", "IMPACT", "GT"]

_REF_CALLS      = frozenset({"0/0", "./.", "0|0", ".|."})
_RANK_TO_IMPACT = {v: k for k, v in IMPACT_RANK.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read(path: Path, wanted: list[str]) -> pd.DataFrame:
    """Read only *wanted* columns from a Parquet file, skipping absent ones."""
    available = set(pq.read_schema(path).names)
    cols = [c for c in wanted if c in available]
    return pd.read_parquet(path, columns=cols)


def _filter_gt(df: pd.DataFrame) -> pd.DataFrame:
    """Drop homozygous-reference and missing genotype rows."""
    if "GT" not in df.columns:
        return df
    return df[~df["GT"].isin(_REF_CALLS)]


def _prep_pos(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce POS to int, drop non-numeric rows in-place on a copy."""
    df = df.copy()
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    return df.dropna(subset=["POS"]).astype({"POS": int})


# ---------------------------------------------------------------------------
# Pass 1 — locus + gene accumulation
# ---------------------------------------------------------------------------

def _pass1(
    parquet_files: list[Path],
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Stream all patient Parquet files once.

    Returns
    -------
    locus_count   : pd.Series  keyed by (CHROM, POS, REF, ALT) MultiIndex
                    value = number of patients carrying that locus
    locus_ann     : pd.DataFrame with the same MultiIndex, annotation columns
                    populated from the first patient that carries each locus
    all_gene_stats: pd.DataFrame with one row per (patient × gene); the caller
                    reduces this with a single groupby
    """
    locus_count: pd.Series | None    = None
    locus_ann:   pd.DataFrame | None = None
    gene_agg:    pd.DataFrame | None = None

    for f in tqdm(parquet_files, desc="  Pass 1/2 – locus + gene", unit="pt"):
        df = _read(f, _PASS1_COLS)
        df = _filter_gt(df)
        if df.empty:
            continue

        avail_locus = [c for c in _LOCUS_KEY if c in df.columns]
        avail_ann   = [c for c in _ANN_COLS  if c in df.columns]

        # ---- Locus accumulation (vectorised Series.add) ------------------
        patient_loci = df.drop_duplicates(subset=avail_locus)
        new_idx      = pd.MultiIndex.from_frame(patient_loci[avail_locus])
        new_counts   = pd.Series(1, index=new_idx, dtype="int32")

        if locus_count is None:
            locus_count = new_counts.rename("patient_count")
            locus_ann   = patient_loci[avail_locus + avail_ann].set_index(avail_locus)
        else:
            locus_count = locus_count.add(new_counts, fill_value=0).astype("int32")
            truly_new   = new_counts.index.difference(locus_ann.index)
            if len(truly_new):
                new_ann   = patient_loci[avail_locus + avail_ann].set_index(avail_locus)
                locus_ann = pd.concat([locus_ann, new_ann.loc[new_ann.index.isin(truly_new)]])

        # ---- Gene stats for this patient ---------------------------------
        if "SYMBOL" not in df.columns:
            continue

        g = df.dropna(subset=["SYMBOL"])
        g = g[g["SYMBOL"].astype(str).str.strip() != ""]
        g = _prep_pos(g)
        if g.empty:
            continue

        dedup_cols  = ["SYMBOL", "CHROM", "POS"] + [c for c in ["REF", "ALT"] if c in g.columns]
        locus_dedup = g.drop_duplicates(subset=dedup_cols)

        gene_stats = (
            locus_dedup
            .groupby(["SYMBOL", "CHROM"], sort=False)
            .agg(variant_count=("POS", "count"),
                 gene_start    =("POS", "min"),
                 gene_end      =("POS", "max"))
            .reset_index()
        )
        gene_stats["patient_count"] = 1  # one patient per frame

        if "IMPACT" in g.columns:
            # Vectorised: map string → rank, then groupby max
            g = g.copy()
            g["_rank"] = g["IMPACT"].map(IMPACT_RANK).fillna(0).astype("int8")
            impact_df  = (
                g.groupby(["SYMBOL", "CHROM"], sort=False)["_rank"]
                .max()
                .reset_index(name="worst_impact_rank")
            )
            gene_stats = gene_stats.merge(impact_df, on=["SYMBOL", "CHROM"], how="left")
            gene_stats["worst_impact_rank"] = gene_stats["worst_impact_rank"].fillna(0).astype("int8")
        else:
            gene_stats["worst_impact_rank"] = 0

        gs = gene_stats.set_index(["SYMBOL", "CHROM"])
        if gene_agg is None:
            gene_agg = gs
        else:
            gene_agg = (
                pd.concat([gene_agg, gs])
                .groupby(level=["SYMBOL", "CHROM"], sort=False)
                .agg({"patient_count":     "sum",
                      "variant_count":     "sum",
                      "gene_start":        "min",
                      "gene_end":          "max",
                      "worst_impact_rank": "max"})
            )

    gene_agg_df = gene_agg.reset_index() if gene_agg is not None else pd.DataFrame()
    return locus_count, locus_ann, gene_agg_df


# ---------------------------------------------------------------------------
# Pass 2 — bin accumulation
# ---------------------------------------------------------------------------

def _pass2(
    parquet_files:  list[Path],
    gene_bounds_df: pd.DataFrame,
    n_patients:     int,
    bin_length:     int,
    overlap:        int,
) -> pd.DataFrame:
    """
    Stream all patient Parquet files a second time, computing per-bin stats.

    Gene boundaries from Pass 1 enable fully vectorised bin assignment for
    the no-overlap case and a fast list-comprehension explode for overlap > 0.
    Only ~7 columns are read from each Parquet file.
    """
    step    = bin_length - overlap
    bin_agg: pd.DataFrame | None = None

    for f in tqdm(parquet_files, desc="  Pass 2/2 – bins      ", unit="pt"):
        df = _read(f, _PASS2_COLS)
        df = _filter_gt(df)
        if df.empty or "SYMBOL" not in df.columns:
            continue

        g = df.dropna(subset=["SYMBOL"])
        g = g[g["SYMBOL"].astype(str).str.strip() != ""]
        g = _prep_pos(g)
        if g.empty:
            continue

        # Deduplicate to unique loci for this patient
        dedup_cols = ["SYMBOL", "CHROM", "POS"] + [c for c in ["REF", "ALT"] if c in g.columns]
        g = g.drop_duplicates(subset=dedup_cols).copy()

        # Vectorised IMPACT → rank
        g["_rank"] = (
            g["IMPACT"].map(IMPACT_RANK).fillna(0).astype("int8")
            if "IMPACT" in g.columns else 0
        )

        # Join gene boundaries; discard loci outside observed gene span
        g = g.merge(gene_bounds_df[["SYMBOL", "CHROM", "gene_start", "gene_end"]],
                    on=["SYMBOL", "CHROM"], how="inner")
        g = g[(g["POS"] >= g["gene_start"]) & (g["POS"] <= g["gene_end"])]
        if g.empty:
            continue

        if overlap == 0:
            # ---- Fast path: one bin per locus, fully vectorised ----------
            g["bin_id"]    = (g["POS"] - g["gene_start"]) // bin_length
            g["bin_start"] = g["gene_start"] + g["bin_id"] * bin_length
            g["bin_end"]   = (g["bin_start"] + bin_length - 1).clip(upper=g["gene_end"])

            bin_stats = (
                g.groupby(["SYMBOL", "CHROM", "bin_id", "bin_start", "bin_end"], sort=False)
                .agg(variant_count    =("POS",   "count"),
                     worst_impact_rank=("_rank", "max"))
                .reset_index()
            )

        else:
            # ---- Overlap path: each locus may belong to multiple bins ----
            # b_max = last bin index containing POS
            # b_min = first bin index containing POS (ceiling division via
            #         the identity  ceil(a/b) = -(−a // b)  for integer b > 0)
            g["b_max"] = (g["POS"] - g["gene_start"]) // step
            g["b_min"] = (
                -((g["gene_start"] + bin_length - 1 - g["POS"]) // step)
            ).clip(lower=0)

            # Explode bin index range per row (list-comprehension, no apply)
            g["bin_ids"] = [
                list(range(lo, hi + 1))
                for lo, hi in zip(g["b_min"], g["b_max"])
            ]
            g = g.explode("bin_ids").rename(columns={"bin_ids": "bin_id"})
            g["bin_id"]    = g["bin_id"].astype(int)
            g["bin_start"] = g["gene_start"] + g["bin_id"] * step
            g["bin_end"]   = (g["bin_start"] + bin_length - 1).clip(upper=g["gene_end"])
            g = g[g["bin_start"] <= g["gene_end"]]  # drop out-of-gene bins

            bin_stats = (
                g.groupby(["SYMBOL", "CHROM", "bin_id", "bin_start", "bin_end"], sort=False)
                .agg(variant_count    =("POS",   "count"),
                     worst_impact_rank=("_rank", "max"))
                .reset_index()
            )

        bin_stats["patient_count"] = 1
        bs = bin_stats.set_index(["SYMBOL", "CHROM", "bin_id", "bin_start", "bin_end"])
        if bin_agg is None:
            bin_agg = bs
        else:
            bin_agg = (
                pd.concat([bin_agg, bs])
                .groupby(level=["SYMBOL", "CHROM", "bin_id", "bin_start", "bin_end"], sort=False)
                .agg({"patient_count":     "sum",
                      "variant_count":     "sum",
                      "worst_impact_rank": "max"})
            )

    if bin_agg is None:
        return pd.DataFrame()

    bin_df = bin_agg.reset_index()

    bin_df["frequency"]          = bin_df["patient_count"] / n_patients
    bin_df["bin_length_actual"]  = bin_df["bin_end"] - bin_df["bin_start"] + 1
    bin_df["normalized_density"] = bin_df["variant_count"] / (bin_length * n_patients)
    bin_df["worst_impact"]       = bin_df["worst_impact_rank"].map(_RANK_TO_IMPACT).fillna("MODIFIER")
    bin_df = bin_df.drop(columns=["worst_impact_rank"])

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
        "patient_count", "frequency", "variant_count",
        "normalized_density", "worst_impact", "POS",
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Two-pass streaming aggregation over all patient Parquet files.

    Pass 1 reads ~9 columns; Pass 2 reads ~7 columns.
    Peak memory is dominated by the locus accumulator, which is proportional
    to the number of *unique* loci across the cohort, not to
    n_patients × rows_per_patient.

    Returns
    -------
    freq_df    : per-locus  variant frequency DataFrame
    gene_df    : per-gene   variant frequency DataFrame
    bin_df     : per-bin    variant frequency DataFrame
    n_patients : number of patient files processed
    """
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        sys.exit(f"[ERROR] No Parquet files found in: {input_dir.resolve()}")

    n_patients = len(parquet_files)
    print(f"  Found {n_patients} patient Parquet files in {input_dir.resolve()}")

    # ---- Pass 1 ----------------------------------------------------------
    locus_count, locus_ann, all_gene_stats = _pass1(parquet_files)

    if locus_count is None:
        sys.exit("[ERROR] No variant data found after filtering.")

    # ---- Variant-level result --------------------------------------------
    avail_locus = list(locus_count.index.names)
    avail_ann   = [c for c in _ANN_COLS if locus_ann is not None and c in locus_ann.columns]

    freq_df = locus_count.rename("patient_count").reset_index()
    if avail_ann:
        freq_df = freq_df.merge(locus_ann[avail_ann].reset_index(),
                                on=avail_locus, how="left")
    freq_df["frequency"] = freq_df["patient_count"] / n_patients
    freq_df = _prep_pos(freq_df).sort_values(["CHROM", "POS"]).reset_index(drop=True)

    print(f"  Unique variant loci : {len(freq_df):,}")

    # ---- Gene-level result -----------------------------------------------
    gene_df = all_gene_stats.copy()
    gene_df["frequency"]          = gene_df["patient_count"] / n_patients
    gene_df["gene_length"]        = (gene_df["gene_end"] - gene_df["gene_start"] + 1).clip(lower=1)
    gene_df["normalized_density"] = gene_df["variant_count"] / (gene_df["gene_length"] * n_patients)
    gene_df["worst_impact"]       = gene_df["worst_impact_rank"].map(_RANK_TO_IMPACT).fillna("MODIFIER")
    gene_df["POS"]                = gene_df["gene_start"].fillna(0).astype(int)
    gene_df = (
        gene_df
        .drop(columns=["worst_impact_rank"])
        .sort_values(["CHROM", "POS"])
        .reset_index(drop=True)
    )
    print(f"  Unique genes        : {len(gene_df):,}")

    # ---- Pass 2: bins ----------------------------------------------------
    gene_bounds_df = gene_df[["SYMBOL", "CHROM", "gene_start", "gene_end"]]
    bin_df = _pass2(parquet_files, gene_bounds_df, n_patients, bin_length, overlap)
    print(f"  Total bins          : {len(bin_df):,}")

    return freq_df, gene_df, bin_df, n_patients


# ---------------------------------------------------------------------------
# Chromosome layout helpers
# ---------------------------------------------------------------------------

def build_chrom_offsets(freq_df: pd.DataFrame) -> dict[str, int]:
    """Cumulative x-axis offset per chromosome for Manhattan-style layout."""
    present   = [c for c in CHROM_ORDER if c in freq_df["CHROM"].values]
    chrom_max = freq_df.groupby("CHROM")["POS"].max().to_dict()
    gap       = 10_000_000
    offsets: dict[str, int] = {}
    cursor = 0
    for chrom in present:
        offsets[chrom] = cursor
        cursor += chrom_max.get(chrom, 0) + gap
    return offsets


def assign_cumulative_pos(freq_df: pd.DataFrame, offsets: dict[str, int]) -> pd.DataFrame:
    """Vectorised cumulative position assignment (no row-wise apply)."""
    freq_df = freq_df.copy()
    freq_df["cum_pos"] = freq_df["CHROM"].map(offsets).fillna(0).astype(int) + freq_df["POS"]
    return freq_df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(freq_df: pd.DataFrame, n_patients: int, top_n: int = 10) -> None:
    print("\n" + "=" * 60)
    print("  VARIANT DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"  Patients in cohort          : {n_patients}")
    print(f"  Unique variant loci         : {len(freq_df):,}")

    if "IMPACT" in freq_df.columns:
        for impact in ["HIGH", "MODERATE", "LOW", "MODIFIER"]:
            n = (freq_df["IMPACT"] == impact).sum()
            print(f"  {impact:<10} impact loci     : {n:,}")

    print(f"\n  Frequency statistics:")
    print(f"    Mean   : {freq_df['frequency'].mean():.3f}")
    print(f"    Median : {freq_df['frequency'].median():.3f}")
    print(f"    Max    : {freq_df['frequency'].max():.3f}")

    top = freq_df.nlargest(top_n, "frequency")
    print(f"\n  Top {top_n} most frequent variants:")
    print(f"  {'Gene':<12} {'CHROM':<8} {'POS':<12} {'REF>ALT':<14} "
          f"{'Freq':>7} {'Patients':>9} {'Impact'}")
    print("  " + "-" * 72)
    for _, row in top.iterrows():
        gene   = str(row.get("SYMBOL", ""))[:12] or "—"
        chrom  = str(row.get("CHROM",  ""))
        pos    = str(int(row["POS"]))
        allele = f"{row.get('REF', '?')}>{row.get('ALT', '?')}"[:14]
        freq   = f"{row['frequency']:.1%}"
        pts    = f"{int(row['patient_count'])}/{n_patients}"
        impact = str(row.get("IMPACT", ""))
        print(f"  {gene:<12} {chrom:<8} {pos:<12} {allele:<14} "
              f"{freq:>7} {pts:>9} {impact}")
    print("=" * 60)


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
    p.add_argument("--output-name",      default="variant_distribution.png",
                   help="Filename for the variant-level plot.")
    p.add_argument("--gene-output-name", default="gene_distribution.png",
                   help="Filename for the gene-level plot.")
    p.add_argument("--top",      type=int,   default=15,
                   help="Number of top variants/genes to annotate.")
    p.add_argument("--min-freq", type=float, default=0.0,
                   help="Minimum frequency threshold to display (0–1).")
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

    print(f"\n[1/4] Streaming patient Parquet files "
          f"(bin_length={args.bin_length:,} bp, overlap={args.bin_overlap:,} bp) …")
    freq_df, gene_df, bin_df, n_patients = stream_aggregate(
        args.input_dir, args.bin_length, args.bin_overlap,
    )

    print("\n[2/4] Building genomic layout …")
    offsets = build_chrom_offsets(freq_df)
    freq_df = assign_cumulative_pos(freq_df, offsets)
    gene_df = assign_cumulative_pos(gene_df, offsets)
    if not bin_df.empty:
        bin_df = assign_cumulative_pos(bin_df, offsets)

    print_summary(freq_df, n_patients, top_n=args.top)

    print("\n[3/4] Saving aggregation results …")
    run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    gene_csv = run_dir / "gene_level_analysis.csv"
    bin_csv  = run_dir / args.bin_output_name
    gene_df.to_csv(gene_csv, index=False)
    if not bin_df.empty:
        bin_df.to_csv(bin_csv, index=False)
    print(f"  Gene-level results  : {gene_csv.resolve()}")
    if not bin_df.empty:
        print(f"  Bin-level results   : {bin_csv.resolve()}")

    print("\n[4/4] Generating plots …")
    make_gene_figure(
        gene_df, offsets, n_patients,
        top_n=args.top,
        min_freq=args.min_freq,
        output_path=run_dir / args.gene_output_name,
    )
    make_bin_figure(
        bin_df, offsets, n_patients,
        bin_length=args.bin_length,
        overlap=args.bin_overlap,
        top_n=args.top,
        min_freq=args.min_freq,
        output_path=run_dir / args.bin_plot_name,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
