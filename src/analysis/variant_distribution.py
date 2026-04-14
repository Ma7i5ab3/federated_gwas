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
Each Parquet file is read in batches of _BATCH_SIZE rows so that only one
chunk is in memory at a time.  Per-patient deduplication is maintained via
lightweight Python sets of tuple keys — these hold only the *variant* rows
(non-ref calls), which are a small fraction of the raw VCF rows.

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
from collections.abc import Iterator
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

# Rows read from a Parquet file per iteration — keeps peak RSS manageable.
_BATCH_SIZE = 100_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_batches(path: Path, wanted: list[str]) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks of *wanted* columns from a Parquet file."""
    available = set(pq.read_schema(path).names)
    cols = [c for c in wanted if c in available]
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=_BATCH_SIZE, columns=cols):
        yield batch.to_pandas()


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
    Stream all patient Parquet files once, reading _BATCH_SIZE rows at a time.

    Returns
    -------
    locus_count   : pd.Series  keyed by (CHROM, POS, REF, ALT) MultiIndex
                    value = number of patients carrying that locus
    locus_ann     : pd.DataFrame with the same MultiIndex, annotation columns
                    populated from the first patient that carries each locus
    gene_agg_df   : pd.DataFrame with one row per unique (SYMBOL, CHROM)
    """
    locus_count: pd.Series | None    = None
    locus_ann:   pd.DataFrame | None = None
    gene_agg:    pd.DataFrame | None = None

    for f in tqdm(parquet_files, desc="  Pass 1/2 – locus + gene", unit="pt"):
        # Per-patient dedup state (sets of tuple keys for seen variant loci)
        seen_loci:      set[tuple] = set()
        seen_gene_loci: set[tuple] = set()
        # Per-patient gene accumulator: (SYMBOL, CHROM) -> stats dict
        patient_gene: dict[tuple, dict] = {}

        avail_locus: list[str] = []
        avail_ann:   list[str] = []

        for df in _iter_batches(f, _PASS1_COLS):
            df = _filter_gt(df)
            if df.empty:
                continue

            if not avail_locus:
                avail_locus = [c for c in _LOCUS_KEY if c in df.columns]
                avail_ann   = [c for c in _ANN_COLS  if c in df.columns]

            # ---- Locus accumulation: deduplicate within this patient ------
            loci_tuples = list(zip(*(df[c] for c in avail_locus)))
            new_mask    = [t not in seen_loci for t in loci_tuples]
            seen_loci.update(t for t, m in zip(loci_tuples, new_mask) if m)

            new_loci_df = df.loc[[i for i, m in zip(df.index, new_mask) if m]]
            if not new_loci_df.empty:
                new_idx    = pd.MultiIndex.from_frame(new_loci_df[avail_locus])
                new_counts = pd.Series(1, index=new_idx, dtype="int32")
                if locus_count is None:
                    locus_count = new_counts.rename("patient_count")
                    locus_ann   = new_loci_df[avail_locus + avail_ann].set_index(avail_locus)
                else:
                    locus_count = locus_count.add(new_counts, fill_value=0).astype("int32")
                    truly_new   = new_counts.index.difference(locus_ann.index)
                    if len(truly_new):
                        new_ann   = new_loci_df[avail_locus + avail_ann].set_index(avail_locus)
                        locus_ann = pd.concat(
                            [locus_ann, new_ann.loc[new_ann.index.isin(truly_new)]]
                        )

            # ---- Gene stats for this patient / batch ---------------------
            if "SYMBOL" not in df.columns:
                continue

            g = df.dropna(subset=["SYMBOL"])
            g = g[g["SYMBOL"].astype(str).str.strip() != ""]
            g = _prep_pos(g)
            if g.empty:
                continue

            if "IMPACT" in g.columns:
                g = g.copy()
                g["_rank"] = g["IMPACT"].map(IMPACT_RANK).fillna(0).astype("int8")
            else:
                g = g.assign(_rank=0)

            dedup_cols     = ["SYMBOL", "CHROM", "POS"] + [c for c in ["REF", "ALT"] if c in g.columns]
            gene_loci_keys = list(zip(*(g[c] for c in dedup_cols)))
            new_gene_mask  = [t not in seen_gene_loci for t in gene_loci_keys]
            seen_gene_loci.update(t for t, m in zip(gene_loci_keys, new_gene_mask) if m)

            ng = g.loc[[i for i, m in zip(g.index, new_gene_mask) if m]]
            if ng.empty:
                continue

            for (symbol, chrom), grp in ng.groupby(["SYMBOL", "CHROM"], sort=False):
                key = (symbol, chrom)
                vc  = len(grp)
                gs_val = int(grp["POS"].min())
                ge_val = int(grp["POS"].max())
                ir     = int(grp["_rank"].max())
                if key not in patient_gene:
                    patient_gene[key] = {
                        "variant_count":     vc,
                        "gene_start":        gs_val,
                        "gene_end":          ge_val,
                        "worst_impact_rank": ir,
                    }
                else:
                    s = patient_gene[key]
                    s["variant_count"]     += vc
                    s["gene_start"]         = min(s["gene_start"], gs_val)
                    s["gene_end"]           = max(s["gene_end"],   ge_val)
                    s["worst_impact_rank"]  = max(s["worst_impact_rank"], ir)

        # Merge this patient's gene stats into the global accumulator
        if patient_gene:
            records = [
                {"SYMBOL": sym, "CHROM": chrom, "patient_count": 1, **stats}
                for (sym, chrom), stats in patient_gene.items()
            ]
            gs_df = pd.DataFrame(records).set_index(["SYMBOL", "CHROM"])
            if gene_agg is None:
                gene_agg = gs_df
            else:
                gene_agg = (
                    pd.concat([gene_agg, gs_df])
                    .groupby(level=["SYMBOL", "CHROM"], sort=False)
                    .agg({
                        "patient_count":     "sum",
                        "variant_count":     "sum",
                        "gene_start":        "min",
                        "gene_end":          "max",
                        "worst_impact_rank": "max",
                    })
                )

    if gene_agg is not None:
        gene_agg_df = gene_agg.reset_index()
    else:
        gene_agg_df = pd.DataFrame(columns=[
            "SYMBOL", "CHROM", "patient_count", "variant_count",
            "gene_start", "gene_end", "worst_impact_rank",
        ])
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
    Stream all patient Parquet files a second time, reading _BATCH_SIZE rows
    at a time and computing per-bin stats.

    Gene boundaries from Pass 1 enable fully vectorised bin assignment for
    the no-overlap case and a fast list-comprehension explode for overlap > 0.
    Only ~7 columns are read from each Parquet file.
    """
    step    = bin_length - overlap
    bin_agg: pd.DataFrame | None = None

    _BIN_IDX = ["SYMBOL", "CHROM", "bin_id", "bin_start", "bin_end"]

    for f in tqdm(parquet_files, desc="  Pass 2/2 – bins      ", unit="pt"):
        # Per-patient dedup state
        seen_loci: set[tuple] = set()
        # Per-patient bin accumulator (no patient_count yet)
        patient_bin: pd.DataFrame | None = None

        for df in _iter_batches(f, _PASS2_COLS):
            df = _filter_gt(df)
            if df.empty or "SYMBOL" not in df.columns:
                continue

            g = df.dropna(subset=["SYMBOL"])
            g = g[g["SYMBOL"].astype(str).str.strip() != ""]
            g = _prep_pos(g)
            if g.empty:
                continue

            # Deduplicate to unique loci for this patient (across batches)
            dedup_cols     = ["SYMBOL", "CHROM", "POS"] + [c for c in ["REF", "ALT"] if c in g.columns]
            loci_tuples    = list(zip(*(g[c] for c in dedup_cols)))
            new_mask       = [t not in seen_loci for t in loci_tuples]
            seen_loci.update(t for t, m in zip(loci_tuples, new_mask) if m)

            g = g.loc[[i for i, m in zip(g.index, new_mask) if m]].copy()
            if g.empty:
                continue

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
                # ---- Fast path: one bin per locus, fully vectorised ------
                g["bin_id"]    = (g["POS"] - g["gene_start"]) // bin_length
                g["bin_start"] = g["gene_start"] + g["bin_id"] * bin_length
                g["bin_end"]   = (g["bin_start"] + bin_length - 1).clip(upper=g["gene_end"])

                bin_stats = (
                    g.groupby(_BIN_IDX, sort=False)
                    .agg(variant_count    =("POS",   "count"),
                         worst_impact_rank=("_rank", "max"))
                    .reset_index()
                )

            else:
                # ---- Overlap path: each locus may belong to multiple bins
                g["b_max"] = (g["POS"] - g["gene_start"]) // step
                g["b_min"] = (
                    -((g["gene_start"] + bin_length - 1 - g["POS"]) // step)
                ).clip(lower=0)

                g["bin_ids"] = [
                    list(range(lo, hi + 1))
                    for lo, hi in zip(g["b_min"], g["b_max"])
                ]
                g = g.explode("bin_ids").rename(columns={"bin_ids": "bin_id"})
                g["bin_id"]    = g["bin_id"].astype(int)
                g["bin_start"] = g["gene_start"] + g["bin_id"] * step
                g["bin_end"]   = (g["bin_start"] + bin_length - 1).clip(upper=g["gene_end"])
                g = g[g["bin_start"] <= g["gene_end"]]

                bin_stats = (
                    g.groupby(_BIN_IDX, sort=False)
                    .agg(variant_count    =("POS",   "count"),
                         worst_impact_rank=("_rank", "max"))
                    .reset_index()
                )

            bs = bin_stats.set_index(_BIN_IDX)
            if patient_bin is None:
                patient_bin = bs
            else:
                patient_bin = (
                    pd.concat([patient_bin, bs])
                    .groupby(level=_BIN_IDX, sort=False)
                    .agg({"variant_count": "sum", "worst_impact_rank": "max"})
                )

        # Merge this patient's bins into the global accumulator
        if patient_bin is not None:
            patient_bin = patient_bin.copy()
            patient_bin["patient_count"] = 1
            if bin_agg is None:
                bin_agg = patient_bin
            else:
                bin_agg = (
                    pd.concat([bin_agg, patient_bin])
                    .groupby(level=_BIN_IDX, sort=False)
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
