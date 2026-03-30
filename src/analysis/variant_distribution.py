"""
variant_distribution.py
-----------------------
GWAS-style variant frequency analysis across a cohort of patients.

Loads all patient CSV files from a selectable folder, builds a single in-memory
DataFrame, computes per-position mutation frequencies across the cohort, and
produces a publication-quality Manhattan-style plot showing which genomic
positions harbour the most recurrent variants.

Usage
-----
    python analysis/variant_distribution.py
    python analysis/variant_distribution.py --input-dir output/synthetic_patients
    python analysis/variant_distribution.py --input-dir /path/to/csvs --top 20
    python analysis/variant_distribution.py --min-freq 0.05 --output-dir output/distribution_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from plots import (
    CHROM_ORDER,
    IMPACT_RANK,
    make_figure,
    make_gene_figure,
    make_bin_figure,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_patient_csvs(input_dir: Path) -> pd.DataFrame:
    """
    Iterate every *.csv file in *input_dir* and concatenate them into
    a single in-memory DataFrame.  The CSV files are expected to follow
    the schema produced by parse_vcf.py (Patient_ID, CHROM, POS, …).
    """
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        sys.exit(f"[ERROR] No CSV files found in: {input_dir.resolve()}")

    print(f"  Found {len(csv_files)} patient CSV files in {input_dir.resolve()}")

    frames: list[pd.DataFrame] = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as exc:
            print(f"  [WARN] Could not read {f.name}: {exc}")

    if not frames:
        sys.exit("[ERROR] All CSV files failed to load.")

    df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows loaded   : {len(df):,}")
    print(f"  Unique patients     : {df['Patient_ID'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# Mutation filtering
# ---------------------------------------------------------------------------

def filter_mutations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows that represent a true mutation, i.e. GT is NOT
    homozygous reference (0/0) or missing (./.).
    """
    if "GT" not in df.columns:
        print("  [WARN] GT column not found – using all rows as mutations.")
        return df

    ref_calls = {"0/0", "./.", "0|0", ".|."}
    mask = ~df["GT"].isin(ref_calls)
    mutated = df[mask].copy()
    print(f"  Mutated rows (GT ≠ ref): {len(mutated):,}  "
          f"({100 * len(mutated) / max(len(df), 1):.1f}% of total)")
    return mutated


# ---------------------------------------------------------------------------
# Frequency computation
# ---------------------------------------------------------------------------

def compute_variant_frequency(df: pd.DataFrame, n_patients: int) -> pd.DataFrame:
    """
    For each unique (CHROM, POS, REF, ALT) locus, compute:
      - patient_count   : number of distinct patients carrying the variant
      - frequency       : patient_count / n_patients
      - SYMBOL / Consequence / IMPACT from the most common annotation at that locus
    """
    # Deduplicate per patient (a patient may have multiple transcripts per locus)
    locus_cols = ["Patient_ID", "CHROM", "POS", "REF", "ALT"]
    available = [c for c in locus_cols if c in df.columns]
    dedup = df.drop_duplicates(subset=available)

    # Aggregate
    agg_cols = {c: "first" for c in ["SYMBOL", "Consequence", "IMPACT", "Gene"] if c in df.columns}
    agg_cols["Patient_ID"] = "nunique"

    freq_df = (
        dedup.groupby(["CHROM", "POS", "REF", "ALT"], dropna=False)
             .agg({"Patient_ID": "nunique",
                   **{c: "first" for c in ["SYMBOL", "Consequence", "IMPACT", "Gene"]
                      if c in dedup.columns}})
             .rename(columns={"Patient_ID": "patient_count"})
             .reset_index()
    )

    freq_df["frequency"] = freq_df["patient_count"] / n_patients
    freq_df["POS"] = pd.to_numeric(freq_df["POS"], errors="coerce")
    freq_df = freq_df.dropna(subset=["POS"])
    freq_df["POS"] = freq_df["POS"].astype(int)

    return freq_df.sort_values(["CHROM", "POS"]).reset_index(drop=True)


def compute_gene_frequency(mut_df: pd.DataFrame, n_patients: int) -> pd.DataFrame:
    """
    Aggregate mutations to the gene level.

    For each (SYMBOL, CHROM) pair compute:
      - patient_count      : distinct patients with ≥ 1 variant anywhere in that gene
      - frequency          : patient_count / n_patients  (used as allele-frequency proxy)
      - variant_count      : total (patient × locus) observations across the cohort
      - gene_start         : minimum POS of any variant locus in the gene (x-axis order)
      - gene_length        : max(POS) - min(POS) + 1  (observed span, proxy for gene size)
      - normalized_density : variant_count / (gene_length * n_patients)
                             → variants per base per patient (removes length & cohort bias)
      - worst_impact       : highest VEP IMPACT seen across all variants in the gene
    """
    if "SYMBOL" not in mut_df.columns:
        return pd.DataFrame()

    df = mut_df.dropna(subset=["SYMBOL"]).copy()
    df = df[df["SYMBOL"].astype(str).str.strip() != ""]
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    df = df.dropna(subset=["POS"])
    df["POS"] = df["POS"].astype(int)

    # --- gene start + gene length from observed variant span --------------
    locus_dedup = df.drop_duplicates(subset=["SYMBOL", "CHROM", "POS", "REF", "ALT"])
    gene_span = (
        locus_dedup.groupby(["SYMBOL", "CHROM"])["POS"]
        .agg(gene_start="min", gene_end="max")
        .reset_index()
    )
    gene_span["gene_length"] = (gene_span["gene_end"] - gene_span["gene_start"] + 1).clip(lower=1)

    # --- variant_count: total (patient, locus) pairs across the cohort ----
    patient_locus = df.drop_duplicates(
        subset=["Patient_ID", "SYMBOL", "CHROM", "POS", "REF", "ALT"]
    )
    variant_count = (
        patient_locus.groupby(["SYMBOL", "CHROM"])
        .size()
        .reset_index(name="variant_count")
    )

    # --- worst IMPACT per gene --------------------------------------------
    def _worst(series):
        ranked = [(IMPACT_RANK.get(str(v), 0), str(v)) for v in series.dropna()]
        return max(ranked)[1] if ranked else "MODIFIER"

    if "IMPACT" in df.columns:
        worst_impact = (
            df.groupby(["SYMBOL", "CHROM"])["IMPACT"]
            .apply(_worst)
            .reset_index(name="worst_impact")
        )
    else:
        worst_impact = None

    # --- patient count per gene (one patient counted once per gene) -------
    patient_dedup = df.drop_duplicates(subset=["Patient_ID", "SYMBOL", "CHROM"])
    patient_count = (
        patient_dedup.groupby(["SYMBOL", "CHROM"])["Patient_ID"]
        .nunique()
        .reset_index(name="patient_count")
    )
    patient_count["frequency"] = patient_count["patient_count"] / n_patients

    # --- merge ------------------------------------------------------------
    gene_df = patient_count.merge(gene_span,    on=["SYMBOL", "CHROM"], how="left")
    gene_df = gene_df.merge(variant_count,       on=["SYMBOL", "CHROM"], how="left")
    if worst_impact is not None:
        gene_df = gene_df.merge(worst_impact,    on=["SYMBOL", "CHROM"], how="left")
    else:
        gene_df["worst_impact"] = "MODIFIER"

    # --- normalized density: variants per base per patient ----------------
    gene_df["normalized_density"] = (
        gene_df["variant_count"] / (gene_df["gene_length"] * n_patients)
    )

    # POS is the gene_start: drives both chromosomal ordering and x-axis placement
    gene_df["POS"] = gene_df["gene_start"].fillna(0).astype(int)
    return gene_df.sort_values(["CHROM", "POS"]).reset_index(drop=True)


def compute_bin_frequency(
    mut_df: pd.DataFrame,
    n_patients: int,
    bin_length: int,
    overlap: int = 0,
) -> pd.DataFrame:
    """
    Aggregate mutations to fixed-length genomic bins within each gene.

    Bins are constructed per (SYMBOL, CHROM) using the observed variant span
    [min(POS), max(POS)].  Bins never cross gene boundaries.

    Parameters
    ----------
    bin_length : int
        Length of each bin in base pairs.
    overlap : int
        Number of bases shared between consecutive bins (0 = non-overlapping).
        Must be strictly less than bin_length.

    For each bin the following metrics are computed:
      - patient_count      : distinct patients with ≥ 1 variant inside the bin
      - frequency          : patient_count / n_patients
      - variant_count      : total (patient × locus) observations inside the bin
      - bin_length_actual  : effective length (last bin of a gene may be shorter)
      - normalized_density : variant_count / (bin_length_actual * n_patients)
      - worst_impact       : highest VEP IMPACT seen inside the bin

    Bin naming convention: {CHROM}_{SYMBOL}_bin{N}  (N is 1-based, ordered by
    genomic position within the gene).
    """
    if "SYMBOL" not in mut_df.columns:
        return pd.DataFrame()

    step = bin_length - overlap

    df = mut_df.dropna(subset=["SYMBOL"]).copy()
    df = df[df["SYMBOL"].astype(str).str.strip() != ""]
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    df = df.dropna(subset=["POS"])
    df["POS"] = df["POS"].astype(int)

    dedup_cols = ["Patient_ID", "SYMBOL", "CHROM", "POS"]
    if "REF" in df.columns and "ALT" in df.columns:
        dedup_cols += ["REF", "ALT"]
    # One row per (patient, locus) — basis for both variant_count and patient_count
    patient_locus = df.drop_duplicates(subset=dedup_cols)

    # ------------------------------------------------------------------
    # Fast path: overlap=0 — each row belongs to exactly one bin,
    # so bin assignment is a single vectorised integer division followed
    # by one groupby.  No Python loop over genes or bins.
    # ------------------------------------------------------------------
    if overlap == 0:
        # Attach gene_start / gene_end to every row
        gene_bounds = (
            patient_locus.groupby(["SYMBOL", "CHROM"])["POS"]
            .agg(gene_start="min", gene_end="max")
            .reset_index()
        )
        pl = patient_locus.merge(gene_bounds, on=["SYMBOL", "CHROM"], how="left")

        # Bin index (0-based) and exact boundaries — fully vectorised
        pl["bin_id"]          = (pl["POS"] - pl["gene_start"]) // bin_length
        pl["bin_start"]       = pl["gene_start"] + pl["bin_id"] * bin_length
        pl["bin_end"]         = (pl["bin_start"] + bin_length - 1).clip(upper=pl["gene_end"])
        pl["bin_length_actual"] = pl["bin_end"] - pl["bin_start"] + 1

        # Map IMPACT to numeric rank so max() works as an aggregation
        if "IMPACT" in pl.columns:
            pl["_impact_rank"] = pl["IMPACT"].map(
                lambda v: IMPACT_RANK.get(str(v), 0)
            )
            impact_agg = {"_impact_rank": ("_impact_rank", "max")}
        else:
            impact_agg = {}

        group_keys = ["SYMBOL", "CHROM", "bin_id", "bin_start", "bin_end", "bin_length_actual"]
        bin_df = (
            pl.groupby(group_keys, sort=False)
            .agg(
                patient_count=("Patient_ID", "nunique"),
                variant_count=("POS",        "count"),
                **impact_agg,
            )
            .reset_index()
        )

        # Decode numeric rank back to impact label
        rank_to_impact = {v: k for k, v in IMPACT_RANK.items()}
        if "_impact_rank" in bin_df.columns:
            bin_df["worst_impact"] = bin_df["_impact_rank"].map(
                lambda r: rank_to_impact.get(r, "MODIFIER")
            )
            bin_df.drop(columns=["_impact_rank"], inplace=True)
        else:
            bin_df["worst_impact"] = "MODIFIER"

        bin_df["frequency"]          = bin_df["patient_count"] / n_patients
        bin_df["normalized_density"] = (
            bin_df["variant_count"] / (bin_length * n_patients)
        )

        # Sort by (CHROM, bin_start) and assign 1-based bin_index per gene
        bin_df = bin_df.sort_values(["CHROM", "SYMBOL", "bin_start"]).reset_index(drop=True)
        bin_df["bin_index"] = (
            bin_df.groupby(["SYMBOL", "CHROM"]).cumcount() + 1
        )
        bin_df["bin_name"] = (
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
        return bin_df[col_order].sort_values(["CHROM", "bin_start"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Slow path: overlap > 0 — each row can belong to multiple bins;
    # iterate gene by gene with a tqdm progress bar.
    # ------------------------------------------------------------------
    def _worst(series: pd.Series) -> str:
        ranked = [(IMPACT_RANK.get(str(v), 0), str(v)) for v in series.dropna()]
        return max(ranked)[1] if ranked else "MODIFIER"

    rows: list[dict] = []

    groups = list(patient_locus.groupby(["SYMBOL", "CHROM"]))
    for (symbol, chrom), group in tqdm(groups, desc="  Binning genes", unit="gene"):
        gene_start = int(group["POS"].min())
        gene_end   = int(group["POS"].max())

        bin_idx   = 1
        bin_start = gene_start

        while bin_start <= gene_end:
            bin_end_full      = bin_start + bin_length - 1
            bin_end_actual    = min(bin_end_full, gene_end)
            bin_length_actual = bin_end_actual - bin_start + 1

            mask      = (group["POS"] >= bin_start) & (group["POS"] <= bin_end_actual)
            bin_group = group[mask]

            rows.append({
                "bin_name":           f"{chrom}_{symbol}_bin{bin_idx}",
                "CHROM":              chrom,
                "SYMBOL":             symbol,
                "bin_index":          bin_idx,
                "bin_start":          bin_start,
                "bin_end":            bin_end_actual,
                "bin_length_actual":  bin_length_actual,
                "patient_count":      int(bin_group["Patient_ID"].nunique()),
                "frequency":          bin_group["Patient_ID"].nunique() / n_patients,
                "variant_count":      len(bin_group),
                "normalized_density": len(bin_group) / (bin_length * n_patients),
                "worst_impact":       (
                    _worst(bin_group["IMPACT"])
                    if "IMPACT" in bin_group.columns and not bin_group.empty
                    else "MODIFIER"
                ),
            })

            bin_idx   += 1
            bin_start += step

    if not rows:
        return pd.DataFrame()

    bin_df = pd.DataFrame(rows)
    bin_df["POS"] = bin_df["bin_start"].astype(int)
    return bin_df.sort_values(["CHROM", "bin_start"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chromosome layout helpers
# ---------------------------------------------------------------------------

def build_chrom_offsets(freq_df: pd.DataFrame) -> dict[str, int]:
    """
    Assign a cumulative x-axis offset to each chromosome so all chromosomes
    are laid out contiguously along the x-axis (standard Manhattan layout).
    Uses the actual max POS observed per chromosome to compute spacing.
    """
    present = [c for c in CHROM_ORDER if c in freq_df["CHROM"].values]
    chrom_max = freq_df.groupby("CHROM")["POS"].max().to_dict()

    gap = 10_000_000  # 10 Mb spacing between chromosomes
    offsets: dict[str, int] = {}
    cursor = 0
    for chrom in present:
        offsets[chrom] = cursor
        cursor += chrom_max.get(chrom, 0) + gap

    return offsets


def assign_cumulative_pos(freq_df: pd.DataFrame, offsets: dict[str, int]) -> pd.DataFrame:
    freq_df = freq_df.copy()
    freq_df["cum_pos"] = freq_df.apply(
        lambda r: offsets.get(r["CHROM"], 0) + r["POS"], axis=1
    )
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
        chrom  = str(row.get("CHROM", ""))
        pos    = str(int(row["POS"]))
        allele = f"{row.get('REF','?')}>{row.get('ALT','?')}"[:14]
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
    project_root = Path(__file__).resolve().parent.parent.parent
    default_input  = project_root / "output" / "synthetic_patients"
    default_output = project_root / "output" / "distribution_analysis"

    p = argparse.ArgumentParser(
        description="GWAS-style variant frequency analysis for a patient cohort.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", type=Path, default=default_input,
        help="Folder containing per-patient CSV files.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=default_output,
        help="Directory where the output plot will be saved.",
    )
    p.add_argument(
        "--output-name", default="variant_distribution.png",
        help="Filename for the variant-level plot.",
    )
    p.add_argument(
        "--gene-output-name", default="gene_distribution.png",
        help="Filename for the gene-level plot.",
    )
    p.add_argument(
        "--top", type=int, default=15,
        help="Number of top variants to annotate / highlight.",
    )
    p.add_argument(
        "--min-freq", type=float, default=0.0,
        help="Minimum frequency threshold to display in the Manhattan plot (0–1).",
    )
    p.add_argument(
        "--no-show", action="store_true",
        help="Skip the interactive plot window; only save to file.",
    )
    p.add_argument(
        "--bin-length", type=int, default=1000000,
        help="Bin size in base pairs for bin-level aggregation.",
    )
    p.add_argument(
        "--bin-overlap", type=int, default=0,
        help="Overlap between consecutive bins in base pairs (must be < bin-length).",
    )
    p.add_argument(
        "--bin-output-name", default="bin_distribution.csv",
        help="Filename for the bin-level aggregation CSV.",
    )
    p.add_argument(
        "--bin-plot-name", default="bin_distribution.png",
        help="Filename for the bin-level plot.",
    )
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

    print("\n[1/8] Loading patient CSV files …")
    raw_df = load_patient_csvs(args.input_dir)
    n_patients = raw_df["Patient_ID"].nunique()

    print("\n[2/8] Filtering for true mutations (GT ≠ 0/0) …")
    mut_df = filter_mutations(raw_df)

    print("\n[3/8] Computing per-locus mutation frequencies …")
    freq_df = compute_variant_frequency(mut_df, n_patients)
    print(f"  Unique variant loci         : {len(freq_df):,}")

    print("\n[4/8] Computing gene-level mutation frequencies …")
    gene_df = compute_gene_frequency(mut_df, n_patients)
    print(f"  Unique genes                : {len(gene_df):,}")

    print(f"\n[5/8] Computing bin-level frequencies "
          f"(bin_length={args.bin_length:,} bp, overlap={args.bin_overlap:,} bp) …")
    bin_df = compute_bin_frequency(mut_df, n_patients,
                                   bin_length=args.bin_length,
                                   overlap=args.bin_overlap)
    print(f"  Total bins                  : {len(bin_df):,}")

    print("\n[6/8] Building genomic layout …")
    offsets = build_chrom_offsets(freq_df)
    freq_df = assign_cumulative_pos(freq_df, offsets)
    gene_df = assign_cumulative_pos(gene_df, offsets)
    if not bin_df.empty:
        bin_df = assign_cumulative_pos(bin_df, offsets)

    print_summary(freq_df, n_patients, top_n=args.top)

    print("\n[7/8] Saving aggregation results …")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant_csv = args.output_dir / "variant_level_analysis.csv"
    gene_csv    = args.output_dir / "gene_level_analysis.csv"
    bin_csv     = args.output_dir / args.bin_output_name
    freq_df.to_csv(variant_csv, index=False)
    gene_df.to_csv(gene_csv,    index=False)
    if not bin_df.empty:
        bin_df.to_csv(bin_csv, index=False)
    print(f"  Variant-level results saved to : {variant_csv.resolve()}")
    print(f"  Gene-level results saved to    : {gene_csv.resolve()}")
    if not bin_df.empty:
        print(f"  Bin-level results saved to     : {bin_csv.resolve()}")

    print("\n[8/8] Generating plots …")
    make_figure(
        freq_df, offsets, n_patients,
        top_n=args.top,
        min_freq=args.min_freq,
        output_path=args.output_dir / args.output_name,
    )
    make_gene_figure(
        gene_df, offsets, n_patients,
        top_n=args.top,
        min_freq=args.min_freq,
        output_path=args.output_dir / args.gene_output_name,
    )
    make_bin_figure(
        bin_df, offsets, n_patients,
        bin_length=args.bin_length,
        overlap=args.bin_overlap,
        top_n=args.top,
        min_freq=args.min_freq,
        output_path=args.output_dir / args.bin_plot_name,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
