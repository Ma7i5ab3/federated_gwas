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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Human chromosome order (standard cytogenetic order)
CHROM_ORDER = [
    "chr1", "chr2", "chr3", "chr4", "chr5",
    "chr6", "chr7", "chr8", "chr9", "chr10",
    "chr11", "chr12", "chr13", "chr14", "chr15",
    "chr16", "chr17", "chr18", "chr19", "chr20",
    "chr21", "chr22", "chrX", "chrY", "chrM",
]

# Alternating color palette (dark/light per chromosome, GWAS-style)
CHROM_PALETTE = [
    "#1B4F72", "#2E86C1",   # chr1-2
    "#1A5276", "#2980B9",   # chr3-4
    "#154360", "#1F618D",   # chr5-6
    "#0E6655", "#17A589",   # chr7-8
    "#145A32", "#1E8449",   # chr9-10
    "#7B241C", "#C0392B",   # chr11-12
    "#6E2F1A", "#BA4A00",   # chr13-14
    "#6C3483", "#8E44AD",   # chr15-16
    "#4A235A", "#7D3C98",   # chr17-18
    "#1B2631", "#2C3E50",   # chr19-20
    "#4D5656", "#717D7E",   # chr21-22
    "#922B21", "#CB4335",   # chrX-chrY
    "#784212",              # chrM
]

# Consequence impact colours for secondary plots
IMPACT_COLORS = {
    "HIGH":     "#C0392B",
    "MODERATE": "#E67E22",
    "LOW":      "#27AE60",
    "MODIFIER": "#2980B9",
}

CONSEQUENCE_ORDER = [
    "splice_acceptor_variant", "splice_donor_variant", "stop_gained",
    "frameshift_variant", "stop_lost", "start_lost",
    "missense_variant", "inframe_insertion", "inframe_deletion",
    "synonymous_variant", "splice_region_variant",
    "5_prime_UTR_variant", "3_prime_UTR_variant",
    "intron_variant", "upstream_gene_variant", "downstream_gene_variant",
    "intergenic_variant", "non_coding_transcript_variant",
]

# Ranking used to pick the worst impact level per gene
IMPACT_RANK = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}


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
# Plotting
# ---------------------------------------------------------------------------

def make_figure(freq_df: pd.DataFrame,
                offsets: dict[str, int],
                n_patients: int,
                top_n: int = 15,
                min_freq: float = 0.0,
                output_path: Path | None = None) -> None:
    """
    Create a multi-panel figure:
      Panel A  – Manhattan-style scatter  (main plot)
      Panel B  – Top-N most frequent variants (horizontal bar chart)
      Panel C  – Consequence-type frequency breakdown (stacked bar)
    """
    present_chroms = [c for c in CHROM_ORDER if c in freq_df["CHROM"].values]
    chrom_to_color = {c: CHROM_PALETTE[i % len(CHROM_PALETTE)]
                      for i, c in enumerate(present_chroms)}

    # Apply minimum frequency filter for display
    plot_df = freq_df[freq_df["frequency"] >= min_freq].copy() if min_freq > 0 else freq_df.copy()

    # ------------------------------------------------------------------ layout
    fig = plt.figure(figsize=(22, 16), facecolor="#F8F9FA")
    fig.suptitle(
        "Genomic Variant Distribution across Patient Cohort",
        fontsize=20, fontweight="bold", color="#1C1C1C", y=0.98
    )

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.8, 1],
        hspace=0.42,
        wspace=0.32,
        left=0.06, right=0.97, top=0.93, bottom=0.07,
    )

    ax_manhattan = fig.add_subplot(gs[0, :])   # full-width top
    ax_top       = fig.add_subplot(gs[1, 0])   # bottom-left
    ax_conseq    = fig.add_subplot(gs[1, 1])   # bottom-right

    # ===================================================================
    # PANEL A – Manhattan plot
    # ===================================================================
    _draw_manhattan(ax_manhattan, plot_df, present_chroms, chrom_to_color,
                    offsets, n_patients, top_n)

    # ===================================================================
    # PANEL B – Top-N most frequent variants
    # ===================================================================
    _draw_top_variants(ax_top, freq_df, top_n, n_patients)

    # ===================================================================
    # PANEL C – Consequence breakdown
    # ===================================================================
    _draw_consequence_breakdown(ax_conseq, freq_df)

    # ------------------------------------------------------------------ save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"\n  Plot saved to: {output_path.resolve()}")

    plt.show()


def _draw_manhattan(ax, plot_df, present_chroms, chrom_to_color,
                    offsets, n_patients, top_n):
    """Draw the main Manhattan scatter panel."""

    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
    ax.tick_params(colors="#555555")

    # Alternating background bands per chromosome
    for i, chrom in enumerate(present_chroms):
        chrom_data = plot_df[plot_df["CHROM"] == chrom]
        if chrom_data.empty:
            continue
        x_min = chrom_data["cum_pos"].min()
        x_max = chrom_data["cum_pos"].max()
        band_color = "#F0F4F8" if i % 2 == 0 else "#FFFFFF"
        ax.axvspan(x_min - 5e6, x_max + 5e6, color=band_color, zorder=0, alpha=0.6)

    # Scatter per chromosome
    for chrom in present_chroms:
        sub = plot_df[plot_df["CHROM"] == chrom]
        if sub.empty:
            continue
        color = chrom_to_color[chrom]
        # Scale dot size relative to the global max so differences are visible
        global_max = plot_df["frequency"].max()
        norm_freq = sub["frequency"] / global_max if global_max > 0 else sub["frequency"]
        sizes = 15 + (norm_freq ** 1.2) * 120

        ax.scatter(
            sub["cum_pos"], sub["frequency"],
            c=color, s=sizes, alpha=0.75, linewidths=0,
            zorder=2, rasterized=True,
        )

    # Annotate top-N variants
    top = plot_df.nlargest(top_n, "frequency")
    for _, row in top.iterrows():
        label = row.get("SYMBOL") or f"chr{row['CHROM']}:{row['POS']}"
        impact = str(row.get("IMPACT", ""))
        font_color = IMPACT_COLORS.get(impact, "#333333")

        ax.annotate(
            label,
            xy=(row["cum_pos"], row["frequency"]),
            xytext=(0, 10), textcoords="offset points",
            fontsize=7.5, fontweight="bold", color=font_color,
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.7),
            zorder=5,
        )

    # Dynamic y-axis: scale to the actual maximum frequency observed
    y_max = plot_df["frequency"].max()
    y_ceil = y_max * 1.30   # 30 % headroom for annotations
    y_floor = -y_max * 0.08  # small negative margin so bottom dots are visible

    # Frequency reference lines – evenly spaced within the observed range
    n_lines = 5
    ref_levels = np.linspace(0, y_max, n_lines + 1)[1:]  # exclude 0
    for level in ref_levels:
        ax.axhline(level, color="#DDDDDD", linewidth=0.8, linestyle="--", zorder=1)
        ax.text(
            plot_df["cum_pos"].max() * 1.002, level,
            f"{level:.1%}",
            va="center", fontsize=7.5, color="#888888"
        )

    # X-axis: chromosome labels at midpoints
    xtick_positions, xtick_labels = [], []
    for chrom in present_chroms:
        sub = plot_df[plot_df["CHROM"] == chrom]
        if sub.empty:
            continue
        mid = (sub["cum_pos"].min() + sub["cum_pos"].max()) / 2
        xtick_positions.append(mid)
        label = chrom.replace("chr", "")
        xtick_labels.append(label)

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=8.5, rotation=0, color="#333333")
    ax.set_xlim(plot_df["cum_pos"].min() - 2e7, plot_df["cum_pos"].max() + 2e7)
    ax.set_ylim(y_floor, y_ceil)

    # Format y-axis ticks with 1 decimal place when range is small
    if y_max < 0.10:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xlabel("Chromosome", fontsize=12, color="#333333", labelpad=8)
    ax.set_ylabel("Mutation Frequency\n(fraction of cohort)", fontsize=12, color="#333333")

    # Title with cohort size
    ax.set_title(
        f"Variant Frequency across Genome  ·  Cohort: {n_patients} patients  ·  "
        f"Loci shown: {len(plot_df):,}",
        fontsize=11, color="#555555", pad=8,
    )

    # Legend: IMPACT colors
    legend_handles = [
        mpatches.Patch(color=c, label=f"{k} impact")
        for k, c in IMPACT_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles, loc="upper right",
        fontsize=8.5, framealpha=0.9,
        title="Variant Impact", title_fontsize=9,
    )


def _draw_top_variants(ax, freq_df, top_n, n_patients):
    """Horizontal bar chart of the top-N most frequent variants."""

    top = freq_df.nlargest(top_n, "frequency").copy()
    top["label"] = top.apply(
        lambda r: (
            f"{r.get('SYMBOL', '') or 'Unknown'}  "
            f"{r['CHROM']}:{r['POS']}  "
            f"({r['REF']}>{r['ALT']})"
        ),
        axis=1,
    )
    top = top.sort_values("frequency")

    impact_colors = [IMPACT_COLORS.get(str(i), "#95A5A6") for i in top["IMPACT"]]

    bars = ax.barh(
        top["label"], top["frequency"],
        color=impact_colors, edgecolor="white", linewidth=0.6, height=0.7,
    )

    # Value labels – offset is a fraction of the x range so it stays proportional
    x_max = freq_df["frequency"].max()
    label_offset = x_max * 0.02
    for bar, (_, row) in zip(bars, top.iterrows()):
        pct = f"{row['frequency']:.1%}  ({int(row['patient_count'])}/{n_patients})"
        ax.text(
            bar.get_width() + label_offset, bar.get_y() + bar.get_height() / 2,
            pct, va="center", ha="left", fontsize=7.5, color="#333333",
        )

    ax.set_facecolor("#FFFFFF")
    ax.set_xlabel("Mutation Frequency", fontsize=10, color="#333333")
    ax.set_title(f"Top {top_n} Most Frequent Variants", fontsize=12,
                 fontweight="bold", color="#1C1C1C", pad=10)

    x_max = top["frequency"].max()
    ax.set_xlim(0, min(x_max * 1.30, 1.0))

    # Choose decimal precision so ticks never collapse into duplicate labels
    if x_max < 0.01:
        decimals = 3
    elif x_max < 0.10:
        decimals = 2
    elif x_max < 0.50:
        decimals = 1
    else:
        decimals = 0
    fmt_str = f"{{:.{decimals}%}}"
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_str.format(x)))
    # Cap at 6 ticks so they never crowd or repeat
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune="both"))

    ax.tick_params(axis="y", labelsize=7.5)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Legend for IMPACT
    handles = [mpatches.Patch(color=c, label=k) for k, c in IMPACT_COLORS.items()]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right",
              title="Impact", title_fontsize=8, framealpha=0.85)


def _draw_consequence_breakdown(ax, freq_df):
    """
    Stacked bar chart: for each consequence type present in the data,
    show the number of variant loci and colour by IMPACT.
    """
    if "Consequence" not in freq_df.columns:
        ax.text(0.5, 0.5, "No consequence data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888888")
        ax.set_title("Consequence Type Breakdown", fontsize=12, fontweight="bold")
        return

    # Split multi-consequence annotations (VEP uses & as separator)
    exploded = (
        freq_df.assign(Consequence=freq_df["Consequence"].str.split("&"))
               .explode("Consequence")
               .dropna(subset=["Consequence"])
    )
    exploded["Consequence"] = exploded["Consequence"].str.strip()

    # Count loci per (Consequence, IMPACT)
    counts = (
        exploded.groupby(["Consequence", "IMPACT"])
                .size()
                .reset_index(name="count")
    )

    # Pivot so each IMPACT is a column
    pivot = counts.pivot_table(
        index="Consequence", columns="IMPACT", values="count", fill_value=0
    )

    # Reorder rows by total count
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=True).drop(columns="_total")

    # Keep top consequences to avoid clutter
    if len(pivot) > 18:
        pivot = pivot.tail(18)

    impact_order = ["HIGH", "MODERATE", "LOW", "MODIFIER"]
    cols_present = [c for c in impact_order if c in pivot.columns]
    pivot = pivot[cols_present]

    colors = [IMPACT_COLORS.get(c, "#95A5A6") for c in cols_present]

    pivot.plot(
        kind="barh", stacked=True, ax=ax,
        color=colors, edgecolor="white", linewidth=0.4,
        legend=True, width=0.72,
    )

    ax.set_facecolor("#FFFFFF")
    ax.set_xlabel("Number of Variant Loci", fontsize=10, color="#333333")
    ax.set_title("Variant Consequence Distribution", fontsize=12,
                 fontweight="bold", color="#1C1C1C", pad=10)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(title="Impact", fontsize=7.5, title_fontsize=8,
              loc="lower right", framealpha=0.85)


def make_gene_figure(gene_df: pd.DataFrame,
                     offsets: dict[str, int],
                     n_patients: int,
                     top_n: int = 20,
                     min_freq: float = 0.0,
                     output_path: Path | None = None) -> None:
    """
    Three-panel gene-level figure:
      Panel A  – Gene Manhattan (one dot per gene, y = cohort frequency)
      Panel B  – Top-N most affected genes (horizontal bar, coloured by worst impact)
      Panel C  – Bubble chart: variant count per gene vs gene frequency
    """
    if gene_df.empty:
        print("  [WARN] Gene dataframe is empty – skipping gene figure.")
        return

    present_chroms = [c for c in CHROM_ORDER if c in gene_df["CHROM"].values]
    chrom_to_color = {c: CHROM_PALETTE[i % len(CHROM_PALETTE)]
                      for i, c in enumerate(present_chroms)}

    plot_df = gene_df[gene_df["frequency"] >= min_freq].copy() if min_freq > 0 else gene_df.copy()

    fig = plt.figure(figsize=(22, 16), facecolor="#F8F9FA")
    fig.suptitle(
        "Gene-Level Variant Distribution across Patient Cohort",
        fontsize=20, fontweight="bold", color="#1C1C1C", y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.8, 1],
        hspace=0.42,
        wspace=0.34,
        left=0.06, right=0.97, top=0.93, bottom=0.07,
    )

    ax_manhattan = fig.add_subplot(gs[0, :])
    ax_top       = fig.add_subplot(gs[1, 0])
    ax_bubble    = fig.add_subplot(gs[1, 1])

    _draw_gene_manhattan(ax_manhattan, plot_df, present_chroms,
                         chrom_to_color, offsets, n_patients, top_n)
    _draw_top_genes(ax_top, gene_df, top_n, n_patients)
    _draw_gene_bubble(ax_bubble, gene_df, top_n)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Gene plot saved to: {output_path.resolve()}")

    plt.show()


def _draw_gene_manhattan(ax, plot_df, present_chroms, chrom_to_color,
                         offsets, n_patients, top_n):
    """
    Gene-level Manhattan:
      - x     : rank-based position within each chromosome (equal spacing, no overlap)
      - y     : normalized_density auto-scaled to variants/Mb/patient or variants/kb/patient
      - color : continuous colormap (plasma) encoding 1/allele_frequency (rarer = brighter)
      - size  : uniform
    """

    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
    ax.tick_params(colors="#555555")

    y_col = "normalized_density" if "normalized_density" in plot_df.columns else "frequency"
    plot_df = plot_df.copy()

    # --- rank-based x: each gene gets an integer slot, sorted by gene_start ----
    # This guarantees equal spacing so no two genes ever overlap visually.
    ordered_chroms = [c for c in CHROM_ORDER if c in plot_df["CHROM"].values]
    gap = max(4, len(plot_df) // 40)   # adaptive gap between chromosomes
    rank_x: dict[int, int] = {}
    chrom_bounds: dict[str, tuple[int, int]] = {}
    cursor = 0
    for chrom in ordered_chroms:
        sub = plot_df[plot_df["CHROM"] == chrom].sort_values("POS")
        if sub.empty:
            continue
        for rank, idx in enumerate(sub.index):
            rank_x[idx] = cursor + rank
        chrom_bounds[chrom] = (cursor, cursor + len(sub) - 1)
        cursor += len(sub) + gap
    plot_df["rank_x"] = pd.Series(rank_x)

    # --- alternating chromosome bands (visual separation) -----------------
    for i, chrom in enumerate(ordered_chroms):
        if chrom not in chrom_bounds:
            continue
        lo, hi = chrom_bounds[chrom]
        ax.axvspan(lo - gap * 0.4, hi + gap * 0.4,
                   color="#F0F4F8" if i % 2 == 0 else "#FFFFFF",
                   zorder=0, alpha=0.7)

    # --- auto-scale y to a readable unit ----------------------------------
    y_median = plot_df[y_col].median()
    if y_median < 1e-4:
        scale, unit = 1e6, "Mb"
    elif y_median < 0.1:
        scale, unit = 1e3, "kb"
    else:
        scale, unit = 1.0, "bp"
    y_scaled = plot_df[y_col] * scale

    # --- colormap: inverse allele frequency (rarer gene = brighter color) -
    min_af = 1.0 / n_patients
    inv_af = 1.0 / plot_df["frequency"].clip(lower=min_af)
    # Cap at 95th percentile so a few extreme outliers don't compress the palette
    v_lo, v_hi = inv_af.min(), inv_af.quantile(0.95)
    if v_hi <= v_lo:
        v_hi = v_lo + 1
    af_norm = Normalize(vmin=v_lo, vmax=v_hi, clip=True)
    cmap = plt.cm.plasma
    dot_colors = cmap(af_norm(inv_af.values))

    # --- scatter (uniform size, color = inv AF) ---------------------------
    ax.scatter(
        plot_df["rank_x"].values, y_scaled.values,
        c=dot_colors, s=35, alpha=0.85,
        linewidths=0.3, edgecolors="white",
        zorder=2, rasterized=True,
    )

    # --- colorbar for the inv-AF scale ------------------------------------
    sm = ScalarMappable(cmap=cmap, norm=af_norm)
    sm.set_array([])
    cbar = ax.get_figure().colorbar(sm, ax=ax, fraction=0.018, pad=0.01, aspect=30)
    # Tick labels: show as "1/x" so the reader sees actual AF denominator
    cbar_ticks = np.linspace(v_lo, v_hi, 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"1/{t:.0f}" if t >= 2 else f"{1/t:.2f}" for t in cbar_ticks])
    cbar.set_label("Allele frequency  (1/AF, rarer →)", fontsize=8, color="#555555")
    cbar.ax.tick_params(labelsize=7, colors="#555555")

    # --- annotate top-N genes by (scaled) density -------------------------
    top = plot_df.nlargest(top_n, y_col)
    used: list[tuple[float, float]] = []
    y_max_val = y_scaled.max()
    for _, row in top.iterrows():
        label      = str(row.get("SYMBOL", "")) or "?"
        impact     = str(row.get("worst_impact", ""))
        font_color = IMPACT_COLORS.get(impact, "#333333")
        y_val      = row[y_col] * scale

        y_offset = 14
        for px, py in used:
            if abs(row["rank_x"] - px) < gap * 3 and abs(y_val - py) < y_max_val * 0.05:
                y_offset += 12
        used.append((row["rank_x"], y_val))

        ax.annotate(
            label,
            xy=(row["rank_x"], y_val),
            xytext=(0, y_offset), textcoords="offset points",
            fontsize=8, fontweight="bold", color=font_color,
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6),
            zorder=5,
        )

    # --- dynamic y-axis ---------------------------------------------------
    y_max   = y_scaled.max()
    y_ceil  = y_max * 1.35
    y_floor = -y_max * 0.05
    ref_levels = np.linspace(0, y_max, 6)[1:]
    x_right = plot_df["rank_x"].max()
    for level in ref_levels:
        ax.axhline(level, color="#DDDDDD", linewidth=0.8, linestyle="--", zorder=1)
        ax.text(x_right * 1.001, level, f"{level:.2f}",
                va="center", fontsize=7.5, color="#888888")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune="both"))

    # --- chromosome x-ticks (midpoint of each chromosome band) -----------
    xtick_pos, xtick_lab = [], []
    for chrom in ordered_chroms:
        if chrom not in chrom_bounds:
            continue
        lo, hi = chrom_bounds[chrom]
        xtick_pos.append((lo + hi) / 2)
        xtick_lab.append(chrom.replace("chr", ""))

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=8.5, color="#333333")
    ax.set_xlim(-gap, x_right + gap)
    ax.set_ylim(y_floor, y_ceil)

    ax.set_xlabel("Chromosome  (genes ordered by genomic position within each chromosome)",
                  fontsize=11, color="#333333", labelpad=8)
    ax.set_ylabel(f"Mutation Density\n(variants / {unit} / patient)",
                  fontsize=11, color="#333333")
    ax.set_title(
        f"Gene-Level Variant Density  ·  Cohort: {n_patients} patients  ·  "
        f"Genes shown: {len(plot_df):,}  ·  "
        "color = 1/AF (rarer genes brighter)  ·  label color = worst impact",
        fontsize=11, color="#555555", pad=8,
    )

    # Small legend for impact colors (annotation labels only)
    legend_handles = [
        mpatches.Patch(color=IMPACT_COLORS[k], label=f"{k}")
        for k in ["HIGH", "MODERATE", "LOW", "MODIFIER"]
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.9, title="Label color = worst impact", title_fontsize=8.5)


def _draw_top_genes(ax, gene_df, top_n, n_patients):
    """Horizontal bar chart of top-N genes by normalized mutation density."""

    sort_col = "normalized_density" if "normalized_density" in gene_df.columns else "frequency"
    top = gene_df.nlargest(top_n, sort_col).sort_values(sort_col)
    impact_col = "worst_impact" if "worst_impact" in top.columns else None
    colors = (
        [IMPACT_COLORS.get(str(i), "#95A5A6") for i in top[impact_col]]
        if impact_col else ["#2980B9"] * len(top)
    )

    bars = ax.barh(
        top["SYMBOL"], top[sort_col],
        color=colors, edgecolor="white", linewidth=0.6, height=0.7,
    )

    x_max = gene_df[sort_col].max()
    offset = x_max * 0.02
    for bar, (_, row) in zip(bars, top.iterrows()):
        n_vars = int(row["variant_count"]) if "variant_count" in row else "?"
        label  = (f"{row[sort_col]:.2e}  "
                  f"({int(row['patient_count'])}/{n_patients} pts,  "
                  f"{n_vars} vars)")
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=7, color="#333333")

    ax.set_facecolor("#FFFFFF")
    ax.set_xlabel("Mutation Density (variants / base / patient)", fontsize=10, color="#333333")
    ax.set_title(f"Top {top_n} Genes by Mutation Density", fontsize=12,
                 fontweight="bold", color="#1C1C1C", pad=10)

    ax.set_xlim(0, x_max * 1.45)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}"))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune="both"))
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if impact_col:
        handles = [mpatches.Patch(color=c, label=k) for k, c in IMPACT_COLORS.items()]
        ax.legend(handles=handles, fontsize=7.5, loc="lower right",
                  title="Worst Impact", title_fontsize=8, framealpha=0.85)


def _draw_gene_bubble(ax, gene_df, top_n):
    """
    Bubble chart: x = number of distinct variant loci in the gene,
                  y = gene mutation frequency,
                  size & colour = worst impact level.
    Annotates the top-N genes by frequency.
    """
    if "variant_count" not in gene_df.columns:
        ax.text(0.5, 0.5, "variant_count not available",
                ha="center", va="center", transform=ax.transAxes)
        return

    df = gene_df.dropna(subset=["variant_count", "frequency"]).copy()
    impact_col = "worst_impact" if "worst_impact" in df.columns else None

    for impact in ["MODIFIER", "LOW", "MODERATE", "HIGH"]:
        sub = df[df[impact_col] == impact] if impact_col else df
        if sub.empty:
            continue
        ax.scatter(
            sub["variant_count"], sub["frequency"],
            c=IMPACT_COLORS.get(impact, "#95A5A6"),
            s=40 + sub["frequency"] / df["frequency"].max() * 180,
            alpha=0.65, linewidths=0.3, edgecolors="white",
            label=impact, zorder=2, rasterized=True,
        )

    # Annotate top genes
    top = df.nlargest(top_n, "frequency")
    for _, row in top.iterrows():
        ax.annotate(
            str(row["SYMBOL"]),
            xy=(row["variant_count"], row["frequency"]),
            xytext=(5, 3), textcoords="offset points",
            fontsize=7, color="#333333", zorder=5,
        )

    ax.set_facecolor("#FFFFFF")
    ax.set_xlabel("Number of Distinct Variant Loci in Gene", fontsize=10, color="#333333")
    ax.set_ylabel("Gene Mutation Frequency", fontsize=10, color="#333333")
    ax.set_title("Variant Richness vs Gene Frequency", fontsize=12,
                 fontweight="bold", color="#1C1C1C", pad=10)

    y_max = df["frequency"].max()
    decimals = 2 if y_max < 0.10 else (1 if y_max < 0.50 else 0)
    fmt = f"{{:.{decimals}%}}"
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: fmt.format(y)))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune="both"))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune="both", integer=True))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(title="Worst Impact", fontsize=7.5, title_fontsize=8,
              loc="lower right", framealpha=0.85)


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
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.no_show:
        plt.switch_backend("Agg")

    print("\n[1/6] Loading patient CSV files …")
    raw_df = load_patient_csvs(args.input_dir)
    n_patients = raw_df["Patient_ID"].nunique()

    print("\n[2/6] Filtering for true mutations (GT ≠ 0/0) …")
    mut_df = filter_mutations(raw_df)

    print("\n[3/6] Computing per-locus mutation frequencies …")
    freq_df = compute_variant_frequency(mut_df, n_patients)
    print(f"  Unique variant loci         : {len(freq_df):,}")

    print("\n[4/6] Computing gene-level mutation frequencies …")
    gene_df = compute_gene_frequency(mut_df, n_patients)
    print(f"  Unique genes                : {len(gene_df):,}")

    print("\n[5/6] Building genomic layout …")
    offsets = build_chrom_offsets(freq_df)
    freq_df = assign_cumulative_pos(freq_df, offsets)
    gene_df = assign_cumulative_pos(gene_df, offsets)

    print_summary(freq_df, n_patients, top_n=args.top)

    print("\n[6/6] Generating plots …")
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
    print("\nDone.")


if __name__ == "__main__":
    main()
