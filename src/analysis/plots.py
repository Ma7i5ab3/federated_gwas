"""
plots.py
--------
Plotting functions for GWAS-style variant frequency analysis.

All figure-generation logic is here; import make_figure / make_gene_figure
from this module to produce publication-quality Manhattan-style plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
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
# Variant-level figure
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
    plt.close()


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


# ---------------------------------------------------------------------------
# Gene-level figure
# ---------------------------------------------------------------------------

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
    plt.close()


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
# Bin-level figure
# ---------------------------------------------------------------------------

def make_bin_figure(bin_df: pd.DataFrame,
                    offsets: dict[str, int],
                    n_patients: int,
                    bin_length: int,
                    overlap: int = 0,
                    top_n: int = 20,
                    min_freq: float = 0.0,
                    output_path: Path | None = None) -> None:
    """
    Three-panel bin-level figure:
      Panel A  – Bin Manhattan (one dot per bin, y = normalized_density)
      Panel B  – Top-N bins by mutation density (horizontal bar)
      Panel C  – Bubble chart: variant count per bin vs bin frequency
    """
    if bin_df.empty:
        print("  [WARN] Bin dataframe is empty – skipping bin figure.")
        return

    present_chroms = [c for c in CHROM_ORDER if c in bin_df["CHROM"].values]
    chrom_to_color = {c: CHROM_PALETTE[i % len(CHROM_PALETTE)]
                      for i, c in enumerate(present_chroms)}

    plot_df = bin_df[bin_df["frequency"] >= min_freq].copy() if min_freq > 0 else bin_df.copy()

    step = bin_length - overlap
    title_detail = (f"bin={bin_length:,} bp"
                    + (f", overlap={overlap:,} bp" if overlap > 0 else ", no overlap"))

    fig = plt.figure(figsize=(22, 16), facecolor="#F8F9FA")
    fig.suptitle(
        f"Bin-Level Variant Distribution across Patient Cohort  ·  {title_detail}",
        fontsize=19, fontweight="bold", color="#1C1C1C", y=0.98,
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

    _draw_bin_manhattan(ax_manhattan, plot_df, present_chroms,
                        chrom_to_color, offsets, n_patients, top_n)
    _draw_top_bins(ax_top, bin_df, top_n, n_patients)
    _draw_bin_bubble(ax_bubble, bin_df, top_n)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Bin plot saved to: {output_path.resolve()}")

    plt.show()
    plt.close()


def _draw_bin_manhattan(ax, plot_df, present_chroms, chrom_to_color,
                        offsets, n_patients, top_n):
    """
    Bin-level Manhattan:
      - x     : rank-based position within each chromosome (equal spacing)
      - y     : normalized_density, auto-scaled to /Mb, /kb, or /bp per patient
      - color : plasma colormap encoding 1/frequency (rarer bins = brighter)
      - size  : uniform
    Annotations use the short form  SYMBOL[binN].
    """
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
    ax.tick_params(colors="#555555")

    y_col = "normalized_density" if "normalized_density" in plot_df.columns else "frequency"
    plot_df = plot_df.copy()

    # --- rank-based x: one integer slot per bin, sorted by bin_start within each chrom ---
    ordered_chroms = [c for c in CHROM_ORDER if c in plot_df["CHROM"].values]
    gap = max(4, len(plot_df) // 80)
    rank_x: dict[int, int] = {}
    chrom_bounds: dict[str, tuple[int, int]] = {}
    cursor = 0
    for chrom in ordered_chroms:
        sub = plot_df[plot_df["CHROM"] == chrom].sort_values("bin_start"
              if "bin_start" in plot_df.columns else "POS")
        if sub.empty:
            continue
        for rank, idx in enumerate(sub.index):
            rank_x[idx] = cursor + rank
        chrom_bounds[chrom] = (cursor, cursor + len(sub) - 1)
        cursor += len(sub) + gap
    plot_df["rank_x"] = pd.Series(rank_x)

    # --- alternating chromosome bands ---
    for i, chrom in enumerate(ordered_chroms):
        if chrom not in chrom_bounds:
            continue
        lo, hi = chrom_bounds[chrom]
        ax.axvspan(lo - gap * 0.4, hi + gap * 0.4,
                   color="#F0F4F8" if i % 2 == 0 else "#FFFFFF",
                   zorder=0, alpha=0.7)

    # --- auto-scale y ---
    y_median = plot_df[y_col].median()
    if y_median < 1e-4:
        scale, unit = 1e6, "Mb"
    elif y_median < 0.1:
        scale, unit = 1e3, "kb"
    else:
        scale, unit = 1.0, "bp"
    y_scaled = plot_df[y_col] * scale

    # --- colormap: inverse frequency ---
    min_af   = 1.0 / n_patients
    inv_af   = 1.0 / plot_df["frequency"].clip(lower=min_af)
    v_lo, v_hi = inv_af.min(), inv_af.quantile(0.95)
    if v_hi <= v_lo:
        v_hi = v_lo + 1
    af_norm    = Normalize(vmin=v_lo, vmax=v_hi, clip=True)
    cmap       = plt.cm.plasma
    dot_colors = cmap(af_norm(inv_af.values))

    ax.scatter(
        plot_df["rank_x"].values, y_scaled.values,
        c=dot_colors, s=20, alpha=0.80,
        linewidths=0.3, edgecolors="white",
        zorder=2, rasterized=True,
    )

    # --- colorbar ---
    sm = ScalarMappable(cmap=cmap, norm=af_norm)
    sm.set_array([])
    cbar = ax.get_figure().colorbar(sm, ax=ax, fraction=0.018, pad=0.01, aspect=30)
    cbar_ticks = np.linspace(v_lo, v_hi, 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"1/{t:.0f}" if t >= 2 else f"{1/t:.2f}" for t in cbar_ticks])
    cbar.set_label("Allele frequency  (1/AF, rarer →)", fontsize=8, color="#555555")
    cbar.ax.tick_params(labelsize=7, colors="#555555")

    # --- annotate top-N bins by density (short label: SYMBOL[binN]) ---
    top      = plot_df.nlargest(top_n, y_col)
    used: list[tuple[float, float]] = []
    y_max_val = y_scaled.max()
    for _, row in top.iterrows():
        symbol = str(row.get("SYMBOL", "?"))
        bidx   = int(row["bin_index"]) if "bin_index" in row else "?"
        label  = f"{symbol}[bin{bidx}]"
        impact = str(row.get("worst_impact", ""))
        font_color = IMPACT_COLORS.get(impact, "#333333")
        y_val  = row[y_col] * scale

        y_offset = 14
        for px, py in used:
            if abs(row["rank_x"] - px) < gap * 3 and abs(y_val - py) < y_max_val * 0.05:
                y_offset += 12
        used.append((row["rank_x"], y_val))

        ax.annotate(
            label,
            xy=(row["rank_x"], y_val),
            xytext=(0, y_offset), textcoords="offset points",
            fontsize=7.5, fontweight="bold", color=font_color,
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6),
            zorder=5,
        )

    # --- axes ---
    y_max  = y_scaled.max()
    y_ceil = y_max * 1.35
    ref_levels = np.linspace(0, y_max, 6)[1:]
    x_right = plot_df["rank_x"].max()
    for level in ref_levels:
        ax.axhline(level, color="#DDDDDD", linewidth=0.8, linestyle="--", zorder=1)
        ax.text(x_right * 1.001, level, f"{level:.2f}",
                va="center", fontsize=7.5, color="#888888")

    ax.set_ylim(-y_max * 0.05, y_ceil)
    ax.set_xlim(-gap, x_right + gap)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune="both"))

    xtick_pos, xtick_lab = [], []
    for chrom in ordered_chroms:
        if chrom not in chrom_bounds:
            continue
        lo, hi = chrom_bounds[chrom]
        xtick_pos.append((lo + hi) / 2)
        xtick_lab.append(chrom.replace("chr", ""))
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=8.5, color="#333333")

    ax.set_xlabel("Chromosome  (bins ordered by genomic position within each chromosome)",
                  fontsize=11, color="#333333", labelpad=8)
    ax.set_ylabel(f"Mutation Density\n(variants / {unit} / patient)",
                  fontsize=11, color="#333333")
    ax.set_title(
        f"Bin-Level Variant Density  ·  Cohort: {n_patients} patients  ·  "
        f"Bins shown: {len(plot_df):,}  ·  "
        "color = 1/AF (rarer bins brighter)  ·  label color = worst impact",
        fontsize=11, color="#555555", pad=8,
    )

    legend_handles = [
        mpatches.Patch(color=IMPACT_COLORS[k], label=k)
        for k in ["HIGH", "MODERATE", "LOW", "MODIFIER"]
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.9, title="Label color = worst impact", title_fontsize=8.5)


def _draw_top_bins(ax, bin_df, top_n, n_patients):
    """Horizontal bar chart of top-N bins by normalized mutation density."""

    sort_col   = "normalized_density" if "normalized_density" in bin_df.columns else "frequency"
    impact_col = "worst_impact" if "worst_impact" in bin_df.columns else None

    top = bin_df.nlargest(top_n, sort_col).sort_values(sort_col)
    colors = (
        [IMPACT_COLORS.get(str(i), "#95A5A6") for i in top[impact_col]]
        if impact_col else ["#2980B9"] * len(top)
    )

    bars = ax.barh(
        top["bin_name"], top[sort_col],
        color=colors, edgecolor="white", linewidth=0.6, height=0.7,
    )

    x_max  = bin_df[sort_col].max()
    offset = x_max * 0.02
    for bar, (_, row) in zip(bars, top.iterrows()):
        n_vars = int(row["variant_count"]) if "variant_count" in row else "?"
        label  = (f"{row[sort_col]:.2e}  "
                  f"({int(row['patient_count'])}/{n_patients} pts,  "
                  f"{n_vars} vars,  len={int(row.get('bin_length_actual', 0)):,} bp)")
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=6.5, color="#333333")

    ax.set_facecolor("#FFFFFF")
    ax.set_xlabel("Mutation Density (variants / base / patient)", fontsize=10, color="#333333")
    ax.set_title(f"Top {top_n} Bins by Mutation Density", fontsize=12,
                 fontweight="bold", color="#1C1C1C", pad=10)
    ax.set_xlim(0, x_max * 1.55)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}"))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune="both"))
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if impact_col:
        handles = [mpatches.Patch(color=c, label=k) for k, c in IMPACT_COLORS.items()]
        ax.legend(handles=handles, fontsize=7.5, loc="lower right",
                  title="Worst Impact", title_fontsize=8, framealpha=0.85)


def _draw_bin_bubble(ax, bin_df, top_n):
    """
    Bubble chart:  x = variant_count per bin,  y = bin frequency,
                   size & colour = worst_impact.
    Annotates the top-N bins by frequency using the short label SYMBOL[binN].
    """
    if "variant_count" not in bin_df.columns:
        ax.text(0.5, 0.5, "variant_count not available",
                ha="center", va="center", transform=ax.transAxes)
        return

    df         = bin_df.dropna(subset=["variant_count", "frequency"]).copy()
    impact_col = "worst_impact" if "worst_impact" in df.columns else None
    freq_max   = df["frequency"].max()

    for impact in ["MODIFIER", "LOW", "MODERATE", "HIGH"]:
        sub = df[df[impact_col] == impact] if impact_col else df
        if sub.empty:
            continue
        ax.scatter(
            sub["variant_count"], sub["frequency"],
            c=IMPACT_COLORS.get(impact, "#95A5A6"),
            s=30 + sub["frequency"] / max(freq_max, 1e-9) * 160,
            alpha=0.65, linewidths=0.3, edgecolors="white",
            label=impact, zorder=2, rasterized=True,
        )

    top = df.nlargest(top_n, "frequency")
    for _, row in top.iterrows():
        symbol = str(row.get("SYMBOL", "?"))
        bidx   = int(row["bin_index"]) if "bin_index" in row else "?"
        ax.annotate(
            f"{symbol}[bin{bidx}]",
            xy=(row["variant_count"], row["frequency"]),
            xytext=(5, 3), textcoords="offset points",
            fontsize=6.5, color="#333333", zorder=5,
        )

    ax.set_facecolor("#FFFFFF")
    ax.set_xlabel("Number of Variant Observations in Bin", fontsize=10, color="#333333")
    ax.set_ylabel("Bin Mutation Frequency", fontsize=10, color="#333333")
    ax.set_title("Variant Richness vs Bin Frequency", fontsize=12,
                 fontweight="bold", color="#1C1C1C", pad=10)

    y_max    = df["frequency"].max()
    decimals = 2 if y_max < 0.10 else (1 if y_max < 0.50 else 0)
    fmt      = f"{{:.{decimals}%}}"
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: fmt.format(y)))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune="both"))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune="both", integer=True))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(title="Worst Impact", fontsize=7.5, title_fontsize=8,
              loc="lower right", framealpha=0.85)
