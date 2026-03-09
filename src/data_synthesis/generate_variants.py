"""
generate_variants.py
--------------------
Synthesise realistic VCF-derived variant data for N patients,
each with M variants distributed across all human chromosomes.

Produces one CSV per patient under output/synthetic_patients/.

Usage
-----
    python src/data_synthesis/generate_variants.py --patients 10 --variants 500
    python src/data_synthesis/generate_variants.py -p 5 -v 1000 --seed 42 --output output/synthetic_patients
"""

import argparse
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# Human chromosome names and approximate lengths (GRCh38, in bp)
CHROMOSOMES: dict[str, int] = {
    "chr1":  248_956_422, "chr2":  242_193_529, "chr3":  198_295_559,
    "chr4":  190_214_555, "chr5":  181_538_259, "chr6":  170_805_979,
    "chr7":  159_345_973, "chr8":  145_138_636, "chr9":  138_394_717,
    "chr10": 133_797_422, "chr11": 135_086_622, "chr12": 133_275_309,
    "chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189,
    "chr16":  90_338_345, "chr17":  83_257_441, "chr18":  80_373_285,
    "chr19":  58_617_616, "chr20":  64_444_167, "chr21":  46_709_983,
    "chr22":  50_818_468, "chrX":  156_040_895, "chrY":   57_227_415,
}

CHROM_NAMES = list(CHROMOSOMES.keys())
# Weight chromosomes by length so longer chromosomes get more variants
CHROM_WEIGHTS = np.array(list(CHROMOSOMES.values()), dtype=float)
CHROM_WEIGHTS /= CHROM_WEIGHTS.sum()

# --- Nucleotides & alleles ---
BASES = ["A", "C", "G", "T"]

# SNP substitution table: given REF, possible ALT bases
SNP_ALT: dict[str, list[str]] = {b: [x for x in BASES if x != b] for b in BASES}

# --- VEP consequences and their impact levels ---
# (consequence, IMPACT, typical BIOTYPE, requires coding annotation)
CONSEQUENCE_CATALOG: list[tuple[str, str, str, bool]] = [
    ("upstream_gene_variant",         "MODIFIER", "transcribed_unprocessed_pseudogene", False),
    ("downstream_gene_variant",       "MODIFIER", "protein_coding",                     False),
    ("intron_variant",                "MODIFIER", "protein_coding",                     False),
    ("synonymous_variant",            "LOW",      "protein_coding",                     True),
    ("missense_variant",              "MODERATE", "protein_coding",                     True),
    ("stop_gained",                   "HIGH",     "protein_coding",                     True),
    ("frameshift_variant",            "HIGH",     "protein_coding",                     True),
    ("splice_region_variant",         "LOW",      "protein_coding",                     False),
    ("splice_donor_variant",          "HIGH",     "protein_coding",                     False),
    ("splice_acceptor_variant",       "HIGH",     "protein_coding",                     False),
    ("3_prime_UTR_variant",           "MODIFIER", "protein_coding",                     False),
    ("5_prime_UTR_variant",           "MODIFIER", "protein_coding",                     False),
    ("non_coding_transcript_variant", "MODIFIER", "retained_intron",                    False),
    ("intergenic_variant",            "MODIFIER", "N/A",                                False),
]
# Weights: most variants are non-coding / MODIFIER
CONSEQUENCE_WEIGHTS = np.array([
    15, 10, 20, 8, 12, 1, 1, 5, 1, 1, 8, 5, 8, 5
], dtype=float)
CONSEQUENCE_WEIGHTS /= CONSEQUENCE_WEIGHTS.sum()

# --- Real human gene symbols (a curated representative subset) ---
GENE_SYMBOLS = [
    "TP53", "BRCA1", "BRCA2", "EGFR", "KRAS", "PIK3CA", "APC", "PTEN",
    "RB1",  "VHL",   "MLH1",  "MSH2", "CDH1", "STK11",  "CDKN2A", "SMAD4",
    "ATM",  "PALB2", "RAD51C","RAD51D","CHEK2","NBN",    "BARD1",  "BRIP1",
    "MEN1", "RET",   "NF1",   "NF2",   "TSC1", "TSC2",   "MUTYH",  "EPCAM",
    "MSH6", "PMS2",  "BMPR1A","GREM1", "POLD1","POLE",   "NTHL1",  "AXIN2",
    "DDX11L1","WASH7P","MIR1302","FAM138A","OR4F5","LINC01409","SAMD11",
    "NOC2L","KLHL17","PLEKHN1","PERM1", "HES4", "ISG15",  "AGRN",   "C1orf159",
    "TNFRSF18","TNFRSF4","SDF4","B3GALT6","FAM132A","UBE2J2","SCNN1D",
    "ACAP3", "PUSL1", "INTS11","CPTP",  "TAS1R3","DVL1",  "MXRA8",  "AURKAIP1",
    "CCNL2","DFFB",  "CGN",   "INTS6L","ERRFI1","SLC45A1","CATSPERE","TSSK3",
    "ACTRT2","AGRN",  "AGTRAP","ATAD3A","ATAD3B","ATAD3C","AURKAIP1","B3GALT6",
]
GENE_SYMBOLS = sorted(set(GENE_SYMBOLS))

# --- Feature BIOTYPEs ---
BIOTYPES = [
    "protein_coding", "retained_intron", "processed_transcript",
    "nonsense_mediated_decay", "transcribed_unprocessed_pseudogene",
    "lncRNA", "miRNA", "snRNA", "snoRNA",
]

# --- Feature types ---
FEATURE_TYPES = ["Transcript", "RegulatoryFeature", "MotifFeature"]

# --- FILTER values ---
FILTERS = ["PASS", "RefCall", "LowQual"]
FILTER_WEIGHTS = [0.75, 0.20, 0.05]

# --- Amino acids (single letter) ---
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# --- Codons ---
CODON_BASES = ["A", "C", "G", "T"]
ALL_CODONS = [a + b + c for a in CODON_BASES for b in CODON_BASES for c in CODON_BASES]
STOP_CODONS = {"TAA", "TAG", "TGA"}
SENSE_CODONS = [c for c in ALL_CODONS if c not in STOP_CODONS]

# --- gnomAD population tags ---
GNOMAD_POPS = ["afr", "ami", "amr", "asj", "eas", "fin", "nfe", "sas"]


# ---------------------------------------------------------------------------
# Low-level generators
# ---------------------------------------------------------------------------

def _random_rs_id(rng: np.random.Generator) -> str:
    return f"rs{rng.integers(100_000, 999_999_999)}"


def _random_ensg(rng: np.random.Generator) -> str:
    return f"ENSG{rng.integers(0, 99_999_999_999):011d}"


def _random_enst(rng: np.random.Generator) -> str:
    return f"ENST{rng.integers(0, 99_999_999_999):011d}"


def _random_hgnc_id(rng: np.random.Generator) -> str:
    return f"HGNC:{rng.integers(1, 60_000)}"


def _random_allele(rng: np.random.Generator, ref: str, is_indel: bool) -> str:
    """Generate a plausible ALT allele."""
    if is_indel:
        # insertion or deletion relative to ref
        if rng.random() < 0.5:
            # insertion: ref + extra base(s)
            extra = "".join(rng.choice(BASES) for _ in range(rng.integers(1, 4)))
            return ref + extra
        else:
            # deletion: ref truncated (keep first base, remove rest)
            if len(ref) > 1:
                return ref[0]
            # single-base ref → insert 1 base deletion using longer ref
            return ref + rng.choice([b for b in BASES if b != ref])
    else:
        # SNP
        return rng.choice(SNP_ALT[ref[0]])


def _random_ref(rng: np.random.Generator, is_indel: bool) -> str:
    base = rng.choice(BASES)
    if is_indel:
        extra = "".join(rng.choice(BASES) for _ in range(rng.integers(1, 4)))
        return base + extra
    return base


def _random_gnomad_af(rng: np.random.Generator) -> float | None:
    """Most variants are rare; use a log-uniform distribution."""
    if rng.random() < 0.15:      # 15 % chance of no gnomAD entry
        return None
    # log-uniform in [1e-6, 0.5]
    log_af = rng.uniform(np.log10(1e-6), np.log10(0.5))
    return round(float(10 ** log_af), 8)


def _random_pop_afs(rng: np.random.Generator, overall_af: float | None
                    ) -> dict[str, float | None]:
    """Generate per-population AFs that roughly sum consistently."""
    if overall_af is None:
        return {f"gnomADg_AF_{p}": None for p in GNOMAD_POPS}
    result = {}
    for pop in GNOMAD_POPS:
        # perturb AF per population
        log_base = np.log10(max(overall_af, 1e-7))
        delta = rng.uniform(-1.5, 1.5)
        af = float(10 ** np.clip(log_base + delta, -7, np.log10(0.999)))
        result[f"gnomADg_AF_{pop}"] = round(af, 8)
    return result


def _random_genotype(rng: np.random.Generator, is_low_qual: bool
                     ) -> tuple[str, int | None, int, str, float, str]:
    """
    Returns (GT, GQ, DP, AD, VAF, PL).
    Weights: 0/0 (60 %), 0/1 (30 %), 1/1 (8 %), ./. (2 %)
    """
    if is_low_qual and rng.random() < 0.4:
        gt = "./."
        gq = None
        dp = int(rng.integers(1, 15))
        ref_ad = int(rng.integers(0, dp + 1))
        alt_ad = dp - ref_ad
        vaf = round(alt_ad / dp, 6) if dp > 0 else 0.0
        pl = "0,0,0"
        return gt, gq, dp, f"{ref_ad},{alt_ad}", vaf, pl

    choice = rng.choice(["0/0", "0/1", "1/1"], p=[0.60, 0.32, 0.08])
    dp = int(rng.integers(20, 120))
    gq = int(rng.integers(10, 99))

    if choice == "0/0":
        ref_ad = dp - int(rng.integers(0, max(1, int(dp * 0.05))))
        alt_ad = dp - ref_ad
        vaf = round(alt_ad / dp, 6)
        pl_hom_ref = 0
        pl_het = int(rng.integers(50, 500))
        pl_hom_alt = int(rng.integers(200, 1000))
    elif choice == "0/1":
        ref_ad = int(dp * rng.uniform(0.35, 0.65))
        alt_ad = dp - ref_ad
        vaf = round(alt_ad / dp, 6)
        pl_hom_ref = int(rng.integers(50, 500))
        pl_het = 0
        pl_hom_alt = int(rng.integers(50, 500))
    else:  # 1/1
        alt_ad = dp - int(rng.integers(0, max(1, int(dp * 0.05))))
        ref_ad = dp - alt_ad
        vaf = round(alt_ad / dp, 6)
        pl_hom_ref = int(rng.integers(200, 1000))
        pl_het = int(rng.integers(50, 500))
        pl_hom_alt = 0

    ad = f"{ref_ad},{alt_ad}"
    pl = f"{pl_hom_ref},{pl_het},{pl_hom_alt}"
    return choice, gq, dp, ad, vaf, pl


def _random_hgvsc(rng: np.random.Generator, enst: str, pos: int,
                  ref: str, alt: str) -> str:
    cdna_pos = rng.integers(1, 5000)
    if len(ref) == 1 and len(alt) == 1:
        return f"{enst}.1:c.{cdna_pos}{ref}>{alt}"
    elif len(alt) > len(ref):
        ins = alt[len(ref):]
        return f"{enst}.1:c.{cdna_pos}_{cdna_pos + 1}ins{ins}"
    else:
        del_len = len(ref) - len(alt)
        return f"{enst}.1:c.{cdna_pos}_{cdna_pos + del_len - 1}del"


def _random_hgvsp(rng: np.random.Generator, symbol: str,
                  consequence: str) -> str | None:
    if consequence not in ("missense_variant", "stop_gained", "synonymous_variant",
                           "frameshift_variant"):
        return None
    prot_pos = rng.integers(1, 800)
    aa_ref = rng.choice(AMINO_ACIDS)
    aa_alt = rng.choice(AMINO_ACIDS)
    if consequence == "stop_gained":
        return f"{symbol}:p.{aa_ref}{prot_pos}*"
    elif consequence == "frameshift_variant":
        return f"{symbol}:p.{aa_ref}{prot_pos}fs"
    elif consequence == "synonymous_variant":
        return f"{symbol}:p.{aa_ref}{prot_pos}="
    else:
        return f"{symbol}:p.{aa_ref}{prot_pos}{aa_alt}"


# ---------------------------------------------------------------------------
# Single-variant row generator
# ---------------------------------------------------------------------------

def _generate_variant(
        rng: np.random.Generator,
        patient_id: str,
        chrom: str,
        pos: int,
) -> dict:
    # Variant type
    is_indel = rng.random() < 0.15   # ~15 % indels
    ref = _random_ref(rng, is_indel)
    alt = _random_allele(rng, ref, is_indel)
    allele = alt  # VEP Allele field mirrors ALT for simple variants

    # Quality & filter
    filter_val = rng.choice(FILTERS, p=FILTER_WEIGHTS)
    is_low_qual = filter_val == "LowQual"
    qual = round(float(rng.uniform(0, 5)), 2) if filter_val == "RefCall" \
        else (round(float(rng.uniform(0, 15)), 2) if is_low_qual
              else round(float(rng.uniform(20, 500)), 2))

    # Consequence
    csq_idx = int(rng.choice(len(CONSEQUENCE_CATALOG), p=CONSEQUENCE_WEIGHTS))
    consequence, impact, biotype_hint, has_coding = CONSEQUENCE_CATALOG[csq_idx]

    # Gene / transcript
    symbol = str(rng.choice(GENE_SYMBOLS))
    gene_id = _random_ensg(rng)
    enst = _random_enst(rng)

    # Feature type
    feat_type = "Transcript"  # most variants annotated against transcripts

    biotype = biotype_hint if biotype_hint != "N/A" else rng.choice(BIOTYPES[:5])

    # Exon / Intron
    exon = intron = None
    if consequence in ("synonymous_variant", "missense_variant",
                       "stop_gained", "frameshift_variant",
                       "5_prime_UTR_variant", "3_prime_UTR_variant"):
        exon_n = rng.integers(1, 20)
        exon_total = rng.integers(exon_n, exon_n + 10)
        exon = f"{exon_n}/{exon_total}"
    elif consequence == "intron_variant":
        intron_n = rng.integers(1, 19)
        intron_total = rng.integers(intron_n, intron_n + 10)
        intron = f"{intron_n}/{intron_total}"

    # HGVS
    hgvsc = _random_hgvsc(rng, enst, pos, ref, alt) if has_coding else None
    hgvsp = _random_hgvsp(rng, symbol, consequence) if has_coding else None

    # cDNA / CDS / Protein positions
    cdna_position = str(rng.integers(1, 5000)) if has_coding else None
    cds_position  = str(rng.integers(1, 3000)) if has_coding else None
    prot_position = str(rng.integers(1, 800))  if has_coding else None

    # Amino acids & codons
    amino_acids = codons = None
    if consequence in ("missense_variant", "stop_gained", "synonymous_variant"):
        aa_ref = rng.choice(AMINO_ACIDS)
        aa_alt = rng.choice(AMINO_ACIDS)
        amino_acids = f"{aa_ref}/{aa_alt}"
        cod_ref = rng.choice(SENSE_CODONS)
        cod_alt = rng.choice(SENSE_CODONS if consequence != "stop_gained"
                             else list(STOP_CODONS))
        codons = f"{cod_ref}/{cod_alt}"

    # Known variation (rsID)
    existing_variation = _random_rs_id(rng) if rng.random() < 0.3 else None

    # Distance (upstream / downstream)
    distance = str(rng.integers(1, 5000)) \
        if consequence in ("upstream_gene_variant", "downstream_gene_variant") else None

    strand = rng.choice([1, -1])
    symbol_source = "HGNC"
    hgnc_id = _random_hgnc_id(rng)
    source = "Ensembl"

    # gnomAD
    gnomad_af = _random_gnomad_af(rng)
    gnomad_id = (f"gnomADg_{chrom}_{pos}_{ref}_{alt}" if gnomad_af is not None else None)
    pop_afs = _random_pop_afs(rng, gnomad_af)

    # Genotype
    gt, gq, dp, ad, vaf, pl = _random_genotype(rng, is_low_qual)

    # Variant ID (rsID ~30 % of the time)
    vid = _random_rs_id(rng) if rng.random() < 0.3 else None

    return {
        "Patient_ID":         patient_id,
        "CHROM":              chrom,
        "POS":                pos,
        "ID":                 vid,
        "REF":                ref,
        "ALT":                alt,
        "QUAL":               qual,
        "FILTER":             filter_val,
        # VEP fields
        "Allele":             allele,
        "Consequence":        consequence,
        "IMPACT":             impact,
        "SYMBOL":             symbol,
        "Gene":               gene_id,
        "Feature_type":       feat_type,
        "Feature":            enst,
        "BIOTYPE":            biotype,
        "EXON":               exon,
        "INTRON":             intron,
        "HGVSc":              hgvsc,
        "HGVSp":              hgvsp,
        "cDNA_position":      cdna_position,
        "CDS_position":       cds_position,
        "Protein_position":   prot_position,
        "Amino_acids":        amino_acids,
        "Codons":             codons,
        "Existing_variation": existing_variation,
        "DISTANCE":           distance,
        "STRAND":             strand,
        "FLAGS":              None,
        "SYMBOL_SOURCE":      symbol_source,
        "HGNC_ID":            hgnc_id,
        "SOURCE":             source,
        "gnomADg":            gnomad_id,
        "gnomADg_AF":         gnomad_af,
        **pop_afs,
        # FORMAT / sample
        "GT":                 gt,
        "GQ":                 gq,
        "DP":                 dp,
        "AD":                 ad,
        "VAF":                vaf,
        "PL":                 pl,
    }


# ---------------------------------------------------------------------------
# Patient-level generator
# ---------------------------------------------------------------------------

def generate_patient(
        patient_id: str,
        n_variants: int,
        rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate *n_variants* variants for a single patient, distributed across
    all chromosomes proportionally to chromosome length.

    Returns a DataFrame with the same column order as parse_vcf.py output.
    """
    # Assign variants to chromosomes (multinomial draw → proportional to length)
    counts = rng.multinomial(n_variants, CHROM_WEIGHTS)

    rows: list[dict] = []
    for chrom, chrom_count in zip(CHROM_NAMES, counts):
        if chrom_count == 0:
            continue
        chrom_len = CHROMOSOMES[chrom]
        # Draw positions without replacement (if possible) within the chromosome
        positions = sorted(
            rng.choice(chrom_len, size=chrom_count, replace=chrom_count > chrom_len).tolist()
        )
        for pos in positions:
            rows.append(_generate_variant(rng, patient_id, chrom, int(pos) + 1))

    df = pd.DataFrame(rows)

    # Enforce column order (matches parse_vcf.py output)
    ordered_cols = [
        "Patient_ID", "CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER",
        "Allele", "Consequence", "IMPACT", "SYMBOL", "Gene", "Feature_type",
        "Feature", "BIOTYPE", "EXON", "INTRON", "HGVSc", "HGVSp",
        "cDNA_position", "CDS_position", "Protein_position", "Amino_acids",
        "Codons", "Existing_variation", "DISTANCE", "STRAND", "FLAGS",
        "SYMBOL_SOURCE", "HGNC_ID", "SOURCE",
        "gnomADg", "gnomADg_AF",
        "gnomADg_AF_afr", "gnomADg_AF_ami", "gnomADg_AF_amr",
        "gnomADg_AF_asj", "gnomADg_AF_eas", "gnomADg_AF_fin",
        "gnomADg_AF_nfe", "gnomADg_AF_sas",
        "GT", "GQ", "DP", "AD", "VAF", "PL",
    ]
    present = [c for c in ordered_cols if c in df.columns]
    extra   = [c for c in df.columns    if c not in ordered_cols]
    return df[present + extra]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic VCF-derived variant CSVs for N patients."
    )
    parser.add_argument(
        "-p", "--patients", type=int, required=True,
        help="Number of patients to generate.",
    )
    parser.add_argument(
        "-v", "--variants", type=int, required=True,
        help="Number of variants per patient.",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: random).",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path(__file__).parents[2] / "output" / "synthetic_patients",
        help="Output directory (default: output/synthetic_patients/).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Generating data for {args.patients} patient(s), "
          f"{args.variants} variants each.")
    print(f"Output directory: {args.output.resolve()}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")

    for i in range(1, args.patients + 1):
        patient_id = f"PATIENT_{i:04d}"
        df = generate_patient(patient_id, args.variants, rng)
        out_path = args.output / f"{patient_id}.csv"
        df.to_csv(out_path, index=False)
        chrom_counts = df["CHROM"].value_counts().to_dict()
        print(f"  [{i}/{args.patients}] {patient_id}: {len(df)} variants "
              f"across {df['CHROM'].nunique()} chromosomes → {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
