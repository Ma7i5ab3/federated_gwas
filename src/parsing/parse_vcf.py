"""
parse_vcf.py
------------
Parse VEP-annotated VCF files (Sarek / DeepVariant output) from data/*.vcf
and produce a tidy pandas DataFrame with one row per variant per patient.

Columns produced
----------------
Standard VCF fields : Patient_ID, CHROM, POS, ID, REF, ALT, QUAL, FILTER
VEP CSQ sub-fields  : Allele … gnomADg_AF_sas  (parsed from the INFO/CSQ tag)
Sample FORMAT fields: GT, GQ, DP, AD, VAF, PL
"""

import os
import re
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_csq_header(info_meta_line: str) -> list[str]:
    """
    Extract the ordered list of CSQ sub-field names from the ##INFO CSQ
    meta-information line.

    Example input:
        ##INFO=<ID=CSQ,...,Description="... Format: Allele|Consequence|...">
    Returns:
        ['Allele', 'Consequence', ...]
    """
    match = re.search(r"Format: ([^\"]+)", info_meta_line)
    if not match:
        raise ValueError("Could not find CSQ format in header line:\n" + info_meta_line)
    return match.group(1).strip().split("|")


def _parse_info_csq(info_str: str, csq_fields: list[str]) -> list[dict]:
    """
    Extract the CSQ annotations from the INFO column string.

    The CSQ tag can contain multiple comma-separated transcripts.
    Each transcript is a pipe-delimited string aligned to csq_fields.

    Returns a list of dicts (one per transcript).
    Missing values are stored as None.
    """
    # Pull the raw CSQ value: everything after "CSQ="
    csq_match = re.search(r"CSQ=([^;]+)", info_str)
    if not csq_match:
        # Variant has no annotation – return one empty record
        return [{f: None for f in csq_fields}]

    csq_raw = csq_match.group(1)
    transcripts = csq_raw.split(",")   # multiple transcripts separated by ","

    records = []
    for transcript in transcripts:
        values = transcript.split("|")
        # Zip with field names; fall back to None if fewer values than fields
        record = {
            field: (values[i] if i < len(values) and values[i] != "" else None)
            for i, field in enumerate(csq_fields)
        }
        records.append(record)

    return records


def _parse_format_sample(format_str: str, sample_str: str) -> dict:
    """
    Parse the FORMAT / SAMPLE columns into a dict.

    FORMAT : "GT:GQ:DP:AD:VAF:PL"
    SAMPLE : "0/1:45:120:60,60:0.5:100,0,95"

    Returns {'GT': '0/1', 'GQ': '45', 'DP': '120', ...}
    Missing / dot values are stored as None.
    """
    keys = format_str.split(":")
    vals = sample_str.split(":")
    result = {}
    for i, key in enumerate(keys):
        v = vals[i] if i < len(vals) else None
        result[key] = None if v in (".", "", None) else v
    return result


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_vcf(vcf_path: str | Path) -> pd.DataFrame:
    """
    Parse a single VCF file and return a DataFrame.

    Parameters
    ----------
    vcf_path : path to the .vcf file

    Returns
    -------
    pd.DataFrame with all requested columns.
    """
    vcf_path = Path(vcf_path)
    patient_id = vcf_path.stem          # filename without extension

    csq_fields: list[str] = []          # filled when we find the CSQ header line
    rows: list[dict] = []

    with vcf_path.open("r") as fh:
        for line in fh:
            line = line.rstrip("\n")

            # --- Meta-information lines ---
            if line.startswith("##"):
                # Grab the CSQ field order from the VEP annotation line
                if "ID=CSQ" in line:
                    csq_fields = _parse_csq_header(line)
                continue

            # --- Column header line ---
            if line.startswith("#CHROM"):
                # We don't need sample name from header for now
                continue

            # --- Data lines ---
            parts = line.split("\t")
            if len(parts) < 9:
                continue                # skip malformed lines

            chrom, pos, vid, ref, alt, qual, filt, info, fmt = parts[:9]
            sample = parts[9] if len(parts) > 9 else ""

            # Parse VEP annotations (one dict per transcript)
            csq_records = _parse_info_csq(info, csq_fields)

            # Parse FORMAT/SAMPLE (GT, GQ, DP, AD, VAF, PL)
            fmt_dict = _parse_format_sample(fmt, sample)

            # Build one output row per CSQ transcript
            # (with --pick VEP flag there is usually exactly one)
            for csq in csq_records:
                row = {
                    "Patient_ID": patient_id,
                    "CHROM":      chrom,
                    "POS":        pos,
                    "ID":         vid if vid != "." else None,
                    "REF":        ref,
                    "ALT":        alt,
                    "QUAL":       qual if qual != "." else None,
                    "FILTER":     filt,
                }
                row.update(csq)         # VEP sub-fields
                row.update(fmt_dict)    # GT, GQ, DP, AD, VAF, PL
                rows.append(row)

    return pd.DataFrame(rows)


def parse_all_vcfs(data_dir: str | Path = "data") -> pd.DataFrame:
    """
    Parse all .vcf files found in *data_dir* and concatenate the results
    into a single DataFrame.

    Parameters
    ----------
    data_dir : directory that contains the .vcf files (default: "data/")

    Returns
    -------
    pd.DataFrame combining all patients, reset index.
    """
    data_dir = Path(data_dir)
    vcf_files = sorted(data_dir.glob("*.vcf"))

    if not vcf_files:
        raise FileNotFoundError(f"No .vcf files found in {data_dir.resolve()}")

    frames = [parse_vcf(f) for f in vcf_files]
    df = pd.concat(frames, ignore_index=True)

    # Define the desired column order (columns not present stay as-is at the end)
    desired_columns = [
        "Patient_ID", "CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER",
        # VEP CSQ fields
        "Allele", "Consequence", "IMPACT", "SYMBOL", "Gene", "Feature_type",
        "Feature", "BIOTYPE", "EXON", "INTRON", "HGVSc", "HGVSp",
        "cDNA_position", "CDS_position", "Protein_position", "Amino_acids",
        "Codons", "Existing_variation", "DISTANCE", "STRAND", "FLAGS",
        "SYMBOL_SOURCE", "HGNC_ID", "SOURCE",
        "gnomADg", "gnomADg_AF",
        "gnomADg_AF_afr", "gnomADg_AF_ami", "gnomADg_AF_amr",
        "gnomADg_AF_asj", "gnomADg_AF_eas", "gnomADg_AF_fin",
        "gnomADg_AF_nfe", "gnomADg_AF_sas",
        # FORMAT / sample fields
        "GT", "GQ", "DP", "AD", "VAF", "PL",
    ]
    # Keep only the columns that actually exist in the dataframe
    ordered = [c for c in desired_columns if c in df.columns]
    extra   = [c for c in df.columns if c not in desired_columns]
    df = df[ordered + extra]

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run from the project root: python src/parse_vcf.py
    parsing_dir = Path(__file__).parent
    script_dir = parsing_dir.parent
    data_dir   = script_dir.parent / "data"
    output_dir = script_dir.parent / "output"
    output_dir.mkdir(exist_ok=True)

    print(f"Scanning for VCF files in: {data_dir.resolve()}")
    df = parse_all_vcfs(data_dir)

    print(f"Parsed {len(df)} variant-transcript rows from {df['Patient_ID'].nunique()} patient(s).")
    print(df.head())

    out_path = output_dir / "variants.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")
