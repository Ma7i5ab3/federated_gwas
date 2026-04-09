"""
parse_vcf.py
------------
Parse VEP-annotated VCF files (Sarek / DeepVariant output) from data/*.vcf
and produce a Parquet file per patient with one row per variant per transcript.

Columns produced
----------------
Standard VCF fields : Patient_ID, CHROM, POS, ID, REF, ALT, QUAL, FILTER
VEP CSQ sub-fields  : Allele … gnomADg_AF_sas  (parsed from the INFO/CSQ tag)
Sample FORMAT fields: GT, GQ, DP, AD, VAF, PL

Memory model
------------
Rows are buffered in memory up to CHUNK_SIZE, then flushed to the Parquet file
via a PyArrow ParquetWriter. Peak memory per file is bounded to
~CHUNK_SIZE × row_size regardless of VCF size.
"""

import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


CHUNK_SIZE = 10_000  # rows to buffer before flushing to disk

DESIRED_COLUMNS = [
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
    csq_match = re.search(r"CSQ=([^;]+)", info_str)
    if not csq_match:
        return [{f: None for f in csq_fields}]

    csq_raw = csq_match.group(1)
    transcripts = csq_raw.split(",")

    records = []
    for transcript in transcripts:
        values = transcript.split("|")
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


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the desired column order to a DataFrame chunk."""
    ordered = [c for c in DESIRED_COLUMNS if c in df.columns]
    extra   = [c for c in df.columns if c not in DESIRED_COLUMNS]
    return df[ordered + extra]


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_vcf(vcf_path: str | Path, output_path: str | Path,
              chunk_size: int = CHUNK_SIZE) -> int:
    """
    Parse a single VCF file and write a Parquet file incrementally.

    Rows are buffered in memory up to *chunk_size*, then flushed to disk.
    Peak memory is bounded to chunk_size × row_size, not the full file size.

    Parameters
    ----------
    vcf_path    : path to the input .vcf file
    output_path : path for the output .parquet file (overwritten if exists)
    chunk_size  : number of rows to buffer before flushing (default: 10 000)

    Returns
    -------
    Total number of rows written.
    """
    vcf_path    = Path(vcf_path)
    output_path = Path(output_path)
    patient_id  = vcf_path.stem

    csq_fields: list[str] = []
    buffer: list[dict] = []
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    def _flush(buf: list[dict]) -> None:
        nonlocal writer
        df    = _reorder_columns(pd.DataFrame(buf))
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

    with vcf_path.open("r") as fh:
        for line in fh:
            line = line.rstrip("\n")

            if line.startswith("##"):
                if "ID=CSQ" in line:
                    csq_fields = _parse_csq_header(line)
                continue

            if line.startswith("#CHROM"):
                continue

            parts = line.split("\t")
            if len(parts) < 9:
                continue

            chrom, pos, vid, ref, alt, qual, filt, info, fmt = parts[:9]
            sample = parts[9] if len(parts) > 9 else ""

            csq_records = _parse_info_csq(info, csq_fields)
            fmt_dict    = _parse_format_sample(fmt, sample)

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
                row.update(csq)
                row.update(fmt_dict)
                buffer.append(row)

            if len(buffer) >= chunk_size:
                _flush(buffer)
                total_rows += len(buffer)
                buffer.clear()

    # Flush any remaining rows
    if buffer:
        _flush(buffer)
        total_rows += len(buffer)

    if writer is not None:
        writer.close()

    return total_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_all_vcfs(data_dir: str | Path = "data",
                   output_dir: str | Path = "output/parsed",
                   chunk_size: int = CHUNK_SIZE) -> None:
    """
    Parse all .vcf files in *data_dir* and write one Parquet file per patient
    to *output_dir*.

    Patients are processed one at a time; no DataFrames are kept in memory
    between files.

    Parameters
    ----------
    data_dir   : directory containing the .vcf files (default: "data/")
    output_dir : directory for the output .parquet files
    chunk_size : rows to buffer per flush (default: CHUNK_SIZE)
    """
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vcf_files = sorted(data_dir.glob("*.vcf"))
    if not vcf_files:
        raise FileNotFoundError(f"No .vcf files found in {data_dir.resolve()}")

    for vcf_file in vcf_files:
        out_path = output_dir / f"{vcf_file.stem}.parquet"
        print(f"Parsing {vcf_file.name} -> {out_path.name} ...", flush=True)
        n = parse_vcf(vcf_file, out_path, chunk_size=chunk_size)
        print(f"  {n:,} rows written.", flush=True)


if __name__ == "__main__":
    parsing_dir = Path(__file__).parent
    script_dir  = parsing_dir.parent
    data_dir    = script_dir.parent / "data"
    output_dir  = script_dir.parent / "output" / "parsed"

    print(f"Scanning for VCF files in: {data_dir.resolve()}")
    parse_all_vcfs(data_dir, output_dir)
    print("Done.")
