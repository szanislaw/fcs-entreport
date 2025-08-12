#!/usr/bin/env python3
"""
CSV -> SQL converter with simple type inference and batched INSERTs.

Usage:
  python csv_to_sql.py \
    --input "Job Detail Listing (Cross Day)_DR_20250811_111137.csv" \
    --output job_detail_listing.sql \
    --table job_detail_listing \
    --dialect sqlite \
    --sample-rows 5000 \
    --batch-size 500 \
    --dayfirst true
"""

import argparse
import csv
import math
import os
import re
import sys
from datetime import datetime

# Optional date parsing (recommended): pip install python-dateutil
try:
    from dateutil import parser as dateparser  # type: ignore
except Exception:
    dateparser = None

# ------------------------
# Identifier helpers
# ------------------------
SQL_RESERVED = set([
    # Minimal set; add more if needed
    "select","from","where","group","order","by","insert","update","delete",
    "into","values","table","create","drop","null","and","or","not","in","as"
])

def sanitize_identifier(s: str) -> str:
    s = s.strip().replace("\ufeff", "")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    if not s or s.lower() in SQL_RESERVED:
        s = f"col_{s}" if s else "col_unnamed"
    if re.match(r"^\d", s):  # cannot start with digit
        s = f"col_{s}"
    return s.lower()

def dedupe_names(names):
    seen = {}
    out = []
    for n in names:
        base = n
        i = 1
        while n in seen:
            i += 1
            n = f"{base}_{i}"
        seen[n] = True
        out.append(n)
    return out

def quote_ident(ident: str, dialect: str) -> str:
    if dialect == "mysql":
        return "`" + ident.replace("`", "``") + "`"
    # SQLite/Postgres default: double quotes
    return '"' + ident.replace('"', '""') + '"'

# ------------------------
# Type detection
# ------------------------
def is_bool(s: str) -> bool:
    t = s.strip().lower()
    return t in ("true","false","yes","no","y","n","1","0")

def bool_to_int(s: str):
    t = s.strip().lower()
    if t in ("true","yes","y","1"): return 1
    if t in ("false","no","n","0"): return 0
    return None

def is_int(s: str) -> bool:
    t = s.strip()
    if not t: return False
    if re.fullmatch(r"[+-]?\d+", t): return True
    return False

def is_float(s: str) -> bool:
    t = s.strip().replace(",", "")
    if not t: return False
    try:
        float(t)
        # exclude pure integers
        return not re.fullmatch(r"[+-]?\d+", t)
    except Exception:
        return False

def datetime_kind(s: str, dayfirst: bool):
    """Return 'DATE', 'DATETIME', or None."""
    if not s or not s.strip(): return None
    if dateparser is None:
        return None  # no parser available
    try:
        dt = dateparser.parse(s.strip(), dayfirst=dayfirst, fuzzy=False)
        # heuristic: if time components present or explicit time-like chars
        if (dt.hour or dt.minute or dt.second) or re.search(r"\d:\d", s):
            return "DATETIME"
        return "DATE"
    except Exception:
        return None

def infer_type(values, dayfirst: bool):
    """Infer a column type from a list of sample strings (non-empty)."""
    n = len(values)
    if n == 0:
        return "TEXT"

    def ratio(cnt): return cnt / n if n else 0.0

    b_cnt = sum(1 for v in values if is_bool(v))
    if ratio(b_cnt) >= 0.95:
        return "BOOLEAN"

    i_cnt = sum(1 for v in values if is_int(v))
    if ratio(i_cnt) >= 0.95:
        return "INTEGER"

    f_cnt = sum(1 for v in values if is_float(v) or is_int(v))
    if ratio(f_cnt) >= 0.95:
        return "REAL"

    if dateparser is not None:
        kinds = [datetime_kind(v, dayfirst) for v in values]
        dt_cnt = sum(1 for k in kinds if k is not None)
        if ratio(dt_cnt) >= 0.85:
            dtime_detail = sum(1 for k in kinds if k == "DATETIME")
            return "DATETIME" if dtime_detail >= dt_cnt / 2 else "DATE"

    return "TEXT"

def sql_literal(val, col_type: str, dialect: str, dayfirst: bool):
    if val is None:
        return "NULL"
    s = str(val)
    if s.strip() == "" or s.strip().lower() in ("nan","none","null"):
        return "NULL"

    if col_type == "INTEGER":
        try:
            return str(int(float(s.replace(",", ""))))
        except Exception:
            return "NULL"

    if col_type == "REAL":
        try:
            return str(float(s.replace(",", "")))
        except Exception:
            return "NULL"

    if col_type == "BOOLEAN":
        b = bool_to_int(s)
        return "1" if b == 1 else "0" if b == 0 else "NULL"

    if col_type in ("DATE","DATETIME") and dateparser is not None:
        try:
            dt = dateparser.parse(s.strip(), dayfirst=dayfirst, fuzzy=False)
            if col_type == "DATE":
                iso = dt.date().isoformat()
            else:
                iso = dt.strftime("%Y-%m-%d %H:%M:%S")
            return "'" + iso.replace("'", "''") + "'"
        except Exception:
            # fall back to text if parsing fails
            pass

    # TEXT default (escape single quotes)
    return "'" + s.replace("'", "''") + "'"

# ------------------------
# CSV reading & inference
# ------------------------
def sniff_dialect(path, has_header=True):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(4096)
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        dialect.skipinitialspace = True
        return dialect
    except Exception:
        # fallback to comma
        class D(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
            skipinitialspace = True
        return D

def read_header(path, dialect):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, dialect=dialect)
        header = next(reader, None)
    if header is None:
        raise ValueError("CSV appears to be empty.")
    return header

def sample_for_inference(path, dialect, sample_rows: int):
    header = read_header(path, dialect)
    col_samples = [[] for _ in header]
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            for i, h in enumerate(header):
                v = row.get(h, "")
                if v is not None and str(v).strip() != "":
                    col_samples[i].append(v)
            count += 1
            if count >= sample_rows:
                break
    return header, col_samples

# ------------------------
# Main conversion
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="Convert CSV to SQL with type inference.")
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--output", required=True, help="Path to output .sql")
    ap.add_argument("--table", required=False, help="Destination table name")
    ap.add_argument("--dialect", choices=["sqlite","postgres","mysql"], default="sqlite",
                    help="SQL dialect for quoting/booleans")
    ap.add_argument("--sample-rows", type=int, default=5000, help="Rows to sample for type inference")
    ap.add_argument("--batch-size", type=int, default=500, help="INSERT VALUES batch size")
    ap.add_argument("--dayfirst", type=str, default="true", help="Interpret ambiguous dates as day-first (true/false)")
    args = ap.parse_args()

    dayfirst = str(args.dayfirst).lower() in ("1","true","t","yes","y")

    # Sniff CSV dialect
    csvdialect = sniff_dialect(args.input)

    # Read header & infer names
    raw_header = read_header(args.input, csvdialect)
    sanitized = [sanitize_identifier(h) for h in raw_header]
    sanitized = dedupe_names(sanitized)

    # Choose table name
    if args.table:
        table_name = sanitize_identifier(args.table)
    else:
        base = os.path.basename(args.input)
        base = re.sub(r"\.[^.]*$", "", base)
        table_name = sanitize_identifier(base) or "imported_table"

    # Sample to infer types
    header, samples = sample_for_inference(args.input, csvdialect, args.sample_rows)
    inferred_types = []
    for i, col in enumerate(header):
        vals = samples[i]
        vals = [v for v in vals if v is not None and str(v).strip() != ""]
        inferred_types.append(infer_type(vals, dayfirst) if vals else "TEXT")

    # Write SQL
    with open(args.output, "w", encoding="utf-8") as out:
        # CREATE TABLE
        cols_def = []
        for name, typ in zip(sanitized, inferred_types):
            # Postgres prefers BOOLEAN; SQLite accepts it too; MySQL uses TINYINT(1) commonly
            if typ == "BOOLEAN" and args.dialect == "mysql":
                typ_sql = "TINYINT(1)"
            else:
                typ_sql = typ
            cols_def.append(f"{quote_ident(name, args.dialect)} {typ_sql}")
        create_stmt = f"CREATE TABLE {quote_ident(table_name, args.dialect)} (\n  " + ",\n  ".join(cols_def) + "\n);\n"
        out.write(create_stmt)

        # INSERTs (stream rows; no full-file load)
        with open(args.input, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, dialect=csvdialect)
            cols_q = ", ".join(quote_ident(c, args.dialect) for c in sanitized)
            batch = []
            def flush_batch():
                if not batch: return
                out.write(f"INSERT INTO {quote_ident(table_name, args.dialect)} ({cols_q}) VALUES\n  ")
                out.write(",\n  ".join(batch))
                out.write(";\n")
                batch.clear()

            for row in reader:
                vals = []
                for raw_name, name, typ in zip(raw_header, sanitized, inferred_types):
                    v = row.get(raw_name, None)
                    vals.append(sql_literal(v, typ, args.dialect, dayfirst))
                batch.append("(" + ", ".join(vals) + ")")
                if len(batch) >= args.batch_size:
                    flush_batch()
            flush_batch()

    print(f"Done. Wrote SQL to: {args.output}")
    print(f"Table: {table_name}")
    print("Column types:")
    for n, t in zip(sanitized, inferred_types):
        print(f"  - {n}: {t}")

if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Handle piping to tools like `head`
        try:
            sys.stderr.close()
        except Exception:
            pass
        try:
            sys.stdout.close()
        except Exception:
            pass
