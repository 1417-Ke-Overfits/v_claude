"""
clean_dataset.py — stream a raw source .tsv into a cleaned Parquet table.

Applies :mod:`normalize` to every record and writes canonical fields + the
pieces needed to build blocking keys and matching features downstream. Runs a
multiprocessing pool over line chunks (string normalization is CPU-bound and
GIL-heavy), writing one Parquet row-group per chunk so memory stays flat.

Usage (from the student_resource/ project root):

    python code/business_entity_resolution/src/clean_dataset.py \
        --input dataset/train/train_source2.tsv \
        --output data/processed/train_source2.parquet \
        --source 2

Or clean everything at once:

    python code/business_entity_resolution/src/clean_dataset.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import clean_name, clean_address, is_domain_name  # noqa: E402

CHUNK = 50_000
NONASCII = None  # set in _init

SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("country", pa.string()),
    ("raw_name", pa.string()),
    ("raw_addr", pa.string()),
    ("name_canon", pa.string()),
    ("name_core", pa.string()),     # space-joined SORTED core tokens
    ("name_legal", pa.string()),    # space-joined legal tokens
    ("addr_canon", pa.string()),    # space-joined SORTED address tokens
    ("street_number", pa.string()),
    ("state", pa.string()),
    ("is_domain", pa.bool_()),
    ("is_native", pa.bool_()),      # name contained non-ASCII (native script)
])


def _init():
    global NONASCII
    import re
    NONASCII = re.compile(r"[^\x00-\x7F]")


def _process_chunk(lines):
    """Normalize a list of raw TSV lines -> column-oriented dict of lists."""
    cols = {k: [] for k in SCHEMA.names}
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            parts = (parts + ["", "", "", ""])[:4]
        eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
        canon, core, legal = clean_name(name)
        acanon, atoks, snum, state = clean_address(addr, country)
        cols["entity_id"].append(eid)
        cols["country"].append(country)
        cols["raw_name"].append(name)
        cols["raw_addr"].append(addr)
        cols["name_canon"].append(canon)
        cols["name_core"].append(" ".join(sorted(core)))
        cols["name_legal"].append(" ".join(sorted(legal)))
        cols["addr_canon"].append(acanon)
        cols["street_number"].append(snum)
        cols["state"].append(state)
        cols["is_domain"].append(is_domain_name(name))
        cols["is_native"].append(bool(NONASCII.search(name)))
    return cols


def _chunks(path):
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        buf = []
        for line in f:
            buf.append(line)
            if len(buf) >= CHUNK:
                yield buf
                buf = []
        if buf:
            yield buf


def clean_file(inp, out, workers):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    n = 0
    writer = pq.ParquetWriter(out, SCHEMA, compression="zstd")
    with Pool(workers, initializer=_init) as pool:
        for cols in pool.imap(_process_chunk, _chunks(inp), chunksize=1):
            batch = pa.record_batch([pa.array(cols[k], type=SCHEMA.field(k).type)
                                     for k in SCHEMA.names], schema=SCHEMA)
            writer.write_batch(batch)
            n += batch.num_rows
            if n % 500_000 < CHUNK:
                print(f"  {os.path.basename(inp)}: {n:,} rows "
                      f"({n/(time.time()-t0):,.0f}/s)", flush=True)
    writer.close()
    dt = time.time() - t0
    sz = os.path.getsize(out) / 1e6
    print(f"DONE {os.path.basename(inp)} -> {out}: {n:,} rows in {dt:.1f}s, "
          f"{sz:.0f} MB", flush=True)


ALL_FILES = [
    ("dataset/train/train_source1.tsv", "data/processed/train_source1.parquet"),
    ("dataset/train/train_source2.tsv", "data/processed/train_source2.parquet"),
    ("dataset/train/train_source3.tsv", "data/processed/train_source3.parquet"),
    ("dataset/test/test_source1.tsv", "data/processed/test_source1.parquet"),
    ("dataset/test/test_source2.tsv", "data/processed/test_source2.parquet"),
    ("dataset/test/test_source3.tsv", "data/processed/test_source3.parquet"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--source", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    if args.all:
        for inp, out in ALL_FILES:
            if os.path.isfile(inp):
                clean_file(inp, out, args.workers)
            else:
                print(f"SKIP (missing): {inp}")
    else:
        if not (args.input and args.output):
            ap.error("provide --input and --output, or --all")
        clean_file(args.input, args.output, args.workers)


if __name__ == "__main__":
    main()
