"""
build_training.py — build a labeled feature matrix from candidate pairs.

Joins candidate pairs (from generate_candidates) with cleaned record fields,
labels each pair (positive iff in ground truth), featurizes with features.py,
and writes Parquet. Entities are split train/val by a hash of the S1 id (no S1
leaks across the split). Training rows subsample negatives (candidates are ~1%
positive); validation rows keep ALL candidates so the F0.5 threshold is tuned on
a realistic base rate.

Usage:
    python .../build_training.py \
        --candidates output/candidate_pairs_train.tsv \
        --gt dataset/train/train_ground_truth.tsv \
        --out-train data/interim/train_feats.parquet \
        --out-val   data/interim/val_feats.parquet \
        --n-train-s1 300000 --n-val-s1 60000 --neg-per-s1 30
"""
from __future__ import annotations
import argparse, os, sys, time, random
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import prep, pair_features, FEATURE_NAMES  # noqa: E402

PROC = "data/processed"
COLS = ["entity_id", "name_canon", "name_core", "addr_canon", "street_number",
        "state", "country", "is_native", "is_domain", "name_legal"]
random.seed(0)


def load_table(src):
    t = pq.read_table(f"{PROC}/{src}.parquet", columns=COLS)
    idnum = np.array([int(e.split('-', 1)[1]) for e in t.column("entity_id").to_pylist()],
                     dtype=np.uint64)
    row = {int(v): i for i, v in enumerate(idnum)}
    return t, row


def prep_rows(table, indices, s3):
    """arrow.take(indices) -> list of prepped dicts (aligned to indices)."""
    sub = table.take(pa.array(indices)).to_pydict()
    out = []
    for i in range(len(indices)):
        out.append(prep(sub["name_canon"][i], sub["name_core"][i],
                        sub["addr_canon"][i], sub["street_number"][i],
                        sub["state"][i], sub["country"][i],
                        sub["is_native"][i], sub["is_domain"][i],
                        sub["name_legal"][i], s3))
    return out


def val_hash(idnum, mod=5):
    return (idnum * 2654435761) % mod == 0   # ~20% -> validation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--n-train-s1", type=int, default=300000)
    ap.add_argument("--n-val-s1", type=int, default=60000)
    ap.add_argument("--neg-per-s1", type=int, default=30)
    ap.add_argument("--max-s1", type=int, default=0, help="debug: cap S1 scanned")
    args = ap.parse_args()
    t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

    t1, row1 = load_table("train_source1")
    t2, row2 = load_table("train_source2")
    t3, row3 = load_table("train_source3")
    log(f"loaded tables S1={t1.num_rows:,} S2={t2.num_rows:,} S3={t3.num_rows:,}")

    # GT positives per S1 id_num
    pos = {}
    with open(args.gt, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            pos[int(s1.split('-', 1)[1])] = set(rest.split(",")) if rest.strip() else set()
    log(f"gt entities {len(pos):,}")

    schema = pa.schema([("s1", pa.string()), ("cand", pa.string()),
                        ("label", pa.int8())] +
                       [(n, pa.float32()) for n in FEATURE_NAMES])
    wtr_tr = pq.ParquetWriter(args.out_train, schema, compression="zstd")
    wtr_va = pq.ParquetWriter(args.out_val, schema, compression="zstd")

    n_tr_s1 = n_va_s1 = 0
    buf = {"train": [], "val": []}
    BUF = 300_000

    def flush(kind, writer):
        rows = buf[kind]
        if not rows:
            return
        cols = list(zip(*rows))
        arrays = [pa.array(cols[0]), pa.array(cols[1]), pa.array(cols[2], type=pa.int8())]
        for j in range(len(FEATURE_NAMES)):
            arrays.append(pa.array(cols[3 + j], type=pa.float32()))
        writer.write_table(pa.table(arrays, schema=schema))
        buf[kind] = []

    def prep_unique(table, rows_needed, s3):
        """Take unique row indices once, prep once -> {row: prepped}."""
        uniq = sorted(set(rows_needed))
        preps = prep_rows(table, uniq, s3) if uniq else []
        return {r: p for r, p in zip(uniq, preps)}

    # Accumulate a chunk of S1 (with their emit lists), then batch-take & prep.
    CHUNK = 4000
    chunk = []  # list of (s1, s1num, is_val, gtset, emit_ok, srcs)

    def process_chunk():
        if not chunk:
            return
        need1, need2, need3 = [], [], []
        for _, s1num, _, _, emit_ok, _ in chunk:
            need1.append(row1[s1num])
            for c in emit_ok:
                (need2 if c[1] == '2' else need3).append(int(c.split('-', 1)[1]))
        # need2/need3 hold id_nums -> convert to rows
        rows2 = [row2[n] for n in need2]; rows3 = [row3[n] for n in need3]
        P1 = prep_unique(t1, need1, False)
        P2 = prep_unique(t2, rows2, False)
        P3 = prep_unique(t3, rows3, True)
        for s1, s1num, is_val, gtset, emit_ok, _ in chunk:
            a = P1[row1[s1num]]
            kind = "val" if is_val else "train"
            for c in emit_ok:
                num = int(c.split('-', 1)[1])
                b = P2[row2[num]] if c[1] == '2' else P3[row3[num]]
                lab = 1 if c in gtset else 0
                buf[kind].append((s1, c, lab, *pair_features(a, b)))
        if len(buf["train"]) >= BUF:
            flush("train", wtr_tr)
        if len(buf["val"]) >= BUF:
            flush("val", wtr_va)
        chunk.clear()

    scanned = 0
    with open(args.candidates, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            s1num = int(s1.split('-', 1)[1])
            if row1.get(s1num) is None or not rest.strip():
                continue
            cands = [c for c in rest.split(",") if c]
            if not cands:
                continue
            is_val = val_hash(s1num)
            if is_val and n_va_s1 >= args.n_val_s1:
                continue
            if not is_val and n_tr_s1 >= args.n_train_s1:
                continue
            gtset = pos.get(s1num, set())
            if is_val:
                emit = cands
            else:
                p = [c for c in cands if c in gtset]
                n = [c for c in cands if c not in gtset]
                random.shuffle(n)
                emit = p + n[:args.neg_per_s1]
            emit_ok = [c for c in emit
                       if (row2.get(int(c.split('-', 1)[1])) if c[1] == '2'
                           else row3.get(int(c.split('-', 1)[1]))) is not None]
            if not emit_ok:
                continue
            chunk.append((s1, s1num, is_val, gtset, emit_ok, None))
            if is_val:
                n_va_s1 += 1
            else:
                n_tr_s1 += 1
            scanned += 1
            if len(chunk) >= CHUNK:
                process_chunk()
                log(f"  scanned {scanned:,} (train_s1={n_tr_s1:,} val_s1={n_va_s1:,})")
            if args.max_s1 and scanned >= args.max_s1:
                break
            if n_tr_s1 >= args.n_train_s1 and n_va_s1 >= args.n_val_s1:
                break
    process_chunk()
    flush("train", wtr_tr); flush("val", wtr_va)
    wtr_tr.close(); wtr_va.close()
    log(f"DONE train_s1={n_tr_s1:,} val_s1={n_va_s1:,} -> {args.out_train}, {args.out_val}")


if __name__ == "__main__":
    main()
