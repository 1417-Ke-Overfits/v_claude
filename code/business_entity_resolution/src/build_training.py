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

    from features import compute_relative, ALL_FEATURE_NAMES, REL_FEATURE_NAMES
    schema = pa.schema([("s1", pa.string()), ("cand", pa.string()),
                        ("label", pa.int8())] +
                       [(n, pa.float32()) for n in ALL_FEATURE_NAMES])
    NF = len(ALL_FEATURE_NAMES)
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
        for j in range(NF):
            arrays.append(pa.array(cols[3 + j], type=pa.float32()))
        writer.write_table(pa.table(arrays, schema=schema))
        buf[kind] = []

    def core_set(name_core):
        return frozenset(name_core.split()) if name_core else frozenset()

    CHUNK = 4000
    chunk = []  # (s1, s1num, is_val, gtset, cands_all)

    def process_chunk():
        if not chunk:
            return
        # --- gather ALL candidate rows for cheap core sets (relative features) ---
        rows2_all, rows3_all = [], []
        for _, _, _, _, cands in chunk:
            for c in cands:
                (rows2_all if c[1] == '2' else rows3_all).append(int(c.split('-', 1)[1]))
        u2 = sorted(set(row2[n] for n in rows2_all))
        u3 = sorted(set(row3[n] for n in rows3_all))
        sub2 = t2.take(pa.array(u2)).to_pydict() if u2 else {"name_core": [], "addr_canon": []}
        sub3 = t3.take(pa.array(u3)).to_pydict() if u3 else {"name_core": [], "addr_canon": []}
        core2 = {r: core_set(v) for r, v in zip(u2, sub2["name_core"])}
        core3 = {r: core_set(v) for r, v in zip(u3, sub3["name_core"])}
        addr2 = {r: core_set(v) for r, v in zip(u2, sub2["addr_canon"])}
        addr3 = {r: core_set(v) for r, v in zip(u3, sub3["addr_canon"])}

        # --- decide WRITTEN candidates and full-prep only those ---
        need1 = [row1[s1num] for _, s1num, _, _, _ in chunk]
        writes = []  # per S1: list of (cand, label, full_b_key)
        need2w, need3w = [], []
        for s1, s1num, is_val, gtset, cands in chunk:
            if is_val:
                sel = cands
            else:
                p = [c for c in cands if c in gtset]
                ng = [c for c in cands if c not in gtset]
                if len(ng) > args.neg_per_s1:
                    ng = random.sample(ng, args.neg_per_s1)
                sel = p + ng
            sset = set(sel)
            writes.append((s1, s1num, is_val, gtset, cands, sset))
            for c in sel:
                num = int(c.split('-', 1)[1])
                (need2w if c[1] == '2' else need3w).append(num)
        P1 = {r: p for r, p in zip(sorted(set(need1)),
              prep_rows(t1, sorted(set(need1)), False))} if need1 else {}
        rw2 = sorted(set(row2[n] for n in need2w)); rw3 = sorted(set(row3[n] for n in need3w))
        P2 = {r: p for r, p in zip(rw2, prep_rows(t2, rw2, False))} if rw2 else {}
        P3 = {r: p for r, p in zip(rw3, prep_rows(t3, rw3, True))} if rw3 else {}

        for s1, s1num, is_val, gtset, cands, sset in writes:
            a = P1[row1[s1num]]
            # relative features over ALL candidates (cheap core + addr sets)
            cand_cores, cand_addrs, cand_s3 = [], [], []
            for c in cands:
                num = int(c.split('-', 1)[1])
                if c[1] == '2':
                    cand_cores.append(core2.get(row2[num], frozenset()))
                    cand_addrs.append(addr2.get(row2[num], frozenset())); cand_s3.append(False)
                else:
                    cand_cores.append(core3.get(row3[num], frozenset()))
                    cand_addrs.append(addr3.get(row3[num], frozenset())); cand_s3.append(True)
            rel = compute_relative(a["core"], cand_cores, cand_addrs, cand_s3)
            kind = "val" if is_val else "train"
            for i, c in enumerate(cands):
                if c not in sset:
                    continue
                num = int(c.split('-', 1)[1])
                b = P2[row2[num]] if c[1] == '2' else P3[row3[num]]
                lab = 1 if c in gtset else 0
                buf[kind].append((s1, c, lab, *pair_features(a, b), *rel[i]))
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
            cands = [c for c in rest.split(",") if c and
                     (row2.get(int(c.split('-', 1)[1])) if c[1] == '2'
                      else row3.get(int(c.split('-', 1)[1]))) is not None]
            if not cands:
                continue
            is_val = val_hash(s1num)
            if is_val and n_va_s1 >= args.n_val_s1:
                continue
            if not is_val and n_tr_s1 >= args.n_train_s1:
                continue
            chunk.append((s1, s1num, is_val, pos.get(s1num, set()), cands))
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
