"""
generate_candidates.py — scalable candidate generation (blocking v2).

Given cached blocking keys (from precompute_keys), build an IDF-weighted
sparse key matrix and, for each Source-1 entity, retrieve the top-N Source-2/3
records by summed IDF of shared keys — a BM25-style candidate ranking. Uses
batched sparse matrix multiplication (scipy) so it scales to millions of
entities in minutes instead of hours.

Rule (chosen empirically — see docs/09_blocking_v2.md):
  * keys per record: n2, sn, ns, nn (composite/specific) + nt, at (DF-pruned)
  * candidate score(S1, r) = sum over shared kept keys k of IDF(k)
  * keep the top-N candidates per S1 (N configurable; default 300)

IDF weighting is essential: a shared *rare* key (an exact street#+state, a rare
name token) is far more indicative than a shared common one, so ranking by
summed IDF puts true matches near the top and lets a modest N retain most recall.

Usage:
    python .../generate_candidates.py --split train --topn 300 \
        --out output/candidate_pairs_train.tsv [--gt dataset/train/train_ground_truth.tsv]
"""
from __future__ import annotations
import argparse, math, os, time
import numpy as np
import scipy.sparse as sp

INT = "data/interim"

def typ(kh): return kh & 7

def load(src):
    z = np.load(f"{INT}/keys_{src}.npz")
    return z["id_num"], z["off"], z["val"]

def sources_for(split):
    return (f"{split}_source1", f"{split}_source2", f"{split}_source3")

def eid(source_digit, idnum):
    return f"S{source_digit}-{int(idnum)}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--topn", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", default=None, help="ground-truth tsv for recall eval (train)")
    ap.add_argument("--batch", type=int, default=4000)
    ap.add_argument("--nt_cap", type=int, default=10000)
    ap.add_argument("--at_cap", type=int, default=5000)
    ap.add_argument("--ns_cap", type=int, default=50000)
    ap.add_argument("--sn_cap", type=int, default=20000)
    ap.add_argument("--n2_cap", type=int, default=20000)
    args = ap.parse_args()
    t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

    s1n, s2n, s3n = sources_for(args.split)
    id1, off1, val1 = load(s1n)
    id2, off2, val2 = load(s2n)
    id3, off3, val3 = load(s3n)
    M = len(id2) + len(id3)                 # S2+S3 record count (columns of B space)
    n2n = len(id2)
    log(f"loaded: S1={len(id1):,} S2={len(id2):,} S3={len(id3):,}")

    # DF over S2+S3, per-type caps -> kept keys -> column ids + idf
    from collections import Counter
    df = Counter()
    for val in (val2, val3):
        for kh in val: df[kh] += 1
    cap = {1: args.nt_cap, 3: args.at_cap, 5: args.ns_cap, 2: args.sn_cap,
           0: args.n2_cap, 6: 10**9}
    colid = {}; idf = []
    for kh, c in df.items():
        if c <= cap.get(typ(kh), 10**9):
            colid[kh] = len(idf)
            idf.append(math.log(M / c))
    K = len(colid)
    idf = np.asarray(idf, dtype=np.float32)
    log(f"kept keys (columns) K={K:,}")

    # Build B (S23 x K) binary CSR, and remember each column-row's entity id
    def build_matrix(off, val, weighted):
        indptr = np.empty(len(off), dtype=np.int64); indptr[0] = 0
        idx = []; dat = []
        for i in range(len(off) - 1):
            row_cols = set()
            for kh in val[off[i]:off[i+1]]:
                c = colid.get(kh)
                if c is not None: row_cols.add(c)
            for c in row_cols:
                idx.append(c); dat.append(idf[c] if weighted else 1.0)
            indptr[i+1] = len(idx)
        return sp.csr_matrix((np.asarray(dat, dtype=np.float32),
                              np.asarray(idx, dtype=np.int32), indptr),
                             shape=(len(off)-1, K))
    B2 = build_matrix(off2, val2, weighted=False)
    B3 = build_matrix(off3, val3, weighted=False)
    B = sp.vstack([B2, B3], format="csr")     # (M x K)
    BT = B.T.tocsr()                           # (K x M)
    log(f"B built nnz={B.nnz:,}")
    A = build_matrix(off1, val1, weighted=True)  # (S1 x K) idf-weighted
    log(f"A built nnz={A.nnz:,}")

    # code -> entity id (for output)
    def code_eid(g):
        return eid(2, id2[g]) if g < n2n else eid(3, id3[g - n2n])

    # optional GT for recall
    gt = None
    if args.gt:
        s1row = {int(v): i for i, v in enumerate(id1)}
        gt = {}
        with open(args.gt, encoding="utf-8") as f:
            next(f)
            for line in f:
                s1, _, rest = line.rstrip("\n").partition("\t")
                r = s1row.get(int(s1.split('-', 1)[1]))
                if r is None: continue
                gt[r] = set(m for m in rest.split(",") if m) if rest.strip() else set()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    N = args.topn
    true_tot = cov = 0
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("source1_entity_id\tcandidate_entity_ids\n")
        for start in range(0, len(id1), args.batch):
            end = min(start + args.batch, len(id1))
            S = A[start:end] @ BT             # (b x M) csr, entries = shared idf score
            S = S.tocsr()
            for bi in range(end - start):
                r = start + bi
                lo, hi = S.indptr[bi], S.indptr[bi+1]
                cols = S.indices[lo:hi]; scores = S.data[lo:hi]
                if len(cols) > N:
                    part = np.argpartition(scores, len(cols) - N)[-N:]
                    cols = cols[part]
                cand_ids = [code_eid(g) for g in cols]
                out.write(eid(1, id1[r]) + "\t" + ",".join(cand_ids) + "\n")
                if gt is not None:
                    truth = gt.get(r, set())
                    true_tot += len(truth)
                    cov += len(truth & set(cand_ids))
            if start % (args.batch * 25) == 0:
                log(f"  {end:,}/{len(id1):,} S1 done")
    log(f"wrote {args.out}")
    if gt is not None and true_tot:
        log(f"RECALL over all train pairs: {cov/true_tot:.4%} ({cov:,}/{true_tot:,})")

if __name__ == "__main__":
    main()
