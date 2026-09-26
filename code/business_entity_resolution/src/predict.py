"""
predict.py — score test candidates with the trained model and write the
submission file (output/matching_results.tsv).

Reads candidate_pairs (score-sorted), joins with cleaned record fields, computes
the same features as build_training (base + relative), scores each candidate with
the trained XGBoost model, keeps those above the decision threshold, and writes
one row per Source-1 entity (empty when nothing clears the bar — singletons).

Because the F0.5-optimal threshold is high (~0.99) the model only accepts
strong, top-ranked candidates, so `--cand-cap` limits featurisation to the top-K
candidates per S1 (they are written in descending score order) with negligible
recall loss and a large speed-up.

Usage:
    python .../predict.py --candidates output/candidate_pairs.tsv \
        --split test --model models_rel/xgb.json \
        --out output/matching_results.tsv --threshold 0.99 --cand-cap 100
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import prep, pair_features, compute_relative, ALL_FEATURE_NAMES  # noqa: E402

PROC = "data/processed"
COLS = ["entity_id", "name_canon", "name_core", "addr_canon", "street_number",
        "state", "country", "is_native", "is_domain", "name_legal"]


def load_table(src):
    t = pq.read_table(f"{PROC}/{src}.parquet", columns=COLS)
    idnum = np.array([int(e.split('-', 1)[1]) for e in t.column("entity_id").to_pylist()],
                     dtype=np.uint64)
    return t, {int(v): i for i, v in enumerate(idnum)}


def prep_rows(table, indices, s3):
    sub = table.take(pa.array(indices)).to_pydict()
    return [prep(sub["name_canon"][i], sub["name_core"][i], sub["addr_canon"][i],
                 sub["street_number"][i], sub["state"][i], sub["country"][i],
                 sub["is_native"][i], sub["is_domain"][i], sub["name_legal"][i], s3)
            for i in range(len(indices))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--cand-cap", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4000)
    args = ap.parse_args()
    t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

    import xgboost as xgb
    bst = xgb.Booster(); bst.load_model(args.model)
    log(f"model loaded ({args.model})")

    t1, row1 = load_table(f"{args.split}_source1")
    t2, row2 = load_table(f"{args.split}_source2")
    t3, row3 = load_table(f"{args.split}_source3")
    log(f"tables: S1={t1.num_rows:,} S2={t2.num_rows:,} S3={t3.num_rows:,}")

    def core_set(name_core):
        return frozenset(name_core.split()) if name_core else frozenset()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = open(args.out, "w", encoding="utf-8")
    out.write("source1_entity_id\tmatched_entity_ids\n")

    chunk = []   # (s1, s1num, cands)
    n_written = n_match = 0

    def process_chunk():
        nonlocal n_written, n_match
        if not chunk:
            return
        cap = args.cand_cap
        # cheap core sets for ALL candidates (relative features must match training,
        # which computed them over the FULL candidate set).
        r2a = [int(c.split('-', 1)[1]) for _, _, cands in chunk for c in cands if c[1] == '2']
        r3a = [int(c.split('-', 1)[1]) for _, _, cands in chunk for c in cands if c[1] == '3']
        u2 = sorted(set(row2[n] for n in r2a)); u3 = sorted(set(row3[n] for n in r3a))
        sub2 = t2.take(pa.array(u2)).to_pydict() if u2 else {"name_core": [], "addr_canon": []}
        sub3 = t3.take(pa.array(u3)).to_pydict() if u3 else {"name_core": [], "addr_canon": []}
        core2 = {r: core_set(v) for r, v in zip(u2, sub2["name_core"])}
        core3 = {r: core_set(v) for r, v in zip(u3, sub3["name_core"])}
        addr2 = {r: core_set(v) for r, v in zip(u2, sub2["addr_canon"])}
        addr3 = {r: core_set(v) for r, v in zip(u3, sub3["addr_canon"])}
        # full prep for S1 + only the TOP-`cap` candidates we actually score
        need1 = sorted(set(row1[s1num] for _, s1num, _ in chunk))
        sc2 = sorted(set(row2[int(c.split('-', 1)[1])]
                     for _, _, cands in chunk for c in cands[:cap] if c[1] == '2'))
        sc3 = sorted(set(row3[int(c.split('-', 1)[1])]
                     for _, _, cands in chunk for c in cands[:cap] if c[1] == '3'))
        P1 = dict(zip(need1, prep_rows(t1, need1, False)))
        P2 = dict(zip(sc2, prep_rows(t2, sc2, False))) if sc2 else {}
        P3 = dict(zip(sc3, prep_rows(t3, sc3, True))) if sc3 else {}

        feats = []; index = []  # index: (chunk_pos, cand_str)
        for ci, (s1, s1num, cands) in enumerate(chunk):
            a = P1[row1[s1num]]
            cand_cores, cand_addrs, cand_s3 = [], [], []
            for c in cands:
                num = int(c.split('-', 1)[1])
                if c[1] == '2':
                    cand_cores.append(core2.get(row2[num], frozenset()))
                    cand_addrs.append(addr2.get(row2[num], frozenset())); cand_s3.append(False)
                else:
                    cand_cores.append(core3.get(row3[num], frozenset()))
                    cand_addrs.append(addr3.get(row3[num], frozenset())); cand_s3.append(True)
            rel = compute_relative(a["core"], cand_cores, cand_addrs, cand_s3)  # over FULL set
            for i, c in enumerate(cands[:cap]):                     # score only top-cap
                num = int(c.split('-', 1)[1])
                b = P2[row2[num]] if c[1] == '2' else P3[row3[num]]
                feats.append(pair_features(a, b) + list(rel[i]))
                index.append((ci, c))
        # score all pairs in this chunk at once
        matched = {ci: [] for ci in range(len(chunk))}
        if feats:
            dm = xgb.DMatrix(np.asarray(feats, dtype=np.float32), feature_names=ALL_FEATURE_NAMES)
            scores = bst.predict(dm)
            for (ci, c), sc in zip(index, scores):
                if sc >= args.threshold:
                    matched[ci].append(c)
        for ci, (s1, s1num, cands) in enumerate(chunk):
            ids = matched[ci]
            out.write(s1 + "\t" + ",".join(ids) + "\n")
            n_written += 1; n_match += len(ids)
        chunk.clear()

    with open(args.candidates, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            s1num = int(s1.split('-', 1)[1])
            # keep the FULL candidate list (relative features are computed over it);
            # only the top `cand-cap` are actually featurised/scored in process_chunk.
            cands = [c for c in rest.split(",") if c and
                     (row2.get(int(c.split('-', 1)[1])) if c[1] == '2'
                      else row3.get(int(c.split('-', 1)[1]))) is not None] if rest.strip() else []
            chunk.append((s1, s1num, cands))
            if len(chunk) >= args.batch:
                process_chunk()
                if n_written % (args.batch * 20) < args.batch:
                    log(f"  {n_written:,} S1 done, {n_match:,} matches")
    process_chunk()
    out.close()
    log(f"DONE wrote {args.out}: {n_written:,} S1 rows, {n_match:,} total matches "
        f"({n_match/max(1,n_written):.2f}/S1)")


if __name__ == "__main__":
    main()
