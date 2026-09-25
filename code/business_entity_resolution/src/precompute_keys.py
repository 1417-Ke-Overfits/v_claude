"""
precompute_keys.py — cache blocking keys per source as CSR int arrays.

Computes the rich keyset (n2, nt, sn, at, ns, nn) for every record of a split
and stores it compactly so candidate-generation and experiments run fast
(loading arrays instead of recomputing keys). The key-type is encoded in the low
3 bits of each key int so downstream code can DF-prune per type without keeping
the string.

Encoding: key_int = (hash(payload) << 3) | type_id
  type_id: n2=0 nt=1 sn=2 at=3 ns=5 nn=6

Output: data/interim/keys_<split>_source{1,2,3}.npz with
  id_num (uint64), off (int64, n+1), val (int64), cc (int8 country code)

Usage:
    python .../precompute_keys.py --split train
    python .../precompute_keys.py --split test
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import BlockConfig, gen_keys  # noqa: E402

PROC = "data/processed"
OUT = "data/interim"
COLS = ["entity_id", "name_core", "name_canon", "street_number", "state",
        "addr_canon", "is_domain", "country"]
CFG = BlockConfig(use_n2=True, use_nt=True, use_sn=True, use_at=True,
                  use_ns=True, use_nn=True, use_lsh=False)
TYPE = {'n2': 0, 'nt': 1, 'sn': 2, 'at': 3, 'ns': 5, 'nn': 6}
CC = {'US': 0, 'India': 1, 'France': 2}


def enc(k):
    return ((hash(k) & ((1 << 58) - 1)) << 3) | TYPE[k.split(':', 1)[0]]


def process(src):
    t = time.time()
    id_num = []; off = [0]; val = []; cc = []
    pf = pq.ParquetFile(f"{PROC}/{src}.parquet")
    for b in pf.iter_batches(batch_size=200_000, columns=COLS):
        d = b.to_pydict()
        e = d["entity_id"]
        for i in range(len(e)):
            nc = d["name_core"][i]
            toks = nc.split() if nc else []
            at = d["addr_canon"][i].split() if d["addr_canon"][i] else []
            ks = gen_keys(toks, d["name_canon"][i], d["street_number"][i],
                          d["state"][i], at, d["is_domain"][i], d["country"][i], CFG)
            val.extend(enc(k) for k in ks)
            off.append(len(val))
            id_num.append(int(e[i].split('-', 1)[1]))
            cc.append(CC.get(d["country"][i], 3))
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/keys_{src}.npz",
             id_num=np.array(id_num, dtype=np.uint64),
             off=np.array(off, dtype=np.int64),
             val=np.array(val, dtype=np.int64),
             cc=np.array(cc, dtype=np.int8))
    print(f"{src}: {len(id_num):,} records, {len(val):,} keys, "
          f"{time.time()-t:.0f}s -> {OUT}/keys_{src}.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    args = ap.parse_args()
    for s in (f"{args.split}_source1", f"{args.split}_source2", f"{args.split}_source3"):
        process(s)


if __name__ == "__main__":
    main()
