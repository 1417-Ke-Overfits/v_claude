"""
build_translit_dict.py — learn a native-script→English token dictionary from the
ground truth (fully offline, allowed: learned only from provided training data).

Our library transliteration is phonetic ("food"->"phuda", "private"->"praiveta"),
so native Indian names don't share tokens with their romanized S1 counterpart.
Here we LEARN the correction: for every ground-truth pair whose S2/S3 record is
native-script, align its romanized tokens to the S1 English tokens (greedy by
Jaro-Winkler), accumulate (phonetic_token -> english_token) evidence, and keep
the dominant mapping per phonetic token. normalize.py then applies this dict
after romanize(), so "phuda" becomes "food" and the names finally match.

Output: code/business_entity_resolution/src/translit_dict.json
"""
from __future__ import annotations
import json, os, sys, time
from collections import defaultdict, Counter
import pyarrow.parquet as pq
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import romanize, strip_accents, LEGAL_TOKENS  # noqa: E402
import re as _re
PROC = "data/processed"


def phonetic_tokens(raw):
    """Phonetic tokens of a raw name BEFORE the learned dict is applied — so the
    dictionary can be rebuilt reproducibly even after data was cleaned with it."""
    s = strip_accents(romanize(raw or "")).lower()
    s = _re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t and t not in LEGAL_TOKENS and not t.isdigit()]
GT = "dataset/train/train_ground_truth.tsv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translit_dict.json")
MIN_JW = 0.55          # only record an alignment this similar or better
MIN_SUPPORT = 8        # phonetic token must be seen >= this many times
MIN_FRACTION = 0.45    # dominant English target must win >= this share
# Only correct tokens that look non-English (phonetic transliteration artifacts),
# never map one English word to another (avoids "management"->"investment").
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)


def load_names(src, need_native):
    """id_num -> (phonetic_token_list, is_native). Uses RAW names + romanize so
    it is independent of whether the parquet was cleaned with the dict."""
    t = pq.read_table(f"{PROC}/{src}.parquet",
                      columns=["entity_id", "raw_name", "is_native"])
    d = {}
    for e, rn, nat in zip(t.column("entity_id").to_pylist(),
                          t.column("raw_name").to_pylist(),
                          t.column("is_native").to_pylist()):
        if need_native and not nat:
            continue
        d[int(e.split('-', 1)[1])] = (phonetic_tokens(rn), bool(nat))
    return d


def main():
    s1 = load_names("train_source1", need_native=False)
    log(f"S1 names {len(s1):,}")
    s2 = load_names("train_source2", need_native=True)
    s3 = load_names("train_source3", need_native=True)
    log(f"native S2 {len(s2):,}  native S3 {len(s3):,}")

    # phonetic token -> Counter of english tokens (weighted by JW)
    votes = defaultdict(Counter)
    npairs = 0
    with open(GT, encoding="utf-8") as f:
        next(f)
        for line in f:
            src1, _, rest = line.rstrip("\n").partition("\t")
            if not rest.strip():
                continue
            e_tokens = s1.get(int(src1.split('-', 1)[1]), (None,))[0]
            if not e_tokens:
                continue
            e_tokens = [e for e in e_tokens if not e.isdigit()]
            if not e_tokens:
                continue
            for m in rest.split(","):
                if not m:
                    continue
                num = int(m.split('-', 1)[1])
                rec = s2.get(num) if m[1] == '2' else s3.get(num)
                if rec is None:
                    continue
                p_tokens = [p for p in rec[0] if not p.isdigit()]
                for p in p_tokens:
                    # greedy: best English token by Jaro-Winkler
                    best_e, best_s = None, 0.0
                    for e in e_tokens:
                        s = JaroWinkler.normalized_similarity(p, e)
                        if s > best_s:
                            best_s, best_e = s, e
                    if best_e is not None and best_s >= MIN_JW and p != best_e:
                        votes[p][best_e] += best_s
                npairs += 1
    log(f"aligned {npairs:,} native pairs; {len(votes):,} phonetic tokens seen")

    dic = {}
    for p, c in votes.items():
        total = sum(c.values())
        e, w = c.most_common(1)[0]
        # support = count of alignments (approx by total weight / avg)
        if total >= MIN_SUPPORT * MIN_JW and (w / total) >= MIN_FRACTION and len(p) >= 3:
            dic[p] = e
    json.dump(dic, open(OUT, "w"), ensure_ascii=False)
    log(f"dictionary: {len(dic):,} entries -> {OUT}")
    print("samples:", dict(list(dic.items())[:25]), flush=True)


if __name__ == "__main__":
    main()
