"""
features.py — pairwise similarity features for (S1, S2/S3) candidate pairs.

Given the cleaned fields of two records, produce a fixed-length numeric feature
vector for the matching classifier. Features are grounded in the signal analysis
(docs/04): char n-grams beat tokens on typos, address is the strongest anchor
when present, transliteration is phonetic (so fuzzy > exact for native names).

Design:
  * `prep(record_fields)` precomputes the per-record sets (core tokens, name
    char-trigrams, address tokens) ONCE, so pairwise features are cheap set ops.
  * `pair_features(a, b)` returns a list of floats in the fixed order of
    FEATURE_NAMES.

Only offline string computation — no external lookup. RapidFuzz (MIT) provides
the edit-distance / token scorers.
"""
from __future__ import annotations
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES = [
    # name
    "name_token_jaccard", "name_trigram_jaccard", "name_ratio",
    "name_jaro_winkler", "name_token_sort", "name_token_set",
    "name_partial_ratio", "name_len_ratio", "name_prefix4", "name_exact_canon",
    "name_core_len_a", "name_core_len_b",
    # address
    "addr_token_jaccard", "addr_ratio", "addr_token_set",
    "street_num_exact", "street_num_both", "state_match", "state_both",
    "addr_len_ratio", "addr_present_a", "addr_present_b", "addr_present_both",
    # context / flags
    "same_country", "b_is_native", "b_is_domain",
    "legal_overlap", "legal_conflict", "src_is_s3",
]
N_FEATURES = len(FEATURE_NAMES)

# Relative / competitive features — computed per S1 across ALL its candidates,
# so they need cross-candidate context (added in build_training, not pair_features):
#   namejac_rank        : rank of this cand by name-core Jaccard to S1 (0=best), normalised
#   is_top_namejac      : 1 if it is the single best name match for this S1
#   namejac_margin      : its name Jaccard minus the best OTHER candidate's
#   other_src_best      : best name-Jaccard-to-S1 among OTHER-source candidates
#                         (cross-source corroboration: does the other source also match S1?)
#   cross_twin          : 1 if it shares a name signature with an other-source candidate
#                         (both S2 & S3 point at the same business -> strong)
#   n_cands_log         : log(1+#candidates for this S1) — context on crowding
REL_FEATURE_NAMES = [
    "namejac_rank", "is_top_namejac", "namejac_margin",
    "other_src_best", "cross_twin", "n_cands_log",
]
ALL_FEATURE_NAMES = FEATURE_NAMES + REL_FEATURE_NAMES


def compute_relative(a_core, cand_cores, cand_s3):
    """Relative/competitive + cross-source features over ALL of one S1's
    candidates, computed from cheap core-token sets only (no full prep needed).

    a_core     : S1 core token frozenset
    cand_cores : list of each candidate's core token frozenset
    cand_s3    : list of bools (candidate is from Source 3)
    Returns a list of REL feature vectors aligned to the candidates.
    """
    import math
    n = len(cand_cores)
    njac = [(_jac(a_core, cc)) for cc in cand_cores]
    sigs = []
    for cc in cand_cores:
        toks = sorted(cc)
        sigs.append("|".join(toks[:2]) if toks else "")
    sig_src = {}
    for i in range(n):
        sig_src.setdefault(sigs[i], set()).add(bool(cand_s3[i]))
    order = sorted(range(n), key=lambda i: -njac[i])
    rank = {idx: r for r, idx in enumerate(order)}
    best = njac[order[0]] if n else 0.0
    second = njac[order[1]] if n > 1 else 0.0
    # best name-jaccard among each source, to get "other source" cheaply
    best_s2 = max((njac[j] for j in range(n) if not cand_s3[j]), default=0.0)
    best_s3 = max((njac[j] for j in range(n) if cand_s3[j]), default=0.0)
    n_log = math.log1p(n)
    out = []
    for i in range(n):
        other_best = best_s2 if cand_s3[i] else best_s3
        twin = 1.0 if (sigs[i] and len(sig_src.get(sigs[i], ())) >= 2) else 0.0
        margin = njac[i] - (best if i != order[0] else second)
        out.append([rank[i] / n if n else 0.0,
                    1.0 if i == order[0] else 0.0,
                    margin, other_best, twin, n_log])
    return out


def _trigrams(s: str):
    s = s.replace(" ", "")
    if len(s) < 3:
        return frozenset([s]) if s else frozenset()
    return frozenset(s[i:i+3] for i in range(len(s) - 2))


def _jac(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def prep(name_canon, name_core, addr_canon, street_number, state,
         country, is_native, is_domain, name_legal, src_is_s3):
    """Precompute a record's reusable pieces for fast pairwise features."""
    core = frozenset(name_core.split()) if name_core else frozenset()
    return {
        "name_canon": name_canon or "",
        "core": core,
        "tri": _trigrams(name_canon or ""),
        "addr_canon": addr_canon or "",
        "atok": frozenset(addr_canon.split()) if addr_canon else frozenset(),
        "snum": street_number or "",
        "state": state or "",
        "country": country or "",
        "native": bool(is_native),
        "domain": bool(is_domain),
        "legal": frozenset(name_legal.split()) if name_legal else frozenset(),
        "s3": bool(src_is_s3),
        "core_len": len(core),
        "addr_present": bool(addr_canon),
    }


def pair_features(a: dict, b: dict):
    """Return the feature vector (list[float]) for S1 record `a` vs cand `b`."""
    na, nb = a["name_canon"], b["name_canon"]
    aa, ab = a["addr_canon"], b["addr_canon"]

    # name
    ntj = _jac(a["core"], b["core"])
    ntri = _jac(a["tri"], b["tri"])
    nr = fuzz.ratio(na, nb) / 100.0 if na and nb else 0.0
    njw = JaroWinkler.normalized_similarity(na, nb) if na and nb else 0.0
    nts = fuzz.token_sort_ratio(na, nb) / 100.0 if na and nb else 0.0
    ntset = fuzz.token_set_ratio(na, nb) / 100.0 if na and nb else 0.0
    npr = fuzz.partial_ratio(na, nb) / 100.0 if na and nb else 0.0
    la, lb = len(na), len(nb)
    nlen = (min(la, lb) / max(la, lb)) if max(la, lb) else 0.0
    npref = 1.0 if (na[:4] == nb[:4] and len(na) >= 4 and len(nb) >= 4) else 0.0
    nexact = 1.0 if (na and na == nb) else 0.0

    # address
    atj = _jac(a["atok"], b["atok"])
    ar = fuzz.ratio(aa, ab) / 100.0 if aa and ab else 0.0
    atset = fuzz.token_set_ratio(aa, ab) / 100.0 if aa and ab else 0.0
    sn_both = 1.0 if (a["snum"] and b["snum"]) else 0.0
    sn_exact = 1.0 if (a["snum"] and a["snum"] == b["snum"]) else 0.0
    st_both = 1.0 if (a["state"] and b["state"]) else 0.0
    st_match = 1.0 if (a["state"] and a["state"] == b["state"]) else 0.0
    laa, lab = len(aa), len(ab)
    alen = (min(laa, lab) / max(laa, lab)) if max(laa, lab) else 0.0
    apa = 1.0 if a["addr_present"] else 0.0
    apb = 1.0 if b["addr_present"] else 0.0
    apboth = 1.0 if (a["addr_present"] and b["addr_present"]) else 0.0

    # context
    same_country = 1.0 if a["country"] == b["country"] else 0.0
    b_native = 1.0 if b["native"] else 0.0
    b_domain = 1.0 if b["domain"] else 0.0
    legal_ov = _jac(a["legal"], b["legal"])
    legal_conf = 1.0 if (a["legal"] and b["legal"] and not (a["legal"] & b["legal"])) else 0.0
    s3 = 1.0 if b["s3"] else 0.0

    return [
        ntj, ntri, nr, njw, nts, ntset, npr, nlen, npref, nexact,
        float(a["core_len"]), float(b["core_len"]),
        atj, ar, atset, sn_exact, sn_both, st_match, st_both,
        alen, apa, apb, apboth,
        same_country, b_native, b_domain, legal_ov, legal_conf, s3,
    ]
