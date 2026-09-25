"""
blocking.py — candidate-generation (blocking) key generation.

Given a cleaned record, produce a small set of *blocking keys*. Two records are
compared as a candidate pair iff they share at least one key. Keys are string-
tagged by type so the caller can apply type-aware document-frequency (DF)
pruning (drop keys that match too many records — they are non-discriminative and
cause O(n^2) candidate explosion).

Key types (see docs/06 and docs/09):
  n2:  sorted first-2 core name tokens        — selective name signal
  nt:  each core name token >= NT_MIN_LEN      — reordering / partial name (DF-pruned)
  sn:  street_number + state                   — strong address anchor
  at:  each address token >= AT_MIN_LEN        — locality/street signal (DF-pruned)
  lsh: MinHash bands over name char n-grams     — typo/scramble/translit/domain (DF-pruned)

DF pruning and index building live in the driver (candidate generation over the
whole corpus); this module only defines *how to generate keys for one record*,
so the exact same logic is reused for S1, S2, S3, train and test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ------------------------------------------------------------------ MinHash --
# Lightweight, dependency-free MinHash for char-n-gram LSH. We compute B band
# keys per record: for band b, the key is the minimum salted hash over the
# record's n-grams. Two records sharing ANY band-min are LSH candidates. This is
# the "single-min per band" variant — cheap and effective for near-duplicate
# short strings (names), and far faster than k-permutation MinHash on 10M+ rows.

_MASK = (1 << 32) - 1


def _mix(x: int) -> int:
    # a fast 32-bit integer hash (xorshift-multiply); deterministic across runs
    x &= _MASK
    x ^= (x >> 16)
    x = (x * 0x7feb352d) & _MASK
    x ^= (x >> 15)
    x = (x * 0x846ca68b) & _MASK
    x ^= (x >> 16)
    return x & _MASK


@dataclass
class BlockConfig:
    use_n2: bool = True
    use_nt: bool = True
    nt_min_len: int = 4
    use_sn: bool = True
    use_at: bool = False
    at_min_len: int = 4
    # composite (low-cardinality) keys — a single name token is too coarse
    # (blocks up to 400k); combining it with the state or street number yields
    # far more specific keys, cutting candidate volume by orders of magnitude
    # while preserving recall.
    use_ns: bool = False   # name_token + state
    use_nn: bool = False   # name_token + street_number
    use_lsh: bool = False
    lsh_bands: int = 6
    lsh_ngram: int = 3
    lsh_salts: tuple = field(default=None)

    def __post_init__(self):
        if self.lsh_salts is None:
            # fixed salts so LSH keys are stable across S1/S2/S3 and runs
            self.lsh_salts = tuple(0x9E3779B1 * (i + 1) & _MASK for i in range(self.lsh_bands))


def _char_ngrams(s: str, n: int):
    s = s.replace(" ", "")
    if len(s) < n:
        return [s] if s else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def _lsh_band_keys(name_canon: str, cfg: BlockConfig):
    grams = _char_ngrams(name_canon, cfg.lsh_ngram)
    if not grams:
        return []
    # base 32-bit hash per n-gram (well-mixed, deterministic)
    ghash = [_mix(hash(g) & _MASK) for g in grams]
    # each band uses a different bijection (XOR with a fixed salt) of the base
    # hashes, then takes the min -> a MinHash-style band signature. XOR-min is a
    # valid, very cheap LSH band (one XOR per n-gram per band).
    keys = []
    for b, salt in enumerate(cfg.lsh_salts):
        m = min(h ^ salt for h in ghash)
        keys.append(f"l{b}:{m:08x}")
    return keys


def gen_keys(name_core_tokens, name_canon, street_number, state,
             addr_tokens, is_domain, country, cfg: BlockConfig):
    """Return the set of blocking keys for one record.

    name_core_tokens : iterable of core name tokens (already normalised)
    name_canon       : canonical name string (for LSH n-grams)
    street_number    : str ("" if none)
    state            : str ("" if none)
    addr_tokens      : iterable of address tokens
    is_domain        : bool
    country          : str (kept generic; may namespace keys if needed)
    """
    keys = set()
    toks = sorted(t for t in name_core_tokens if t)

    if cfg.use_n2 and toks:
        keys.add("n2:" + "|".join(toks[:2]))

    if cfg.use_nt:
        for t in toks:
            if len(t) >= cfg.nt_min_len:
                keys.add("nt:" + t)

    if cfg.use_sn and street_number:
        keys.add("sn:" + street_number + ":" + (state or ""))

    # composite name keys: pair each name token with an address anchor so a
    # common token (e.g. "center") only collides within the same state / house
    # number, not globally.
    if cfg.use_ns and state:
        for t in toks:
            if len(t) >= cfg.nt_min_len:
                keys.add("ns:" + t + ":" + state)
    if cfg.use_nn and street_number:
        for t in toks:
            if len(t) >= cfg.nt_min_len:
                keys.add("nn:" + t + ":" + street_number)

    if cfg.use_at and addr_tokens:
        for t in addr_tokens:
            # skip pure numbers here (street number already captured by sn:)
            if len(t) >= cfg.at_min_len and not t.isdigit():
                keys.add("at:" + t)

    if cfg.use_lsh and name_canon:
        keys.update(_lsh_band_keys(name_canon, cfg))

    return keys
