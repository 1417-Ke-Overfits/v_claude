# 04 — Signal Analysis & Blocking Recall Ceiling

Before building anything, we quantified **how much signal each field carries**
and **what recall a blocking strategy could possibly reach**. This tells us where
to invest and sets the ceiling for the whole pipeline.

## Method

We sampled **30,000 matched S1 clusters** (139,943 records; **109,943 true
S1↔match pairs**) uniformly at random from the ground truth, fetched the actual
records from all three sources, and computed similarity between each S1 entity
and each of its true matches under our normalisation. Sampling (not the full
7.6M pairs) kept it fast; 110k pairs is more than enough for stable estimates.

Similarity measures:
- **Name token Jaccard** — overlap of normalised core token *sets*.
- **Name char-trigram Jaccard** — overlap of character 3-grams (typo-robust).
- **Address token Jaccard** — overlap of normalised address token *sets*.

## Results — how similar are true matches?

Distribution of each similarity over the 109,943 true pairs (share of pairs in
each bucket):

### Name token Jaccard
| bucket | pairs | share |
|--------|------:|------:|
| 0.0 (no shared token) | 16,181 | 14.7% |
| 0–0.2 | 32 | 0.0% |
| 0.2–0.4 | 8,124 | 7.4% |
| 0.4–0.6 | 12,106 | 11.0% |
| 0.6–0.8 | 15,172 | 13.8% |
| 0.8–1.0 | 1,211 | 1.1% |
| 1.0 (identical token set) | 57,117 | 52.0% |

→ **52% of true matches have an identical core-token set**, but **14.7% share no
token at all** (native scripts, aliases, heavy scrambles). Token matching alone
misses roughly 1 in 7.

### Name character-trigram Jaccard
| bucket | pairs | share |
|--------|------:|------:|
| 0.0 | 9,327 | 8.5% |
| 0–0.2 | 2,147 | 2.0% |
| 0.2–0.4 | 9,229 | 8.4% |
| 0.4–0.6 | 21,487 | 19.5% |
| 0.6–0.8 | 30,922 | 28.1% |
| 0.8–1.0 | 15,231 | 13.9% |
| 1.0 | 21,600 | 19.6% |

→ Only **8.5% have zero trigram overlap** (vs 14.7% for tokens). **Character
n-grams are strictly more robust than tokens** — they survive typos, spacing,
and word reordering. This is why char-n-gram features and blocking are central.

### Address token Jaccard
| bucket | pairs | share |
|--------|------:|------:|
| 0.0 | 4,918 | 4.5% |
| 0–0.2 | 2,003 | 1.8% |
| 0.2–0.4 | 15,791 | 14.4% |
| 0.4–0.6 | 26,699 | 24.3% |
| 0.6–0.8 | 30,263 | 27.5% |
| 0.8–1.0 | 15,977 | 14.5% |
| 1.0 | 14,292 | 13.0% |

→ Only **4.5% have zero address overlap** — and that's almost exactly the
**4.46%** of matches whose match-side address is *empty*. **When an address
exists, it overlaps strongly.** The address is our single most reliable anchor.

### Cross-cutting rates on the 109,943 pairs
- Match-side **address empty:** 4,899 (4.46%) → the name-only path must cover
  these.
- Match-side **native-script name:** 15,231 (13.85%) → transliteration path.
- Country split of pairs: India 43,172 / US 66,771.

## The blocking recall ceiling

"Recall ceiling" = the fraction of true pairs that share at least one blocking
signal, i.e. the best recall any downstream matcher could achieve given the
candidate set. We tested individual signals and their union:

| Blocking signal | True-pair recall |
|-----------------|-----------------:|
| Share ≥1 core **name token** | 85.28% |
| Same **street number** | 67.06% |
| **Union**: name-token ∨ street-number ∨ addr-Jaccard ≥ 0.34 ∨ name-trigram ≥ 0.4 | **99.30%** |

**Key takeaways:**
1. **No single signal is enough.** Name tokens miss 15%; street number misses
   33% (empty/typo'd numbers, and huge blocks if used alone).
2. **Their union reaches 99.3%.** Combining name-token, address, and char-n-gram
   signals leaves only ~0.7% of true pairs uncatchable — the alias + broken
   address cases that no offline method can recover.
3. This 99.3% is the **target for our blocking stage**. Our first, simple
   implementation reached 94.9% (doc 06); closing the gap to ~99% is the main
   blocking work remaining, and doc 04 tells us exactly which signals to add
   (address-token and char-n-gram blocking).

## What this dictates for features & model

The similarity distributions map directly onto the matching-model feature set:

| Evidence | Feature(s) it justifies |
|----------|-------------------------|
| 52% identical tokens, but 14.7% zero-token | token Jaccard **and** char-trigram Jaccard (complementary) |
| Trigrams robust (8.5% zero) | char-n-gram cosine / Jaccard; Jaro-Winkler; Levenshtein ratio |
| Address strongest when present (4.5% zero) | address token overlap, street-number exact match, state match |
| 4.46% empty address | an "address_present" flag so the model can weight name more when address is absent |
| 13.85% native names | transliteration before featurising; an "is_native" flag |
| 85% both-source agreement (doc 02) | cross-source corroboration feature (later stage) |

Feature engineering detail lives with the matcher (doc 08 roadmap → future doc).
