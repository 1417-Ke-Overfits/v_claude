# 06 — Blocking Recall Baseline

Blocking (candidate generation) reduces the ~10¹³ possible pairs to a small
candidate set. Its **recall is the hard ceiling** on the whole pipeline: a true
pair not present as a candidate can never be matched. Its **selectivity**
(reduction ratio) determines whether the classifier stage is even tractable.

This doc reports our **first, deliberately simple** blocking scheme, measured
against the full training ground truth, and the analysis that tells us how to
improve it.

**Code:** blocking keys are produced by `blocking_keys()` in
[`normalize.py`](../code/business_entity_resolution/src/normalize.py); the recall
measurement script streams the cleaned Parquet and checks every true pair.

---

## 1. The v1 blocking keys

For each record we emit a small set of keys; two records are candidate-compared
if they **share any key**:

| Key | Form | Rationale |
|-----|------|-----------|
| `n2` | `n2:<sorted first 2 core tokens>` | order-invariant name signal |
| `nt` | `nt:<token>` for each core token ≥4 chars | catches reordering / partial name overlap |
| `sn` | `sn:<street_number>:<state>` | strong, country-agnostic address anchor |

Keys are computed from the cleaned fields (`name_core`, `street_number`,
`state`) so blocking is consistent with everything downstream.

---

## 2. Measured recall (full train ground truth)

Checked all **7,638,365** true pairs: does each S1 entity share ≥1 key with each
of its true matches?

| Metric | Value |
|--------|------:|
| True pairs | 7,638,365 |
| Covered (share a key) | 7,251,082 |
| **Blocking recall** | **94.93%** |
| — US | **97.89%** (4,481,986 / 4,578,522) |
| — India | **90.50%** (2,769,096 / 3,059,843) |

**Reading it:**
- 94.9% is a strong baseline for three keys, but **below the 99.3% ceiling**
  (doc 04). The gap is ~4.4 points of recall we can still recover.
- **India is the weak spot (90.5% vs US 97.9%)** — exactly as predicted: native
  script romanisation diverges from English spelling, so name tokens don't align,
  and when the address is also weak/native the pair is missed.

---

## 3. Miss analysis — *why* the 5% is missed

Sample of missed true matches (raw S2/S3 name shown):

| Miss type | Examples | Why v1 misses it |
|-----------|----------|------------------|
| **Domain-form names** | `agrosheronhotels.com`, `chandraindiafund.com`, `glyphifylaw.com`, `jgsreal.com`, `KASEYHEREDIATAILWIND.COM` | name collapses to one long token; `nt`/`n2` keys don't match the S1 word tokens |
| **Native-script names, weak address** | `शक्ति अर्बन प्रोडक्ट्स…`, `फर्स्ट फूड…`, `जैन वेंचर्स…`, `मां इंफ्रा…`, `एपेक्स फाइनेंस एलएलपी` | romanised tokens (`epeksa`,`phrsta`) share no ≥4-char token with English S1, and `street#+state` didn't align |
| **Ultra-short / alias names** | `MT`, `Vantageavigild` | nothing to tokenise, or a genuine alias with no shared signal |

The common thread: v1 relies on **name-token** and **street#+state** keys. It has
**no address-token blocking** and **no character-n-gram blocking**, which are
exactly the two signals doc 04 showed are needed to reach 99.3%.

---

## 4. Block-size distribution — the explosion problem

Over all S2+S3 records:

| Metric | Value |
|--------|------:|
| Unique blocking keys | 4,890,931 |
| Total postings (S2+S3) | 43,723,159 |
| Mean block size | 8.9 |
| Median | 1 |
| p95 | 13 |
| p99 | 81 |
| **Max** | **424,349** |
| Keys with >1,000 postings | 3,407 |
| Keys with >10,000 postings | 408 |

**Largest blocks (all `nt:` name-token keys):**
```
center 424,349 | partners 320,743 | services 318,499 | group 289,405
india 229,501 | holdings 197,901 | care 190,666 | associates 155,438
service 121,514 | health 108,480 | enterprises 108,317 | industries 103,423
clinic 98,099 | ventures 95,750 | solutions 89,958
```

**The problem:** a block of size *n* generates up to *n²* candidate pairs. The
`center` block alone (424k records) would generate ~9×10¹⁰ pairs — catastrophic.
A handful of generic descriptive tokens dominate the cost while contributing
**almost no discriminating power** (knowing two businesses both contain
"services" tells you almost nothing).

**The median block is 1 and p99 is 81** — the vast majority of blocks are tiny
and healthy. The entire problem is a **long tail of ~3,400 generic tokens**.

---

## 5. Fixes queued for blocking v2 (see doc 08)

Directly implied by §3 and §4:

1. **Document-frequency (DF) pruning** — drop `nt:` keys whose posting count
   exceeds a threshold (candidates them via other keys instead). Eliminates the
   n² explosion at essentially zero recall cost, because a match that *only*
   shares "services" is unusable anyway. **Highest priority, cheapest win.**
2. **Address-token blocking** — key on distinctive (low-DF) address tokens,
   optionally combined with the street number. Indian native-name records
   usually have intact addresses, so this should recover most of the India gap.
3. **Domain word-splitting** — split domain labels into word-pieces so
   `agrosheronhotels` can share a token with `agro`/`hotels`; and/or add
   char-n-gram/substring blocking for domains.
4. **Character-n-gram LSH** (MinHash banding on name char-trigrams) — a
   principled way to block typo'd/scrambled/transliterated names that share no
   whole token, targeting the residual India misses.

**Target:** lift recall from 94.9% toward the **99.3% ceiling** while keeping the
max block under a few thousand, so candidate generation stays tractable and the
classifier sees a clean, high-recall candidate set.

---

## 6. Why report a "only 94.9%" baseline at all?

Because it is **honest, measured, and diagnostic**. A simple scheme that we fully
understand — and whose every miss we can explain and attribute — is worth far
more than an untested complex one. The baseline:
- confirms the cleaning works (US already at 97.9%),
- localises the remaining problem precisely (India + domains + generic-token
  explosion),
- and gives each v2 change a clear, falsifiable recall target to beat.
