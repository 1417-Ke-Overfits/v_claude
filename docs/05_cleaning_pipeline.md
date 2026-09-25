# 05 — Cleaning & Normalization Pipeline

**Goal:** map every raw, noisy record into a canonical representation where two
records for the same business look as similar as possible, *without* collapsing
genuinely different businesses. This is where most of the eventual score is won
or lost — good normalisation raises both blocking recall and feature quality.

**Code:**
[`code/business_entity_resolution/src/normalize.py`](../code/business_entity_resolution/src/normalize.py)
(the logic) and
[`clean_dataset.py`](../code/business_entity_resolution/src/clean_dataset.py)
(the streaming driver).

---

## 1. What the pipeline produces

Each raw record → one Parquet row with these columns:

| Column | Meaning |
|--------|---------|
| `entity_id`, `country` | passed through (country kept as an open label) |
| `raw_name`, `raw_addr` | **originals preserved** — needed for exact-match features |
| `name_canon` | cleaned name, core tokens space-joined (order preserved) |
| `name_core` | **sorted** core tokens (order-invariant key material) |
| `name_legal` | legal/entity-type tokens found (Inc/Ltd/SARL/…) |
| `addr_canon` | sorted, normalised address tokens (order-invariant) |
| `street_number` | first standalone house/building number (leading zeros stripped) |
| `state` | canonical state/region token |
| `is_domain` | name was a web domain |
| `is_native` | name contained non-ASCII (native script) |

Keeping **both** raw and canonical fields is deliberate: canonical fields drive
blocking and fuzzy features; raw fields let the model reward exact agreement,
which canonicalisation would otherwise hide.

---

## 2. Name canonicalization — step by step

For `clean_name(raw)`:

1. **Romanize** any native script → Latin (see §4).
2. **Strip accents** via Unicode NFKD (`Énterprises`→`Enterprises`).
3. **Lowercase.**
4. **Split domains:** `foo.com` → keep label `foo`, drop the TLD.
5. **Fold ampersands:** `&` and `+` → `and`.
6. **Strip phone numbers:** long digit runs removed.
7. **Punctuation → space**, collapse whitespace.
8. **Separate legal tokens** (Inc/Corp/LLC/LLP/Ltd/Limited/Pvt/Private, and
   France's SARL/SAS/SASU/SA/EURL/SCI, plus international GmbH/AG/BV/…, plus
   stopwords and/the/of, plus romanised Indic suffixes like `praiveta limiteda`)
   from the identifying **core** tokens.
9. Return `(canon, core_token_set, legal_token_set)`. If a name is *only* legal
   tokens, we keep them so it is never blank.

### Design decision: keep descriptive words in the core
Early on we put words like *Ventures, Enterprises, Services, Solutions,
Partners, Group* into the strip list. **This was wrong** — it collapsed
"Red Ventures" → "red", making it indistinguishable from "Red Trading",
"Red Solutions", etc., which would destroy precision and explode block sizes.

**Fix:** only *genuine legal-entity types* and pure stopwords are stripped.
Descriptive words stay in the core (they carry identity). The problem those
words *do* cause — a few of them are extremely common and create giant blocks —
is handled at **blocking time by document-frequency pruning** (doc 06), not by
deletion here. This cleanly separates "identity signal" from "blocking
selectivity".

---

## 3. Address canonicalization — step by step

For `clean_address(raw, country)`:

1. Romanize + strip accents + lowercase.
2. **Find the state/region** by scanning **all** comma fields (not just the
   last, because addresses are reordered) and matching against the
   state/region map; keep the last match found.
3. Punctuation → space; tokenise.
4. **Extract street number:** first standalone 1–6 digit token; strip leading
   zeros (`0684`→`684`).
5. Build an **order-invariant token set**, applying per token:
   - drop stopwords (`null`, `na`, `none`, `po`, `box`, `no`, `door`, bare
     ordinal suffixes…);
   - **street-type abbreviation** map (`rd`→`road`, `st`→`street`, `ave`→
     `avenue`, `blvd`→`boulevard`, `rue`→`rue`, …);
   - **ordinal normalisation** (`45th`/`45nd`/`2nd`→`45`/`2`);
   - **leading-zero strip** on numeric tokens;
   - **fold full-form state variants** (`hariyana`→`haryana`) wherever they
     appear, so native-romanised state names align;
   - drop single letters.
6. Return `(sorted_token_string, token_set, street_number, state)`.

### State/region normalisation (multi-country, open set)
A map collapses many surface forms to one canonical token:
- **US:** all 50 states + DC, abbrev ↔ full.
- **India:** all states/UTs, abbrev ↔ full ↔ **romanised native** variants
  (e.g. `தமிழ்நாடு`→`dhamilnadhu`→`tamilnadu`; `महाराष्ट्र`→`maharastra`→
  `maharashtra`).
- **France:** the main regions (Nouvelle-Aquitaine, Hauts-de-France, Nord, …).

**Ambiguity handled:** some 2-letter codes collide across countries (`GA` =
Georgia/Goa, `OR` = Oregon/Odisha, `IN` = Indiana). `normalize_state` is
**country-aware** — for India rows it prefers the Indian interpretation. Only
*unambiguous full-form* variants (≥4 chars) are folded when they appear as
arbitrary address tokens, so common English words are never mis-mapped.

---

## 4. Transliteration — why and how

**Why:** ~12–15% of S2/S3 names (and ~9% of addresses) are in native Indic
scripts while S1 is romanized English (doc 02 §2.3). Without bridging scripts,
those Indian entities are unmatchable — and India is 40% of the data.

**How:** `indic-transliteration` (MIT, offline). We:
1. detect the script by Unicode range (Devanagari/Bengali/Gurmukhi/Gujarati/
   Tamil/Telugu/Kannada/Malayalam);
2. pre-fold Devanagari **candra vowels** (`ॉ`,`ॅ`) to their standard vowel signs
   (the IAST scheme doesn't map them — see problem P1 below);
3. transliterate to **IAST**, then strip diacritics → plain Latin.

**Fair-play:** this is a deterministic, offline *script conversion*, not an
identity lookup. It is the same category of tool as the string-similarity
features the brief explicitly recommends.

**Quality:** romanization is *phonetic*, not spelling-exact:
```
एसएस फूड प्राइवेट लिमिटेड → "esaesa phuda praiveta limiteda"   (≈ SS Food Private Limited)
தமிழ்நாடு                 → "dhamilnadhu"                     (≈ Tamil Nadu)
```
`phuda`≈food, `praiveta`≈private, `limiteda`≈limited — close enough for
char-n-gram/fuzzy matching, which is exactly why we pair transliteration with
n-gram features rather than expecting exact equality.

---

## 5. Problems faced & how we solved them

| # | Problem | Symptom | Fix |
|---|---------|---------|-----|
| P1 | IAST scheme doesn't map Devanagari candra-O `ॉ` | `मॉडर्न`→`maॉdarna` (raw script left in output) | pre-fold candra vowels `ॉ→ो`, `ॅ→े` before transliterating → `modarna` |
| P2 | Over-aggressive legal stripping | "Red Ventures"→"red"; distinct businesses collide, precision risk, block explosion | strip only true legal suffixes + stopwords; keep descriptive words; prune common tokens at blocking time instead |
| P3 | Native-script state ≠ English state | `தமிழ்நாடு`→`dhamilnadhu` but `Tamil Nadu`→`tamilnadu`; `street#+state` key breaks | add romanised-native→canonical state variant map; fold state variants on address tokens too |
| P4 | State only read from last field | reordered/native addresses put state elsewhere → state lost | scan **all** comma fields for the state, keep last match |
| P5 | Leading-zero numeric mismatch | address token `0684` ≠ `684` | strip leading zeros on numeric tokens (and on street number) |
| P6 | Romanised legal-suffix spelling guessed wrong | `एलएलपी`→`elaelapi` (we'd listed `elelapi`), suffix left in core | corrected romanised-suffix list to observed spellings |
| P7 | Python 3.14 default lacks ML wheels | pip installs fail / slow source builds | use a Python 3.11 venv (doc 01) |
| P8 | 24.2M rows too slow single-threaded | ~5 min/pass, painful to iterate | multiprocessing pool over line chunks → ~2.5 min for everything |

Every fix was verified by re-running the normaliser on the real cluster examples
(e.g. after P3–P5 both "Chordia & Partners" records normalise to the *identical*
address `1038 9 faridabad haryana sector`).

---

## 6. The streaming driver

`clean_dataset.py`:
- reads a source `.tsv` in 50k-line chunks,
- normalises each chunk in a **multiprocessing pool** (string work is CPU-bound
  and GIL-heavy, so processes beat threads),
- writes **one Parquet row-group per chunk** with `pyarrow.ParquetWriter`, so
  peak memory stays flat regardless of file size.

```bash
python code/business_entity_resolution/src/clean_dataset.py --all
```

### Measured performance
| File | Rows | Time | Rate |
|------|-----:|-----:|-----:|
| train_source1 | 2.21M | 9.8s | 210k/s |
| train_source2 | 5.03M | 29.5s | 171k/s |
| train_source3 | 5.29M | 30.9s | 171k/s |
| test_source1 | 1.73M | 9.3s | 166k/s |
| test_source2 | 4.89M | 33.5s | 145k/s |
| test_source3 | 5.08M | 32.9s | 153k/s |
| **Total** | **24.2M** | **~2.5 min** | ~150–210k/s (8 workers) |

Output: `data/processed/{train,test}_source{1,2,3}.parquet`, 165–520 MB each,
zstd-compressed (gitignored; regenerate with the command above).

---

## 7. Why this is a sound design (and its limits)

**Sound because:**
- It attacks *every* noise class from the taxonomy (doc 03) with a targeted,
  cheap, deterministic transform, and each fix is validated on real matches.
- It is **country-agnostic and script-aware**, so it generalises to France
  (unseen) and any other label without code changes to the control flow.
- It keeps raw + canonical, so no information is destroyed — the model decides
  how to weight exact vs fuzzy agreement.
- It is fully offline and permissively licensed (fair-play compliant).
- It is fast and reproducible from raw data.

**Known limits (tracked for improvement — doc 08):**
- Transliteration is phonetic; `epeksa`≠`apex` means some native names still
  won't share a token with S1 (mitigated by address + n-gram blocking, not
  eliminated).
- Domain-form names aren't yet word-split, so they under-tokenise (doc 06 miss
  analysis).
- `1056c` vs `1056` and city typos are left to the fuzzy classifier, not fixed
  in cleaning.
- State/region maps cover observed variants; rare romanisations may slip through
  (redundant name/number keys cover most of these).
