# 02 — Dataset Exploration (Full EDA)

Every number here was measured directly from the provided files with streaming
scripts (no sampling unless stated). This is the empirical foundation for every
downstream decision.

---

## 1. Scale

| File | Rows (records) |
|------|---------------:|
| `train/train_source1.tsv` | 2,206,821 |
| `train/train_source2.tsv` | 5,034,616 |
| `train/train_source3.tsv` | 5,285,603 |
| `train/train_ground_truth.tsv` | 2,206,821 |
| `test/test_source1.tsv` | 1,732,544 |
| `test/test_source2.tsv` | 4,887,273 |
| `test/test_source3.tsv` | 5,082,316 |

- **Total records: 24.2M** (train 12.5M + test 11.7M). Raw data is 2.3 GB.
- Ground truth has exactly one row per S1 entity (2,206,821) — confirming S1 is
  the deduplicated reference and **every** S1 entity has a ground-truth row
  (including singletons with an empty match list).
- S2 and S3 are each ~2.3–2.4× the size of S1, consistent with each real entity
  appearing 0–2 times per noisy source.

**Implication:** naive all-pairs comparison is impossible. Train alone would be
2.2M × 10.3M ≈ **2.3 × 10¹³ pairs**. Blocking is not optional; it is the
backbone of the solution.

---

## 2. Fields & data quality

All files have exactly 4 columns (`entity_id, business_name, business_address,
country`); **zero malformed rows** were found in any file.

### 2.1 Country distribution (train)

| Source | US | India |
|--------|---:|------:|
| S1 | 1,323,633 (60.0%) | 883,188 (40.0%) |
| S2 | 3,016,817 (59.9%) | 2,017,799 (40.1%) |
| S3 | 3,170,056 (60.0%) | 2,115,547 (40.0%) |

Country mix is stable across sources (~60% US / 40% India). **Test adds France**
(see §6).

### 2.2 Missingness

| Field | S1 | S2 | S3 |
|-------|---:|---:|---:|
| empty `business_name` | 0 | 0 | 0 |
| empty `business_address` | 0 | 168,967 (3.36%) | 175,916 (3.33%) |

- **Names are never empty** — the name is always at least a weak signal.
- **~3.3% of S2/S3 addresses are empty** — for those records, matching must rely
  entirely on the name. This directly shapes feature design (we cannot assume an
  address is present) and is the reason we keep a name-only matching path.

### 2.3 Non-ASCII content & scripts

| Field | S1 | S2 | S3 |
|-------|---:|---:|---:|
| non-ASCII `business_name` | 0 (0.00%) | 764,608 (15.19%) | 606,737 (11.48%) |
| non-ASCII `business_address` | 554 (0.03%) | 478,453 (9.50%) | 476,588 (9.02%) |

**S1 is fully romanized/ASCII; the noise lives in S2/S3.** Script breakdown
(records containing each script, name or address):

| Script | S2 | S3 |
|--------|---:|---:|
| Devanagari (Hindi/Marathi) | 479,500 | 394,374 |
| Kannada | 66,637 | 55,457 |
| Telugu | 65,393 | 52,500 |
| Tamil | 60,220 | 49,864 |
| Gujarati | 54,763 | 45,006 |
| Bengali | 54,757 | 45,061 |
| Malayalam | 30,640 | 24,342 |
| Gurmukhi (Punjabi) | 12,114 | 10,319 |

**This is the single most important structural finding.** Roughly one in seven
S2/S3 records carries a name in a *native Indic script* while its S1 counterpart
is romanized English. Matching those requires bridging scripts — hence
**transliteration is mandatory**, not a nice-to-have. (How we do it: doc 05.)

### 2.4 Other name/address characteristics

| Signal | S1 | S2 | S3 |
|--------|---:|---:|---:|
| domain-style names (`foo.com`) | 0 (0.00%) | 201,187 (4.00%) | 210,838 (3.99%) |
| addresses containing a digit | 96.5% | 90.6% | 90.8% |

- **~4% of S2/S3 names are web domains** (e.g. `wilfordhancock.com`). A domain
  packs the whole name into one token with no spaces — a distinct challenge for
  tokenised matching (see doc 03, doc 06).
- **Most addresses contain a number** (house/building/street number), confirming
  the street number is a broadly available, high-value anchor.

### 2.5 Length distributions (character counts)

Names cluster at 11–40 chars; addresses are longer and more variable.

Name length (buckets): the bulk of every source is 11–40 chars. S1 has a small
tail of very long names (24 names > 80 chars); S2/S3 similar.

Address length: S1 addresses are compact (mostly 21–80 chars); S2/S3 have a
heavier long tail (S2: 580,687 addresses > 80 chars) driven by verbose Indian
addresses with landmarks and multi-part building references.

---

## 3. Ground-truth structure (the target)

Measured over all 2,206,821 GT rows:

| Quantity | Value |
|----------|------:|
| S1 entities total | 2,206,821 |
| Singletons (no match) | 123,247 (**5.58%**) |
| Entities with ≥1 match | 2,083,574 (**94.42%**) |
| Total matched pairs | 7,638,365 |
| Avg matches per *matched* entity | **3.67** |

### 3.1 Match-list size distribution

| # matches | S1 entities |
|----------:|------------:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11+ | 37 |

The mode is **3–4 matches**; clusters larger than 7 are rare.

### 3.2 How matches split across sources

| # S2 per S1 | count | | # S3 per S1 | count |
|---:|---:|---|---:|---:|
| 0 | 164,498 | | 0 | 143,029 |
| 1 | 789,108 | | 1 | 716,417 |
| 2 | 652,779 | | 2 | 668,375 |
| 3 | 333,957 | | 3 | 372,443 |
| 4 | 119,078 | | 4 | 145,116 |
| 5 | 24,154 | | 5 | 35,378 |
| | | | 6+ | 2,816 |

Among matched entities:
- **both S2 & S3:** 1,776,047 (85.2%)
- only S2: 143,029
- only S3: 164,498

**Implications:**
- Most entities have **1–2 matches from each source** — small clusters. Getting
  these right dominates the macro F₀.₅.
- 85% of matched entities appear in *both* noisy sources, so cross-source
  agreement (S2 record and S3 record both pointing at the same S1) is a strong
  corroborating signal we can exploit.
- Only 5.58% singletons — so a "predict nothing" baseline scores poorly, but a
  reckless "match everything" baseline is punished hard by F₀.₅. The sweet spot
  is confident, precise matching.

---

## 4. What matches actually look like (real clusters)

Pulled directly from the training data (S1 + its true matches):

**Example A — heavy name noise, US:**
```
S1  Maure Williams Colombier Inc     85 Wayne Avenue, Ticonderoga, NY
S2  Maure Wilblims Colombier Inc     (empty address)          typo: Williams→Wilblims
S2  Maure Williams Colombier         (empty address)          suffix dropped
S3  Dréxkor                          85 Wanye Avenue, Ticonderoga Townshiip, NY   ALIAS name, address typo'd
S3  maurewilliamscolombier.com       Wayne Ave, Ticonderoga Townshiip, NY         domain form
S3  Maure Williams Inc Center        (empty address)          word inserted
```
Here one match (`Dréxkor`) has **no name overlap at all** — only the address
saves it. Another has an **empty address** — only the name saves it.

**Example B — native-script transliteration, India:**
```
S1  Raj Investments LLP                          6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu
S2  ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி              (same address, Tamil state)
S2  Raj Investments LLP                          (same address)
S3  Raj Investments எல்எல்பி                     (same address, state as TN)
S3  ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி              (same address, state in Tamil script)
```
Same address across all; names range from clean English to full Tamil script to
mixed. The address is the reliable anchor; the name needs transliteration.

**Example C — address reordering + `null`, US:**
```
S1  Dahlia Power Reliable Scientific LLC   630 45th Terrace, Kansas City, MO
S2  Dahlia Power Reliable                  KANSAS CITY, MO, 630 45ND TERRACE, null
S2  Dahlia Power Reliable Scientific       45ND TERRACE, null, KANSAS CITY, MO
S3  Dahlia Ponr Reliable Scientific LLC    Missouri, 630 45th Terrace, Kansas City   typo: Power→Ponr
```
Address components are shuffled and contain a literal `null`; `45th`→`45ND`.
Treating the address as an **order-invariant token bag** (after dropping `null`
and normalising ordinals) makes these align.

These three cases motivate the whole cleaning design (doc 03 catalogs the full
noise taxonomy).

---

## 5. Singletons (records with no match)

Singletons look like perfectly ordinary S1 rows — there is no surface feature
distinguishing them:
```
S1  International Automation Consultants Inc   329 Rev Walton Drive, Lockport, IL
S1  Gabriella's Preferred Security             1013 Girard Avenue, Indianola, IA
S1  Twyla's Liquor Corp                        320 Flannery Lane, Silver Spring, MD
```
We cannot detect singletons a priori; they emerge naturally when **no S2/S3
candidate clears the matching threshold**. Because a singleton scores 1.0 only if
we emit nothing for it, the precision-first threshold is exactly what protects
these 123k entities (5.58%) from false-merge damage.

---

## 6. Test set & the France surprise

The test set mirrors train's structure but introduces a **third country not seen
in training**:

| File | France records |
|------|---------------:|
| `test_source1.tsv` | 259,452 |
| `test_source2.tsv` | 703,378 |
| `test_source3.tsv` | 731,615 |

≈15% of the test set is French. Examples:
```
S1  << Team Ecole          175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine
S1  ZNB Club SARL          Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac
S2  Marina Ecole France Sarl   63 R. DE DIEPPE, LILLE, Hauts-de-France
S3  Fractales Amis Groupe S.A.S   23 Rue Icmre, La Teste-de-buch, Gironde
```
French data brings its own **legal suffixes** (SARL, SAS, SASU, SA, EURL, SCI),
**address vocabulary** (Rue, Boulevard, Avenue, "bis"), and **regions**
(Nouvelle-Aquitaine, Hauts-de-France, Nord, Gironde) — plus the same accent,
typo, and reordering noise as the other countries.

**Critical implication:** the pipeline must be **country-agnostic**. We must not
hard-code, filter, or one-hot to {US, India}. Every normalisation table
(legal suffixes, states/regions, street types) is built as an open, extensible
set, and every test entity — France included — must appear in the submission.

---

## 7. Summary of EDA-driven design mandates

| Finding | Design consequence |
|---------|--------------------|
| 24.2M records, ~10¹³ naive pairs | Blocking is mandatory; recall ceiling set by blocking |
| S1 clean, S2/S3 noisy | Normalise everything to a canonical space before comparison |
| 12–15% native-script names | **Offline transliteration** is required (esp. for India) |
| 4% domain-form names | Special handling for domains (split / substring) |
| 3.3% empty addresses | Keep a **name-only** matching path |
| Addresses reordered, contain `null` | Order-invariant address **token bag**; drop `null` |
| Legal-suffix drift, `&/+/and`, accents, typos | Canonicalise suffixes, fold ampersands, strip accents, fuzzy match |
| France unseen in train | Country-agnostic pipeline, open normalisation tables |
| 94.4% have matches, mode 3–4, 85% both sources | Recall matters; cross-source agreement is a usable signal |
| F₀.₅ precision-heavy; 5.6% singletons | High, F₀.₅-tuned decision threshold; never over-emit |

Quantitative similarity/recall evidence for these choices is in
[doc 04](04_signal_analysis.md); the cleaning that acts on them is in
[doc 05](05_cleaning_pipeline.md).
