# 03 — Noise Taxonomy

A catalog of every noise pattern observed in the data, each with real examples
and the concrete pipeline response it demands. This is the "threat model" the
cleaning (doc 05) and blocking (doc 06) are designed against.

All examples are verbatim from the training data.

---

## A. Business-name noise

### A1. Character-level typos & scrambles
The most common and most damaging for exact/token matching.
```
Williams   → Wilblims          (Maure Williams Colombier)
Power      → Ponr              (Dahlia ... Scientific)
Enterprises→ Enterpires / Etrepndiels / ENRTPRMISES
Partners   → Pagnters
Golden     → Gbn
Telecommunication → Tetlecommunication
```
Sometimes near-anagram-level corruption (`PAYNE-ENRTPRMISES`).
**Response:** character-level fuzzy similarity (Levenshtein / Jaro-Winkler) and
**character n-grams** rather than whole tokens. Char-trigrams of `Enterpires`
and `Enterprises` still overlap heavily; tokens do not.

### A2. Added / spurious accents
```
Enterprises → Énterprises
Boral       → Bóral
Learning    → Léarning
Partners    → Pártners
Drexkor     → Dréxkor
```
**Response:** Unicode NFKD normalisation + **accent stripping** so accented and
plain forms collapse to the same characters.

### A3. Native-script transliteration (the big one — India)
Full or partial rendering of the name in an Indic script:
```
एसएस फूड प्राइवेट लिमिटेड              = "SS Food Private Limited"
राम मार्केटिंग प्राइवेट लिमिटेड          = "Ram Marketing Private Limited"
ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி        = "Raj Investments LLP" (Tamil)
Raj Investments எல்எல்பி              = mixed: Latin name + Tamil "LLP"
```
**Response:** offline **script→Latin transliteration** (indic-transliteration,
IAST scheme) applied before any other name processing. Handles the 8 scripts
seen in the data. Romanization is phonetic, not spelling-exact
(`फूड`→`phuda`≈"food"), so it must be paired with fuzzy/n-gram matching.

### A4. Legal-suffix drift
Legal-entity types are added, dropped, reordered, or mis-placed freely:
```
Maure Williams Colombier Inc  vs  Maure Williams Colombier
Payne Enterprises             vs  Payne Enterprises  LLC
Orellana Investments LLC      vs  LLC Orellana Invsmbens   (prefix + typo)
Pvt. EFS Print Ventures Ltd.  (two suffixes at once)
```
Multi-country suffixes: Inc/Corp/Co/LLC/LLP/Ltd/Limited/Pvt/Private (US, India);
SARL/SAS/SASU/SA/EURL/SCI (France).
**Response:** maintain a multi-country legal-suffix list; **separate** suffix
tokens from the identifying *core* tokens and match on the core. Record which
suffixes were present as a light feature (a suffix *conflict* is a weak
negative signal).

### A5. Domain-form names (~4%)
```
wilfordhancock.com
maurewilliamscolombier.com
agrosheronhotels.com
```
The entire name is one lowercase run with no spaces + a TLD.
**Response:** strip the TLD; treat the label as a token. **Known weakness:** a
concatenated label (`agrosheronhotels`) does not tokenise into the S1 words
(`agro sheron hotels`), so token/first-2 blocking misses it (see doc 06,
misses). Planned fix: word-piece splitting and/or char-n-gram/substring
blocking for domains.

### A6. `&` vs `+` vs `and`
```
Chordia & Partners  /  Chordia + Pagnters  /  Chordia &-Pártners Ltd
Hendricks and Flowers Inc
```
**Response:** fold `&` and `+` to the word `and`, then treat `and` as a stopword.

### A7. Word insertion / reordering
```
Maure Williams Inc Center       (word "Center" inserted)
Hendricks and Inc Flowers       ("Inc" inserted mid-name, order changed)
Hotel Limited Services          (vs "Hotel Enterprises Limited")
Smt Chordia  & Center           (honorific + word inserted)
```
**Response:** treat the core name as an **order-invariant token set**; use token
Jaccard alongside sequence-sensitive measures.

### A8. Appended phone numbers / junk
```
Chordia + Pagnters - 7306204978
-- Holloway Peak Inc Seafood       (leading punctuation junk)
<< Team Ecole                      (leading "<<")
```
**Response:** regex-strip long digit runs (phone numbers) and leading/trailing
punctuation before tokenising.

### A9. Alias / entirely different name
```
S1 "Maure Williams Colombier Inc"  ↔  S3 "Dréxkor"   (true match, address-linked)
```
The name carries **zero** identifying signal; only the address links them.
**Response:** this is inherently unrecoverable from the name — the **address
matching path** must carry these, and some (name-different *and* address-broken)
are simply lost. They define the practical recall ceiling.

---

## B. Business-address noise

### B1. Empty address (~3.3% of S2/S3)
```
International South Consultants Private Ltd   ""      (India)
Maure Williams Colombier                      ""      (US)
```
**Response:** name-only matching path; never assume an address exists.

### B2. Component reordering
```
S1  630 45th Terrace, Kansas City, MO
S2  KANSAS CITY, MO, 630 45ND TERRACE, null
S2  45ND TERRACE, null, KANSAS CITY, MO
S3  Missouri, 630 45th Terrace, Kansas City
```
**Response:** **order-invariant token bag**; do not rely on field position
(with one exception — we still *scan* fields to find the state, but check all
positions, not just the last).

### B3. Literal `null` tokens
```
630 45ND TERRACE, null   /   45ND TERRACE, null, KANSAS CITY, MO
```
**Response:** drop `null`/`na`/`none` as address stopwords.

### B4. State / region variants (abbrev ↔ full ↔ native script)
```
US:     IL ↔ Illinois   NY ↔ New York   IA ↔ Iowa   MO ↔ Missouri
India:  TN ↔ Tamil Nadu ↔ தமிழ்நாடு     KA ↔ Karnataka ↔ ಕರ್ನಾಟಕ
        UP ↔ Uttar Pradesh ↔ उत्तर प्रदेश   HR ↔ Haryana ↔ हरियाणा
France: Nouvelle-Aquitaine, Hauts-de-France, Nord, Gironde
```
**Response:** a **state/region normalisation map** collapsing all variants to a
single canonical token — including romanised native forms (`தமிழ்நாடு`→
`dhamilnadhu`→canonical `tamilnadu`). Country-aware, because some 2-letter codes
collide (e.g. `GA` = Georgia in US, Goa in India). See doc 05 §3.

### B5. House / building number drift
```
684  → 0684      (leading zero)
45th → 45ND      (ordinal typo)
1056 → 1056c → 1056-1060   (suffix letter / range)
AF-684 → AF-0684
```
**Response:** normalise ordinals (`45th/45nd/2nd`→`45/2`), strip leading zeros on
numeric tokens, extract the first standalone number as the "street number"
anchor. (`1056c` vs `1056` remains a partial mismatch — handled softly by the
classifier, not blocking.)

### B6. City / locality typos
```
Akron        → AKON
Ticonderoga  → Ticonderoga Townshiip
Faridabad    → FARIABAD
```
**Response:** fuzzy/char-n-gram address similarity in the classifier; the city
token still contributes partial n-gram overlap.

### B7. Street-type abbreviations
```
Rd ↔ Road    St ↔ Street ↔ SAINT(!)   Ave/Av ↔ Avenue   Blvd ↔ Boulevard
Rue (FR)     Dr ↔ Drive    Ln ↔ Lane
```
**Response:** street-type abbreviation map to a canonical form. (`SAINT` for
"St" is a genuine OCR/expansion error we normalise.)

### B8. Extra components: PO boxes, landmarks, phones
```
PO BOX 8807
Near Mother India Public School. Ph. 989, 9487203
New Bridge Business Centre'S 11Th Floor, N1 Block Embassy Manyata Business Tech Park
```
**Response:** these add tokens but rarely hurt (token-bag overlap is robust to
extra tokens); phone digit-runs in addresses are left as tokens (harmless) or
could be stripped. Landmarks are kept — they sometimes match across sources.

---

## C. Cross-cutting observations

- **Corruption is asymmetric:** S1 is clean; S2 and S3 independently corrupt the
  same entity in *different* ways. So an S2 record and an S3 record for the same
  business often disagree with each other as much as with S1 — but usually agree
  on *at least one* of {core name tokens, street number, address tokens}. This is
  why a **union of several blocking keys** achieves high recall (doc 04).
- **The hardest ~1%:** name-different *and* address-empty-or-broken. No offline
  method links these without external data (which is banned). They cap recall.
- **Noise is language/script-entangled:** Indian records concentrate the native
  script, address-reordering, and landmark noise; US records concentrate typos
  and suffix drift; France adds its own suffixes/regions. A country-agnostic but
  script-aware pipeline covers all three.
