"""
normalize.py — text cleaning & canonicalization for Business Entity Resolution.

This module is the single source of truth for turning a raw, noisy record
(business_name, business_address, country) into canonical fields and blocking
keys. It is deliberately dependency-light (only ``indic_transliteration``) and
country-agnostic: it works for US / India / France and any other label without
hard-coding a closed country set.

Everything here is pure, offline string processing — no external data lookup,
no network, no geocoding — so it complies with the challenge fair-play rules.

Design notes discovered during EDA:
  * S1 is the clean, romanized reference; S2/S3 carry the noise.
  * ~12-15% of S2/S3 names are in native scripts (Devanagari/Tamil/Telugu/
    Kannada/Bengali/Gujarati/Malayalam/Gurmukhi) -> transliterate to Latin.
  * Addresses are frequently REORDERED and contain literal "null" tokens, so we
    treat the address as an order-invariant token bag plus a few extracted
    anchors (street number, state).
  * Names carry legal-suffix noise (Inc/Ltd/Pvt/LLC/SARL/SAS...), added accents,
    domain forms (foo.com), appended phone numbers, and &/+/and variation.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# --------------------------------------------------------------------------- #
# Script detection & transliteration (native Indic script -> Latin)
# --------------------------------------------------------------------------- #

# (unicode range) -> sanscript scheme, covering the scripts seen in EDA.
_SCRIPT_RANGES = [
    ((0x0900, 0x097F), sanscript.DEVANAGARI),
    ((0x0980, 0x09FF), sanscript.BENGALI),
    ((0x0A00, 0x0A7F), sanscript.GURMUKHI),
    ((0x0A80, 0x0AFF), sanscript.GUJARATI),
    ((0x0B80, 0x0BFF), sanscript.TAMIL),
    ((0x0C00, 0x0C7F), sanscript.TELUGU),
    ((0x0C80, 0x0CFF), sanscript.KANNADA),
    ((0x0D00, 0x0D7F), sanscript.MALAYALAM),
]

# Widest span of "any Indic script" for a fast membership pre-check.
_INDIC_LO, _INDIC_HI = 0x0900, 0x0DFF

# Devanagari candra vowel signs used for English loan sounds are not mapped by
# the IAST scheme; fold them to their nearest standard vowel sign first so
# e.g. "मॉडर्न" romanizes as "modarna" rather than leaving a raw "ॉ".
_CANDRA_FOLD = {
    "ॅ": "े",  # candra E  -> E
    "ॉ": "ो",  # candra O  -> O
}
_CANDRA_RE = re.compile("[" + "".join(_CANDRA_FOLD) + "]")


def _has_indic(s: str) -> bool:
    return any(_INDIC_LO <= ord(c) <= _INDIC_HI for c in s)


def _detect_scheme(s: str):
    for (lo, hi), scheme in _SCRIPT_RANGES:
        if any(lo <= ord(c) <= hi for c in s):
            return scheme
    return None


def strip_accents(s: str) -> str:
    """Remove combining diacritics (Énterprises -> Enterprises)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def romanize(s: str) -> str:
    """Transliterate any native-script content in ``s`` to accent-free Latin.

    Latin/ASCII text is returned unchanged (fast path). Mixed strings (e.g.
    "Raj Investments எல்எல்பி") are transliterated as a whole; the Latin part
    survives the round-trip.
    """
    if not s or not _has_indic(s):
        return s
    scheme = _detect_scheme(s)
    if scheme is None:
        return s
    s = _CANDRA_RE.sub(lambda m: _CANDRA_FOLD[m.group()], s)
    try:
        out = transliterate(s, scheme, sanscript.IAST)
    except Exception:
        return s
    return strip_accents(out)


# --------------------------------------------------------------------------- #
# Legal suffixes / entity-type tokens (multi-country, open-ended)
# --------------------------------------------------------------------------- #

# True legal-entity type tokens + a few pure stopwords. These carry little
# identifying signal (they are added/dropped/reordered freely across sources),
# so we record them as a light feature but drop them from the CORE token set.
#
# IMPORTANT: only genuine legal suffixes and stopwords go here. Descriptive
# words like "ventures/enterprises/services/solutions" are kept in the core —
# stripping them collapses distinct businesses ("Red Ventures" vs "Red Trading")
# and destroys precision. Over-common core tokens are instead handled at
# BLOCKING time via document-frequency pruning, not by deletion here.
LEGAL_TOKENS = {
    # English / US / India entity types
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "lc",
    "llp", "ltd", "limited", "pvt", "private", "plc", "pc", "pllc", "lp",
    # France entity types
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scp", "selarl", "gie",
    "ei", "eirl", "earl", "gaec", "selas", "sem", "scop", "scic", "sca",
    # generic international entity types
    "gmbh", "ag", "bv", "nv", "srl", "spa", "oy", "ab", "as", "kk",
    # pure stopwords
    "and", "the", "of",
    # phonetic romanizations of Indic legal suffixes (from transliteration), so
    # "praiveta limiteda" folds to the same core as "private limited".
    "praiveta", "limiteda", "elaelapi", "elaelabhi", "elaelasi", "elelapi",
    "elelbhi", "kampani", "karporesana", "praivet", "limited", "praiveta",
    "praiveṭa", "inda",
}


def _fold_ampersand(s: str) -> str:
    """Unify '&', '+', ' and ' so 'A & B' == 'A and B' == 'A + B'."""
    s = s.replace("&", " and ").replace("+", " and ")
    return s


# Name-token abbreviation synonyms (mostly French/English), folded to a canonical
# form so e.g. "Ets" and "Établissements" match.
NAME_SYNONYMS = {
    "ets": "etablissements", "etabl": "etablissements", "etab": "etablissements",
    "asso": "association", "assoc": "association", "intl": "international",
    "intnl": "international", "mfg": "manufacturing", "cie": "compagnie",
}


_PHONE_RE = re.compile(r"[-+(]?\b\d[\d\s().-]{6,}\d\b")
_DOMAIN_RE = re.compile(r"\b([a-z0-9][a-z0-9-]*)\.(?:com|net|org|io|co|in|us|biz|info|shop|store)\b", re.I)
_WS_RE = re.compile(r"\s+")


def is_domain_name(raw: str) -> bool:
    return bool(_DOMAIN_RE.search(raw or ""))


def clean_name(raw: str):
    """Canonicalize a business name.

    Returns ``(canon, core_tokens, legal_tokens_present)`` where
      * ``canon``  — space-joined core tokens (order-preserved, deduped-adjacent);
      * ``core_tokens`` — frozenset of identifying tokens (legal suffixes removed);
      * ``legal_tokens_present`` — frozenset of legal/entity tokens found.
    """
    if raw is None:
        raw = ""
    s = romanize(raw)
    s = strip_accents(s).lower()
    # split a domain like "maurewilliamscolombier.com" -> keep the label
    m = _DOMAIN_RE.search(s)
    if m:
        s = s.replace(m.group(0), " " + m.group(1) + " ")
    s = _fold_ampersand(s)
    s = s.replace("'", "").replace("’", "")  # JOIN apostrophes: L'Epicerie->lepicerie, Moyna's->moynas
    s = _PHONE_RE.sub(" ", s)                 # drop appended phone numbers
    s = re.sub(r"[^a-z0-9\s]", " ", s)        # punctuation -> space
    s = _WS_RE.sub(" ", s).strip()

    tokens = s.split()
    core, legal = [], []
    for t in tokens:
        t = NAME_SYNONYMS.get(t, t)           # fold abbreviations (Ets->etablissements)
        if t in LEGAL_TOKENS:
            legal.append(t)
        else:
            core.append(t)
    # if a name is ONLY legal tokens, keep them so it isn't blank
    if not core:
        core = tokens
    canon = " ".join(core)
    return canon, frozenset(core), frozenset(legal)


# --------------------------------------------------------------------------- #
# State / region normalization (US states, Indian states, French regions)
# --------------------------------------------------------------------------- #
# Maps many surface forms -> one canonical token. Native-script state names are
# handled because romanize() runs first (e.g. "தமிழ்நாடு" -> "tamilnadu").

def _sm(canon, *variants):
    return {v: canon for v in variants}


_STATE_MAP = {}
# --- US states (abbrev + full) ---
_US = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "newhampshire", "nj": "newjersey", "nm": "newmexico", "ny": "newyork",
    "nc": "northcarolina", "nd": "northdakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhodeisland",
    "sc": "southcarolina", "sd": "southdakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "westvirginia", "wi": "wisconsin", "wy": "wyoming", "dc": "districtofcolumbia",
}
for ab, full in _US.items():
    _STATE_MAP[ab] = full
    _STATE_MAP[full] = full

# --- Indian states / UTs (abbrev + full + common romanized forms) ---
_IN = {
    "ap": "andhrapradesh", "ar": "arunachalpradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachalpradesh", "jh": "jharkhand", "ka": "karnataka", "kl": "kerala",
    "mp": "madhyapradesh", "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya",
    "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "odisha",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamilnadu",
    "tg": "telangana", "ts": "telangana", "tr": "tripura", "up": "uttarpradesh",
    "uk": "uttarakhand", "ua": "uttarakhand", "wb": "westbengal", "dl": "delhi",
    "jk": "jammuandkashmir", "la": "ladakh", "ch": "chandigarh", "py": "puducherry",
}
# NOTE: some 2-letter codes (GA, IN, OR, LA, CH) collide with US codes. Country
# context resolves this at call time (see normalize_state), so we keep both maps.
_IN_FULL = {
    "andhrapradesh", "arunachalpradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachalpradesh", "jharkhand", "karnataka", "kerala",
    "madhyapradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "punjab", "rajasthan", "sikkim", "tamilnadu", "telangana", "tripura",
    "uttarpradesh", "uttarakhand", "westbengal", "delhi", "newdelhi",
    "jammuandkashmir", "ladakh", "chandigarh", "puducherry",
    # frequent romanized/native-transliterated variants
    "maharastra", "hariyana", "dhamilnadhu", "karnataka",
}
for f in _IN_FULL:
    _STATE_MAP.setdefault(f, "delhi" if f == "newdelhi" else f)

# French regions (as an open set — we canonicalize the common ones seen).
_FR_FULL = {
    "nouvelleaquitaine", "hautsdefrance", "iledefrance", "occitanie",
    "grandest", "normandie", "bretagne", "paysdelaloire", "centrevaldeloire",
    "bourgognefranchecomte", "provencealpescotedazur", "auvergnerhonealpes",
    "corse",
}
for f in _FR_FULL:
    _STATE_MAP[f] = f

# French DÉPARTEMENT -> RÉGION. S1 tends to carry the région ("Nouvelle-Aquitaine")
# while S2/S3 carry the département ("Gironde", "Nord"); mapping both to the région
# makes them normalize identically (same idea as US state abbrev -> full).
_FR_DEPT_TO_REGION = {
    # Nouvelle-Aquitaine
    "gironde": "nouvelleaquitaine", "dordogne": "nouvelleaquitaine",
    "landes": "nouvelleaquitaine", "pyreneesatlantiques": "nouvelleaquitaine",
    "charente": "nouvelleaquitaine", "charentemaritime": "nouvelleaquitaine",
    "viennent": "nouvelleaquitaine", "vienne": "nouvelleaquitaine",
    "deuxsevres": "nouvelleaquitaine", "correze": "nouvelleaquitaine",
    "creuse": "nouvelleaquitaine", "hautevienne": "nouvelleaquitaine",
    "lotetgaronne": "nouvelleaquitaine",
    # Hauts-de-France
    "nord": "hautsdefrance", "pasdecalais": "hautsdefrance",
    "somme": "hautsdefrance", "aisne": "hautsdefrance", "oise": "hautsdefrance",
    # Île-de-France
    "paris": "iledefrance", "seineetmarne": "iledefrance",
    "yvelines": "iledefrance", "essonne": "iledefrance",
    "hautsdeseine": "iledefrance", "seinesaintdenis": "iledefrance",
    "valdemarne": "iledefrance", "valdoise": "iledefrance",
    # Occitanie
    "hautegaronne": "occitanie", "herault": "occitanie", "gard": "occitanie",
    "aude": "occitanie", "tarn": "occitanie", "gers": "occitanie",
    "lot": "occitanie", "aveyron": "occitanie", "lozere": "occitanie",
    "pyreneesorientales": "occitanie", "hautespyrenees": "occitanie",
    "ariege": "occitanie", "tarnetgaronne": "occitanie",
    # Auvergne-Rhône-Alpes
    "rhone": "auvergnerhonealpes", "isere": "auvergnerhonealpes",
    "loire": "auvergnerhonealpes", "ain": "auvergnerhonealpes",
    "drome": "auvergnerhonealpes", "ardeche": "auvergnerhonealpes",
    "puydedome": "auvergnerhonealpes", "hautesavoie": "auvergnerhonealpes",
    "savoie": "auvergnerhonealpes", "allier": "auvergnerhonealpes",
    "cantal": "auvergnerhonealpes", "hauteloire": "auvergnerhonealpes",
    # Provence-Alpes-Côte d'Azur
    "bouchesdurhone": "provencealpescotedazur", "var": "provencealpescotedazur",
    "alpesmaritimes": "provencealpescotedazur", "vaucluse": "provencealpescotedazur",
    "alpesdehauteprovence": "provencealpescotedazur", "hautesalpes": "provencealpescotedazur",
    # Grand Est
    "basrhin": "grandest", "hautrhin": "grandest", "moselle": "grandest",
    "meurtheetmoselle": "grandest", "marne": "grandest", "aube": "grandest",
    "vosges": "grandest", "ardennes": "grandest", "meuse": "grandest", "hautemarne": "grandest",
    # Bretagne
    "finistere": "bretagne", "morbihan": "bretagne", "cotesdarmor": "bretagne",
    "illeetvilaine": "bretagne",
    # Pays de la Loire
    "loireatlantique": "paysdelaloire", "maineetloire": "paysdelaloire",
    "vendee": "paysdelaloire", "sarthe": "paysdelaloire", "mayenne": "paysdelaloire",
    # Normandie
    "seinemaritime": "normandie", "calvados": "normandie", "manche": "normandie",
    "eure": "normandie", "orne": "normandie",
    # Bourgogne-Franche-Comté
    "cotedor": "bourgognefranchecomte", "saoneetloire": "bourgognefranchecomte",
    "doubs": "bourgognefranchecomte", "yonne": "bourgognefranchecomte",
    "nievre": "bourgognefranchecomte", "jura": "bourgognefranchecomte",
    "hautesaone": "bourgognefranchecomte", "territoiredebelfort": "bourgognefranchecomte",
    # Centre-Val de Loire
    "loiret": "centrevaldeloire", "indreetloire": "centrevaldeloire",
    "loiretcher": "centrevaldeloire", "cher": "centrevaldeloire",
    "eureetloir": "centrevaldeloire", "indre": "centrevaldeloire",
}
_STATE_MAP.update(_FR_DEPT_TO_REGION)
_FR_FULL |= set(_FR_DEPT_TO_REGION.values())

# Romanized-native-script variants -> canonical. When an Indic state name is
# written in its own script (e.g. "தமிழ்நாடு", "महाराष्ट्र"), romanize() yields a
# phonetic Latin form that differs from the English spelling ("dhamilnadhu",
# "maharastra"). Map those to the same canonical token so the street#+state
# blocking anchor aligns across sources.
_STATE_VARIANTS = {
    "dhamilnadhu": "tamilnadu", "tamilnadu": "tamilnadu", "tamilnad": "tamilnadu",
    "maharastra": "maharashtra", "maharashtra": "maharashtra", "maharasta": "maharashtra",
    "hariyana": "haryana", "haryana": "haryana",
    "karnataka": "karnataka", "karnatak": "karnataka",
    "uttarapradesa": "uttarpradesh", "uttarpradesh": "uttarpradesh",
    "uttarapradesh": "uttarpradesh", "uttaraprades": "uttarpradesh",
    "madhyapradesa": "madhyapradesh", "madhyapradesh": "madhyapradesh",
    "rajasthana": "rajasthan", "rajasthan": "rajasthan",
    "dilli": "delhi", "dilee": "delhi", "delhi": "delhi",
    "gujarata": "gujarat", "gujarat": "gujarat",
    "keralam": "kerala", "kerala": "kerala", "kerela": "kerala",
    "pascimabangala": "westbengal", "pascimabanga": "westbengal",
    "andhrapradesa": "andhrapradesh", "andhrapradesh": "andhrapradesh",
    "telamgana": "telangana", "telangana": "telangana",
    "bihara": "bihar", "bihar": "bihar",
    "asama": "assam", "assam": "assam",
    "pambaba": "punjab", "panjaba": "punjab", "punjab": "punjab",
    "gova": "goa", "goa": "goa",
    "odisa": "odisha", "odisha": "odisha", "udisa": "odisha",
    "jharakhanda": "jharkhand", "jharkhand": "jharkhand",
    "chattisagadha": "chhattisgarh", "chhattisgarh": "chhattisgarh",
    "himacalaprades": "himachalpradesh", "himachalpradesh": "himachalpradesh",
    "uttarakhanda": "uttarakhand", "uttarakhand": "uttarakhand",
}
_STATE_MAP.update(_STATE_VARIANTS)

# Unambiguous FULL-FORM state variants (>=4 chars) -> canonical, safe to apply
# to arbitrary address tokens (unlike 2-letter codes, which collide with common
# words). Used to fold e.g. "hariyana"->"haryana" wherever it appears.
_FULL_STATE_CANON = {k: v for k, v in _STATE_MAP.items() if len(k) >= 4}

# Country-specific overrides where a 2-letter code is ambiguous.
_IN_OVERRIDE = dict(_IN)   # applied when country == India


def normalize_state(token: str, country: str = "") -> str:
    """Map a raw state/region token to a canonical form (country-aware)."""
    t = re.sub(r"[^a-z]", "", romanize(token).lower())
    if not t:
        return ""
    if (country or "").lower() in ("india", "in") and t in _IN_OVERRIDE:
        return _IN_OVERRIDE[t]
    return _STATE_MAP.get(t, t)


# --------------------------------------------------------------------------- #
# Address canonicalization
# --------------------------------------------------------------------------- #

# Street-type abbreviations -> canonical, so "Rd"=="Road", "St"=="Street".
_STREET_ABBR = {
    "rd": "road", "st": "street", "str": "street", "saint": "street",
    "ave": "avenue", "av": "avenue", "blvd": "boulevard", "boulevard": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle", "pl": "place",
    "sq": "square", "ter": "terrace", "trl": "trail", "hwy": "highway",
    "pkwy": "parkway", "apt": "apartment", "ste": "suite", "fl": "floor",
    "rue": "rue", "r": "rue", "bd": "boulevard",
}
_STOP_ADDR = {"null", "na", "none", "po", "box", "pobox", "no", "number", "door",
              "th", "nd", "rd", "st", "numero", "num", "ndeg", "deg"}
# NOTE: "rd"/"st" appear in both street-abbr and ordinal-suffix contexts; the
# street-abbr map wins (applied first), so these stopwords only catch leftovers.

_HOUSE_NUM_RE = re.compile(r"\b(\d{1,6})\b")


def clean_address(raw: str, country: str = ""):
    """Canonicalize an address into order-invariant, matchable pieces.

    Returns ``(canon, addr_tokens, street_number, state)``:
      * ``canon`` — normalized, sorted token string (order-invariant);
      * ``addr_tokens`` — frozenset of content tokens (abbrev-normalized);
      * ``street_number`` — the first standalone number, or "" (house/building);
      * ``state`` — canonical state/region token, or "".
    """
    if raw is None:
        raw = ""
    s = romanize(raw)
    s = strip_accents(s).lower()
    s = s.replace("'", "").replace("’", "")   # join apostrophes (l'eglise->leglise)
    s = re.sub(r"n\s*[°o]\s*(?=\d)", " ", s)   # drop "N°"/"No" before a number

    # Find the state/region by scanning ALL comma-separated fields (addresses
    # are frequently reordered, so it is not always last). Prefer the last
    # matching field, which is the most common position.
    state = ""
    fields = [f.strip() for f in s.split(",") if f.strip()]
    _valid_states = set(_STATE_MAP.values()) | _FR_FULL
    for fld in fields:
        cand = normalize_state(fld, country)
        if cand in _valid_states:
            state = cand

    s = re.sub(r"[^a-z0-9\s]", " ", s)
    raw_tokens = _WS_RE.sub(" ", s).strip().split()

    # first standalone house/building number
    street_number = ""
    for t in raw_tokens:
        if t.isdigit() and 1 <= len(t) <= 6:
            street_number = t.lstrip("0") or t
            break

    tokens = set()
    for t in raw_tokens:
        if t in _STOP_ADDR:
            continue
        t = _STREET_ABBR.get(t, t)
        # normalize ordinal house-number forms: 45th/45nd/2nd -> 45/2
        m = re.fullmatch(r"(\d+)(st|nd|rd|th)", t)
        if m:
            t = m.group(1)
        # strip leading zeros on numeric tokens so 0684 == 684
        if t.isdigit():
            t = t.lstrip("0") or "0"
        # fold unambiguous full-form state variants (hariyana -> haryana)
        t = _FULL_STATE_CANON.get(t, t)
        if len(t) == 1 and t.isalpha():
            continue
        tokens.add(t)
    if state:
        tokens.add(state)

    canon = " ".join(sorted(tokens))
    return canon, frozenset(tokens), street_number, state


# --------------------------------------------------------------------------- #
# Blocking keys & similarity helpers
# --------------------------------------------------------------------------- #

def char_ngrams(s: str, n: int = 3) -> frozenset:
    """Character n-grams over the space-stripped string (typo-robust)."""
    s = s.replace(" ", "")
    if len(s) < n:
        return frozenset([s]) if s else frozenset()
    return frozenset(s[i : i + n] for i in range(len(s) - n + 1))


def blocking_keys(canon_name_tokens, street_number, state, city_token=""):
    """Produce a small set of blocking keys for a record.

    Keys are strings; two records are candidate-compared if they share ANY key.
    Chosen from EDA to reach a ~99% recall ceiling while keeping block sizes
    manageable:
      * sorted first-2 name tokens (name signal, order-invariant)
      * each individual name token >= 4 chars (catches reordering / partial)
      * street_number + state (strong address anchor, country-agnostic)
    """
    keys = set()
    toks = sorted(canon_name_tokens)
    if toks:
        keys.add("n2:" + "|".join(toks[:2]))
    for t in toks:
        if len(t) >= 4:
            keys.add("nt:" + t)
    if street_number:
        keys.add("sn:" + street_number + ":" + (state or ""))
    return keys


@lru_cache(maxsize=100_000)
def _cached_clean_name(raw):
    return clean_name(raw)
