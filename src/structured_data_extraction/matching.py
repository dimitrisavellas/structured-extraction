"""Same-fact equivalence and field comparison.

Decides whether two extracted records describe the same real-world fact so that
rater.py can cluster them for consensus scoring. The rule: two records match
when every non-metadata field either agrees or is empty on at least one side
(empty = wildcard, so a sparse record still matches a fuller one).

Numeric fields must match exactly (after float parsing). Unit fields match after
Unicode/dash normalisation and synonym expansion (e.g. "millimeters" == "mm").
Descriptor text (conditions, qualifiers) uses a rapidfuzz token-set similarity
threshold of 85 — fuzzy because models phrase conditions differently
("TA = −40°C" vs "T_A = -40°C"), but exact on the number so distinct specs are
never merged.

Also provides normalize_key (user key → valid Pydantic field name) and
validate_keys (trust-boundary cleaning of user input).
"""

import logging
import re
import unicodedata
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

FUZZ_THRESHOLD = 85

MATCH, MISMATCH, GAP = "match", "mismatch", "gap"
_DASHES = str.maketrans("–—−‐‑‒", "------")


def _norm(value):
    return unicodedata.normalize("NFKC", str(value)).translate(_DASHES).strip().casefold()


def _empty(value):
    return value is None or _norm(value) == ""


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


_LEADING_NUM_RE = re.compile(r"^-?\d+(?:[.\-]\d+)*\s*")
_UNIT_SYNONYMS = {"millimeters": "mm", "millimeter": "mm", "inches": "in", "inch": "in"}


def _norm_unit(value):
    # drop a leading value number so a baked-in "5 µM" matches a standalone unit "µM"
    u = _LEADING_NUM_RE.sub("", _norm(value)).replace(" ", "").replace("μ", "u")
    return _UNIT_SYNONYMS.get(u, u)


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_NONALNUM_RE = re.compile(r"[^0-9a-z]+")


def _numbers(value):
    return frozenset(round(float(x), 6) for x in _NUM_RE.findall(str(value).translate(_DASHES)))


_SYNONYMS = {"typical": "typ", "minimum": "min", "maximum": "max"}


def _norm_text(value):
    # strip underscores first so subscripts collapse ("R_L" == "RL"), then tokenise
    s = unicodedata.normalize("NFKC", str(value)).translate(_DASHES).casefold().replace("_", "")
    return " ".join(_SYNONYMS.get(t, t) for t in _NONALNUM_RE.sub(" ", s).split())


def normalize_key(key):
    # strip("_") so edge punctuation (a quote, a leading "+") can't produce a leading/trailing
    # underscore — Pydantic rejects field names that start with "_".
    return re.sub(r"[^a-z0-9_]", "_", unicodedata.normalize("NFKC", key).lower()).strip("_")


def validate_keys(keys):
    """Clean + validate user keys at the trust boundary, returning the cleaned key list.

    Repairs the common cases (strips surrounding quotes/whitespace, drops blanks and exact
    duplicates) and rejects what can't become a valid, unique Pydantic field name:
    a key with no alphanumerics (empty field name) or two distinct keys that normalize to the
    same field name (a silent collision that would otherwise drop one key's data). Raises
    ValueError with a user-facing message on unrepairable input.
    """
    cleaned = []
    for k in keys:
        c = k.strip().strip("\"'").strip()
        if not c:
            if k.strip():
                logger.info(f"Dropped key {k!r} (empty after cleaning)")
            continue
        if c != k:
            logger.info(f"Cleaned key {k!r} -> {c!r}")
        cleaned.append(c)

    cleaned = list(dict.fromkeys(cleaned))  # drop exact duplicates, keep order
    if not cleaned:
        raise ValueError("no usable keys provided")

    by_field = {}
    for k in cleaned:
        field = normalize_key(k)
        if not field:
            raise ValueError(f"key {k!r} has no letters or digits to form a field name — rename it")
        if field in by_field:
            raise ValueError(
                f"keys {by_field[field]!r} and {k!r} both map to field name "
                f"{field!r} — make them distinct"
            )
        by_field[field] = k
    return cleaned


def model_slug(model_name):
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def compare(name, a, b):
    if _empty(a) or _empty(b):
        return GAP

    na, nb = _as_number(a), _as_number(b)
    if na is not None and nb is not None:
        return MATCH if na == nb else MISMATCH

    if name == "unit" or name.endswith("_unit"):
        return MATCH if _norm_unit(a) == _norm_unit(b) else MISMATCH

    # one side a number, the other a string carrying exactly that number ("10 mL" vs 10)
    if na is not None and _numbers(b) == {na}:
        return MATCH
    if nb is not None and _numbers(a) == {nb}:
        return MATCH

    numa, numb = _numbers(a), _numbers(b)
    if not (numa <= numb or numb <= numa):
        return MISMATCH
    return (
        MATCH if fuzz.token_set_ratio(_norm_text(a), _norm_text(b)) >= FUZZ_THRESHOLD else MISMATCH
    )


_META_FIELDS = {"metadata", "confidence", "models"}


def same_fact(a, b):
    fields = (set(a) | set(b)) - _META_FIELDS
    for f in fields:
        if compare(f, a.get(f), b.get(f)) == MISMATCH:
            return False
    return True
