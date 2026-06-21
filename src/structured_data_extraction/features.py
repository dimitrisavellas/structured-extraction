import re
import numpy as np
from scipy import sparse

DIST_CAP = 20

BULLET_RE = re.compile(r"^[•●○▪►→■□◆◇★☆\-\*]\s*")
N_BULLET_RE = re.compile(r"^n\s+\S")
FIG_RE = re.compile(r"^(Fig\.?|Figure|Table|Tab\.)\s*\d", re.I)
UNITS = set("ΩμµΩ°±")

HASH_RE = re.compile(r"^(#{1,6})\s")
FIG_TABLE_PREFIX_RE = re.compile(r"^(Figure|Fig\.|Table|Tab\.)\s+[A-Z0-9][A-Z0-9\-\.]*[\.\:]\s*", re.I)
SENTENCE_BOUNDARY_RE = re.compile(r"[a-z\)\"\']\.\s+[A-Z]")
NUM_LIST_RE = re.compile(r"^\d+[\.\)]\s")
DASH_RE = re.compile(r"^[-–—]\s")
PAREN_RE = re.compile(r"\([^)]{1,40}\)")


def _sentence_count(text):
    stripped = FIG_TABLE_PREFIX_RE.sub("", text.strip())
    return len(SENTENCE_BOUNDARY_RE.findall(stripped)) + 1


def regex_features(text):
    n_sent = _sentence_count(text)
    words = text.split()
    wc = max(len(words), 1)
    n = len(text)
    letters = [c for c in text if c.isalpha()]
    upper_ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
    title_words = sum(1 for w in words if len(w) > 1 and w[0].isupper() and w[1:].islower())
    hash_m = HASH_RE.match(text)
    hash_depth = len(hash_m.group(1)) if hash_m else 0
    return [
        float("|" in text), float("\n" in text), float(text.count("\n") > 2),
        float(bool(NUM_LIST_RE.match(text))), float(bool(BULLET_RE.match(text))),
        float(bool(N_BULLET_RE.match(text))), float(bool(DASH_RE.match(text))),
        float(bool(FIG_RE.match(text))), float(n < 80), float(n < 40), float(n < 20),
        float(text.endswith(".")), float(text.endswith(":")),
        float(text[-1] in ".:;?!") if text else 0.0, float(text.count(".") <= 1),
        float(bool(PAREN_RE.search(text))),
        float(sum(1 for w in words[:3] if w.isupper() and len(w) > 1)),
        float(text.isupper() and any(c.isalpha() for c in text)),
        upper_ratio, title_words / wc, float(text[0].isupper()) if text else 0.0,
        sum(c.isdigit() for c in text) / max(n, 1), float(any(c in UNITS for c in text)),
        sum(len(w) for w in words) / wc, float(n), float(wc),
        float(hash_depth > 0), float(hash_depth) / 6.0,
        float(n_sent == 1), float(min(n_sent, 10)),
    ]


def _normalize_text(text, dl):
    # restore bullet marker on list_items that lost it
    if dl == "list_item" and text and not BULLET_RE.match(text):
        return "- " + text
    return text


def build_features(data, dl_idx, word_vec, char_vec, fit):
    texts = [_normalize_text(r["text"], r["dl"]) for r in data]
    word = word_vec.fit_transform(texts) if fit else word_vec.transform(texts)
    char = char_vec.fit_transform(texts) if fit else char_vec.transform(texts)
    k = len(dl_idx)
    ctx = np.zeros((len(data), 3 * k + 1), dtype=np.float32)
    for i, r in enumerate(data):
        ctx[i, dl_idx.get(r["dl"], 0)] = 1.0
        ctx[i, k + dl_idx.get(r["prev"], 0)] = 1.0
        ctx[i, 2 * k + dl_idx.get(r["next"], 0)] = 1.0
        ctx[i, 3 * k] = r["dist"]
    regex = sparse.csr_matrix([regex_features(t) for t in texts])
    return sparse.hstack([word, char, regex, sparse.csr_matrix(ctx)]).tocsr()


def context_rows(chunks, text_key="text"):
    chunks = sorted(chunks, key=lambda c: int(c["section_id"]))
    n = len(chunks)
    rows = []
    for i, c in enumerate(chunks):
        dist = next((min(i - b, DIST_CAP) for b in range(i, -1, -1)
                     if chunks[b]["docling_label"] == "section_header"), DIST_CAP)
        rows.append({"text": c.get(text_key, c.get("text", "")), "dl": c["docling_label"],
                     "prev": chunks[i - 1]["docling_label"] if i > 0 else "",
                     "next": chunks[i + 1]["docling_label"] if i < n - 1 else "", "dist": dist})
    return chunks, rows
