import re
import numpy as np
from scipy import sparse

# How far (in items) we look back for the nearest section header before giving up.
DIST_CAP = 20

# A line that starts with a bullet (or "-"/"*") then optional space -> list item.
BULLET_RE = re.compile(r"^[•●○▪►→■□◆◇★☆\-\*]\s*")
# A lone leading "n" + space + non-space: docling sometimes mangles a "•" bullet into "n".
N_BULLET_RE = re.compile(r"^n\s+\S")
# Start of a figure/table caption: "Figure 1", "Fig. 2", "Table 3", "Tab. 4".
FIG_RE = re.compile(r"^(Fig\.?|Figure|Table|Tab\.)\s*\d", re.I)
# Characters that signal engineering/scientific units: ohm, micro (two code points), degree, plus-minus.
UNITS = set("ΩμµΩ°±")

# Markdown ATX heading: 1-6 "#" then a space. The captured "#"s give the heading depth.
HASH_RE = re.compile(r"^(#{1,6})\s")
# A caption label prefix like "Figure 1: " or "Table 3.2." Stripped before sentence counting
# so the "1." in "Figure 1." isn't mistaken for a sentence boundary.
FIG_TABLE_PREFIX_RE = re.compile(r"^(Figure|Fig\.|Table|Tab\.)\s+[A-Z0-9][A-Z0-9\-\.]*[\.\:]\s*", re.I)
# A sentence boundary: lowercase/quote/paren, then ". ", then a capital letter.
SENTENCE_BOUNDARY_RE = re.compile(r"[a-z\)\"\']\.\s+[A-Z]")
# A numbered list item: "1." or "2)" at the start, then a space.
NUM_LIST_RE = re.compile(r"^\d+[\.\)]\s")
# A dash bullet: hyphen / en-dash / em-dash at the start, then a space.
DASH_RE = re.compile(r"^[-–—]\s")
# A short parenthetical anywhere, e.g. "(typical)", "(see note)" — 1-40 non-")" chars.
PAREN_RE = re.compile(r"\([^)]{1,40}\)")


def _sentence_count(text):
    """Estimate how many sentences `text` contains.

    Strips a leading figure/table caption label first (so "Figure 1." reads as 0 boundaries,
    not 1), then counts sentence boundaries. N boundaries means N+1 sentences, hence the `+ 1`.
    A header or list item scores 1; a paragraph scores higher.
    """
    stripped = FIG_TABLE_PREFIX_RE.sub("", text.strip())
    return len(SENTENCE_BOUNDARY_RE.findall(stripped)) + 1


def regex_features(text):
    """Return the 30 hand-crafted layout/typography features for one text chunk.

    The vector mixes structural flags (is this a bullet? a heading? a table row?) with shape
    stats (length, uppercase ratio, digit density). See the module docstring: the order and
    length here are fixed by the trained model — do not change without retraining.
    """
    n_sent = _sentence_count(text)
    words = text.split()
    wc = max(len(words), 1)                                  # word count, floored at 1 to avoid /0
    n = len(text)
    letters = [c for c in text if c.isalpha()]
    upper_ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
    # Count Title Case words like Voltage. Used as a fraction below
    # (title_words / wc): one sentence-initial capital barely registers, but a heading where
    # most words are Title Case scores high.
    title_words = sum(1 for w in words if len(w) > 1 and w[0].isupper() and w[1:].islower())
    hash_m = HASH_RE.match(text)
    hash_depth = len(hash_m.group(1)) if hash_m else 0       # markdown heading level, 0 if none
    return [
        float("|" in text),                                  # has a pipe -> table row
        float("\n" in text),                                 # spans multiple lines
        float(text.count("\n") > 2),                         # has many lines -> list/table block
        float(bool(NUM_LIST_RE.match(text))),                # starts "1." / "2)" -> numbered list
        float(bool(BULLET_RE.match(text))),                  # starts with a bullet glyph
        float(bool(N_BULLET_RE.match(text))),                # starts "n " -> mangled bullet
        float(bool(DASH_RE.match(text))),                    # starts with a dash bullet
        float(bool(FIG_RE.match(text))),                     # "Figure 1" / "Table 2" caption
        float(n < 80), float(n < 40), float(n < 20),         # short / shorter / very short
        float(text.endswith(".")),                           # ends with a period (sentence-like)
        float(text.endswith(":")),                           # ends with a colon (label / lead-in)
        float(text[-1] in ".:;?!") if text else 0.0,         # ends with terminal punctuation
        float(text.count(".") <= 1),                         # at most one period 
        float(bool(PAREN_RE.search(text))),                  # contains a short parenthetical
        float(sum(1 for w in words[:3] if w.isupper() and len(w) > 1)),  # count of ALL-CAPS words in first 3
        float(text.isupper() and any(c.isalpha() for c in text)),        # whole text is uppercase
        upper_ratio,                                         # fraction of letters uppercase
        title_words / wc,                                    # fraction of Title-Case words
        float(text[0].isupper()) if text else 0.0,           # first character is uppercase
        sum(c.isdigit() for c in text) / max(n, 1),          # digit density (table/spec signal)
        float(any(c in UNITS for c in text)),                # contains a unit/symbol char
        sum(len(w) for w in words) / wc,                     # mean word length
        float(n),                                            # character length
        float(wc),                                           # word count
        float(hash_depth > 0),                               # is a markdown "# " heading
        float(hash_depth) / 6.0,                             # heading depth, normalised to 0..1
        float(n_sent == 1),                                  # single sentence
        float(min(n_sent, 10)),                              # sentence count, capped at 10
    ]


def _normalize_text(text, dl):
    """Restore the leading bullet on a list item that lost its marker.

    Docling occasionally strips the "-" off a `list_item`; we add it back so the tf-idf
    vectorizers see the same surface shape they were fit on (and so the bullet regex flags fire).
    """
    if dl == "list_item" and text and not BULLET_RE.match(text):
        return "- " + text
    return text


def build_features(data, dl_idx, word_vec, char_vec, fit):
    """Assemble the full sparse feature matrix for a batch of chunks.

    `data` rows come from `context_rows` (each has text, dl, prev, next, dist). `dl_idx` maps
    a docling label to a column. `word_vec`/`char_vec` are the tf-idf vectorizers; pass
    `fit=True` at training time to fit them, `fit=False` at inference to transform with the
    loaded ones. Output columns are [word tf-idf | char tf-idf | regex flags | context].
    """
    texts = [_normalize_text(r["text"], r["dl"]) for r in data]
    word = word_vec.fit_transform(texts) if fit else word_vec.transform(texts)
    char = char_vec.fit_transform(texts) if fit else char_vec.transform(texts)
    # Context block: three one-hot label slots (own / prev / next) of width k, plus one distance scalar.
    k = len(dl_idx)
    ctx = np.zeros((len(data), 3 * k + 1), dtype=np.float32)
    for i, r in enumerate(data):
        ctx[i, dl_idx.get(r["dl"], 0)] = 1.0                 # this item's own docling label
        ctx[i, k + dl_idx.get(r["prev"], 0)] = 1.0           # the previous item's label
        ctx[i, 2 * k + dl_idx.get(r["next"], 0)] = 1.0       # the next item's label
        ctx[i, 3 * k] = r["dist"]                            # items since the last section header
    regex = sparse.csr_matrix([regex_features(t) for t in texts])
    return sparse.hstack([word, char, regex, sparse.csr_matrix(ctx)]).tocsr()


def context_rows(chunks, text_key="text"):
    """Order chunks by `section_id` and attach each one's neighbourhood context.

    For every chunk we record its own docling label, its previous/next labels, and `dist` =
    how many items back the nearest `section_header` is (capped at DIST_CAP, also used when no
    header precedes it). Returns the sorted chunks alongside the context rows that feed
    `build_features`.
    """
    chunks = sorted(chunks, key=lambda c: int(c["section_id"]))
    n = len(chunks)
    rows = []
    for i, c in enumerate(chunks):
        # walk backwards from i to the nearest section_header; DIST_CAP if none is found
        dist = next((min(i - b, DIST_CAP) for b in range(i, -1, -1)
                     if chunks[b]["docling_label"] == "section_header"), DIST_CAP)
        rows.append({"text": c.get(text_key, c.get("text", "")), "dl": c["docling_label"],
                     "prev": chunks[i - 1]["docling_label"] if i > 0 else "",
                     "next": chunks[i + 1]["docling_label"] if i < n - 1 else "", "dist": dist})
    return chunks, rows
