"""Embedding of document sections, table rows, and user keys.

Encodes every section and individual table row into dense vectors with a
SentenceTransformer model (default: nomic-embed-text-v1.5), then encodes the
user's extraction keys into the same space so search.py can match them via
cosine similarity.

nomic requires task-type prefixes: sections are prefixed with "search_document:"
and keys with "search_query:". Without these prefixes similarity scores collapse
and SIMILARITY_THRESHOLD stops discriminating relevant from irrelevant hits.

Sections are stacked before rows in the embedding matrix, search.py relies on
that ordering to tell section-level hits from row-level hits using num_sections
as the boundary index.

Table rows are parsed into per-row text (with group headers prepended) so that
search can match an individual parameter row inside a large table rather than
only matching the table as a whole.
"""
import argparse
import json
import logging
import os
import re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from . import config
from .paths import inter, read_json, write_json

logger = logging.getLogger(__name__)


EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", 4))

_model_cache = {}


def _load_model(model_name=None):
    """Load the embedding model (from the given name or env), caching it so the hundreds-of-MB
    SentenceTransformer is loaded once and reused across PDFs."""
    name = model_name or os.environ["EMBEDDING_MODEL"]
    if name not in _model_cache:
        logger.info(f"Loading embedding model: {name} on {config.EMBEDDING_DEVICE}")
        _model_cache[name] = SentenceTransformer(name, device=config.EMBEDDING_DEVICE, trust_remote_code=True)
    return _model_cache[name]


def _expand_key(key, vocab):
    """Return a tuple of (expanded_key, original_key). If EXPAND_KEYS is False, returns (key, key)."""
    if not config.EXPAND_KEYS:
        return key, key

    key_lower = key.lower()
    key_words = key_lower.split()
    expansions = []
    for item in vocab:
        term_lower = item["term"].lower()
        term_words = term_lower.split()
        if key_lower in term_lower or all(word in term_words for word in key_words):
            expansions.append(item["term"])

    if expansions:
        return " ".join([key, " ".join(expansions[:3])]), key
    return key, key


def _is_group_header(cells):
    """Determine if a row is likely a group header based on its content."""
    if len(cells) <= 1 or not cells[0]:
        return False
    empty_count = sum(1 for cell in cells[1:] if not cell)
    standalone_nums = sum(
        1 for cell in cells[1:]
        if cell and re.match(r'^[\±\-\+]?\d+\.?\d*\s*[a-zA-Zµ°/²³%]*$', cell.strip()) # matches standalone numbers with optional units
    )
    has_equals = '=' in cells[0]
    return empty_count >= len(cells[1:]) * 0.6 and standalone_nums < 2 and not has_equals # if most cells are empty, few standalone numbers,
                                                                                        # and no equals sign in the first cell, it's likely a group header


def _parse_html_rows(content):
    """Parse table rows from HTML content, return a list of row dictionaries."""
    rows = []
    for n, tr in enumerate(re.findall(r"<tr\b[^>]*>(.*?)</tr>", content, re.S | re.I)): # find all <tr>...</tr> blocks
        cells = [re.sub(r"<[^>]+>", "", c).strip()
                 for c in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr, re.S | re.I)] # find all <td> or <th> blocks within the <tr>
        dedup = []
        for c in cells:                       # collapse repeats from rowspan/colspan expansion
            if not dedup or dedup[-1] != c:
                dedup.append(c)
        row_text = " | ".join(c for c in dedup if c)
        if row_text.strip():
            rows.append({"row_number": n, "row_text": row_text, "group_header": ""})
    return rows


def _parse_table_rows(content):
    """Parse table rows from markdown content, return a list of row dictionaries."""
    if "<tr" in content.lower():              # pre-embed refined tables are HTML, not markdown pipes
        return _parse_html_rows(content)
    lines = content.strip().split('\n')
    rows = []
    group_header = ""
    col_headers = []
    row_num = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or not stripped.startswith('|'):
            continue
        if re.match(r'^\|[\s\-:|]+\|$', stripped):
            continue

        cells = [cell.strip() for cell in stripped.split('|')]
        if cells and cells[0] == '':
            cells = cells[1:]
        if cells and cells[-1] == '':
            cells = cells[:-1]
        if not cells:
            continue

        
        non_empty = [cell for cell in cells if cell]
        if non_empty and all(cell in ('Min', 'Typ', 'Max') for cell in non_empty) and len(non_empty) >= 3:
            col_headers = cells
            continue

        if _is_group_header(cells):
            group_header = cells[0]
            row_text = ' | '.join(cell for cell in cells if cell)
            if row_text.strip():
                rows.append({"row_number": row_num, "row_text": row_text, "group_header": ""})
                row_num += 1
            continue

        if col_headers and len(col_headers) == len(cells):
            annotated = []
            for i, cell in enumerate(cells):
                if cell and col_headers[i]:
                    annotated.append(f"[{col_headers[i]}] {cell}")
                else:
                    annotated.append(cell)
            row_text = ' | '.join(annotated)
        else:
            row_text = ' | '.join(cells)

        if row_text.strip():
            rows.append({"row_number": row_num, "row_text": row_text, "group_header": group_header})
            row_num += 1

    return rows


def _build_section_texts(sections, tables):
    """Build per-section text list, replacing table sections with their cleaned markdown content."""
    sec_texts = [f"{section['type']}: {section['content']}" for section in sections]
    table_by_idx = {table["section_idx"]: table for table in tables}
    for i, section in enumerate(sections):
        if section.get("type") == "table" and i in table_by_idx:
            cleaned = table_by_idx[i].get("content", "")
            if cleaned:
                sec_texts[i] = f"table: {cleaned}"
    return sec_texts, table_by_idx


def _parse_all_rows(sections, table_by_idx):
    """Parse table rows from all table sections, return (row_texts, row_map)."""
    row_texts = []
    row_map = []
    for i, section in enumerate(sections):
        if section.get("type") != "table" or i not in table_by_idx:
            continue
        cleaned = table_by_idx[i].get("content", "")
        if not cleaned:
            continue
        for row_info in _parse_table_rows(cleaned):
            txt = row_info["row_text"]
            if row_info["group_header"]:
                txt = f"{row_info['group_header']} > {txt}"
            row_texts.append(f"table_row: {txt}")
            row_map.append({
                "section_idx": i,
                "row_number": row_info["row_number"],
                "row_text": row_info["row_text"],
                "group_header": row_info["group_header"],
            })
    return row_texts, row_map


def _expand_all_keys(user_keys, vocab):
    expansions = [_expand_key(key, vocab) for key in user_keys]
    expanded = [pair[0] for pair in expansions]
    originals = [pair[1] for pair in expansions]
    return expanded, originals


def _save_outputs(stem, out_dir, all_embs, key_embs, all_texts, expanded, originals, row_map,
                  num_sections):
    """Write the embedding + text artifacts for this stem."""
    np.save(str(out_dir / f"{stem}_section_embeddings.npy"), all_embs)
    np.save(str(out_dir / f"{stem}_key_embeddings.npy"), key_embs)
    write_json(out_dir / f"{stem}_section_texts.json", all_texts)
    write_json(out_dir / f"{stem}_key_texts.json", expanded)
    write_json(out_dir / f"{stem}_original_keys.json", originals)
    write_json(out_dir / f"{stem}_row_map.json", {"num_sections": num_sections, "rows": row_map})


def embed_sections_and_keys(stem, user_keys=None, model_name=None):
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    sec_names = ("section_embeddings.npy", "section_texts.json", "row_map.json")
    key_names = ("key_embeddings.npy", "key_texts.json")
    sec_cached = all(inter(stem, n).exists() for n in sec_names) and not config.FORCE
    key_cached = all(inter(stem, n).exists() for n in key_names) and not config.FORCE

    if sec_cached and key_cached:
        logger.info(f"Skipping embeddings — already cached for {stem}")
        return

    sections = read_json(inter(stem, "sections.json"))
    tables = read_json(inter(stem, "tables.json")) if inter(stem, "tables.json").exists() else []
    vocab = read_json(inter(stem, "vocab.json")) if inter(stem, "vocab.json").exists() else []

    if user_keys is None:
        user_keys = []

    model = _load_model(model_name)

    if sec_cached:
        logger.info(f"Reusing cached section embeddings for {stem}")
        all_embs = np.load(str(inter(stem, "section_embeddings.npy")))
        all_texts = read_json(inter(stem, "section_texts.json"))
        rm = read_json(inter(stem, "row_map.json"))
        num_sections = rm.get("num_sections", len(sections))
        row_map = rm.get("rows", [])
    else:
        sec_texts, table_by_idx = _build_section_texts(sections, tables)
        row_texts, row_map = _parse_all_rows(sections, table_by_idx)
        logger.info(f"Parsed {len(row_map)} table rows")
        all_texts = sec_texts + row_texts
        num_sections = len(sec_texts)
        logger.info(f"Embedding {len(sec_texts)} sections + {len(row_texts)} rows...")
        all_embs = model.encode([f"search_document: {t}" for t in all_texts],
                                show_progress_bar=False, batch_size=EMBED_BATCH_SIZE)

    expanded, originals = _expand_all_keys(user_keys, vocab)
    logger.info(f"Embedding {len(expanded)} keys...")
    key_embs = (model.encode([f"search_query: {k}" for k in expanded],
                             show_progress_bar=False, batch_size=EMBED_BATCH_SIZE)
                if expanded else np.array([]))

    _save_outputs(stem, out_dir, all_embs, key_embs, all_texts, expanded, originals,
                  row_map, num_sections)
    logger.info(f"Saved embeddings: {all_embs.shape[0]} content ({num_sections} sections + {len(row_map)} rows), {len(expanded)} keys")


def main():
    parser = argparse.ArgumentParser(description="Embed sections and keys")
    parser.add_argument("--stem", required=True)
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--model")
    args = parser.parse_args()
    embed_sections_and_keys(args.stem, user_keys=args.keys, model_name=args.model)
    print("Done")


if __name__ == "__main__":
    main()
