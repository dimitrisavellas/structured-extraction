import argparse
import json
import logging
import re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from . import config
from .abbreviations import synonyms_for_keys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_model(model_name=None):
    name = model_name or config.EMBEDDING_MODEL
    logger.info(f"Loading embedding model: {name} on {config.EMBEDDING_DEVICE}")
    return SentenceTransformer(name, device=config.EMBEDDING_DEVICE, trust_remote_code=True)


def _expand_key(key, vocab, synonyms=None):
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

    parts = [key]
    if expansions:
        parts.append(" ".join(expansions[:3]))

    if synonyms:
        parts.extend(synonyms)

    if len(parts) > 1:
        return " ".join(parts), key
    return key, key


def _is_group_header(cells):
    """Return True if this row looks like a group-header label rather than a data row."""
    if len(cells) <= 1 or not cells[0]:
        return False
    empty_count = sum(1 for cell in cells[1:] if not cell)
    standalone_nums = sum(
        1 for cell in cells[1:]
        if cell and re.match(r'^[\±\-\+]?\d+\.?\d*\s*[a-zA-Zµ°/²³%]*$', cell.strip())
    )
    has_equals = '=' in cells[0]
    return empty_count >= len(cells[1:]) * 0.6 and standalone_nums < 2 and not has_equals


def _parse_html_rows(content):
    rows = []
    for n, tr in enumerate(re.findall(r"<tr\b[^>]*>(.*?)</tr>", content, re.S | re.I)):
        cells = [re.sub(r"<[^>]+>", "", c).strip()
                 for c in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr, re.S | re.I)]
        dedup = []
        for c in cells:                       # collapse repeats from rowspan/colspan expansion
            if not dedup or dedup[-1] != c:
                dedup.append(c)
        row_text = " | ".join(c for c in dedup if c)
        if row_text.strip():
            rows.append({"row_number": n, "row_text": row_text, "group_header": ""})
    return rows


def _parse_table_rows(content):
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


def _load_inputs(sections_path, tables_path, vocab_path):
    sections = json.loads(sections_path.read_text(encoding="utf-8"))

    tables = []
    if tables_path and Path(tables_path).exists():
        tables = json.loads(Path(tables_path).read_text(encoding="utf-8"))

    vocab = []
    if vocab_path and Path(vocab_path).exists():
        vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))

    return sections, tables, vocab


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


def _expand_all_keys(user_keys, vocab, syn_map):
    expansions = [_expand_key(key, vocab, syn_map.get(key)) for key in user_keys]
    expanded = [pair[0] for pair in expansions]
    originals = [pair[1] for pair in expansions]
    return expanded, originals


def _save_outputs(stem, out_dir, all_embs, key_embs, all_texts, expanded, originals, row_map,
                  num_sections, syn_map):
    """Write all six output files for this stem."""
    np.save(str(out_dir / f"{stem}_section_embeddings.npy"), all_embs)
    np.save(str(out_dir / f"{stem}_key_embeddings.npy"), key_embs)
    (out_dir / f"{stem}_section_texts.json").write_text(json.dumps(all_texts, indent=2), encoding="utf-8")
    (out_dir / f"{stem}_key_texts.json").write_text(json.dumps(expanded, indent=2), encoding="utf-8")
    (out_dir / f"{stem}_original_keys.json").write_text(json.dumps(originals, indent=2), encoding="utf-8")
    rmap_data = {"num_sections": num_sections, "rows": row_map}
    (out_dir / f"{stem}_row_map.json").write_text(
        json.dumps(rmap_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"{stem}_key_synonyms.json").write_text(
        json.dumps(syn_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def embed_sections_and_keys(sections_path, tables_path=None, vocab_path=None,
                             user_keys=None, model_name=None):
    sections_path = Path(sections_path)
    stem = sections_path.stem.replace("_sections", "")
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_emb_path = out_dir / f"{stem}_section_embeddings.npy"
    key_emb_path = out_dir / f"{stem}_key_embeddings.npy"
    sec_txt_path = out_dir / f"{stem}_section_texts.json"
    key_txt_path = out_dir / f"{stem}_key_texts.json"
    rmap_path = out_dir / f"{stem}_row_map.json"

    result = {
        "section_embeddings_path": str(combined_emb_path),
        "key_embeddings_path": str(key_emb_path),
        "section_texts_path": str(sec_txt_path),
        "key_texts_path": str(key_txt_path),
        "row_map_path": str(rmap_path),
    }

    if (combined_emb_path.exists() and key_emb_path.exists() and sec_txt_path.exists()
            and key_txt_path.exists() and rmap_path.exists()):
        logger.info(f"Skipping embeddings — already exists for {stem}")
        return result

    sections, tables, vocab = _load_inputs(sections_path, tables_path, vocab_path)

    sec_texts, table_by_idx = _build_section_texts(sections, tables)
    row_texts, row_map = _parse_all_rows(sections, table_by_idx)
    logger.info(f"Parsed {len(row_map)} table rows")

    if user_keys is None:
        user_keys = []
    # Derive acronym/spelled-out synonyms from the document text and cache them so
    # the extractor prompt can use the same list.
    doc_text = "\n".join(section.get("content", "") for section in sections)
    syn_map = synonyms_for_keys(user_keys, doc_text)

    expanded, originals = _expand_all_keys(user_keys, vocab, syn_map)

    model = _load_model(model_name)

    all_texts = sec_texts + row_texts
    logger.info(f"Embedding {len(sec_texts)} sections + {len(row_texts)} rows and {len(expanded)} keys...")
    all_embs = model.encode(all_texts, show_progress_bar=False)
    key_embs = model.encode(expanded, show_progress_bar=False) if expanded else np.array([])

    _save_outputs(stem, out_dir, all_embs, key_embs, all_texts, expanded, originals,
                  row_map, len(sec_texts), syn_map)

    logger.info(f"Saved combined embeddings: {all_embs.shape[0]} total ({len(sec_texts)} sections + {len(row_map)} rows)")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed sections and keys")
    parser.add_argument("--sections", required=True)
    parser.add_argument("--tables")
    parser.add_argument("--vocab")
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--model")
    args = parser.parse_args()

    result = embed_sections_and_keys(
        sections_path=args.sections, tables_path=args.tables,
        vocab_path=args.vocab, user_keys=args.keys,
        model_name=args.model,
    )
    print(f"Done: {result}")
