import argparse
import json
import logging
from pathlib import Path

import numpy as np

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def _load_inputs(section_embeddings_path, key_embeddings_path,
                 section_texts_path, key_texts_path,
                 sections_path, row_map_path):
    sec_embs = np.load(section_embeddings_path)
    key_embs = np.load(key_embeddings_path)
    sec_texts = json.loads(Path(section_texts_path).read_text(encoding="utf-8"))
    key_texts = json.loads(Path(key_texts_path).read_text(encoding="utf-8"))

    sections = []
    if sections_path and Path(sections_path).exists():
        sections = json.loads(Path(sections_path).read_text(encoding="utf-8"))

    row_map_data = {"num_sections": 0, "rows": []}
    if row_map_path and Path(row_map_path).exists():
        row_map_data = json.loads(Path(row_map_path).read_text(encoding="utf-8"))

    return sec_embs, key_embs, sec_texts, key_texts, sections, row_map_data


def _resolve_original_keys(original_keys, stem):
    if original_keys is not None:
        return original_keys
    # embed.py writes the unexpanded keys here
    orig_keys_path = Path(config.INTERMEDIATE_DIR) / f"{stem}_original_keys.json"
    if orig_keys_path.exists():
        return json.loads(orig_keys_path.read_text(encoding="utf-8"))
    raise ValueError(
        f"original_keys not provided and {orig_keys_path} does not exist. "
        "Pass original_keys directly or ensure _original_keys.json is present."
    )


def _build_match_dict(idx, score, sec_texts, sections, num_secs, rows):
    preview = sec_texts[idx][:150] + "..." if len(sec_texts[idx]) > 150 else sec_texts[idx]

    is_row = idx >= num_secs and len(rows) > 0
    if is_row:
        row_idx = idx - num_secs
        row_info = rows[row_idx] if row_idx < len(rows) else {}
        sec_idx = row_info.get("section_idx", idx)
    else:
        sec_idx = idx

    match = {
        "section_idx": int(sec_idx),
        "similarity": round(score, 4),
        "content_preview": preview,
    }
    if is_row:
        match.update({
            "is_row": True,
            "row_text": row_info.get("row_text", ""),
            "row_number": row_info.get("row_number"),
            "group_header": row_info.get("group_header", ""),
        })
    if sec_idx < len(sections):
        match["type"] = sections[sec_idx].get("type", "unknown")

    return match


def _select_hits_for_key(sims, sec_texts, sections, num_secs, rows, threshold):
    ranked = np.argsort(sims)[::-1]
    key_matches = []
    for idx in ranked:
        score = float(sims[idx])
        if score < threshold:
            break
        if len(key_matches) >= config.SEARCH_TOP_K:
            break
        key_matches.append(_build_match_dict(idx, score, sec_texts, sections, num_secs, rows))
    return key_matches


def search(section_embeddings_path, key_embeddings_path, section_texts_path,
           key_texts_path, sections_path=None, original_keys=None,
           row_map_path=None, threshold=None):

    threshold = threshold if threshold is not None else config.SIMILARITY_THRESHOLD

    stem = Path(section_embeddings_path).stem.replace("_section_embeddings", "")
    out_path = Path(config.INTERMEDIATE_DIR) / f"{stem}_matches.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.info(f"Skipping search — already exists")
        return json.loads(out_path.read_text(encoding="utf-8"))

    sec_embs, key_embs, sec_texts, key_texts, sections, row_map_data = _load_inputs(
        section_embeddings_path, key_embeddings_path,
        section_texts_path, key_texts_path,
        sections_path, row_map_path,
    )

    if key_embs.size == 0:
        logger.warning("No key embeddings. Nothing to search.")
        return {}

    # num_secs is the boundary: idx >= num_secs means a row hit
    num_secs = row_map_data.get("num_sections", len(sections))
    rows = row_map_data.get("rows", [])

    sim_matrix = cosine_similarity(key_embs, sec_embs)
    orig_keys = _resolve_original_keys(original_keys, stem)

    matches = {}
    for i, key_text in enumerate(key_texts):
        orig_key = orig_keys[i] if i < len(orig_keys) else key_text
        key_matches = _select_hits_for_key(
            sim_matrix[i], sec_texts, sections, num_secs, rows, threshold
        )
        matches[orig_key] = key_matches
        logger.info(f"Key '{orig_key}': {len(key_matches)} matches above {threshold}")

    out_path.write_text(json.dumps(matches, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved matches → {out_path}")
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic search")
    parser.add_argument("--stem", required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--original-keys", nargs="+")
    args = parser.parse_args()

    int_dir = Path(config.INTERMEDIATE_DIR)
    result = search(
        section_embeddings_path=str(int_dir / f"{args.stem}_section_embeddings.npy"),
        key_embeddings_path=str(int_dir / f"{args.stem}_key_embeddings.npy"),
        section_texts_path=str(int_dir / f"{args.stem}_section_texts.json"),
        key_texts_path=str(int_dir / f"{args.stem}_key_texts.json"),
        sections_path=str(int_dir / f"{args.stem}_sections.json"),
        original_keys=args.original_keys,
        row_map_path=str(int_dir / f"{args.stem}_row_map.json"),
        threshold=args.threshold,
    )
    print(f"Found matches for {len(result)} keys")
