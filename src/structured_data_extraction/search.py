"""Semantic search : match user keys to document sections and table rows.

Computes cosine similarity between the key embeddings and the section/row
embeddings produced by embed.py, then returns the top-K hits above
SIMILARITY_THRESHOLD for each key.

The embedding matrix has sections first, then rows; a hit index
>= num_sections is a row-level match and carries the originating section_idx,
row_text, and group_header so downstream extraction knows which table and row
the hit came from.

Results are cached to matches.json and reused on reruns unless --force is set.
"""
import argparse
import logging
import numpy as np
from . import config
from .paths import inter, read_json, cached

logger = logging.getLogger(__name__)


def cosine_similarity(a, b):
    # floor the norm so a zero-norm vector stays zero instead of 0/0 = nan, which would
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a_norm @ b_norm.T


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
    key_matches = []
    for idx in np.argsort(sims)[::-1]:
        score = float(sims[idx])
        if score < threshold or len(key_matches) >= config.SEARCH_TOP_K:
            break
        key_matches.append(_build_match_dict(idx, score, sec_texts, sections, num_secs, rows))
    return key_matches


def search(stem, original_keys=None, threshold=None):
    threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold

    def build():
        key_embs = np.load(inter(stem, "key_embeddings.npy"))
        if key_embs.size == 0:
            logger.warning("No key embeddings. Nothing to search.")
            return {}
        sec_embs = np.load(inter(stem, "section_embeddings.npy"))
        sec_texts = read_json(inter(stem, "section_texts.json"))
        sections = read_json(inter(stem, "sections.json"))
        key_texts = read_json(inter(stem, "key_texts.json"))
        row_map = read_json(inter(stem, "row_map.json"))
        # num_secs is the boundary: idx >= num_secs means a row hit
        num_secs = row_map.get("num_sections", len(sections))
        rows = row_map.get("rows", [])
        keys = original_keys if original_keys is not None else read_json(inter(stem, "original_keys.json"))

        sims = cosine_similarity(key_embs, sec_embs)
        matches = {}
        for i, key_text in enumerate(key_texts):
            orig_key = keys[i] if i < len(keys) else key_text
            matches[orig_key] = _select_hits_for_key(
                sims[i], sec_texts, sections, num_secs, rows, threshold)
            logger.info(f"Key '{orig_key}': {len(matches[orig_key])} matches above {threshold}")
        return matches

    return cached(inter(stem, "matches.json"), build)


def main():
    parser = argparse.ArgumentParser(description="Semantic search")
    parser.add_argument("--stem", required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--original-keys", nargs="+")
    args = parser.parse_args()
    result = search(args.stem, original_keys=args.original_keys, threshold=args.threshold)
    print(f"Found matches for {len(result)} keys")


if __name__ == "__main__":
    main()
