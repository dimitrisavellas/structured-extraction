from unittest.mock import patch

import numpy as np
import pytest

from structured_data_extraction.search import (
    _build_match_dict,
    _select_hits_for_key,
    cosine_similarity,
)


def test_cosine_similarity():
    a = np.array([[1.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sims = cosine_similarity(a, b)
    assert sims.shape == (1, 3)
    # identical -> 1, orthogonal -> 0, opposite -> -1
    np.testing.assert_allclose(sims[0], [1.0, 0.0, -1.0], atol=1e-6)


def test_cosine_similarity_scale_invariant():
    assert cosine_similarity(np.array([[2.0, 0.0]]), np.array([[5.0, 0.0]]))[0, 0] == pytest.approx(
        1.0
    )


def test_cosine_similarity_zero_vector_scores_zero():
    # a zero embedding scores 0 against everything (norm is floored, so no 0/0 = nan)
    sims = cosine_similarity(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))
    assert not np.isnan(sims).any()
    assert sims[0, 0] == 0.0


def test_build_match_dict_section():
    m = _build_match_dict(0, 0.9, ["short text"], [{"type": "table"}], num_secs=1, rows=[])
    assert m == {
        "section_idx": 0,
        "similarity": 0.9,
        "content_preview": "short text",
        "type": "table",
    }


def test_build_match_dict_row():
    sec_texts = ["s0", "s1", "table_row: A | B"]
    sections = [{"type": "text"}, {"type": "table"}]
    rows = [{"section_idx": 1, "row_text": "A | B", "row_number": 0, "group_header": "Grp"}]
    # idx past the section boundary -> resolves through the row map back to its section
    m = _build_match_dict(2, 0.7, sec_texts, sections, num_secs=2, rows=rows)
    assert m == {
        "section_idx": 1,
        "similarity": 0.7,
        "content_preview": "table_row: A | B",
        "is_row": True,
        "row_text": "A | B",
        "row_number": 0,
        "group_header": "Grp",
        "type": "table",
    }


def test_build_match_dict_truncates_preview():
    m = _build_match_dict(0, 0.5, ["x" * 200], [], num_secs=1, rows=[])
    assert m["content_preview"] == "x" * 150 + "..."


def test_select_hits_threshold_and_order():
    sims = np.array([0.9, 0.4, 0.7])
    sections = [{"type": "text"}] * 3
    hits = _select_hits_for_key(
        sims, ["a", "b", "c"], sections, num_secs=3, rows=[], threshold=0.55
    )
    # below-threshold hit dropped; remaining sorted by score descending
    assert [h["section_idx"] for h in hits] == [0, 2]
    assert [h["similarity"] for h in hits] == [0.9, 0.7]


def test_select_hits_caps_at_top_k():
    sims = np.array([0.9, 0.8, 0.7])
    sections = [{"type": "text"}] * 3
    with patch("structured_data_extraction.search.config.SEARCH_TOP_K", 2):
        hits = _select_hits_for_key(sims, ["a", "b", "c"], sections, 3, [], 0.55)
    assert len(hits) == 2
