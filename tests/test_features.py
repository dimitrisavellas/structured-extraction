from structured_data_extraction.features import (
    _normalize_text,
    _sentence_count,
    context_rows,
    regex_features,
)


def test_sentence_count():
    assert _sentence_count("Just one sentence.") == 1
    assert _sentence_count("First part. Second part.") == 2
    # a figure/table caption prefix is stripped before counting
    assert _sentence_count("Figure 1: A single caption.") == 1


def test_regex_features_length_is_stable():
    # the trained classifier indexes this vector positionally; its length must not drift
    assert len(regex_features("x")) == 30
    assert len(regex_features("| a | b |")) == len(regex_features("plain prose"))


def test_regex_features_spot_signals():
    # feature 0: contains a pipe (table signal)
    assert regex_features("| a | b |")[0] == 1.0
    assert regex_features("plain prose")[0] == 0.0
    # feature 26: markdown hash depth > 0 (heading signal)
    assert regex_features("## Heading")[26] == 1.0
    assert regex_features("plain prose")[26] == 0.0


def test_normalize_text_restores_list_bullet():
    assert _normalize_text("apples", "list_item") == "- apples"
    assert _normalize_text("- apples", "list_item") == "- apples"  # already bulleted
    assert _normalize_text("body text", "text") == "body text"


def test_context_rows_sorts_and_measures_distance_to_header():
    chunks = [
        {"section_id": 2, "docling_label": "text", "text": "c"},
        {"section_id": 0, "docling_label": "section_header", "text": "a"},
        {"section_id": 1, "docling_label": "text", "text": "b"},
    ]
    ordered, rows = context_rows(chunks)
    assert [c["section_id"] for c in ordered] == [0, 1, 2]
    # distance to the nearest preceding section_header
    assert [r["dist"] for r in rows] == [0, 1, 2]
    assert rows[0]["prev"] == "" and rows[0]["next"] == "text"
    assert rows[1]["prev"] == "section_header"
    assert rows[2]["next"] == ""


def test_context_rows_distance_caps_without_header():
    _, rows = context_rows([{"section_id": 0, "docling_label": "text", "text": "x"}])
    assert rows[0]["dist"] == 20  # DIST_CAP when no section_header exists
