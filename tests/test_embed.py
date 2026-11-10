import json
from unittest.mock import patch

import numpy as np

from structured_data_extraction import embed
from structured_data_extraction.embed import (
    _build_section_texts,
    _expand_all_keys,
    _expand_key,
    _is_group_header,
    _load_model,
    _parse_all_rows,
    _parse_html_rows,
    _parse_table_rows,
    _save_outputs,
    embed_sections_and_keys,
)


def test_load_model_caches():
    """Loading the same name twice hits the SentenceTransformer ctor once and reuses the object."""
    embed._model_cache.clear()
    with patch("structured_data_extraction.embed.SentenceTransformer") as st_ctor:
        st_ctor.side_effect = lambda name, **kw: f"model::{name}"
        m1 = _load_model("fake-model")
        m2 = _load_model("fake-model")
    assert m1 is m2
    assert st_ctor.call_count == 1


def test_expand_key():
    vocab = [
        {"term": "operating temperature"},
        {"term": "junction temperature"},
        {"term": "voltage"},
    ]
    # substring + all-words matches get appended (capped later); the key is repeated by design
    expanded, original = _expand_key("temperature", vocab)
    assert expanded == "temperature operating temperature junction temperature"
    assert original == "temperature"


def test_expand_key_all_words_unordered():
    """Matches when every key word is present, even out of order ('max current' ~ 'current max')."""
    expanded, _ = _expand_key("max current", [{"term": "current max"}])
    assert expanded == "max current current max"


def test_expand_key_no_match_returns_key():
    expanded, original = _expand_key("nonexistent", [{"term": "voltage"}])
    assert expanded == "nonexistent"
    assert original == "nonexistent"


def test_expand_key_caps_at_three():
    vocab = [{"term": f"temperature {i}"} for i in range(5)]
    expanded, _ = _expand_key("temperature", vocab)
    assert expanded == "temperature temperature 0 temperature 1 temperature 2"


def test_expand_key_disabled():
    with patch("structured_data_extraction.embed.config.EXPAND_KEYS", False):
        assert _expand_key("temperature", [{"term": "junction temperature"}]) == (
            "temperature",
            "temperature",
        )


def test_expand_all_keys():
    vocab = [{"term": "junction temperature"}]
    expanded, originals = _expand_all_keys(["temperature", "voltage"], vocab)
    assert expanded == ["temperature junction temperature", "voltage"]
    assert originals == ["temperature", "voltage"]


def test_is_group_header():
    # label row with mostly empty trailing cells -> group header
    assert _is_group_header(["Power Supply", "", "", ""]) is True
    # single cell / empty first cell -> not a header
    assert _is_group_header(["Power Supply"]) is False
    assert _is_group_header(["", "1", "2"]) is False
    # real data row (numbers with units present) -> not a header
    assert _is_group_header(["Vcc", "5V", "10mA"]) is False
    # an equation in the first cell -> not a header
    assert _is_group_header(["Vcc = 5V", "", ""]) is False


def test_parse_html_rows_dedups_spans():
    content = "<tr><td>A</td><td>B</td></tr><tr><td>C</td><td>C</td><td>D</td></tr>"
    assert _parse_html_rows(content) == [
        {"row_number": 0, "row_text": "A | B", "group_header": ""},
        {"row_number": 1, "row_text": "C | D", "group_header": ""},
    ]


def test_parse_table_rows_delegates_to_html():
    rows = _parse_table_rows("<table><tr><td>A</td><td>B</td></tr></table>")
    assert rows == [{"row_number": 0, "row_text": "A | B", "group_header": ""}]


def test_parse_table_rows_annotates_min_typ_max():
    content = "| | Min | Typ | Max |\n| Vcc | 1 | 2 | 3 |"
    rows = _parse_table_rows(content)
    assert rows == [
        {
            "row_number": 0,
            "row_text": "Vcc | [Min] 1 | [Typ] 2 | [Max] 3",
            "group_header": "",
        }
    ]


def test_parse_table_rows_propagates_group_header():
    content = "| Power Supply | | | |\n| Vcc | 1 | 2 | 3 |"
    rows = _parse_table_rows(content)
    # the header row itself carries no group, but the data row beneath inherits it
    assert rows[0]["row_text"] == "Power Supply"
    assert rows[0]["group_header"] == ""
    assert rows[1]["row_text"] == "Vcc | 1 | 2 | 3"
    assert rows[1]["group_header"] == "Power Supply"


def test_build_section_texts_replaces_table_with_cleaned():
    sections = [
        {"type": "text", "content": "hello"},
        {"type": "table", "content": "raw"},
    ]
    tables = [{"section_idx": 1, "content": "CLEANED"}]
    sec_texts, table_by_idx = _build_section_texts(sections, tables)
    assert sec_texts == ["text: hello", "table: CLEANED"]
    assert table_by_idx == {1: tables[0]}


def test_build_section_texts_keeps_raw_when_no_cleaned():
    sections = [{"type": "table", "content": "raw"}]
    sec_texts, _ = _build_section_texts(sections, [{"section_idx": 0, "content": ""}])
    assert sec_texts == ["table: raw"]


def test_parse_all_rows_prefixes_group_header():
    sections = [{"type": "table", "content": "ignored"}]
    table_by_idx = {0: {"section_idx": 0, "content": "| Supply | | | |\n| Vcc | 1 | 2 | 3 |"}}
    row_texts, row_map = _parse_all_rows(sections, table_by_idx)
    assert row_texts == ["table_row: Supply", "table_row: Supply > Vcc | 1 | 2 | 3"]
    assert row_map[1]["section_idx"] == 0
    assert row_map[1]["group_header"] == "Supply"


def test_save_outputs(tmp_path):
    _save_outputs(
        "doc1",
        tmp_path,
        all_embs=np.array([[1.0, 2.0]]),
        key_embs=np.array([[3.0, 4.0]]),
        all_texts=["t1"],
        expanded=["k1"],
        originals=["o1"],
        row_map=[{"row_number": 0}],
        num_sections=1,
    )
    assert np.load(tmp_path / "doc1_section_embeddings.npy").tolist() == [[1.0, 2.0]]
    assert np.load(tmp_path / "doc1_key_embeddings.npy").tolist() == [[3.0, 4.0]]
    assert json.loads((tmp_path / "doc1_section_texts.json").read_text()) == ["t1"]
    assert json.loads((tmp_path / "doc1_key_texts.json").read_text()) == ["k1"]
    assert json.loads((tmp_path / "doc1_original_keys.json").read_text()) == ["o1"]
    assert json.loads((tmp_path / "doc1_row_map.json").read_text()) == {
        "num_sections": 1,
        "rows": [{"row_number": 0}],
    }


class _FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kw):
        texts = list(texts)
        self.calls.append(texts)
        return np.zeros((len(texts), 2))


def test_embed_sections_and_keys(tmp_path):
    """End-to-end glue: sections + table rows + keys get the right task prefixes and artifacts."""
    (tmp_path / "doc_sections.json").write_text(
        json.dumps(
            [
                {"type": "text", "content": "hello"},
                {"type": "table", "content": "raw"},
            ]
        )
    )
    (tmp_path / "doc_tables.json").write_text(
        json.dumps([{"section_idx": 1, "content": "| A | B |\n| C | D |"}])
    )

    fake = _FakeModel()
    with (
        patch("structured_data_extraction.embed.config.INTERMEDIATE_DIR", str(tmp_path)),
        patch("structured_data_extraction.embed._load_model", return_value=fake),
    ):
        embed_sections_and_keys("doc", user_keys=["temperature"])

    # documents carry the search_document: prefix, in section-then-row order
    assert fake.calls[0] == [
        "search_document: text: hello",
        "search_document: table: | A | B |\n| C | D |",
        "search_document: table_row: A | B",
        "search_document: table_row: C | D",
    ]
    # queries carry the search_query: prefix (no vocab.json -> key unchanged)
    assert fake.calls[1] == ["search_query: temperature"]

    assert json.loads((tmp_path / "doc_key_texts.json").read_text()) == ["temperature"]
    row_map = json.loads((tmp_path / "doc_row_map.json").read_text())
    assert row_map["num_sections"] == 2
    assert [r["row_text"] for r in row_map["rows"]] == ["A | B", "C | D"]
