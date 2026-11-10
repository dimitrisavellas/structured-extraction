from typing import get_args
from unittest.mock import MagicMock, patch

from structured_data_extraction import extract
from structured_data_extraction.extract import (
    _build_extraction_model,
    _dedup_extraction,
    _format_sections,
    _merge_cleaned_tables,
    _neighbor_line,
)
from structured_data_extraction.main import use_local_profile


def test_extract_values_per_section_one_call_each_and_merges(monkeypatch):
    monkeypatch.setenv("MAX_LLM_RETRIES", "0")
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(gain=[MagicMock(model_dump=lambda: {"value": 1})]),
        MagicMock(gain=[MagicMock(model_dump=lambda: {"value": 2})]),
    ]
    sections = [{"type": "table", "content": "a"}, {"type": "table", "content": "b"}]
    with patch.object(extract.config, "EXTRACT_PER_SECTION", True):
        out = extract.extract_values(client, sections, [], ["gain"], {}, model="m")
    assert client.chat.completions.create.call_count == 2
    assert out == {"gain": [{"value": 1}, {"value": 2}]}


def _record_model(model, field):
    return get_args(model.model_fields[field].annotation)[0]


def test_build_extraction_model_default_field():
    # no schema entry -> the field falls back to a plain List[ValueRecord]
    model = _build_extraction_model({}, ["Supply Voltage"])
    assert "supply_voltage" in model.model_fields
    rec = _record_model(model, "supply_voltage")
    assert {"value", "unit", "metadata"} <= set(rec.model_fields)


def test_build_extraction_model_acronym_schema_lookup():
    # schema key "PSRR" must be found for the normalized field name "psrr"
    schema = {"PSRR": {"fields": {"gain": {"type": "number"}}}}
    model = _build_extraction_model(schema, ["PSRR"])
    rec = _record_model(model, "psrr")
    assert "gain" in rec.model_fields
    assert "metadata" in rec.model_fields


def test_build_extraction_model_synthesizes_unit_field():
    schema = {"current": {"fields": {"value": {"type": "number", "unit": True}}}}
    rec = _record_model(_build_extraction_model(schema, ["current"]), "current")
    assert "value_unit" in rec.model_fields


def test_build_extraction_model_skips_unit_synthesis_when_present():
    schema = {
        "current": {
            "fields": {"value": {"type": "number", "unit": True}, "unit": {"type": "string"}}
        }
    }
    rec = _record_model(_build_extraction_model(schema, ["current"]), "current")
    assert "unit" in rec.model_fields
    assert "value_unit" not in rec.model_fields


def test_merge_cleaned_tables():
    sections = [
        {"type": "header", "content": "Specs"},
        {"type": "table", "content": "raw", "section_idx": 1},
    ]
    tables = [{"section_idx": 1, "content": "CLEANED", "refined_format": "html"}]
    merged = _merge_cleaned_tables(sections, tables)
    assert merged[1]["content"] == "CLEANED"
    assert merged[1]["refined_format"] == "html"
    assert merged[1]["nearest_header"] == "Specs"
    # the header section itself is untouched and carries no nearest_header
    assert merged[0]["content"] == "Specs"
    assert "nearest_header" not in merged[0]


def test_merge_cleaned_tables_keeps_raw_when_no_cleaned():
    sections = [{"type": "table", "content": "raw", "section_idx": 0}]
    merged = _merge_cleaned_tables(sections, [{"section_idx": 0, "content": ""}])
    assert merged[0]["content"] == "raw"


def test_format_sections_header_and_index():
    sections = [{"type": "text", "content": "hello", "nearest_header": "H", "section_idx": 3}]
    assert (
        _format_sections(sections) == "[Section 3] (text):\nSurrounding section heading: H\nhello"
    )


def test_format_sections_prefers_full_content():
    sections = [{"type": "text", "content": "short", "full_content": "FULL"}]
    assert "FULL" in _format_sections(sections)


def test_format_sections_truncates_text_but_not_tables():
    long_text = [{"type": "text", "content": "x" * 100}]
    assert "x" * 10 + "..." in _format_sections(long_text, max_chars=10)
    long_table = [{"type": "table", "content": "y" * 100}]
    assert "y" * 100 in _format_sections(long_table, max_chars=10)


def test_dedup_extraction_prefers_table_section():
    extraction = {
        "f": [
            {"value": "A", "unit": "mm", "metadata": {"section_type": "text"}},
            {"value": "A", "unit": "mm", "metadata": {"section_type": "table"}},
        ]
    }
    deduped = _dedup_extraction(extraction)
    assert len(deduped["f"]) == 1
    assert deduped["f"][0]["metadata"]["section_type"] == "table"


def test_dedup_extraction_keeps_distinct_values():
    extraction = {"f": [{"value": "A"}, {"value": "B"}]}
    assert len(_dedup_extraction(extraction)["f"]) == 2


def test_neighbor_line():
    sections = [
        {"type": "text", "content": "line1\nline2"},
        {"type": "text", "content": "target"},
        {"type": "text", "content": "next1\nnext2"},
    ]
    assert _neighbor_line(sections, 1, "prev") == "line2"  # last line of prev section
    assert _neighbor_line(sections, 1, "next") == "next1"  # first line of next section


def test_neighbor_line_skips_table_and_header_and_bounds():
    sections = [{"type": "table", "content": "t"}, {"type": "text", "content": "x"}]
    assert _neighbor_line(sections, 1, "prev") == ""  # table neighbor ignored
    assert _neighbor_line(sections, 0, "prev") == ""  # out of bounds


def test_use_local_profile_keeps_explicit_settings(monkeypatch):
    cfg = extract.config
    monkeypatch.setattr(cfg, "EXTRACT_PER_SECTION", False)
    monkeypatch.setattr(cfg, "LLM_EXTRA", {})
    monkeypatch.setattr(cfg, "LLM_TIMEOUT", 600.0)
    use_local_profile()
    assert (cfg.EXTRACT_PER_SECTION, cfg.LLM_EXTRA, cfg.LLM_TIMEOUT) == (
        True,
        {"reasoning_effort": "none"},
        1800.0,
    )
    monkeypatch.setattr(cfg, "LLM_EXTRA", {"reasoning_effort": "low"})
    monkeypatch.setattr(cfg, "LLM_TIMEOUT", 3600.0)
    use_local_profile()
    assert (cfg.LLM_EXTRA, cfg.LLM_TIMEOUT) == ({"reasoning_effort": "low"}, 3600.0)
