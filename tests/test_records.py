import json
from unittest.mock import patch

from structured_data_extraction.records import make_record, write_records


def test_make_record(tmp_path):
    (tmp_path / "doc_run.json").write_text(json.dumps({
        "extracted_at": "2026-01-01T00:00:00Z",
        "user_keys": ["voltage"],
    }))
    (tmp_path / "doc_schema.json").write_text(json.dumps({"document_type": "datasheet"}))

    with patch("structured_data_extraction.records.config.INTERMEDIATE_DIR", str(tmp_path)):
        env = make_record("doc", {"voltage": [{"value": "5V"}]})

    assert env["type"] == "ExtractionResult"
    assert env["generated_at"] == "2026-01-01T00:00:00Z"
    prov = env["provenance"]
    assert prov["source_file"] == "doc.pdf"
    assert prov["document_type"] == "datasheet"
    assert "extraction" not in prov["models"]
    assert prov["user_keys"] == ["voltage"]
    assert env["voltage"] == [{"value": "5V"}]


def test_make_record_defaults_when_run_files_missing(tmp_path):
    with patch("structured_data_extraction.records.config.INTERMEDIATE_DIR", str(tmp_path)):
        env = make_record("doc", {})
    assert env["generated_at"] is None
    assert env["provenance"]["document_type"] == "unknown"
    assert "extraction" not in env["provenance"]["models"]


def test_write_records(tmp_path):
    (tmp_path / "fused.json").write_text(
        json.dumps({"doc": {"voltage": [{"value": "5V"}]}})
    )
    with patch("structured_data_extraction.records.config.INTERMEDIATE_DIR", str(tmp_path)):
        out_path, n = write_records(tmp_path)

    assert n == 1
    records = json.loads(out_path.read_text(encoding="utf-8"))
    assert records["doc"]["voltage"] == [{"value": "5V"}]
    assert records["doc"]["type"] == "ExtractionResult"
