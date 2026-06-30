import json
from unittest.mock import patch

from structured_data_extraction import config
from structured_data_extraction.fdo_records import fdo_envelope, write_records


def test_fdo_envelope(tmp_path):
    (tmp_path / "doc_run.json").write_text(json.dumps({
        "extracted_at": "2026-01-01T00:00:00Z",
        "user_keys": ["voltage"],
        "models": ["m1"],
    }))
    (tmp_path / "doc_schema.json").write_text(json.dumps({"document_type": "datasheet"}))

    with patch("structured_data_extraction.fdo_records.config.INTERMEDIATE_DIR", str(tmp_path)):
        env = fdo_envelope("doc", {"voltage": [{"value": "5V"}]})

    assert env["type"] == "ExtractionResult"
    assert env["generated_at"] == "2026-01-01T00:00:00Z"
    prov = env["provenance"]
    assert prov["source_file"] == "doc.pdf"
    assert prov["document_type"] == "datasheet"
    assert prov["models"]["extraction"] == ["m1"]
    assert prov["user_keys"] == ["voltage"]
    # the extracted data is spread into the envelope alongside provenance
    assert env["voltage"] == [{"value": "5V"}]


def test_fdo_envelope_defaults_when_run_files_missing(tmp_path):
    with patch("structured_data_extraction.fdo_records.config.INTERMEDIATE_DIR", str(tmp_path)):
        env = fdo_envelope("doc", {})
    assert env["generated_at"] is None
    assert env["provenance"]["document_type"] == "unknown"
    # falls back to the configured extraction models when the run stamp is absent
    assert env["provenance"]["models"]["extraction"] == config.EXTRACTION_MODELS


def test_write_records(tmp_path):
    (tmp_path / "user_report.json").write_text(
        json.dumps({"doc": {"voltage": [{"value": "5V"}]}})
    )
    with patch("structured_data_extraction.fdo_records.config.INTERMEDIATE_DIR", str(tmp_path)):
        out_path, n = write_records(tmp_path)

    assert n == 1
    records = json.loads(out_path.read_text(encoding="utf-8"))
    assert records["doc"]["voltage"] == [{"value": "5V"}]
    assert records["doc"]["type"] == "ExtractionResult"
