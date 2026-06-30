import json
from unittest.mock import patch

from structured_data_extraction.rater import (
    _best_section,
    _filled_fields,
    _fields_match,
    _fused_metadata,
    _group_by_doc,
    _merge_into,
    _strict_eq,
    agreement,
    consensus,
    dedup,
    fuse,
    fuse_outputs,
    load_outputs,
    rate,
)


def test_filled_fields_counts_non_meta_nonempty():
    assert _filled_fields({"value": "A", "unit": "mm", "metadata": {"x": 1}, "confidence": 0.9}) == 2
    assert _filled_fields({"value": None, "unit": ""}) == 0


def test_fuse_same_model_records_stay_separate():
    """fuse never puts two records from the same model in one cluster."""
    clusters = fuse({"m1": [{"a": 1}, {"a": 1}]})
    assert len(clusters) == 2


def test_fuse_matches_same_fact_across_models():
    clusters = fuse({"m1": [{"id": 1}], "m2": [{"id": 1}], "m3": [{"id": 1}]})
    assert len(clusters) == 1
    assert set(clusters[0]) == {"m1", "m2", "m3"}


def test_fuse_different_facts_remain_separate():
    clusters = fuse({
        "m1": [{"id": 1, "value": "A"}, {"id": 2, "value": "B"}],
        "m2": [{"id": 1, "value": "A"}, {"id": 3, "value": "C"}],
    })
    # only m1/m2 id=1 are the same fact -> one shared cluster; the other two stand alone
    by_models = {frozenset(c): c for c in clusters}
    assert len(clusters) == 3
    assert by_models[frozenset({"m1", "m2"})]["m1"] == {"id": 1, "value": "A"}


def test_best_section_priority():
    # table beats text beats list; missing type loses to all
    assert _best_section([{"metadata": {"section_type": "text"}},
                          {"metadata": {"section_type": "table"}}]) == "table"
    assert _best_section([{"metadata": {}}, {"metadata": {}}]) == "unknown"


def test_fused_metadata():
    records = [
        {"value": "A", "metadata": {"section_type": "table", "specified": False}},
        {"value": "A", "metadata": {"section_type": "text"}},
    ]
    assert _fused_metadata(records) == {"section_type": "table", "specified": False}
    assert _fused_metadata([{"value": "A"}]) == {"section_type": "unknown", "specified": True}


def test_consensus():
    cluster = {
        "m1": {"value": "10 mm", "metadata": {"section_type": "table"}},
        "m2": {"value": "10 mm", "metadata": {"section_type": "text"}},
    }
    result = consensus(cluster, n_models=3)
    assert result["value"] == "10 mm"
    assert result["confidence"] == round(2 / 3, 2)
    assert result["models"] == ["m1", "m2"]
    assert result["metadata"]["section_type"] == "table"


def test_fields_match():
    assert _fields_match("value", 10, 10.0) is True          # numeric
    assert _fields_match("value", 10, 15) is False
    assert _fields_match("unit", "millimeters", "mm") is True  # unit normalization
    assert _fields_match("name", "Typical", "typ") is True     # text normalization
    assert _fields_match("name", "foo", "bar") is False


def test_strict_eq():
    a = {"value": "10 mm", "unit": "mm"}
    assert _strict_eq(a, {"value": "10 mm", "unit": "millimeters"}) is True   # unit synonym
    assert _strict_eq(a, {"value": "20 mm", "unit": "mm"}) is False           # value differs
    assert _strict_eq(a, {"value": "10 mm"}) is True                         # missing field is a gap, ignored
    # only metadata differs -> still equal
    assert _strict_eq({"value": "A", "metadata": {"x": 1}}, {"value": "A", "metadata": {"x": 2}}) is True


def test_strict_eq_all_empty_is_equal():
    # all shared fields empty -> nothing disagrees -> treated as the same fact (records merge in dedup)
    assert _strict_eq({"value": ""}, {"value": None}) is True
    assert _strict_eq({}, {}) is True


def test_merge_into():
    existing = {"value": "A", "metadata": {"section_type": "table"}, "models": ["m1"]}
    incoming = {"value": "A", "metadata": {"section_type": "text"}, "models": ["m2"]}
    merged = _merge_into(existing, incoming, n_models=2)
    assert merged["models"] == ["m1", "m2"]
    assert merged["confidence"] == 1.0
    assert merged["metadata"]["section_type"] == "table"


def test_dedup():
    facts = [
        {"value": "A", "unit": "mm", "models": ["m1"]},
        {"value": "A", "unit": "millimeters", "models": ["m2"]},  # same fact as above
        {"value": "B", "unit": "mm", "models": ["m3"]},
    ]
    merged = dedup(facts, n_models=3)
    by_value = {f["value"]: f for f in merged}
    assert len(merged) == 2
    assert by_value["A"]["models"] == ["m1", "m2"]
    assert by_value["A"]["confidence"] == round(2 / 3, 2)


def test_load_outputs(tmp_path):
    (tmp_path / "doc1_extraction_model1.json").write_text(
        json.dumps({"field1": [{"value": "A"}]}), encoding="utf-8")
    (tmp_path / "doc1_extraction_model2.json").write_text(
        json.dumps({"field1": [{"value": "A"}], "field2": [{"value": "B"}]}), encoding="utf-8")

    with patch("structured_data_extraction.rater.config.EXTRACTION_MODELS", ["model1", "model2"]), \
         patch("structured_data_extraction.rater.config.OUTPUT_DIR", str(tmp_path)):
        by_file = load_outputs()

    assert set(by_file["doc1"]) == {"field1", "field2"}
    assert by_file["doc1"]["field1"]["model1"][0]["value"] == "A"


def test_fuse_outputs():
    by_file = {"doc1": {"field1": {"model1": [{"value": "A"}], "model2": [{"value": "A"}]}}}
    facts = fuse_outputs(by_file, ["model1", "model2"])
    assert len(facts) == 1
    assert facts[0] == {"source_file": "doc1.pdf", "field": "field1", "value": "A",
                        "confidence": 1.0, "models": ["model1", "model2"],
                        "metadata": {"section_type": "unknown", "specified": True}}


def test_group_by_doc():
    facts = [
        {"source_file": "a.pdf", "field": "f1", "value": "x"},
        {"source_file": "b.pdf", "field": "f1", "value": "z"},
    ]
    grouped = _group_by_doc(facts)
    assert set(grouped) == {"a", "b"}
    assert grouped["a"]["f1"][0]["value"] == "x"
    # routing keys are stripped from the grouped records
    assert "source_file" not in grouped["a"]["f1"][0]


def test_agreement():
    facts = [{"models": ["a", "b"]}, {"models": ["a", "b", "c"]}, {"models": ["a", "c"]}]
    by_pair = {tuple(r["raters"]): r for r in agreement(facts, ["a", "b", "c"])}
    assert by_pair[("a", "b")]["agreement"] == round(2 / 3, 4)
    assert by_pair[("b", "c")]["agreement"] == round(1 / 3, 4)
    # a is always present (constant) -> kappa 0 b vs c genuinely disagree -> negative
    assert by_pair[("a", "b")]["kappa"] == 0.0
    assert by_pair[("b", "c")]["kappa"] == -0.5


def test_agreement_empty():
    assert agreement([], ["a", "b"]) == []


def test_agreement_identical_labels_skips_kappa():
    """cohen_kappa_score needs at least two label classes."""
    rows = agreement([{"models": ["a", "b"]}, {"models": ["a", "b"]}], ["a", "b"])
    assert rows[0]["kappa"] is None


def test_rate(tmp_path):
    by_file = {
        "doc1": {
            "field1": {
                "model1": [{"value": "A", "metadata": {"section_type": "table"}}],
                "model2": [{"value": "A", "metadata": {"section_type": "text"}}],
            },
            "field2": {"model1": [{"value": "B"}]},  # 1 of 2 models -> below majority cutoff
        }
    }
    with patch("structured_data_extraction.rater.config.OUTPUT_DIR", str(tmp_path)):
        rate(by_file, models=["model1", "model2"])

    fused = json.loads((tmp_path / "fused.json").read_text(encoding="utf-8"))
    user = json.loads((tmp_path / "user_report.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "rating_report.json").read_text(encoding="utf-8"))

    # fused keeps the full view (incl. the below-cutoff fact); user_report drops it
    assert fused["doc1"]["field1"][0]["metadata"]["section_type"] == "table"
    assert fused["doc1"]["field2"][0]["value"] == "B"
    assert "field2" not in user["doc1"]
    assert report["facts"] == 2
    assert report["full_agreement"] == 1
    assert [f["field"] for f in report["partial_agreement"]] == ["field2"]
