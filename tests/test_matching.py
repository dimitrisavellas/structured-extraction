from structured_data_extraction.matching import (
    GAP,
    MATCH,
    MISMATCH,
    _as_number,
    _empty,
    _norm_text,
    _norm_unit,
    _numbers,
    compare,
    model_slug,
    normalize_key,
    same_fact,
)


def test_empty():
    assert _empty(None) is True
    assert _empty("") is True
    assert _empty("  ") is True
    assert _empty("x") is False
    assert _empty(0) is False


def test_as_number():
    assert _as_number(10) == 10.0
    assert _as_number("10") == 10.0
    # bools are not numbers
    assert _as_number(True) is None
    # a value with a baked-in unit is not a bare number
    assert _as_number("10 mm") is None
    assert _as_number("abc") is None


def test_numbers_extracts_all():
    assert _numbers("10 to 20") == frozenset({10.0, 20.0})
    assert _numbers("none") == frozenset()


def test_norm_unit():
    assert _norm_unit("millimeters") == "mm"
    assert _norm_unit("inches") == "in"
    # leading value stripped and micro sign folded, so "5 µM" matches a bare "µM"
    assert _norm_unit("5 µM") == _norm_unit("µM") == "um"


def test_norm_text():
    # underscores collapse so subscripts match
    assert _norm_text("R_L") == "rl"
    # synonyms fold
    assert _norm_text("Typical") == "typ"
    assert _norm_text("Max-Min") == "max min"


def test_compare_gap():
    assert compare("value", "", "x") == GAP
    assert compare("value", None, "x") == GAP


def test_compare_numeric():
    assert compare("value", 10, 10.0) == MATCH
    assert compare("value", 10, 15) == MISMATCH


def test_compare_unit():
    assert compare("unit", "millimeters", "mm") == MATCH
    assert compare("width_unit", "inches", "in") == MATCH
    assert compare("unit", "mm", "cm") == MISMATCH


def test_compare_number_in_string():
    """A bare number matches a string carrying exactly that number, either side."""
    assert compare("value", 10, "10 mL") == MATCH
    assert compare("value", "10 mL", 10) == MATCH


def test_compare_text():
    assert compare("name", "Typical", "typ") == MATCH
    assert compare("desc", "foo", "bar") == MISMATCH
    # different embedded numbers can't be the same fact even if the text is close
    assert compare("name", "model 5", "model 7") == MISMATCH


def test_model_slug():
    assert model_slug("Claude 3.5 Sonnet") == "claude_3_5_sonnet"
    assert model_slug("GPT-4o") == "gpt_4o"


def test_normalize_key():
    assert normalize_key("Pin Name") == "pin_name"
    assert normalize_key("V-DD") == "v_dd"


def test_same_fact():
    assert same_fact({"value": "10 mm"}, {"value": "10 mm"}) is True
    assert same_fact({"value": "10"}, {"value": "10.0"}) is True
    assert same_fact({"value": "10 mm"}, {"value": "20 mm"}) is False
    # metadata fields are ignored
    assert same_fact(
        {"value": "A", "metadata": {"x": 1}},
        {"value": "A", "metadata": {"x": 2}},
    ) is True
