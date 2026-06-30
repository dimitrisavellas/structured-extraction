import pytest

from structured_data_extraction.tables import (
    _expand_spans,
    _pipe_stats,
    _word_cell_ratio,
    score_table_complexity,
    strip_code_fence,
)


def test_strip_code_fence():
    assert strip_code_fence("```html\n<table></table>\n```") == "<table></table>"
    assert strip_code_fence("```\nplain\n```") == "plain"
    assert strip_code_fence("no fence") == "no fence"


def test_pipe_stats_none_without_pipes():
    assert _pipe_stats("just prose, no pipes") is None


def test_pipe_stats_simple_table():
    content = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    ragged, sparsity, hierarchy, ncols, merges = _pipe_stats(content)
    assert (ragged, sparsity, hierarchy, merges) == (0.0, 0.0, 0.0, 0.0)
    assert ncols == 2


def test_pipe_stats_vertical_merge():
    # empty cell directly under a filled one in the same column -> a vertical merge
    ragged, sparsity, hierarchy, ncols, merges = _pipe_stats("| A | B |\n| | C |\n| D | E |")
    assert sparsity == pytest.approx(1 / 6)
    assert merges == pytest.approx(1 / 6)


def test_pipe_stats_hierarchy():
    # a row with only its first cell filled is a group/section row
    _, _, hierarchy, ncols, _ = _pipe_stats("| Group | | |\n| a | 1 | 2 |")
    assert hierarchy == 0.5
    assert ncols == 3


def test_score_table_complexity_simple_is_zero():
    assert score_table_complexity("| A | B |\n| --- | --- |\n| 1 | 2 |") == 0.0


def test_score_table_complexity_keywords():
    # each complexity keyword adds 0.15 (counted once), no pipe stats here
    content = "this has colspan and rowspan and nested <img <figure markup"
    assert score_table_complexity(content) == pytest.approx(0.75)


def test_score_table_complexity_clamped_to_one():
    sparse = "\n".join("| x |" + " |" * 10 for _ in range(8))  # very sparse, wide grid
    assert score_table_complexity("colspan rowspan nested " + sparse) <= 1.0


def test_expand_spans_colspan():
    html = "<table><tr><td colspan='2'>A</td></tr><tr><td>B</td><td>C</td></tr></table>"
    out = _expand_spans(html)
    # the spanned value is repeated into every column it covered
    assert out.count("<td>A</td>") == 2
    assert "<td>B</td>" in out and "<td>C</td>" in out


def test_expand_spans_rowspan():
    html = "<table><tr><td rowspan='2'>A</td><td>B</td></tr><tr><td>C</td></tr></table>"
    out = _expand_spans(html)
    assert out.count("<td>A</td>") == 2


def test_expand_spans_passthrough_when_no_table():
    assert _expand_spans("not a table") == "not a table"


def test_word_cell_ratio_small_table_is_one():
    # fewer than 12 cells -> treated as word-bearing (1.0), too small to judge
    assert _word_cell_ratio("| a | b |") == 1.0


def test_word_cell_ratio_symbol_only():
    content = "\n".join(f"| {i} | {i} |" for i in range(7))  # 14 numeric cells
    assert _word_cell_ratio(content) == 0.0


def test_word_cell_ratio_wordy():
    content = "\n".join("| temperature | voltage |" for _ in range(7))
    assert _word_cell_ratio(content) == 1.0
