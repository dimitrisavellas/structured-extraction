from types import SimpleNamespace

from structured_data_extraction.convert import _get_bbox, _get_item_text, _scrub


def test_scrub_removes_control_chars_keeps_whitespace():
    assert _scrub("a\x00b\x07c") == "abc"
    # tabs and newlines are not control chars we strip
    assert _scrub("line1\nline2\tend") == "line1\nline2\tend"
    assert _scrub("") == ""
    assert _scrub(None) is None


def test_get_item_text_plain():
    assert _get_item_text(SimpleNamespace(text="hello"), "text", None) == "hello"


def test_get_item_text_list_item_gets_marker():
    item = SimpleNamespace(text="apples", marker="-")
    assert _get_item_text(item, "list_item", None) == "- apples"


def test_get_item_text_header_depth():
    # section_header depth comes from the docling level (default 2 when absent), title is always 1
    assert _get_item_text(SimpleNamespace(text="Sub"), "section_header", None) == "## Sub"
    assert _get_item_text(SimpleNamespace(text="Deep", level=3), "section_header", None) == "### Deep"
    assert _get_item_text(SimpleNamespace(text="Big"), "title", None) == "# Big"


def test_get_item_text_empty_returns_empty():
    assert _get_item_text(SimpleNamespace(text="", marker="-"), "list_item", None) == ""
    assert _get_item_text(SimpleNamespace(text=""), "section_header", None) == ""


def test_get_item_text_table_passes_doc():
    class Table:
        def export_to_markdown(self, doc=None):
            return f"md:{doc}"

    assert _get_item_text(Table(), "table", "DOC") == "md:DOC"


def test_get_item_text_table_falls_back_without_doc_arg():
    # older docling: export_to_markdown takes no doc argument
    class OldTable:
        def export_to_markdown(self):
            return "old-md"

    assert _get_item_text(OldTable(), "table", "DOC") == "old-md"


def test_get_bbox_none_without_prov():
    assert _get_bbox(SimpleNamespace(prov=[])) == (None, None)


def test_get_bbox_extracts_page_and_box():
    bbox = SimpleNamespace(l=1.0, t=2.0, r=3.0, b=4.0, coord_origin="BOTTOMLEFT")
    item = SimpleNamespace(prov=[SimpleNamespace(page_no=5, bbox=bbox)])
    page, box = _get_bbox(item)
    assert page == 5
    assert box == {"l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0, "coord_origin": "BOTTOMLEFT"}


def test_get_bbox_page_only_when_no_box():
    item = SimpleNamespace(prov=[SimpleNamespace(page_no=2, bbox=None)])
    assert _get_bbox(item) == (2, None)
