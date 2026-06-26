from structured_data_extraction.rater import _best_section

def test_best_section():
    records = [
        {"metadata": {"section_type": "table"}},
        {"metadata": {"section_type": "figure"}},
        {"metadata": {"section_type": "table"}},
    ]
    assert _best_section(records) == "table"

    records = [
        {"metadata": {"section_type": "figure"}},
        {"metadata": {"section_type": "figure"}},
    ]
    assert _best_section(records) == "figure"

    records = [
        {"metadata": {}},
        {"metadata": {}},
    ]
    assert _best_section(records) == "unknown"