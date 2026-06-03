from bs4 import BeautifulSoup

SUBROW_LOOKAHEAD = 4


def filter_markdown_table(table_content, matched_row_numbers):
    # keep only the parameter groups that contain matched rows
    lines = table_content.split("\n")
    if len(lines) < 3:
        return table_content

    header_lines = []
    data_rows = []
    group_id = -1
    row_num = 0

    for line in lines:
        if not line.strip():
            continue
        if "|" not in line:
            header_lines.append(line)
            continue

        cells = line.split("|")
        first_col = cells[1].strip() if len(cells) > 1 else ""

        # separator line
        is_separator = all(c.strip() == "" or set(c.strip()) <= {"-", ":", " "} for c in cells)
        if is_separator:
            header_lines.append(line)
            continue

        # first row is the column header
        if row_num == 0:
            header_lines.append(line)
            row_num += 1
            continue

        # new group when first data column has content
        if first_col:
            group_id += 1

        data_rows.append((line, group_id, row_num))
        row_num += 1

    if not data_rows or group_id < 0:
        return table_content

    # which groups have matched rows
    matched_groups = set()
    for line, group, row_number in data_rows:
        if row_number in matched_row_numbers:
            matched_groups.add(group)

    if not matched_groups:
        return table_content

    # keep a few sub-rows after each matched group
    extra_keep = set()
    last_pos_by_group = {}
    for pos, (line, group, row_number) in enumerate(data_rows):
        if group in matched_groups:
            last_pos_by_group[group] = pos

    for group, last_pos in last_pos_by_group.items():
        added = 0
        for pos in range(last_pos + 1, len(data_rows)):
            if added >= SUBROW_LOOKAHEAD:
                break
            next_line, next_group, next_row_number = data_rows[pos]
            if next_group in matched_groups:
                continue
            # stop at category header rows (all value cells empty)
            cells = next_line.split("|")
            value_cells = [c.strip() for c in cells[2:-1]] if len(cells) > 3 else []
            if value_cells and not any(value_cells):
                break
            extra_keep.add(next_row_number)
            added += 1

    result = list(header_lines)
    for line, group, row_number in data_rows:
        if group in matched_groups or row_number in extra_keep:
            result.append(line)

    return "\n".join(result)


def filter_html_table(html_content, matched_identifiers):
    # keep every group whose first column matches one of the identifiers
    soup = BeautifulSoup(html_content, "lxml")
    table = soup.find("table")
    if not table:
        return html_content

    tbody = table.find("tbody")
    if not tbody:
        return html_content

    rows = tbody.find_all("tr")
    if not rows:
        return html_content

    needles = {s.strip().lower() for s in (matched_identifiers or []) if s and s.strip()}
    if not needles:
        return html_content

    # group rows by first-column content, accounting for rowspan
    data_rows = []
    group_id = -1
    group_first = {}
    rowspan_remaining = 0
    rowspan_group = -1

    for row in rows:
        cells = row.find_all("td")

        if rowspan_remaining > 0:
            data_rows.append((row, rowspan_group))
            rowspan_remaining -= 1
            continue

        first_cell = cells[0] if cells else None
        first_text = first_cell.get_text(strip=True) if first_cell else ""

        if first_text:
            group_id += 1
            group_first[group_id] = first_text.lower()

            rowspan = int(first_cell.get("rowspan", 1)) if first_cell else 1
            if rowspan > 1:
                rowspan_remaining = rowspan - 1
                rowspan_group = group_id

        data_rows.append((row, group_id))

    if not data_rows or group_id < 0:
        return html_content

    # keep every group whose first cell substring-matches any identifier (case-insensitive)
    matched_groups = set()
    for group, first_cell in group_first.items():
        for needle in needles:
            if needle in first_cell or first_cell in needle:
                matched_groups.add(group)
                break

    # nothing matched — vision may have rewritten ids, fall back to keeping whole table
    if not matched_groups:
        return html_content

    # keep a few sub-rows after each matched group (refinement rows like "Over Temperature")
    keep_extra = set()
    last_pos_by_group = {}
    for pos, (row, group) in enumerate(data_rows):
        if group in matched_groups:
            last_pos_by_group[group] = pos

    for group, last_pos in last_pos_by_group.items():
        added = 0
        for pos in range(last_pos + 1, len(data_rows)):
            if added >= SUBROW_LOOKAHEAD:
                break
            next_row, next_group = data_rows[pos]
            if next_group in matched_groups:
                continue
            next_cells = next_row.find_all("td")
            if not next_cells:
                continue
            # stop at category header row (all non-first cells empty)
            rest = "".join(c.get_text(strip=True) for c in next_cells[1:])
            if not rest:
                break
            keep_extra.add(id(next_row))
            added += 1

    for row, group in data_rows:
        if group not in matched_groups and id(row) not in keep_extra:
            row.decompose()

    return str(table)
