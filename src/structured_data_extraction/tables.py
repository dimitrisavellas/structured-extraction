import base64
import concurrent.futures
import json
import logging
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import fitz
from openai import OpenAI
from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _ramp(x, lo, hi):
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


def _keyword_score(text):
    keywords = ("colspan", "rowspan", "<figure", "<img", "nested")
    return sum(0.15 for kw in keywords if kw in text)


def _pipe_stats(content):
    """Return (ragged, sparsity, hierarchy, ncols, merges) for a pipe-delimited table,
    or None if the content has no pipe rows."""
    pipe_rows = [line for line in content.split("\n") if "|" in line]
    if not pipe_rows:
        return None

    delims = [row.count("|") for row in pipe_rows]
    mode = Counter(delims).most_common(1)[0][0]
    ragged = sum(1 for d in delims if d != mode) / len(delims)

    # sparsity + hierarchy over data rows (separators skipped)
    data = []
    for row in pipe_rows:
        inner = [c.strip() for c in row.split("|")[1:-1]]
        if inner and all(set(c) <= {"-", ":", " ", ""} for c in inner):
            continue
        data.append(inner)

    cells = sum(len(r) for r in data)
    empty = sum(1 for r in data for c in r if not c)
    sparsity = empty / cells if cells else 0.0
    group_rows = sum(1 for r in data if r and r[0] and not any(r[1:]))
    hierarchy = group_rows / len(data) if data else 0.0
    ncols = max(mode - 1, 1)

    # vertical merges: empty cells below a filled cell in the same column
    spanned = 0
    width = max((len(r) for r in data), default=0)
    for col in range(width):
        filled_above = False
        for r in data:
            cell = r[col] if col < len(r) else ""
            if cell:
                filled_above = True
            elif filled_above:
                spanned += 1
    merges = spanned / cells if cells else 0.0

    return ragged, sparsity, hierarchy, ncols, merges


def score_table_complexity(content):
    text = content.lower()
    score = _keyword_score(text)

    stats = _pipe_stats(content)
    if stats is not None:
        ragged, sparsity, hierarchy, ncols, merges = stats
        score += _ramp(sparsity, 0.15, 0.60) * 0.40    # sparse grid
        score += ragged * 0.30                         # broken rows
        score += min(hierarchy / 0.30, 1.0) * 0.20     # nested groups
        score += _ramp(ncols, 4, 12) * 0.10            # wide tables
        score += _ramp(merges, 0.05, 0.35) * 0.35      # vertically merged cells

    return min(score, 1.0)


def _render(pdf_path, page_number, bbox=None, dpi=200):
    doc = fitz.open(pdf_path)
    idx = max(0, min(page_number - 1, len(doc) - 1))
    page = doc[idx]

    clip = None
    if bbox:
        h = page.rect.height
        # docling uses BOTTOMLEFT origin, PyMuPDF uses top-left
        origin = bbox.get("coord_origin", "BOTTOMLEFT")
        if "BOTTOM" in origin:
            clip = fitz.Rect(bbox["l"], h - bbox["t"], bbox["r"], h - bbox["b"])
        else:
            clip = fitz.Rect(bbox["l"], bbox["t"], bbox["r"], bbox["b"])

    pix = page.get_pixmap(dpi=dpi, clip=clip)
    b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    doc.close()
    return b64


TABLE_VISION_PROMPT = (
    "You are a table-to-HTML conversion expert. Look at the table in the provided image "
    "and convert it to clean, semantic HTML. Use proper <thead>, <tbody>, <th>, <td> tags. "
    "If a cell is merged/spanned in the original, REPEAT its value in every row and column it "
    "covers — do NOT use rowspan or colspan; every <td> must be filled. "
    "Preserve ALL values exactly as shown — do not omit or invent data. "
    "Do NOT add any explanation, just output the HTML table.\n\n"
    "Here is the markdown version of the table for reference (it may be garbled):\n"
)

TABLE_TEXT_PROMPT = (
    "Convert the following markdown table to clean, semantic HTML. "
    "Use proper <thead>, <tbody>, <th>, <td> tags. "
    "Merge cells only where the original has merged cells. "
    "Do NOT add any explanation, just output the HTML table.\n\n"
)


def strip_code_fence(text):
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text


_ollama_client = None


def _client():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key=config.OLLAMA_API_KEY)
    return _ollama_client


def _call_model(model, messages):
    resp = _client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        seed=config.SEED,
    )
    return strip_code_fence(resp.choices[0].message.content.strip())


def _call_vision_model(b64_img, table_md):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": TABLE_VISION_PROMPT + table_md},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
        ],
    }]
    return _call_model(config.OLLAMA_VISION_MODEL, messages)


def _call_text_model(table_md):
    messages = [{"role": "user", "content": TABLE_TEXT_PROMPT + table_md}]
    return _call_model(config.OLLAMA_EXTRACT_MODEL, messages)


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._row = None
        self._cell = None
        self._attrs = {}

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._attrs = dict(attrs)
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append((tag, "".join(self._cell).strip(),
                              int(self._attrs.get("colspan") or 1),
                              int(self._attrs.get("rowspan") or 1)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _expand_spans(html):
    parser = _TableParser()
    parser.feed(html)
    if not parser.rows:
        return html
    grid = {}
    for r, row in enumerate(parser.rows):
        c = 0
        for tag, text, colspan, rowspan in row:
            while (r, c) in grid:
                c += 1
            for dr in range(rowspan):
                for dc in range(colspan):
                    grid[(r + dr, c + dc)] = (tag, text)
            c += colspan
    width = max((c for _, c in grid), default=-1) + 1
    lines = ["<table>"]
    for r in range(len(parser.rows)):
        lines.append("  <tr>")
        for c in range(width):
            tag, text = grid.get((r, c), ("td", ""))
            lines.append(f"    <{tag}>{text}</{tag}>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


def process_tables(sections_path, pdf_path=None):
    sections_path = Path(sections_path)
    if not sections_path.exists():
        raise FileNotFoundError(f"Sections not found: {sections_path}")

    stem = sections_path.stem.replace("_sections", "")
    out_path = Path(config.INTERMEDIATE_DIR) / f"{stem}_tables.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.info(f"Skipping tables — already exists")
        return json.loads(out_path.read_text(encoding="utf-8"))

    sections = json.loads(sections_path.read_text(encoding="utf-8"))
    table_sections = [(i, s) for i, s in enumerate(sections) if s["type"] == "table"]

    if not table_sections:
        logger.info("No tables found")
        out_path.write_text("[]", encoding="utf-8")
        return []

    # try to find the PDF if not provided
    resolved_pdf = pdf_path
    if not resolved_pdf:
        candidate = Path(config.PDF_DIR) / f"{stem}.pdf"
        if candidate.exists():
            resolved_pdf = str(candidate)

    logger.info(f"Scoring {len(table_sections)} tables...")
    results = []

    for i, (section_idx, section) in enumerate(table_sections):
        content = section["content"]
        complexity = score_table_complexity(content)
        page = section.get("page") or 1

        logger.info(f"  Table {i}: complexity={complexity:.2f}")

        results.append({
            "section_idx": section_idx,
            "complexity_score": round(complexity, 4),
            "page": page,
            "content": content,
            "bbox": section.get("bbox"),
            "pdf_path": resolved_pdf,
        })

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Scored {len(results)} tables → {out_path}")
    return results


def _refine_one_table(table):
    idx = table["section_idx"]
    pdf_path = table.get("pdf_path")
    page = table.get("page", 1)
    content = table.get("content", "")
    bbox = table.get("bbox")

    html = None
    used_vision = False
    if pdf_path and Path(pdf_path).exists():
        try:
            image_b64 = _render(pdf_path, page, bbox)
            html = _call_vision_model(image_b64, content)
            used_vision = True
        except Exception as e:
            logger.warning(f"  Section {idx}: vision failed ({e}), falling back to text LLM")

    if html is None:
        try:
            html = _call_text_model(content)
        except Exception as e:
            logger.error(f"  Section {idx}: failed — {e}")
            return idx, None, False

    return idx, _expand_spans(html), used_vision


_WORD_RE = re.compile(r"[A-Za-z]{4,}")


def _word_cell_ratio(content):
    cells = [c.strip() for line in content.splitlines() for c in line.split("|")
             if c.strip() and set(c.strip()) != {"-"}]
    if len(cells) < 12:
        return 1.0
    return sum(1 for c in cells if _WORD_RE.search(c)) / len(cells)


def refine_symbol_only_tables(tables_path):
    tables = json.loads(Path(tables_path).read_text(encoding="utf-8"))
    idxs = {t["section_idx"] for t in tables
            if t.get("complexity_score", 0) >= config.VISION_COMPLEXITY_THRESHOLD
            and "|" in t.get("content", "") and _word_cell_ratio(t["content"]) < 0.10}
    if idxs:
        logger.info(f"Pre-refining {len(idxs)} symbol-only table(s) before embedding: {sorted(idxs)}")
        refine_matched_tables(tables_path, idxs, finalize=False)


def refine_matched_tables(tables_path, matched_section_indices, finalize=True):
    tables_path = Path(tables_path)
    if not tables_path.exists():
        return []

    tables = json.loads(tables_path.read_text(encoding="utf-8"))

    matched = [t for t in tables if t.get("section_idx") in matched_section_indices]
    to_refine = [t for t in matched
                 if t.get("complexity_score", 0) >= config.VISION_COMPLEXITY_THRESHOLD
                 and not t.get("refined")]

    skipped = len(matched) - len(to_refine)
    if skipped:
        logger.info(f"  Refining {len(to_refine)} matched tables above complexity "
                    f"{config.VISION_COMPLEXITY_THRESHOLD}; {skipped} simpler ones left as markdown")

    if not to_refine:
        if finalize:
            for table in tables:
                table.pop("pdf_path", None)
            tables_path.write_text(json.dumps(tables, indent=2, ensure_ascii=False), encoding="utf-8")
        return tables

    workers = max(1, min(config.VISION_PARALLELISM, len(to_refine)))
    logger.info(f"  Refining {len(to_refine)} tables ({workers} workers in parallel)...")

    by_idx = {table["section_idx"]: table for table in tables}
    vision_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_refine_one_table, table) for table in to_refine]
        for fut in concurrent.futures.as_completed(futures):
            idx, html, used_vision = fut.result()
            if html is None:
                continue
            by_idx[idx]["original_content"] = by_idx[idx].get("content", "")
            by_idx[idx]["content"] = html
            by_idx[idx]["refined"] = True
            by_idx[idx]["refined_format"] = "html"
            if used_vision:
                vision_count += 1
            logger.info(f"  Section {idx}: refined ({len(html)} chars, vision={used_vision})")

    if finalize:
        for table in tables:
            table.pop("pdf_path", None)
    tables_path.write_text(json.dumps(tables, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Refined {vision_count} tables via vision model (of {len(to_refine)} attempted)")
    return tables
