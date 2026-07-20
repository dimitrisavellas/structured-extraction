"""PDF-to-structured-items conversion via Docling.

Converts a PDF into two artifacts: a full-document Markdown file and a
per-item JSON list (_items.json) where each item carries its Docling label,
text, page number, and bounding box.

OCR is disabled because the target documents are born-digital; Docling's table
structure model runs in FAST mode (roughly 2x faster than ACCURATE) because the
downstream vision-refinement step in tables.py re-reads complex tables anyway,
so the speed/accuracy trade-off favours throughput here.

Bounding boxes are stored in Docling's native BOTTOMLEFT coordinate origin.
Any consumer that needs top-left origin (e.g. PyMuPDF rendering in tables.py)
converts on read.
"""
import argparse
import json
import logging
import re
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from . import config

logger = logging.getLogger(__name__)

_converter = None


def _get_converter():
    """Build the Docling converter on first use — it loads vision models, so importing
    this module shouldn't pay that cost."""
    global _converter
    if _converter is None:
        #Custom Docling pipeline configuration for PDF conversion
        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr = False
        pipeline_opts.do_table_structure = True
        pipeline_opts.table_structure_options.mode = TableFormerMode.FAST #reduced runtime by 2x compared to accurate mode,similar output quality
                                                                          #since we pay the price of vision refinement anyways at least we have speed
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_opts,
                    backend=PyPdfiumDocumentBackend, # since we don't need OCR, we can use the faster PyPdfium backend instead of Tesseract
                )
            }
        )
    return _converter


_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]") # control characters that break JSON and Markdown rendering


def _scrub(text):
    """Remove control characters from text, which can break JSON and Markdown rendering."""
    return _CTRL_RE.sub("", text) if text else text


def _get_item_text(item, label, doc):
    """Turn one docling item into Markdown text.

    Docling's whole-doc export gives one big string, we need the text for
    each item on its own so it stays lined up with that item's label, page, and
    bbox. So process each item by type: tables go through docling, list items
    get their marker in front, a title gets one '#' and a section_header gets
    '#' x its docling level, everything else is just its plain text. Gives back
    "" when an item has no text.

    The header depth feeds the section classifier (features.regex_features reads it
    back off the '#' prefix), so keep emitting the level even though the markdown
    export wouldn't strictly need it.
    """
    # tables store content differently, need export_to_markdown
    if label == "table":
        try:
            return item.export_to_markdown(doc)
        except TypeError:
            return item.export_to_markdown()  # older docling: no doc arg

    raw = getattr(item, "text", "") or ""
    if label == "list_item":
        marker = getattr(item, "marker", "") or ""
        return f"{marker} {raw}".strip() if raw else ""
    if label == "section_header":
        lvl = getattr(item, "level", None)
        depth = min(max(int(lvl) if lvl else 2, 1), 6)  # docling level, default 2, clamped to 1..6
        return ("#" * depth + " " + raw) if raw else ""
    if label == "title":
        return ("# " + raw) if raw else ""
    return raw


def _get_bbox(item):
    if not hasattr(item, 'prov') or not item.prov:
        return None, None
    prov = item.prov[0]
    page = prov.page_no
    bbox = None
    if hasattr(prov, 'bbox') and prov.bbox:
        bbox = {
            "l": prov.bbox.l, "t": prov.bbox.t,
            "r": prov.bbox.r, "b": prov.bbox.b,
            "coord_origin": str(prov.bbox.coord_origin),
        }
    return page, bbox


def convert_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_md = Path(config.MARKDOWN_DIR) / f"{pdf_path.stem}.md"
    out_items = Path(config.INTERMEDIATE_DIR) / f"{pdf_path.stem}_items.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_items.parent.mkdir(parents=True, exist_ok=True)

    if out_md.exists() and out_items.exists() and not config.FORCE:
        logger.info(f"Skipping {pdf_path.name} — already exists")
        return str(out_md)

    logger.info(f"Converting {pdf_path.name}...")
    try:
        result = _get_converter().convert(str(pdf_path))
        doc = result.document

        md_text = _scrub(doc.export_to_markdown())
        out_md.write_text(md_text, encoding="utf-8")

        # collect items from docling
        items = []
        for item, _ in doc.iterate_items():
            label = str(item.label) if hasattr(item, 'label') else "unknown"
            text = _scrub(_get_item_text(item, label, doc))
            page, bbox = _get_bbox(item)
            items.append({"label": label, "text": text, "page": page, "bbox": bbox})

        out_items.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

     
        logger.info(f"Saved markdown ({len(md_text)} chars) + {len(items)} items → {out_md}")
        return str(out_md)

    except Exception as e:
        logger.error(f"Failed to convert {pdf_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown")
    parser.add_argument("--input", required=True, help="Path to PDF file")
    args = parser.parse_args()

    result = convert_pdf(args.input)
    if result:
        print(f"Done: {result}")
    else:
        print("Conversion failed.")


if __name__ == "__main__":
    main()
