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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# pypdfium2 backend + FAST tableformer is ~2x faster than defaults on datasheets.
# OCR is off
pipeline_opts = PdfPipelineOptions()
pipeline_opts.do_ocr = False
pipeline_opts.do_table_structure = True
pipeline_opts.table_structure_options.mode = TableFormerMode.FAST

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_opts,
            backend=PyPdfiumDocumentBackend,
        )
    }
)



_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _scrub(text):
    return _CTRL_RE.sub("", text) if text else text


def _get_item_text(item, label, doc):
    # tables store content differently, need export_to_markdown
    if label == "table":
        try:
            return item.export_to_markdown(doc)
        except TypeError:
            return item.export_to_markdown()

    raw = getattr(item, "text", "") or ""
    if label == "list_item":
        marker = getattr(item, "marker", "") or ""
        return f"{marker} {raw}".strip() if raw else ""
    if label == "section_header":
        lvl = getattr(item, "level", None)
        depth = min(max(int(lvl) if lvl else 2, 1), 6)
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

    if out_md.exists() and out_items.exists():
        logger.info(f"Skipping {pdf_path.name} — already exists")
        return str(out_md)

    logger.info(f"Converting {pdf_path.name}...")
    try:
        result = converter.convert(str(pdf_path))
        doc = result.document

        md_text = _scrub(doc.export_to_markdown())
        out_md.write_text(md_text, encoding="utf-8")

        # collect items from docling
        items = []
        for item, level in doc.iterate_items():
            label = str(item.label) if hasattr(item, 'label') else "unknown"
            text = _scrub(_get_item_text(item, label, doc))
            page, bbox = _get_bbox(item)
            items.append({"label": label, "text": text, "level": level, "page": page, "bbox": bbox})

        out_items.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

     
        logger.info(f"Saved markdown ({len(md_text)} chars) + {len(items)} items → {out_md}")
        return str(out_md)

    except Exception as e:
        logger.error(f"Failed to convert {pdf_path.name}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown")
    parser.add_argument("--input", required=True, help="Path to PDF file")
    args = parser.parse_args()

    result = convert_pdf(args.input)
    if result:
        print(f"Done: {result}")
    else:
        print("Conversion failed.")
