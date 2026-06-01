"""Load PDFs and images into LangChain Documents (everything becomes text).

Strategy (per the plan):
  * Digital PDF page  -> extract its text layer with PyMuPDF (type="text").
  * Scanned PDF page  -> render to an image, OCR + vision-caption (type="ocr").
  * Embedded images   -> vision-caption + OCR (type="caption").
  * Standalone image  -> vision-caption + OCR (type="caption"/"ocr").

Each Document carries metadata: source, page, type, doc_id.
"""
from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

from ..config import settings
from .vision import describe_image

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = IMAGE_EXTS | PDF_EXTS


def _image_to_text(image_bytes: bytes, mime: str = "image/png") -> tuple[str, str]:
    """Turn raw image bytes into searchable text + a metadata type label."""
    res = describe_image(image_bytes, mime)
    caption, ocr = res["caption"], res["ocr"]

    parts: list[str] = []
    if ocr:
        parts.append(f"[OCR text]\n{ocr}")
    if caption:
        parts.append(f"[Image description]\n{caption}")
    text = "\n\n".join(parts).strip()

    if caption:
        kind = "caption"
    elif ocr:
        kind = "ocr"
    else:
        kind = "caption"  # produced nothing, but it came from an image
    return text, kind


def _meta(doc_id: str, source: str, page: int, kind: str) -> dict:
    return {"doc_id": doc_id, "source": source, "page": page, "type": kind}


def load_pdf(path: Path, source: str, doc_id: str) -> list[Document]:
    docs: list[Document] = []
    with fitz.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            page_no = page_index + 1
            text = page.get_text("text").strip()

            if len(text) >= settings.scanned_text_threshold:
                # Digital page with a real text layer.
                docs.append(
                    Document(page_content=text, metadata=_meta(doc_id, source, page_no, "text"))
                )
            else:
                # Scanned / image-only page: render and OCR + caption.
                try:
                    pix = page.get_pixmap(dpi=200)
                    rendered = pix.tobytes("png")
                    img_text, _ = _image_to_text(rendered, "image/png")
                    if img_text:
                        docs.append(
                            Document(
                                page_content=img_text,
                                metadata=_meta(doc_id, source, page_no, "ocr"),
                            )
                        )
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed rendering scanned page %s of %s: %s", page_no, source, exc)

            # Embedded images on the page -> caption + OCR.
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    base = pdf.extract_image(xref)
                    img_bytes = base["image"]
                    mime = f"image/{base.get('ext', 'png')}"
                    img_text, kind = _image_to_text(img_bytes, mime)
                    if img_text:
                        docs.append(
                            Document(
                                page_content=img_text,
                                metadata=_meta(doc_id, source, page_no, kind),
                            )
                        )
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed embedded image (xref %s) in %s: %s", xref, source, exc)

    return docs


def load_image(path: Path, source: str, doc_id: str) -> list[Document]:
    img_bytes = path.read_bytes()
    ext = path.suffix.lower().lstrip(".") or "png"
    text, kind = _image_to_text(img_bytes, f"image/{ext}")
    if not text:
        return []
    return [Document(page_content=text, metadata=_meta(doc_id, source, 1, kind))]


def load_file(path: str | Path, source: str, doc_id: str) -> list[Document]:
    """Dispatch on file extension. `source` is the display filename."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        return load_pdf(path, source, doc_id)
    if ext in IMAGE_EXTS:
        return load_image(path, source, doc_id)
    raise ValueError(f"Unsupported file type: {ext}")
