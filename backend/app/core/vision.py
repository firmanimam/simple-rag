"""Image -> text: GPT-4o vision captions + Tesseract OCR.

Both strategies are best-effort: failures (missing Tesseract binary, no API
key, network errors) degrade gracefully to an empty string so ingestion of the
rest of a document never crashes on one bad image.
"""
from __future__ import annotations

import base64
import io
import logging

from ..config import settings

logger = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "You are describing an image extracted from a document for a search index. "
    "Write a thorough, factual description: what the image shows, any objects, "
    "people, charts, diagrams, tables, or UI elements, and the meaning conveyed. "
    "Transcribe any visible text verbatim. Do not speculate beyond what is shown."
)


def _png_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def caption_image(image_bytes: bytes, mime: str = "image/png") -> str:
    """Return a GPT-4o vision caption for the image, or '' on failure."""
    if not settings.enable_vision or not settings.openai_api_key:
        return ""
    try:
        # Imported lazily so the module loads even without the openai extra wired up.
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _CAPTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": _png_data_url(image_bytes, mime)},
                        },
                    ],
                }
            ],
            max_tokens=600,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # pragma: no cover - network/credentials dependent
        logger.warning("Vision caption failed: %s", exc)
        return ""


def ocr_image(image_bytes: bytes) -> str:
    """Run Tesseract OCR on the image bytes, returning extracted text or ''."""
    if not settings.enable_ocr:
        return ""
    try:
        import pytesseract
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as exc:  # pragma: no cover - depends on tesseract binary
        logger.warning("OCR failed: %s", exc)
        return ""


def describe_image(image_bytes: bytes, mime: str = "image/png") -> dict[str, str]:
    """Return {'caption': ..., 'ocr': ...} for an image. Either may be empty."""
    return {
        "caption": caption_image(image_bytes, mime),
        "ocr": ocr_image(image_bytes),
    }
