from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from docx import Document

HEADING_PATTERNS = (
    (1, re.compile(r"^第[一二三四五六七八九十百]+章(?:\s|、|：|:|$)")),
    (3, re.compile(r"^\d+\.\d+\.\d+(?:\s|、|：|:|$)")),
    (2, re.compile(r"^\d+\.\d+(?:\s|、|：|:|$)")),
    (1, re.compile(r"^\d+[\.、]")),
    (1, re.compile(r"^[一二三四五六七八九十]+、")),
    (2, re.compile(r"^[（(][一二三四五六七八九十]+[）)]")),
)


def detect_heading_level(paragraph: Any) -> int | None:
    text = paragraph.text.strip()
    style_name = str(getattr(paragraph.style, "name", "") or "").strip()
    style_match = re.search(r"(?:Heading|标题)\s*([1-4])", style_name, re.IGNORECASE)
    if style_match:
        return int(style_match.group(1))
    for level, pattern in HEADING_PATTERNS:
        if pattern.match(text):
            return level
    return None


def _split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = [item.strip() for item in re.split(r"(?<=[。！？!?；;])", text) if item.strip()]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            pieces.append(current)
            current = ""
        if len(sentence) > max_chars:
            pieces.extend(sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars))
        else:
            current += sentence
    if current:
        pieces.append(current)
    return pieces


def split_docx(document_path: str | Path, max_chars: int = 1800) -> list[dict[str, Any]]:
    path = Path(document_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"文档不存在：{path}")
    if path.suffix.lower() != ".docx":
        raise ValueError("当前切分器只支持DOCX文档")
    if max_chars <= 0:
        raise ValueError("max_chars必须大于0")

    document = Document(path)
    document_title = path.stem
    headings = ["", "", "", ""]
    chunks: list[dict[str, Any]] = []
    paragraphs: list[str] = []

    def append_chunk(text: str) -> None:
        chunks.append({
            "chunk_id": f"{path.stem}_{len(chunks) + 1:06d}",
            "source": str(path),
            "title": document_title,
            "chapter_title": headings[0],
            "section_title": headings[1],
            "third_title": headings[2],
            "fourth_title": headings[3],
            "text": text,
        })

    def flush() -> None:
        nonlocal paragraphs
        text = "\n".join(paragraphs).strip()
        paragraphs = []
        if text:
            for piece in _split_long_text(text, max_chars):
                append_chunk(piece)

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = str(getattr(paragraph.style, "name", "") or "")
        if style_name.lower() == "title" or style_name == "标题":
            flush()
            document_title = text
            continue
        level = detect_heading_level(paragraph)
        if level is not None:
            flush()
            headings[level - 1] = text
            for index in range(level, 4):
                headings[index] = ""
            continue
        if paragraphs and sum(len(item) for item in paragraphs) + len(text) > max_chars:
            flush()
        paragraphs.append(text)

    flush()
    if not chunks:
        raise ValueError("DOCX中没有可切分的正文")
    return chunks
