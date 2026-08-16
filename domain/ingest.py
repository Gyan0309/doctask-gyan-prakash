"""Reading source documents into chunks that carry real character offsets.

Offsets are not decoration. A citation that names a document but cannot point at the
span inside it proves nothing — you cannot check it without re-reading the whole file,
which is exactly the work the citation was supposed to save.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from utils.hashing import bytes_hash

# Declared, supported formats. A capability may be honestly absent from this list;
# it may never be present and broken.
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass(frozen=True)
class RawChunk:
    ordinal: int
    text: str
    char_start: int
    char_end: int
    page: int | None = None


@dataclass(frozen=True)
class LoadedDocument:
    uri: str
    sha256: str
    mime: str
    text: str
    chunks: list[RawChunk]


class UnsupportedFormat(ValueError):
    pass


def _mime_for(path: Path) -> str:
    return {
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".pdf": "application/pdf",
    }[path.suffix.lower()]


def chunk_text(text: str) -> list[RawChunk]:
    """Split on blank lines, preserving exact offsets into the original string.

    Offsets are computed by scanning the source rather than by re-joining the pieces.
    Reconstructing them from lengths drifts the moment separators are not uniform,
    and the drift is silent — citations land a few characters off and nobody notices
    until someone clicks one.
    """
    chunks: list[RawChunk] = []
    ordinal = 0
    for match in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text):
        body = match.group(0).strip()
        if not body:
            continue
        # Re-locate the stripped body inside the match so offsets stay exact.
        offset = match.start() + match.group(0).index(body[:20] if len(body) >= 20 else body)
        chunks.append(
            RawChunk(
                ordinal=ordinal,
                text=body,
                char_start=offset,
                char_end=offset + len(body),
            )
        )
        ordinal += 1
    return chunks


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFormat(
            f"{path.name}: {suffix or 'no extension'} is not a supported format. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )

    data = path.read_bytes()

    if suffix == ".pdf":
        text = _extract_pdf_text(path)
    else:
        text = data.decode("utf-8", errors="replace")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"{path.name}: no readable text found.")

    return LoadedDocument(
        uri=str(path),
        sha256=bytes_hash(data),  # identity is content, not filename
        mime=_mime_for(path),
        text=text,
        chunks=chunks,
    )


def _extract_pdf_text(path: Path) -> str:
    """PDF text extraction.

    Isolated behind its own function so the dependency stays optional and the failure
    is explicit: a PDF we cannot read must say so, not yield an empty string that
    silently produces a document with no facts and no complaint.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise UnsupportedFormat(
            "PDF support requires pypdf. It is in requirements.txt; the environment "
            "running this does not have it installed."
        ) from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(p for p in pages if p)

    if not text.strip():
        raise ValueError(
            f"{path.name}: pypdf extracted no text. This is almost always a scanned "
            f"PDF with no text layer — OCR is out of scope, so the document is "
            f"skipped and reported rather than silently contributing nothing."
        )
    return text
