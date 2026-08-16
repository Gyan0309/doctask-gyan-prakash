"""Generate the PDF document in the seed corpus.

The corpus is committed as files, but a committed binary nobody can regenerate is
opaque — a reviewer cannot see what is in it, and a change means hand-editing a blob.
So the source text lives here in plain sight and the PDF is built from it.

Deliberately dependency-free: this writes the PDF structure directly rather than
pulling in reportlab for one file. The result is a genuine PDF with a real text layer,
which is what matters — a scanned image would exercise nothing, since OCR is out of
scope and honestly declared as such.

Run:  python corpus/build_pdf.py
"""

from __future__ import annotations

import zlib
from pathlib import Path

PAGE_WIDTH, PAGE_HEIGHT = 612, 792  # US Letter, in points
MARGIN_LEFT, MARGIN_TOP = 62, 730
LEADING = 15.5  # line spacing
MAX_LINES_PER_PAGE = 42

COBALT_MSA = """MASTER SERVICES AGREEMENT

Between: Meridian Retail Group ("Client")
And: Cobalt Freight Systems Inc. ("Supplier")
Effective Date: 2024-07-15
Agreement Reference: CF-MSA-2024-0715

1. SERVICES

Supplier shall provide freight forwarding, logistics coordination and
customs brokerage services to the Client.

2. FEES

The hourly rate is $95 per hour for standard logistics coordination
services. Estimated annual fees under this Agreement are $310,000.

3. INVOICING AND PAYMENT

Supplier shall invoice monthly. Payment terms are net 60 days from the
date of invoice.

4. TERM AND RENEWAL

The initial term is twenty-four (24) months from the Effective Date.
Thereafter the Agreement renews automatically for successive periods of
36 months unless written notice of non-renewal is given.

5. TERMINATION

Either party may terminate for convenience on 15 days written notice.

6. LIMITATION OF LIABILITY

The liability cap is $5,000,000 in aggregate for all claims arising under
this Agreement.

7. SERVICE LEVELS

Supplier shall deliver 98% of consignments within the agreed transit
window. Where this is not met, Client is entitled to an SLA credit of 2%
of the monthly fee.

8. NOTICES

Notices under this Agreement shall be given in writing to the addresses
set out in Schedule 1.
"""


def _escape(text: str) -> str:
    """PDF string literals escape backslash and both parentheses."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str]) -> bytes:
    parts = ["BT", "/F1 11 Tf", f"{LEADING} TL", f"1 0 0 1 {MARGIN_LEFT} {MARGIN_TOP} Tm"]
    for line in lines:
        # Tj draws, T* advances — so an empty line still moves the cursor down.
        parts.append(f"({_escape(line)}) Tj" if line else "()Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", errors="replace")


def build_pdf(text: str, destination: Path) -> None:
    raw_lines = text.split("\n")
    pages = [
        raw_lines[i : i + MAX_LINES_PER_PAGE]
        for i in range(0, len(raw_lines), MAX_LINES_PER_PAGE)
    ]

    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # object numbers are 1-based

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # The page tree must be referenced by its children, and vice versa, so its id is
    # reserved before the pages exist.
    pages_id = len(objects) + 1 + 2 * len(pages)

    page_ids: list[int] = []
    for page_lines in pages:
        stream = zlib.compress(_content_stream(page_lines))
        content_id = add(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" /Filter /FlateDecode >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        page_ids.append(
            add(
                f"<< /Type /Page /Parent {pages_id} 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>".encode()
            )
        )

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    actual_pages_id = add(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    )
    assert actual_pages_id == pages_id, "page tree id reservation drifted"

    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()

    destination.write_bytes(bytes(out))


if __name__ == "__main__":
    target = Path(__file__).parent / "seed" / "cobalt-msa.pdf"
    build_pdf(COBALT_MSA, target)
    print(f"wrote {target} ({target.stat().st_size} bytes)")
