"""The register as a document.

Everything else in this system treats the register as rows. This module is the one
place that turns it into prose a person would read — which is what makes SuperDocs the
right editing surface for it rather than a decoration bolted on the side.

Rendering is deliberately deterministic and total: same sections in, same bytes out,
every time. Publishing compares a freshly rendered section against what SuperDocs
currently holds, and a renderer that varied between calls would report edits nobody
made.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def section_ref(section_key: str) -> str:
    """A short, stable, unique anchor for a section.

    A targeted edit needs somewhere unambiguous to land, and a human-readable heading
    is not it. `Vendor — hourly_rate` collides the moment the same term exists at two
    scopes: the register carries both an agreement-level rate and a SOW-scoped one, so
    19 sections rendered as 18 distinct headings, and an instruction to "replace the
    section for this vendor and term" had two candidates and picked wrong.

    That was not a prompt problem. **The document had no unique anchor**, so no
    instruction phrased against it could have been reliable. Derived from the section
    key, which the schema already guarantees is unique per run, and stable across runs
    so the anchor of an unchanged section never moves.
    """
    return hashlib.sha256(section_key.encode("utf-8")).hexdigest()[:8]


def _row(section: dict[str, Any]) -> dict[str, Any]:
    """The section's content, parsed. Sections store canonical JSON."""
    try:
        return json.loads(section["content"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def render_section(section: dict[str, Any]) -> str:
    """One obligation as a short prose paragraph.

    Prose rather than a table row, and that is the point of this module. A table cell
    is not something a targeted document edit can meaningfully improve; a sentence
    naming a value, its source and its effective date is. It also gives an edit
    somewhere unambiguous to land — each obligation is its own heading, so a change to
    one cannot silently reflow another.
    """
    row = _row(section)
    key = section.get("section_key", "")
    vendor = row.get("vendor") or "Unknown vendor"
    term = row.get("term") or key or "unknown term"
    value = row.get("value")
    effective = row.get("effective_date")
    source = row.get("governing_source")
    status = row.get("status") or "unknown"
    superseded = row.get("superseded") or []

    # The ref is what makes this section addressable. It sits in the heading so it
    # survives the markdown → HTML → markdown round trip as ordinary text.
    lines = [f"### {vendor} — {term} [ref:{section_ref(key)}]", ""]

    if value is None:
        # An absent value is stated, never quietly omitted. A register that drops the
        # terms it could not establish reads as complete and is not.
        lines.append(
            f"No supported value for **{term}** was established for {vendor}. "
            "This is recorded rather than omitted."
        )
    else:
        sentence = f"The governing **{term}** for {vendor} is **{value}**"
        if effective:
            sentence += f", effective {effective}"
        if source:
            sentence += f", per the {source}"
        lines.append(sentence + ".")

        if superseded:
            lines.append("")
            lines.append(
                f"This supersedes {', '.join(str(s) for s in superseded)}. "
                "The superseded values are retained as the evidence for why this one governs."
            )

    lines.append("")
    lines.append(f"*Status: {status}.*")
    return "\n".join(lines)


def render_register(
    deliverable: dict[str, Any], *, corpus_name: str, run_id: str
) -> str:
    """The whole register as one markdown document."""
    sections = deliverable.get("sections") or []

    header = [
        "# Vendor Obligation & Exposure Register",
        "",
        f"Corpus **{corpus_name}** · run `{run_id[:8]}` · {len(sections)} obligations.",
        "",
        (
            "Every value below was extracted from a source document and carries a "
            "citation back to the passage it came from. Values that could not be "
            "supported are stated as unsupported rather than omitted."
        ),
        "",
        "---",
        "",
    ]

    body: list[str] = []
    for section in sections:
        body.append(render_section(section))
        body.append("")

    return "\n".join(header + body).rstrip() + "\n"
