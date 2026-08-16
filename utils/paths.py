"""Path helpers that survive crossing an operating system boundary.

`pathlib.Path(...).name` is resolved against the *running* platform's rules, not the
platform the string came from. That is normally invisible and here it is not: documents
are commonly ingested by a tool running on Windows (`C:\\work\\corpus\\msa.md`) while the
API serving them runs in a Linux container. `PurePosixPath` sees no separator in that
string at all, so the "filename" it reports is the entire path.

The symptom is cosmetic until it is not — a provenance panel that answers "which
document did this come from?" with an absolute path containing someone's home directory
is both unreadable and a small information leak.
"""

from __future__ import annotations


def basename(uri: str | None) -> str:
    """The final component of a path, whichever separator wrote it."""
    if not uri:
        return ""
    return uri.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or uri
