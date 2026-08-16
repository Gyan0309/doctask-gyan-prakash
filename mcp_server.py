"""The MCP surface.

Behavior 4 asks that another program can drive the whole flow — the human gate
included — without anyone clicking through a UI. MCP is the shape SuperDocs uses
itself, so this is that surface.

It is a **thin adapter over `services.service`, holding no logic of its own** — the
same relationship `api/routes.py` has to the same module. That is the point rather
than an implementation detail: REST and MCP cannot drift apart because there is
nothing to drift *from*. A tool here that behaved differently from its REST equivalent
would be a bug in this file, not a difference of opinion between two systems.

Named `mcp_server.py` rather than living in an `mcp/` package: a top-level `mcp`
directory shadows the installed SDK of the same name, and the resulting ImportError
points at the SDK rather than at the shadowing.

Run it:

    python mcp_server.py                 # stdio, the usual shape for a local client
    MCP_TRANSPORT=streamable-http python mcp_server.py

Every tool returns plain JSON-serializable data. Errors are raised as ToolError so a
calling model sees the reason rather than a stack trace — and, critically, so a failure
never reads as an empty success.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from database.config import get_settings
from services import service
from utils.logging_config import configure_logging, get_logger, log

_settings = get_settings()
configure_logging(_settings.log_level, _settings.log_format)
logger = get_logger(__name__)

server = MCPServer(
    name="ledger",
    version="0.1.0",
    instructions=(
        "Ledger owns a vendor contract corpus: it extracts cited facts from MSAs, "
        "amendments, SOWs and invoices, detects where those documents contradict each "
        "other, and maintains a Vendor Obligation & Exposure Register.\n\n"
        "The flow is: start_run (or add_documents) -> the run pauses at a human gate "
        "with findings -> submit_decisions approves or rejects each finding "
        "individually -> the register commits.\n\n"
        "A run parked at the gate is waiting for a decision and will wait indefinitely. "
        "Read its findings with get_run, then call submit_decisions with a verdict for "
        "every finding index. Approving everything without reading the findings defeats "
        "the purpose of the gate; each finding carries the explanation and the severity "
        "needed to judge it, and get_provenance shows the exact source passage behind "
        "any row of the register."
    ),
)


def _fail(message: str, *, invalid_params: bool = False) -> ToolError:
    """Surface a failure as a failed tool call rather than as a value.

    `ToolError`, not `MCPError`: the latter is a *protocol* error and tears down the
    session, so one bad argument would kill the connection instead of answering the
    call. This comes back as a result flagged `isError`, which is what a caller can
    actually recover from.

    Returning `{"error": ...}` instead would be worse than either. The caller here is a
    model, and a dict is indistinguishable from a success with unusual data — floor 5
    applies to this surface too: a success must mean the thing succeeded.

    `invalid_params` is kept in the signature because the distinction between "you
    asked wrongly" and "we failed" is worth preserving in the message even though this
    transport carries one error shape.
    """
    prefix = "invalid request: " if invalid_params else ""
    return ToolError(f"{prefix}{message}")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@server.tool()
def list_runs(
    limit: Annotated[int, Field(description="How many runs to return.", ge=1, le=200)] = 25,
) -> list[dict[str, Any]]:
    """Recent runs, newest first.

    Statuses: `running`, `awaiting_review` (parked at the human gate), `completed`,
    `failed`, `interrupted` (its process died; resumable), `escalated`.
    """
    return service.list_runs(limit=limit)


@server.tool()
def get_run(
    run_id: Annotated[str, Field(description="The run's UUID.")],
) -> dict[str, Any]:
    """A run's status, per-stage cost, and — when it is parked at the gate — the
    findings awaiting a verdict.

    `pending_findings` carries each finding's `index`, `severity` and `explanation`.
    Those indexes are what `submit_decisions` expects.
    """
    try:
        return service.get_run(run_id)
    except KeyError:
        raise _fail(f"no run {run_id}", invalid_params=True) from None


@server.tool()
def get_deliverable(
    run_id: Annotated[str, Field(description="The run's UUID.")],
) -> dict[str, Any]:
    """The Vendor Obligation & Exposure Register this run produced.

    Each section carries a `content_hash` and a `carried_forward` flag. Carried-forward
    sections are byte-identical copies of the previous run's — that is how "nothing
    else changed" is proven rather than asserted.
    """
    try:
        return service.get_deliverable(run_id)
    except KeyError:
        raise _fail(f"no run {run_id}", invalid_params=True) from None


@server.tool()
def get_provenance(
    run_id: Annotated[str, Field(description="The run's UUID.")],
    section_key: Annotated[str, Field(description="Section key, e.g. 'Acme Corp::hourly_rate'.")],
) -> dict[str, Any]:
    """Where one row of the register came from: its claim, and every citation behind
    it with the exact source passage.

    Use this before approving a finding that turns on a specific value. A claim whose
    citations no longer resolve is reported `unsupported` rather than dropped.
    """
    try:
        return service.get_provenance(run_id, section_key)
    except KeyError:
        raise _fail(f"no run {run_id} or section {section_key}", invalid_params=True) from None


@server.tool()
def get_changes(
    run_id: Annotated[str, Field(description="The run's UUID.")],
) -> dict[str, Any]:
    """What this run changed against its parent, and because of which document.

    `untouched_fraction` is the incrementality claim, measured: the proportion of the
    register that was carried forward byte-identical rather than re-derived.
    """
    try:
        return service.get_changes(run_id)
    except KeyError:
        raise _fail(f"no run {run_id}", invalid_params=True) from None


@server.tool()
def get_decisions(
    run_id: Annotated[str, Field(description="The run's UUID.")],
) -> list[dict[str, Any]]:
    """Every verdict recorded for this run, approved and rejected alike.

    Rejections are kept, never deleted — a gate whose rejections vanish cannot answer
    "why is this not in the register?" six weeks later.
    """
    return service.get_decisions(run_id)


# ---------------------------------------------------------------------------
# Driving
# ---------------------------------------------------------------------------


@server.tool()
def start_run(
    corpus_name: Annotated[
        str,
        Field(
            description="Logical corpus name; reuse it across runs to get incremental behaviour."
        ),
    ],
    document_paths: Annotated[
        list[str], Field(description="Paths visible to the server process.")
    ],
    thread_id: Annotated[
        str | None,
        Field(
            description=(
                "Isolation key. Defaults to the run id; set it only to deliberately "
                "share checkpoint state."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Start a run over a set of documents and drive it until it completes or parks at
    the human gate.

    Reusing `corpus_name` is what makes a run incremental: only sections whose
    underlying facts changed are re-derived, and the rest are carried forward.

    This blocks for the duration of the run — tens of seconds to a few minutes on a
    cold corpus. That is still processing, not a hang.
    """
    if not document_paths:
        raise _fail(
            "document_paths is empty; a run needs at least one document",
            invalid_params=True,
        )

    try:
        return service.start_run(
            corpus_name=corpus_name,
            document_paths=document_paths,
            thread_id=thread_id,
        )
    except Exception as exc:
        log(logger, logging.ERROR, "mcp start_run failed", error=type(exc).__name__)
        raise _fail(f"{type(exc).__name__}: {exc}") from None


@server.tool()
def submit_decisions(
    run_id: Annotated[str, Field(description="The run's UUID.")],
    decisions: Annotated[
        dict[str, str],
        Field(
            description=(
                "Finding index (as a string) -> 'approved' or 'rejected'. "
                "One entry per pending finding."
            )
        ),
    ],
    actor: Annotated[
        str, Field(description="Who is deciding. Recorded on every verdict for audit.")
    ] = "mcp",
) -> dict[str, Any]:
    """**The human gate.** Approve or reject each finding individually, in one pass.

    Per item, not per batch: rejecting one finding leaves the others exactly as they
    were. Every verdict is recorded against `actor`, so a decision made by a program is
    distinguishable afterwards from one made by a person — which matters, because the
    gate exists to put a responsible party behind the commit.

    Call `get_run` first to read what you are deciding on.
    """
    valid = {"approved", "rejected"}
    bad = {k: v for k, v in decisions.items() if v not in valid}
    if bad:
        raise _fail(
            f"verdicts must be 'approved' or 'rejected'; got {bad}",
            invalid_params=True,
        )

    try:
        return service.resume_run(run_id=run_id, decisions=decisions, actor=actor)
    except Exception as exc:
        log(logger, logging.ERROR, "mcp submit_decisions failed", error=type(exc).__name__)
        raise _fail(f"{type(exc).__name__}: {exc}") from None


@server.tool()
def resume_interrupted_run(
    run_id: Annotated[str, Field(description="The run's UUID.")],
) -> dict[str, Any]:
    """Continue a run whose process was killed, from its last checkpoint.

    Only a run reporting `interrupted` can be resumed. Completed stages are not
    re-computed. If the resume fails — usually a source document that has moved — the
    run stays `interrupted` and remains resumable.
    """
    try:
        return service.resume_interrupted(run_id)
    except KeyError:
        raise _fail(f"no run {run_id}", invalid_params=True) from None
    except ValueError as exc:
        raise _fail(str(exc), invalid_params=True) from None
    except service.ResumeFailed as exc:
        raise _fail(f"resume could not complete: {exc}. The run is still interrupted.") from None


def main() -> None:
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    log(logger, logging.INFO, "ledger mcp server starting", transport=transport)
    server.run(transport=transport)


if __name__ == "__main__":
    main()
