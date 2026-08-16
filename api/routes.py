"""REST surface — a thin adapter over `ledger.service`, holding no logic of its own.

Every operation a human can perform through the UI is reachable here, gate included.
That is behavior 4: another program can drive the entire flow end to end without
touching a browser.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import services.service as service

router = APIRouter()


class StartRunRequest(BaseModel):
    corpus_name: str = Field(..., description="Logical name; reused across runs.")
    document_paths: list[str] = Field(..., description="Paths visible to the API process.")
    thread_id: str | None = Field(
        None, description="Isolation key. Defaults to the run id; set it to run concurrently."
    )


class DecisionsRequest(BaseModel):
    decisions: dict[str, str] = Field(
        ...,
        description=(
            "Finding index → 'approved' | 'rejected'. Per item, not per batch: a "
            "reviewer must be able to accept some findings and reject others in one pass."
        ),
    )


@router.post("/runs")
def start_run(request: StartRunRequest) -> dict[str, Any]:
    return service.start_run(
        corpus_name=request.corpus_name,
        document_paths=request.document_paths,
        thread_id=request.thread_id,
    )


@router.get("/runs")
def list_runs(limit: int = 25) -> list[dict[str, Any]]:
    """Recent runs, newest first."""
    return service.list_runs(limit=limit)


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return service.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no run {run_id}") from None


@router.post("/runs/{run_id}/decisions")
def decide(run_id: str, request: DecisionsRequest) -> dict[str, Any]:
    return service.resume_run(run_id=run_id, decisions=request.decisions)


@router.get("/runs/{run_id}/decisions")
def list_decisions(run_id: str) -> list[dict[str, Any]]:
    return service.get_decisions(run_id)


@router.get("/runs/{run_id}/deliverable")
def get_deliverable(run_id: str) -> dict[str, Any]:
    return service.get_deliverable(run_id)


@router.get("/runs/{run_id}/provenance")
def get_provenance(run_id: str, section_key: str) -> dict[str, Any]:
    """Where a register row's value came from: document, passage, character span.

    The endpoint behind "click a value and see the sentence it was read from". A
    register that cannot answer this is asking to be trusted; one that can is asking
    to be checked.
    """
    try:
        return service.get_provenance(run_id, section_key)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"no section {section_key!r} in run {run_id}"
        ) from None


@router.get("/runs/{run_id}/changes")
def get_changes(run_id: str) -> dict[str, Any]:
    """What this run changed, and which arriving document caused each change."""
    try:
        return service.get_changes(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no run {run_id}") from None


@router.get("/runs/{run_id}/cost")
def get_cost(run_id: str) -> dict[str, Any]:
    try:
        run = service.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no run {run_id}") from None

    stages = run["stages"]
    return {
        "run_id": run_id,
        "total_cost_usd": round(sum(s["cost_usd"] for s in stages), 6),
        "total_cache_hits": sum(s["cache_hits"] for s in stages),
        "total_cache_misses": sum(s["cache_misses"] for s in stages),
        "by_stage": stages,
    }
