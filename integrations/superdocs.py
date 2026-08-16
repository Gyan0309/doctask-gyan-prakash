"""SuperDocs client — upload, chat, approve, export.

This is the editing surface for the register (decision D5). The canonical register
stays ours: sectioned rows and content hashes in Postgres. SuperDocs renders it as a
document and applies targeted edits to it, which is the thing SuperDocs is actually for
— and it means an unreachable API degrades to a locally rendered document rather than
taking a movement down with it.

Nearly every non-obvious line here exists because of something that bit during
integration. Each is commented where it bites; the reproductions are in the Task 2
`BUGS.md`. The short version of what the task brief's cheat-sheet gets wrong:

  * sync `/v1/chat` never returns a `job_id`, and `/approve` requires one — so with
    `ask_every_time` the sync path proposes changes it gives you no way to accept.
    Async is the only path that closes the review loop.
  * export is `POST /v1/documents/export`, not `/v1/export`; it takes a `session_id`,
    not a document id; markdown is spelled `markdown`; and it returns **file bytes,
    not JSON**.
  * pending changes live in `metadata.pending_changes`, while `result` stays null for
    exactly as long as a caller needs them.
  * the working-status vocabulary includes `in_progress`, so a poller written against
    a guessed working set exits immediately and reports nothing to do.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from utils.logging_config import get_logger, log

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.superdocs.app/v1"

REQUEST_TIMEOUT_SECONDS = 120.0
POLL_INTERVAL_SECONDS = 3.0
POLL_CEILING_SECONDS = 600.0

# Expressed as the set of *terminal* states rather than the working ones. The working
# vocabulary includes `in_progress`; a poller listing only pending/processing/running
# treats it as finished, reads a null result, and reports no changes at all — silently,
# and fast, which is the worst combination because it looks like a successful no-op.
TERMINAL_STATUSES = {
    "completed",
    "complete",
    "succeeded",
    "success",
    "failed",
    "error",
    "cancelled",
    "canceled",
    "awaiting_approval",
}

# What counts as "this session is still busy". `awaiting_approval` is deliberately in
# here even though it is also terminal for a *job*: a job parked awaiting approval is
# still the session's active job, and the 409 body names it as one of the states that
# blocks the next instruction. Omitting it made wait_until_idle return immediately in
# the exact window it exists to cover — right after approve, before the server has
# moved the job on — so the guard was only accidentally correct.
ACTIVE_STATUSES = {
    "pending",
    "in_progress",
    "processing",
    "running",
    "queued",
    "awaiting_approval",
}

EXPORT_FORMATS = {"docx", "pdf", "html", "markdown", "txt"}


class SuperDocsError(RuntimeError):
    """A call failed in a way the caller cannot recover from."""


class SuperDocsUnavailable(SuperDocsError):
    """No key, or the service could not be reached.

    Separate from `SuperDocsError` because the caller's response differs: this one
    means "fall back to local rendering and say so", not "something went wrong".
    """


@dataclass
class UploadedDocument:
    session_id: str
    document_id: str
    durable_document_id: str | None
    filename: str
    chunks_count: int = 0
    version_id: str | None = None


@dataclass
class PendingChange:
    change_id: str
    operation: str
    chunk_id: str | None
    old_html: str
    new_html: str
    explanation: str

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> PendingChange:
        return cls(
            change_id=raw["change_id"],
            operation=raw.get("operation", "edit"),
            chunk_id=raw.get("chunk_id"),
            old_html=raw.get("old_html") or "",
            new_html=raw.get("new_html") or "",
            explanation=raw.get("ai_explanation") or "",
        )


@dataclass
class ChatResult:
    session_id: str
    job_id: str
    status: str
    pending_changes: list[PendingChange] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def awaiting_approval(self) -> bool:
        return self.status == "awaiting_approval"


def parse_embedded_json(raw: Any) -> Any:
    """Second-parse a field that arrives as a JSON-encoded string.

    Worth being precise about *where* this applies, because it is not where the brief
    points: `metadata.pending_changes[]` are already objects and need no second parse,
    while `intermediate_responses[]` entries of type `proposed_change_batch` carry
    `content` as a JSON string. "Parse everything twice" and "parse nothing twice" are
    both wrong.
    """
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


class SuperDocsClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if not api_key:
            raise SuperDocsUnavailable(
                "SUPERDOCS_API_KEY is not set — the register can still be rendered "
                "locally, but it cannot be published."
            )
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self.calls_made = 0
        # One client, not one per request. Publishing a 19-section register is 60+
        # requests once job polling and idle waits are counted, and a fresh client per
        # call pays a TLS handshake for every one of them.
        self._http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> SuperDocsClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _redact(self, text: str) -> str:
        """Keys must not reach a log, an exception message, or a screenshot."""
        return text.replace(self._key, "<redacted>") if self._key else text

    def _request(
        self, method: str, path: str, *, json_body: dict | None = None
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            response = self._http.request(
                method,
                f"{self._base}{path}",
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
                json=json_body,
            )
        except httpx.RequestError as exc:
            raise SuperDocsUnavailable(
                f"{method} {path} unreachable: {type(exc).__name__}"
            ) from exc

        self.calls_made += 1
        log(
            logger,
            logging.INFO,
            "superdocs call",
            method=method,
            path=path,
            status=response.status_code,
            elapsed_ms=round((time.monotonic() - started) * 1000),
        )

        if response.status_code >= 400:
            raise SuperDocsError(
                f"{method} {path} -> HTTP {response.status_code}: "
                f"{self._redact(response.text[:500])}"
            )

        return response.json() if response.content else {}

    # -- upload --------------------------------------------------------------

    def upload_markdown(
        self, *, session_id: str, filename: str, body: str
    ) -> UploadedDocument:
        """Upload a markdown document into a session, durably.

        The field is `file_base64`, not `content_base64` — the natural guess 422s.

        **Without a `session_id` the upload is a one-off conversion and is not saved:**
        you get parsed HTML back and nothing persists. The response says so in a
        `how_to_persist` field naming the exact request that would change it, which is
        the best piece of self-documentation on this API. This method always sends one,
        and checks `persisted` rather than assuming it.
        """
        response = self._request(
            "POST",
            "/documents/upload-base64",
            json_body={
                "filename": filename,
                "file_base64": base64.b64encode(body.encode("utf-8")).decode("ascii"),
                "session_id": session_id,
            },
        )

        if not response.get("persisted"):
            raise SuperDocsError(
                f"{filename} was not persisted despite a session_id. "
                f"Server said: {response.get('how_to_persist')!r}"
            )

        # The upload response does not carry the id chat needs — that comes from the
        # session's document list, where ids are session-scoped ("doc_primary") and
        # distinct from the durable UUID.
        listed = self.list_documents(session_id)
        if not listed:
            raise SuperDocsError(f"{filename} uploaded but is not listed in the session")

        # Matched by name, not by position. `listed[-1]` assumes the endpoint returns
        # documents in creation order — undocumented, and if it ever sorted by title or
        # id instead, the returned document_id would belong to a *different* document
        # and every subsequent targeted edit would land in it.
        stem = filename.rsplit(".", 1)[0]
        match = next(
            (d for d in listed if d.filename == filename),
            next((d for d in listed if d.filename.startswith(stem)), None),
        )
        if match is None:
            raise SuperDocsError(
                f"uploaded {filename} but no document in the session matches that name; "
                f"session holds: {[d.filename for d in listed]}"
            )

        match.version_id = response.get("version_id")
        match.chunks_count = response.get("chunks_count", match.chunks_count)
        return match

    def list_documents(self, session_id: str) -> list[UploadedDocument]:
        response = self._request("GET", f"/sessions/{session_id}/documents")
        return [
            UploadedDocument(
                session_id=session_id,
                document_id=d["document_id"],
                durable_document_id=d.get("durable_document_id"),
                filename=d.get("title") or d["document_id"],
                chunks_count=d.get("chunks_count", 0),
            )
            for d in response.get("documents", [])
        ]

    # -- chat ----------------------------------------------------------------

    def propose(
        self, *, session_id: str, message: str, document_id: str | None = None
    ) -> ChatResult:
        """Send one edit instruction and wait for the job to settle.

        Async always. Not merely because long edits exceed the sync gateway timeout —
        sync `/v1/chat` returns no `job_id`, and `/approve` requires one, so under
        `ask_every_time` the sync path produces changes that cannot be approved at all.

        `document_html` is deliberately never sent: the session already holds the
        document, and re-sending it each turn risks clobbering edits already applied.
        """
        body: dict[str, Any] = {
            "session_id": session_id,
            "message": message,
            "approval_mode": "ask_every_time",
        }
        if document_id:
            body["document_id"] = document_id

        started = self._request("POST", "/chat/async", json_body=body)
        job_id = started.get("job_id")
        if not job_id:
            raise SuperDocsError(f"/chat/async returned no job_id: {sorted(started)}")

        return self._wait_for_job(job_id, session_id=session_id)

    def _wait_for_job(self, job_id: str, *, session_id: str) -> ChatResult:
        deadline = time.monotonic() + POLL_CEILING_SECONDS

        while True:
            job = self._request("GET", f"/jobs/{job_id}")
            status = (job.get("status") or "").lower()

            if status in TERMINAL_STATUSES or job.get("error"):
                break
            if time.monotonic() > deadline:
                raise SuperDocsError(
                    f"job {job_id} still {status!r} after {POLL_CEILING_SECONDS:.0f}s"
                )
            # A quiet minute is normal here, not a hang.
            time.sleep(POLL_INTERVAL_SECONDS)

        if job.get("error"):
            raise SuperDocsError(f"job {job_id} failed: {job['error']}")

        # `result` is null for the whole time a job is awaiting approval — which is
        # exactly when a caller needs the changes in order to decide on them.
        metadata = job.get("metadata") or {}
        return ChatResult(
            session_id=session_id,
            job_id=job_id,
            status=(job.get("status") or "").lower(),
            pending_changes=[
                PendingChange.from_api(c) for c in metadata.get("pending_changes") or []
            ],
            raw=job,
        )

    # -- approve -------------------------------------------------------------

    def decide(
        self,
        session_id: str,
        *,
        job_id: str,
        change_id: str | None = None,
        approved: bool = True,
        feedback: str | None = None,
    ) -> dict[str, Any]:
        """Approve or reject one proposed change.

        `job_id` is required and the brief does not mention it; the 422 you get without
        it does at least name both missing fields. The field is `change_id` singular —
        batches go in `changes: [{change_id, approved, feedback}]`, not `change_ids`.
        """
        body: dict[str, Any] = {"job_id": job_id, "approved": approved}
        if change_id:
            body["change_id"] = change_id
        if feedback:
            body["feedback"] = feedback
        return self._request("POST", f"/chat/{session_id}/approve", json_body=body)

    def wait_until_idle(self, session_id: str, *, timeout: float = 300.0) -> None:
        """Block until the session has no active job.

        Approve returns 200 immediately, but the job continues in order to actually
        apply the change, so the next instruction into that session gets a 409
        `session_busy`. That bites the moment you drive a session in a loop, which
        publishing a multi-section register does.

        The 409 body is one of the best error messages on this API — it names the
        condition, lists the states that count as active, and gives three remedies.
        Most of this method is doing what it says.
        """
        deadline = time.monotonic() + timeout
        while True:
            jobs = self._request("GET", f"/sessions/{session_id}/jobs").get("jobs", [])
            if not any((j.get("status") or "").lower() in ACTIVE_STATUSES for j in jobs):
                return
            if time.monotonic() > deadline:
                raise SuperDocsError(
                    f"session {session_id} still has an active job after {timeout:.0f}s"
                )
            time.sleep(POLL_INTERVAL_SECONDS)

    # -- export --------------------------------------------------------------

    def export(self, session_id: str, *, fmt: str = "markdown") -> bytes:
        """Export the session's document, returning file bytes.

        Not `/v1/export` (404), takes a `session_id` not a document id, `markdown` not
        `md` — and the response is the file itself, so a client calling `.json()` on it
        dies on the first byte. Exports do not cost operations.
        """
        if fmt not in EXPORT_FORMATS:
            raise SuperDocsError(
                f"unknown export format {fmt!r}; expected one of {sorted(EXPORT_FORMATS)}"
            )

        try:
            response = self._http.post(
                f"{self._base}/documents/export",
                headers={"Authorization": f"Bearer {self._key}"},
                json={"session_id": session_id, "format": fmt},
            )
        except httpx.RequestError as exc:
            raise SuperDocsUnavailable(f"export unreachable: {type(exc).__name__}") from exc

        self.calls_made += 1
        if response.status_code >= 400:
            raise SuperDocsError(
                f"export -> HTTP {response.status_code}: {self._redact(response.text[:400])}"
            )
        return response.content

    def ops_remaining(self) -> int | None:
        """Promo operations left.

        Read from `/users/me/promotions` because the documented alternative does not
        work: `usage` is null on chat responses, and `/users/me/usage` rejects `sk_`
        keys with a 401. This is the only spend meter that answers.
        """
        for promo in self._request("GET", "/users/me/promotions").get("active", []):
            return promo.get("ops_remaining")
        return None
