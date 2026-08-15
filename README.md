# Ledger

An agentic system that owns a vendor contract file end to end: it reads a pile of
MSAs, amendments, SOWs and invoices, extracts facts that carry citations back to their
source, finds the places where those documents contradict each other, and maintains a
**Vendor Obligation & Exposure Register** that stays correct as new paperwork arrives.

The interesting claim is not that it produces the register. It is that when the
eleventh document lands, the system re-derives only the rows that document affected
and can **prove** the others are byte-identical.

---

## Run it

```bash
docker compose up
```

That is the whole thing. It starts Postgres with pgvector, applies migrations, and
serves the API on <http://localhost:8000>.

**No API key is required to start.** A fresh clone with no `.env` comes up green and
runs the entire test suite. `GET /health` reports which dependencies are configured
and which are not, rather than dying or pretending:

```bash
curl localhost:8000/health
```

To use real models instead of the deterministic offline provider:

```bash
cp .env.example .env    # then paste your Gemini key in
docker compose up
```

### Ports

The database is published on **55432**, not 5432. If you already run Postgres locally
— most reviewers do — the default port is occupied, and the resulting failure is a
liar: Docker still reports the container healthy while your connections quietly reach
the *other* Postgres and fail authentication against credentials that are correct.
Override with `DB_HOST_PORT` if 55432 is also taken.

---

## Tests

```bash
pytest
```

**Every test runs with no API key, no network, and no recorded fixtures.** CI holds no
secrets at all — there is no `secrets.` reference in the workflow and no repository
secret for it to read. A fork gets the same green run we do.

That is a deliberate design constraint rather than a convenience. Tests that assert on
replayed model output mostly prove the recording still parses. These assert on what
the *system* did: which stages executed, which sections changed hash, whether approved
work survived a kill, whether two concurrent runs interleaved.

Tests needing a database are marked `integration` and skip with a usable message when
none is running:

```bash
docker compose up -d db && pytest
```

---

## How it works

Documents arrive → get classified → facts are extracted with character-level
citations → facts are reconciled against what is already known → contradictions are
detected deterministically and adjudicated by a model → the register is composed from
**only the sections whose underlying facts changed** → rules are applied → a verifier
that is not the composer checks every claim against its sources → a human approves or
rejects each finding individually → the result is committed with content hashes.

Five points in that path genuinely change the route taken, rather than logging a label
on a fixed script:

| Decision | Alternate path |
|---|---|
| Classification confidence below threshold | Escalate to a human instead of guessing |
| Extraction fails its schema | Retry twice with a repair prompt, then skip the document and emit a finding |
| No conflict candidates found | Skip adjudication entirely — the honest clean path, and it costs nothing |
| Verification fails | Loop back to compose twice, then escalate |
| Human rejects a finding | Route back to adjudication with the feedback |

### The load-bearing decision

The deliverable is stored **per section**, each with a content hash and a recorded
list of the facts it was derived from. A new document invalidates exactly the sections
depending on facts that changed; everything else is carried forward by reference.

So "nothing else changed" is a hash comparison, not a promise. And "what changed, when,
and because of which source" is a query, not a narrative.

This also means **a full run is the degenerate case of an incremental run** — the case
where the dependency map is empty and every section is therefore invalid. One code
path, not two, so the incremental logic is exercised by every run ever made and cannot
rot unnoticed.

---

## Where the intelligence sits, and where it deliberately does not

Conflict detection is split:

1. **Candidate generation is deterministic** — three general comparators over the
   normalized fact graph (same predicate with different values, temporal precedence
   violations, arithmetic mismatches). No model. Cheap, exhaustive, testable with no key.
2. **Adjudication is the model's job** — is this a real contradiction or two different
   things wearing similar names, how severe is it, and what explanation should a human
   read.

The model only ever sees *candidates*, never the cross product. That is what makes it
affordable, and it keeps hardcoded special cases from wearing intelligence as a costume.

---

## Project layout

```
src/ledger/
  config.py         all environment-driven configuration, model IDs included
  models.py         the schema; three tables carry the invariants
  db.py             engine and transactional session scope
  providers/        model provider interface + Gemini + a deterministic offline stub
  api/              FastAPI surface
migrations/         Alembic; the initial migration creates the vector extension itself
rules/playbook.yaml the contract rules — adding a rule is a data change
tests/              the no-key suite
```

---

## Honest limitations

- **Two ingestion formats** (`.md`, `.pdf`), not five. Declared rather than discovered:
  a capability may be honestly absent, never present and broken.
- **pgvector is insurance, not the retrieval backbone.** At this corpus size, structured
  fact lookup beats embedding search and is deterministic enough to test properly. The
  vector index powers one narrow fallback — "which sections might this new fact affect"
  when exact predicate matching finds nothing. Claiming it as the retrieval story would
  be theater.
- **Checkpointing is node-level.** Kill a node eighteen model calls deep and the graph
  resumes at the node boundary; no approved work is lost, but those calls are re-paid.
  Mitigated by keeping nodes small and putting a content-addressed cache in front of
  every model call — which is also how the "an update costs like an update" claim is
  measured rather than asserted.

---

## A constraint worth knowing before you run this

**The Gemini free tier allows 20 requests per day, per model** — not per minute.
(Quota ID `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, measured 2026-08-15.)
One run over the seven-document corpus makes about 33 calls, so a single model's daily
budget does not cover a single full run.

Three things follow, and they shaped the design rather than being bolted onto it:

- The default model is a **lite** model, not the largest one available.
- `GEMINI_MODEL_FALLBACKS` lists further models tried on exhaustion. Because the cap is
  per model, each is a separate budget. Substitutions are logged at WARNING — silently
  answering with a model you did not configure would be its own kind of dishonesty.
- Per-day and per-minute 429s are told apart by parsing `quotaId` out of the structured
  error, because their human-readable messages are identical and they need opposite
  responses: wait a moment, versus abandon this model until tomorrow.

None of this affects the tests, which use the offline provider and never touch the
network.

---

## Status

**Phase 3 complete.** Documents are classified (with a real escalation branch for low
confidence), facts extracted with verified citations, values normalized, competing
values reconciled to the one that governs, contradictions detected deterministically
and adjudicated by a model, and the register composed from only the sections whose
dependencies changed.

### What it finds

Run against an eight-document corpus spanning two vendors, an amendment chain and two
invoices, it reports three conflicts and nothing else:

| Severity | Finding |
|---|---|
| high | Invoice bills $180/hr when Amendment No. 1 set $195 effective 2025-07-01 |
| medium | Invoice states net 45 when the amendment set net 30 |
| high | Invoice total $20,500 ≠ 100 hrs × $195 = $19,500 — a $1,000 overcharge |

Each is arithmetic, not opinion, and each cites the documents it came from.

### What it costs

| | Run 1 | Run 2, identical input |
|---|---|---|
| Wall time | 23s | **0s** |
| Model calls | 33 | **0** |
| Sections re-derived | 18 | **0** |
| Carried forward | 0 | **18** |
| Content hashes | — | **byte-identical** |

And on a corpus with nothing wrong with it, the adjudication stage does not run at all
— recorded as `skipped`, costing zero model calls. A stage that legitimately did not
run and a stage that was never wired up must not look alike, so the skip is written
down rather than omitted.

Not yet built: the rules engine, the fresh-eyes verifier, and the folder watcher. Those
stages are absent rather than stubbed — a capability may be honestly missing, never
present and broken. This README gains sections as the stages that back them land.
