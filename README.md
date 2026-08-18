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

### Give it something to read

`inbox/` is gitignored, so a fresh clone starts empty. The repository ships eight
synthetic vendor documents — an MSA, two amendments, a SOW, two invoices, a PDF and a
renewal notice — with a deliberate overcharge and two liability breaches in them:

```bash
cp corpus/seed/* inbox/ && curl -X POST localhost:8000/watch/poll
```

Or drag them onto the page, which does the same thing. Either way the run parks at the
human gate with its findings.

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
pytest                      # 414 tests
pytest -m "not integration" # 332 of them need no database either
```

**Every test runs with no API key, no network, and no recorded fixtures.** CI holds no
credentials at all — verify it rather than believe it:

```bash
grep -c 'secrets\.' .github/workflows/ci.yml   # → 0
gh api repos/OWNER/REPO/actions/secrets        # → total_count 0
```

A fork, a clone, or a reviewer with no credentials gets the same green run we do. See
[tests/README.md](tests/README.md) for why there are no cassettes, and for the list of
bugs these tests actually caught.

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

Four points in that path genuinely change the route taken, rather than logging a label
on a fixed script. Three are conditional edges in the graph — visible in
`build_graph()`, observable in the checkpoint, and not an `if` buried inside a node:

| Decision | Alternate path | Where |
|---|---|---|
| Classification confidence below threshold | Escalate to a human instead of guessing | conditional edge |
| No conflict candidates found | Skip adjudication entirely — the honest clean path, and it costs nothing | conditional edge |
| Verification fails | Route to `blocked`: nothing commits, the human gate is **not** offered, and the run records `failed` | conditional edge |
| Extraction fails its schema | Retry twice with a repair prompt, then skip that document and emit a finding | inside the node |

Two paths that would be natural here deliberately **do not** exist, and the reasoning
is part of the design rather than an omission:

- **Verification failure does not loop back to compose.** Re-composing the same facts
  produces the same register, so a retry would burn calls to reach the same verdict.
  A failed verification is a reason to stop, not to try again.
- **Rejecting a finding does not re-run adjudication.** Findings are observations about
  the corpus, not edits waiting to be applied — a rejection is recorded, kept auditable,
  and left out of the committed set. Re-adjudicating would ask the model to overturn a
  human, which is the wrong direction of authority for a review gate.

### A machine can drive all of it — MCP

The whole flow, gate included, is exposed as an MCP server. Eleven tools over
`services/service.py`, which is the same module the REST routes sit on: REST and MCP
cannot drift apart because there is nothing to drift *from*.

```bash
python mcp_server.py                            # stdio
MCP_TRANSPORT=streamable-http python mcp_server.py
```

Point a client at it:

```json
{
  "mcpServers": {
    "ledger": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/doctask-gyan-prakash",
      "env": { "DATABASE_URL": "postgresql+psycopg://ledger:ledger@localhost:55432/ledger" }
    }
  }
}
```

| Tool | |
|---|---|
| `start_run` | run a corpus until it completes or parks at the gate |
| `get_run` | status, per-stage cost, and the findings awaiting a verdict |
| **`submit_decisions`** | **the gate** — approve/reject each finding, recorded against an `actor` |
| `get_deliverable` · `get_provenance` · `get_changes` · `get_decisions` | read the register, its sources, what moved, and every verdict |
| `publish_register` · `render_document` | put the register into SuperDocs as a document, or render it locally for free |
| `resume_interrupted_run` | continue a run whose process died |
| `list_runs` | recent runs |

`submit_decisions` takes an `actor`, so a verdict reached by a program is
distinguishable afterwards from one reached by a person. The gate exists to put a
responsible party behind the commit, and "approved" with no idea who approved it does
not do that.

Failures are `ToolError`, never a value. A tool returning `{"error": ...}` is
indistinguishable from a success with unusual data to the model reading it — floor 5
applies to this surface too.

The tests drive a real `mcp.Client` over the protocol rather than calling the decorated
functions, and one launches the server as a subprocess over stdio. Calling the
functions would prove the service layer works, which other tests already cover, and
would say nothing about whether a client can reach them.

### Built on SuperDocs — the register as an editable document

The register's canonical form is ours: sectioned rows and content hashes in Postgres.
SuperDocs is where it becomes a **document**, and where it is maintained by targeted
edits rather than regeneration.

```bash
curl -X POST localhost:8000/runs/<run_id>/publish
```

The first publish for a corpus uploads the whole document. Every publish after that
**exports the document, diffs it against the register, and sends only the difference** —
one instruction per section that differs, one per section the register has dropped, each
change approved individually. That is the same claim the rest of this system makes,
aimed at someone else's API: an update costs like an update.

The change set comes from the *document*, not from what the run re-derived, and that
distinction does real work. `carried_forward` cannot express "the register used to have
this section and no longer does" — such a section is in neither the run's output nor its
carried-forward set, so nothing would ever mention it and the document would assert a
withdrawn obligation forever. Nor can it express "the document already has this", so a
retry re-paid for every section that had already landed. Sections carry a `rev` marker
(the first eight of their content hash), so both questions are one exact comparison.

If the document cannot be read, publishing **refuses** rather than falling back to run
provenance — a fallback there would silently reintroduce both bugs.

Measured live against `api.superdocs.app`:

| | |
|---|---|
| First publish | 19 sections uploaded, 3 API calls |
| Second publish, after one amendment | **1 section edited, 18 untouched** |
| Republish with nothing changed | **no-op — 2 API calls, 0 operations** |
| Reconciling a 19-section document down to a 4-section register | **19 removed, 4 written**, 4833 → 1109 bytes |
| Verified against the exported document | ✅ |

That last row also produced the most useful failure in the project. Into a session that
had just processed nineteen deletions, an explicit *"reproduce it verbatim"* instruction
came back with the values replaced by **template placeholders** — `Please fill: Annual
Fee Amount` — and one real value written into the neighbouring section. Every status
code was 200 and every count was right. The verification caught it, reported
`published: false`, and named the section. Written up for SuperDocs as finding #11.

`session_id` is caller-chosen, so it is derived from the corpus name and used as an
idempotency handle — republishing continues the *same document* rather than littering
the account with near-duplicates.

**Without a key it renders locally and says so** (`published: false` plus the reason)
rather than failing. Publishing is never automatic: it is an explicitly invoked
operation, because a run must not be able to fail because a third party is having a bad
afternoon, and spending someone's operations budget unasked is how a system teaches
people to distrust it.

#### Two bugs this found, both of which reported success

**The first live edit landed on the wrong vendor's section.** Every status code was
200, every count was right, and the publish reported `published: true`. The cause was
not the prompt: the register renders an agreement-level `hourly_rate` and a SOW-scoped
one identically, so 19 sections produced **18 distinct headings** and the document had
no unique anchor. No phrasing of "replace only this section" could have been reliable
against it. Every section now carries a stable `[ref:…]` derived from its section key,
and the instruction addresses that marker rather than the vendor and term.

**Then the verification failed a correct edit.** `/approve` returns 200 *before* the
change is applied — the same thing the 409 `session_busy` response tells you, applied
to export rather than to the next instruction — so reading the document back
immediately returns the pre-edit version. A check that fails the good case is worse
than no check, because it teaches you to disregard the result.

The lasting fix is neither of those individually: **the write path now verifies
itself.** After publishing it re-exports the document and confirms each edited section
carries its own value, comparing on normalised text rather than bytes (a markdown round
trip legitimately rewrites `_` as `\_`). If a section cannot be found or does not match,
`published` is `false` and the mismatch is named. Exports cost no operations, so the
check is free.

### Killing it, and picking it back up

```bash
curl -X POST localhost:8000/runs/<run_id>/resume
```

A run is driven by an in-process graph, so a killed process leaves a row that says
`running` with nothing running it. At startup every such row is reclassified
`interrupted` — an API reporting work in progress that nothing is progressing is the
same class of untruth as a false success. Resuming re-enters the graph at the node
after the last one that checkpointed, so completed stages are not re-paid for.

That last sentence is only true because the graph is invoked with `durability="sync"`,
which is not LangGraph's default. The default persists checkpoints asynchronously — the
next node starts while the previous one's checkpoint is still being written — so a
`kill -9` takes whatever had not landed, and how much that is depends on how fast the
machine is. The same kill left CI resumable at `compose` and a laptop resumable at
`ingest`, four already-metered stages thrown away. Nothing caught it for a while
because the resumed run still *finishes*; it just quietly re-does the work this section
says it does not. `test_the_kill_leaves_a_checkpoint_at_the_stage_it_died_in` asserts on
the seam rather than the outcome, which is what it takes to see the difference.

If the resume itself fails — most often a source document that moved — the run goes
back to `interrupted` rather than being left at `running`, and the response says which
document and why. A failed rescue must not recreate the ghost it was clearing.

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

Packages sit flat at the repository root and are imported directly (`from domain.rules
import evaluate`). There is no build step and nothing to pip-install — `PYTHONPATH=/app`
in the container, the working directory locally.

```
main.py             entry point: uvicorn main:app
mcp_server.py       the MCP surface — same service layer, no logic of its own
api/                routers mounted by main
services/           orchestration — the run graph, the service layer, metering, watcher
domain/             the logic that has nothing to do with transport:
                      classify · extract · normalize · reconcile
                      conflicts · adjudicate · rules · verify · compose · changes
models/             ORM tables; three of them carry the invariants
database/           engine, session scope, and application configuration
providers/          model provider interface + Gemini + a deterministic offline stub
utils/              hashing and logging configuration
migrations/         Alembic; the first migration creates the vector extension itself
rules/playbook.yaml the contract rules — adding a rule is a data change
corpus/seed/        the synthetic vendor documents
tests/              the no-key suite
```

---

## Honest limitations

- **Two ingestion formats** (`.md`, `.pdf`), not five. Declared rather than discovered:
  a capability may be honestly absent, never present and broken.
- **There is no vector search. pgvector is provisioned and unused.** The extension is
  installed, `chunk.embedding` is a `vector(768)` column, and nothing writes to it or
  reads from it. Retrieval here is structured fact lookup by predicate and vendor —
  deterministic, exhaustive at this corpus size, and testable with no key, which
  embedding similarity is none of.

  It is called out this loudly because the schema looks like the feature exists. A
  provisioned column is the easiest kind of thing to mistake for a working one, and
  the stack asks for vector search, so silence here would read as a claim. If the
  corpus grew past the point where exact predicate matching finds the affected
  sections, the column is where that would go — but that is a plan, not a feature.
- **Checkpointing is node-level.** Kill a node eighteen model calls deep and the graph
  resumes at the node boundary; no approved work is lost, but those calls are re-paid.
  Mitigated by keeping nodes small and putting a content-addressed cache in front of
  every model call — which is also how the "an update costs like an update" claim is
  measured rather than asserted.

  A consequence worth stating: because a node can die *after* writing rows but *before*
  its checkpoint commits, any node that writes must tolerate re-running over its own
  partial output. `compose` clears its own run's section versions before writing for
  exactly this reason. Nodes are not idempotent by accident, and one that isn't fails
  the resume it was supposed to survive.

- **Orphan detection assumes a single process.** At startup any run still marked
  `running` is reclassified `interrupted`, because nothing can be in flight while the
  only process that drives runs is still booting. Correct for the shipped deployment
  (one uvicorn, no `--workers`); under multiple workers this needs a heartbeat, since
  one worker booting says nothing about another's live runs.

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

**All five movements are built.** Documents are classified (with a real escalation
branch for low confidence), facts extracted with verified citations, values normalized,
competing values reconciled to the one that governs, contradictions detected
deterministically and adjudicated by a model, the register composed from only the
sections whose dependencies changed, the contract playbook applied, every claim verified
against its sources before anything is committed, and a watched folder keeps it all
current as new paperwork arrives.

### What it finds

Run against an eight-document corpus spanning two vendors, an amendment chain and two
invoices, it reports ten findings — three contradictions between documents and seven
breaches of the contract playbook:

| Severity | Source | Finding |
|---|---|---|
| high | conflict | Invoice bills $180/hr when Amendment No. 1 set $195 effective 2025-07-01 |
| high | conflict | Invoice total $20,500 ≠ 100 hrs × $195 = $19,500 — a $1,000 overcharge |
| high | LIAB-01 | Cobalt liability cap $5,000,000 exceeds 2 × annual fees = $620,000 |
| high | LIAB-01 | Northwind liability cap $2,500,000 exceeds 2 × annual fees = $1,200,000 |
| medium | conflict | Invoice payment terms deviate from the governing amendment |
| medium | PAY-01 | Cobalt payment terms net 60 exceed net 30 |
| medium | RENEW-01 | Cobalt auto-renewal of 36 months exceeds 12 |
| medium | RENEW-01 | Northwind auto-renewal of 24 months exceeds 12 |
| medium | NOTICE-01 | Cobalt termination notice of 15 days falls short of 30 |
| low | LAW-01 | Cobalt names no governing law |

Each is arithmetic, not opinion, and each cites the documents it came from. `LIAB-01`
computes its bound from another fact about the same vendor, which is why the two
liability findings quote different limits.

### The rules are configuration

`rules/playbook.yaml`. Adding a rule is a data change; adding a rule *kind* is a code
change, and four kinds cover the playbook. An unrecognised kind is refused when the
playbook loads rather than skipped — a rule that never fires looks exactly like a rule
that passes.

The `numeric_bound` expression grammar (`2 * annual_fees`) is a deliberate ~20 lines
rather than `eval()`. A playbook is configuration, and configuration that can execute
Python is a remote code execution hole wearing a friendly name.

### Verification can stop a run

Before anything reaches a human or a commit, a separate pass re-checks every claim: it
has a citation, the cited fact exists, and **the cited value still appears in the
passage it came from**. That last one is the point — a citation naming a document
proves nothing, and the dangerous failure is one that still looks fine while pointing
at text that has since changed.

If verification fails, the run stops. Nothing is committed, the run records `failed`,
and the API reports `blocked_by_verification` — never `completed`. The human gate is
deliberately skipped: showing a reviewer findings drawn from a register known to be
unsound invites an approval that is worse than no approval at all.

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

### Drop a document in a folder and it updates itself

Put a file in `inbox/` and poll:

```bash
curl -X POST localhost:8000/watch/poll
curl localhost:8000/runs/<run_id>/changes
```

Or drag it onto the review page, which posts to `/documents/upload`. That endpoint
writes into the same watched directory and triggers the same poll, so an upload and a
file dropped into the folder converge one line apart — a second ingestion path would
be a second thing to keep correct, and the one used less often is the one that rots.
Uploaded filenames are reduced to their basename before anything touches disk:
`../../x.md` is a path-traversal write, not a document.

Set `WATCH_ENABLED=true` to poll on a timer instead. It is off by default: a system
that starts spending a per-day model budget the moment it boots is one people learn to
distrust.

Measured on a live system — two agreements already processed, then one amendment
dropped into the folder:

| | |
|---|---|
| Wall time | **4 seconds** |
| **Untouched** | **13 of 15 sections (87%)** — byte-identical by hash |
| Changed | **2** — exactly the terms that amendment modifies |
| Model calls | **1 classify, 1 extract** — the new document only |

```
hourly_rate         $180 → $195 per hour   source=amendment  eff=2025-07-01
payment_terms_days  45   → 30              source=amendment  eff=2025-07-01
```

`GET /runs/{id}/changes` names the document responsible for each change, so "what
changed and why" is a query rather than a story assembled afterwards. Detection is by
content hash, not modification time — re-saving a file without editing it is not a
change and must not cost a run.

This is the whole claim of the system in one number: **87% of the register was proven
untouched, not asserted to be.**
