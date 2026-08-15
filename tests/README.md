# The test suite

```bash
docker compose up -d db
pytest
```

**179 tests. No API key, no network, no recorded fixtures.** CI holds no secrets at all
— there is no `secrets.` reference in the workflow and no repository secret for it to
read, so a fork gets the same green run.

---

## Why there are no cassettes

Recording real model responses and replaying them is the usual answer to "tests that
need an LLM". It is not the answer here, for a reason worth stating plainly: a suite
built on replayed fixtures mostly proves the recordings still parse.

So the assertions are about what the *system* did, never about what the model said:

| Instead of | This suite asserts |
|---|---|
| the model returned the right rate | the section's **content hash** did not change |
| adjudication produced good output | the adjudication node **never executed** |
| the run finished after a crash | the model was **not called again** for completed work |
| the register looks right | every claim **resolves to a span of a real document** |

Every one of those is checkable without a model, because every one of them is a fact
about the system's own behaviour.

The offline provider (`ledger.providers.fake`) exists to make that possible. It is not
a mock library: it satisfies the real `ModelProvider` interface, returns
schema-conforming JSON, draws its quotes and values from the actual source text so
citations genuinely resolve, and records every call so "that stage did not run" is
checkable.

---

## Layout

| File | Covers |
|---|---|
| `test_phase0_wiring.py` | Provider selection is configuration; the offline stub behaves |
| `test_phase0_schema.py` | The migration created the constraints that carry the invariants |
| `test_phase1_citations.py` | Citation resolution, including quotes spanning chunks; injection prompt shape |
| `test_phase1_flow.py` | End to end: cited facts → gate → commit; incrementality |
| `test_phase2_normalize.py` | Value normalization and reconciliation — which value governs |
| `test_phase2_classify.py` | Classification, precedence, and the escalation branch |
| `test_phase3_conflicts.py` | The three comparators, and adjudication |
| `test_phase4_rules.py` | The playbook: loading, validation, evaluation |
| `test_phase4_verify.py` | Stage C, and that a failure **blocks** the run |
| `test_phase5_watcher.py` | Folder watching, by content hash |
| `test_phase5_changes.py` | The change ledger — what moved and why |
| `test_floors.py` | The five floors, plus injection and concurrency |

Tests needing a live Postgres are marked `integration` and **skip with a usable
message** rather than erroring, so `pytest` on a laptop with nothing running still
reports the unit tests that did pass instead of burying them in connection errors.

```bash
pytest -m "not integration"   # no database at all
```

---

## The tests worth reading first

**`test_floors.py::TestFloorTwoSurvivesBeingKilled`** — spawns a subprocess, `SIGKILL`s
it at a known stage boundary, then runs again and asserts `cache_misses == 0` for
extraction. The point is that "it finished" proves nothing: a system that silently
restarted from scratch also finishes. Only the call count distinguishes them.

**`test_phase5_changes.py`** — a document arrives, and the assertion is that untouched
sections are byte-identical *and were carried forward rather than regenerated into
matching bytes*. Regeneration that happens to agree still paid full price, which is
exactly the thing being disclaimed.

**`test_floors.py::TestPromptInjectionIsReportedNotObeyed`** — a corpus document
instructs the system to approve everything automatically. Three assertions: it was
reported, the gate still stopped for a human, and the other documents' sections are
byte-identical to a control run. Any one alone would be insufficient.

**`test_phase4_verify.py::TestAFailedVerificationBlocksTheRun`** — forces verification
to fail and asserts the run blocks, the gate is skipped, the run records `failed`, and
a later run refuses to build on it. A verifier whose verdict is ignored downstream is
decoration.

---

## Bugs these tests found

Listed because it is the honest measure of whether a suite is worth its weight:

- **Concurrent runs crashed on ingestion.** Two runs over one corpus both checked
  "document absent", both inserted, one died on a unique constraint. Found by the
  behavior-9 test on its first execution.
- **`resolve_citation("")` returned a span** — `"x".find("")` is `0`, so an empty quote
  "resolved" to a zero-length span while reporting success.
- **Facts were re-minted with fresh IDs every run**, so nothing could ever be carried
  forward. All the incrementality machinery was correct and the claim was still
  impossible.
- **Claims and citations were never written at all** — I1's enforcement table sat empty
  for three phases while everything above it looked correct.
- **Carried-forward sections had no claims**, so verification silently skipped exactly
  the sections asserted to be unchanged. Passing by having nothing to check is the most
  dangerous way for a check to pass.
