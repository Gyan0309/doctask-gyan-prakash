import { useEffect, useState } from "react";
import * as api from "../api";

/**
 * The human gate.
 *
 * Every finding gets its own verdict. That is the requirement, and it is not the same
 * as an "approve all" button with an escape hatch: rejecting one finding must leave the
 * others exactly as they were, and the reviewer must be able to say so in one pass
 * rather than being walked through a wizard.
 *
 * Nothing is submitted until the reviewer presses the button, so changing your mind
 * about item three after deciding item five costs nothing.
 */
const SEVERITIES = ["high", "medium", "low"];
const RANK = { high: 0, medium: 1, low: 2 };

export default function ReviewGate({ runId, run, onDecided, onShowEvidence }) {
  const [verdicts, setVerdicts] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const findings = run?.pending_findings ?? [];
  const awaiting = Boolean(run?.awaiting_review);

  useEffect(() => {
    // Deliberately NOT defaulted to approved.
    //
    // This used to pre-select Approve on every finding, on the argument that a blank
    // queue is tedious and tedium gets rubber-stamped. That argument is real, but it
    // was answered in the wrong place: it made *inaction* mean approval on the one
    // gate that actually commits. A reviewer who submits without reading approved
    // everything, including a $2.5M liability breach, and the UI recorded them as
    // having decided it.
    //
    // The tedium is instead answered by "Approve all" — one click, same outcome, but
    // now a deliberate act with a name rather than a default nobody chose. This is
    // the same rule ClassifyGate already applies to its own options, and the reason
    // it gives there ("pre-selecting turns the question into a confirmation") applies
    // with more force here, not less.
    setVerdicts({});
    setError(null);
  }, [runId, findings.length, awaiting]);

  const resume = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await api.resumeRun(runId);
      await onDecided();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  if (!run) return null;

  if (run.status === "blocked_by_verification") {
    return (
      <div className="error">
        <strong>Blocked: the register failed verification.</strong> Nothing was
        committed and no review is offered — approving findings drawn from a register
        we know is unsound would be worse than not asking.
        {run.verification && (
          <div style={{ marginTop: 6 }}>
            {run.verification.claims_checked} claims checked,{" "}
            {run.verification.failures} failed.
          </div>
        )}
      </div>
    );
  }

  if (run.status === "interrupted") {
    return (
      <div className="finding">
        <strong>This run was interrupted.</strong>
        <p>
          Its process stopped mid-run. Nothing finished was lost — resuming continues
          from the last completed stage rather than starting over.
        </p>
        <button className="primary" onClick={resume} disabled={submitting}>
          {submitting ? "Resuming…" : "Resume this run"}
        </button>
        {error && <div className="error" style={{ marginTop: 10 }}>{error}</div>}
      </div>
    );
  }

  if (!awaiting) {
    return (
      <p className="empty">
        Nothing awaiting review{run.status ? ` — run is ${run.status.replace(/_/g, " ")}` : ""}.
      </p>
    );
  }

  const setVerdict = (index, verdict) =>
    setVerdicts((current) => ({ ...current, [String(index)]: verdict }));

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      // Carried verdicts are sent explicitly. The gate node treats a missing verdict
      // as "approved", so omitting them would quietly turn a previous *rejection*
      // into an approval — the same fail-open shape this file already guards against.
      const payload = {
        ...Object.fromEntries(
          findings
            .filter((f) => f.prior_verdict)
            .map((f) => [String(f.index), f.prior_verdict]),
        ),
        ...verdicts,
      };
      await api.submitDecisions(runId, payload);
      await onDecided();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const setAll = (verdict) =>
    setVerdicts(Object.fromEntries(fresh.map((f) => [String(f.index), verdict])));

  // Findings a human has already ruled on, byte-identical, in an earlier run. The
  // backend only marks one when the explanation string matches exactly, so a finding
  // whose numbers moved is never treated as already-decided.
  const carried = findings.filter((f) => f.prior_verdict);
  const resolved = run?.resolved_findings ?? [];

  // What the reviewer is actually asked. Everything else was ruled on already, on a
  // byte-identical finding, and re-asking is the same mistake as re-deriving a section
  // whose facts never moved: work the system already knows the answer to.
  const fresh = findings.filter((f) => !f.prior_verdict);



  // Counted from the verdicts actually recorded. The old form was
  // `rejected = findings.length - approved`, which silently reported every
  // undecided finding as rejected the moment approval stopped being the default.
  const approved = Object.values(verdicts).filter((v) => v === "approved").length;
  const rejected = Object.values(verdicts).filter((v) => v === "rejected").length;
  const undecided = fresh.length - approved - rejected;

  // Severity order, not the order the pipeline happened to emit them in. A reviewer
  // working top to bottom should meet the $1.2M liability breach before a missing
  // governing-law clause, and attention is highest on the first few items.
  const ordered = [...fresh].sort(
    (a, b) => (RANK[a.severity] ?? 9) - (RANK[b.severity] ?? 9),
  );
  const counts = fresh.reduce((acc, f) => {
    acc[f.severity] = (acc[f.severity] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <section>
      <h2>
        Awaiting review — {fresh.length} new finding{fresh.length === 1 ? "" : "s"}
      </h2>

      {/* A count, not a queue. Findings already ruled on are carried with their
          verdict and never re-listed: re-asking about a byte-identical finding is
          the same waste as re-deriving a section whose facts never moved. */}
      {carried.length > 0 && (
        <p className="empty" style={{ marginTop: -4 }}>
          {carried.length} carried forward from your earlier decisions — unchanged, so
          not re-asked.
        </p>
      )}

      {/* What the document settled. This is usually the most interesting thing a new
          amendment does, and it used to be invisible: the finding simply stopped
          appearing, so noticing required having memorised the previous queue. */}
      {resolved.length > 0 && (
        <div className="resolved">
          <strong>
            {resolved.length} resolved by this run — no longer fires:
          </strong>
          <ul>
            {resolved.map((explanation) => (
              <li key={explanation}>{explanation}</li>
            ))}
          </ul>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      <div className="gatebar">
        <div className="tally">
          {SEVERITIES.filter((s) => counts[s]).map((s) => (
            <span key={s} className={`pill ${s}`}>
              {counts[s]} {s}
            </span>
          ))}
        </div>
        {/* Bulk verdicts still submit per item — the payload is one verdict per
            finding either way. This only saves clicks on the runs where a reviewer
            has genuinely made one decision about everything. */}
        {/* Nothing new to rule on means nothing to bulk-apply. Leaving these live
            offered an action with no target. */}
        {fresh.length > 0 && (
          <div className="bulk">
            <button onClick={() => setAll("approved")}>Approve all</button>
            <button onClick={() => setAll("rejected")}>Reject all</button>
          </div>
        )}
      </div>

      {ordered.map((finding) => {
        const key = String(finding.index);
        const verdict = verdicts[key];
        return (
          <div className={`finding sev-${finding.severity} ${verdict === "rejected" ? "is-rejected" : ""}`} key={key}>
            <div className="finding-head">
              <span className={`pill ${finding.severity}`}>{finding.severity}</span>
              <span className="pill">{finding.rule_code ?? finding.target_kind}</span>
              {finding.subject && <span className="subject">{finding.subject}</span>}
            </div>

            <p>{finding.explanation}</p>

            <div className="verdicts">
              <label>
                <input
                  type="radio"
                  name={`verdict-${key}`}
                  checked={verdict === "approved"}
                  onChange={() => setVerdict(finding.index, "approved")}
                />
                Approve
              </label>
              <label>
                <input
                  type="radio"
                  name={`verdict-${key}`}
                  checked={verdict === "rejected"}
                  onChange={() => setVerdict(finding.index, "rejected")}
                />
                Reject
              </label>

              {/* The evidence, one click away. Approving a finding you have not
                  checked is the failure this gate exists to prevent, and making the
                  reviewer hunt the register for the matching row is how that happens.
                  Absent when the finding is that nothing was found — there is no
                  source passage for a clause that does not exist. */}
              {finding.section_key && onShowEvidence && (
                <button className="link" onClick={() => onShowEvidence(finding.section_key)}>
                  Show evidence
                </button>
              )}
            </div>
          </div>
        );
      })}

      <div className="gatefoot">
        {/* Held closed until every finding has a verdict. Submitting a partial review
            would commit the undecided ones under whatever the server treats as
            missing, and "I never looked at it" must not resolve to a verdict. */}
        <button
          className="primary"
          onClick={submit}
          disabled={submitting || undecided > 0}
        >
          {submitting
            ? "Submitting…"
            : fresh.length === 0
              ? "Commit — nothing new to decide"
              : `Submit ${fresh.length} decision${fresh.length === 1 ? "" : "s"}`}
        </button>
        <span className="tally-text">
          {fresh.length === 0
            ? `${carried.length} carried forward, nothing new to decide`
            : `${approved} approved, ${rejected} rejected`}
          {undecided > 0 && (
            <>
              {" · "}
              <strong>{undecided} still to decide</strong>
            </>
          )}
        </span>
      </div>
    </section>
  );
}
