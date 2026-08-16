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
    // Default every finding to approved, which is the common case, while leaving each
    // one individually changeable. Starting blank would make a reviewer click twice
    // for every item they already agree with, and a queue that is tedious is a queue
    // that gets rubber-stamped.
    setVerdicts(Object.fromEntries(findings.map((f) => [String(f.index), "approved"])));
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
      await api.submitDecisions(runId, verdicts);
      await onDecided();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const setAll = (verdict) =>
    setVerdicts(Object.fromEntries(findings.map((f) => [String(f.index), verdict])));

  const approved = Object.values(verdicts).filter((v) => v === "approved").length;
  const rejected = findings.length - approved;

  // Severity order, not the order the pipeline happened to emit them in. A reviewer
  // working top to bottom should meet the $1.2M liability breach before a missing
  // governing-law clause, and attention is highest on the first few items.
  const ordered = [...findings].sort(
    (a, b) => (RANK[a.severity] ?? 9) - (RANK[b.severity] ?? 9),
  );
  const counts = findings.reduce((acc, f) => {
    acc[f.severity] = (acc[f.severity] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <section>
      <h2>Awaiting review — {findings.length} findings</h2>

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
        <div className="bulk">
          <button onClick={() => setAll("approved")}>Approve all</button>
          <button onClick={() => setAll("rejected")}>Reject all</button>
        </div>
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
        <button className="primary" onClick={submit} disabled={submitting}>
          {submitting ? "Submitting…" : `Submit ${findings.length} decisions`}
        </button>
        <span className="tally-text">
          {approved} approved, {rejected} rejected
        </span>
      </div>
    </section>
  );
}
