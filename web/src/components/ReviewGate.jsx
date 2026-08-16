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
export default function ReviewGate({ runId, run, onDecided }) {
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

  if (!awaiting) {
    return (
      <p className="empty">
        Nothing awaiting review{run.status ? ` — run is ${run.status}` : ""}.
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

  const approved = Object.values(verdicts).filter((v) => v === "approved").length;
  const rejected = findings.length - approved;

  return (
    <section>
      <h2>Awaiting review — {findings.length} findings</h2>

      {error && <div className="error">{error}</div>}

      {findings.map((finding) => {
        const key = String(finding.index);
        return (
          <div className="finding" key={key}>
            <div style={{ marginBottom: 6 }}>
              <span className={`pill ${finding.severity}`}>{finding.severity}</span>{" "}
              <span className="pill">{finding.target_kind}</span>
            </div>
            <p>{finding.explanation}</p>
            <div className="verdicts">
              <label>
                <input
                  type="radio"
                  name={`verdict-${key}`}
                  checked={verdicts[key] === "approved"}
                  onChange={() => setVerdict(finding.index, "approved")}
                />
                Approve
              </label>
              <label>
                <input
                  type="radio"
                  name={`verdict-${key}`}
                  checked={verdicts[key] === "rejected"}
                  onChange={() => setVerdict(finding.index, "rejected")}
                />
                Reject
              </label>
            </div>
          </div>
        );
      })}

      <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 12 }}>
        <button className="primary" onClick={submit} disabled={submitting}>
          {submitting ? "Submitting…" : `Submit ${findings.length} decisions`}
        </button>
        <span className="sub" style={{ color: "var(--muted)", fontSize: 13 }}>
          {approved} approved, {rejected} rejected
        </span>
      </div>
    </section>
  );
}
