import { useEffect, useState } from "react";
import * as api from "../api";

/**
 * The earlier human gate: "what is this document?"
 *
 * This pane exists because the branch behind it started firing. Escalation had not fired
 * once across four runs and twenty-nine documents, so nothing surfaced it and nobody
 * noticed — a run that paused here reported plain "running" with no question anywhere,
 * and the only way to answer was to post document ids by hand.
 *
 * It asks before extraction, not at the review gate, because the answer changes what gets
 * extracted and what the document is allowed to govern. Asking afterwards would mean
 * extracting on a guess and confirming it later, which is the wrong order.
 *
 * `unknown` is a first-class answer and is offered as plainly as the others. A reviewer
 * forced to pick a nearest kind for a data-protection addendum picks `amendment`, and an
 * amendment outranks the agreement it sits beside — which is the exact mistake the
 * classifier was making before it was given the same way out.
 */
/**
 * Fallback help only. The server sends `classification_option_effects`, where whether a
 * kind governs is derived from `OBSERVATIONAL_KINDS` — the list that actually decides it.
 *
 * This map used to be the only description a reviewer ever saw, and it is where the bug
 * lived: `renewal_notice` read "a notice about renewal or termination timing" and never
 * mentioned that choosing it disarms the document. A reviewer followed the model's own
 * displayed reasoning to that answer and unknowingly made a 2026 rate card
 * observation-only, which then produced two high-severity "money may be recoverable"
 * findings against a vendor that had billed correctly.
 */
const KIND_HELP = {
  msa: "a master agreement establishing base terms",
  amendment: "modifies an existing agreement — outranks it where they disagree",
  sow: "a statement of work, binding its own engagement only",
  invoice: "a bill for work done; records what was charged, never what was agreed",
  renewal_notice:
    "a notice about renewal or termination timing — restates terms rather than setting them",
  unknown: "none of these — it is kept and reported, but governs nothing",
};

export default function ClassifyGate({ runId, run, onDecided }) {
  const [choices, setChoices] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const documents = run?.escalations ?? [];
  const awaiting = Boolean(run?.awaiting_classification);

  // Prefer what the server says an answer does. The plain `classification_options` list
  // stays the fallback so an older run still renders, but it cannot say which kinds
  // govern — which is the one thing this pane got wrong.
  const effects = run?.classification_option_effects ?? [];
  const options =
    effects.length > 0
      ? effects
      : (run?.classification_options ?? []).map((kind) => ({
          kind,
          governs: null,
          description: KIND_HELP[kind] ?? kind,
          effect: "",
        }));

  useEffect(() => {
    // Deliberately NOT defaulted to the model's proposal.
    //
    // The whole reason a document is here is that the proposal is not trustworthy —
    // pre-selecting it would turn the question into a confirmation, and the reviewer
    // would be clicking past exactly the judgement they were asked for.
    setChoices({});
    setError(null);
  }, [runId, documents.length, awaiting]);

  if (!awaiting) return null;

  const undecided = documents.filter((d) => !choices[d.document_id]).length;

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await api.submitDecisions(runId, choices);
      await onDecided();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section>
      <h2>What are these documents? — {documents.length} to identify</h2>

      <p className="empty" style={{ margin: "0 0 14px" }}>
        The classifier could not honestly place these. Document kind decides precedence,
        so a wrong answer here produces a register that is confidently incorrect —
        which is worse than one that admits it asked.
      </p>

      {error && <div className="error">{error}</div>}

      {documents.map((doc) => (
        <div className="finding" key={doc.document_id}>
          <div className="fhead">
            <span className="pill medium">needs a kind</span>
            <code>{doc.document}</code>
          </div>

          <p style={{ margin: "8px 0" }}>{doc.reason}</p>

          <p className="hint" style={{ margin: "0 0 10px" }}>
            Proposed <strong>{doc.proposed_kind}</strong> at confidence{" "}
            {Number(doc.confidence).toFixed(2)}
            {doc.reasoning ? ` — ${doc.reasoning}` : ""}
          </p>

          <div className="kinds">
            {options.map((option) => (
              <button
                key={option.kind}
                className={choices[doc.document_id] === option.kind ? "on" : ""}
                title={`${option.description}${option.effect ? ` — ${option.effect}` : ""}`}
                onClick={() =>
                  setChoices((current) => ({
                    ...current,
                    [doc.document_id]: option.kind,
                  }))
                }
              >
                {option.kind.replace(/_/g, " ")}
                {/* Marked on the button itself, not only once chosen. A consequence
                    revealed after the click is a consequence the reviewer already
                    acted without. */}
                {option.governs === false && (
                  <span className="pill low" style={{ marginLeft: 6 }}>
                    governs nothing
                  </span>
                )}
              </button>
            ))}
          </div>

          {choices[doc.document_id] &&
            (() => {
              const chosen = options.find(
                (o) => o.kind === choices[doc.document_id],
              );
              if (!chosen) return null;
              return (
                <p
                  className="hint"
                  style={{
                    marginTop: 8,
                    ...(chosen.governs === false
                      ? { borderLeft: "3px solid var(--warn, #b58900)", paddingLeft: 8 }
                      : {}),
                  }}
                >
                  <strong>{chosen.kind.replace(/_/g, " ")}</strong> — {chosen.description}
                  {chosen.effect ? ` ${chosen.effect}` : ""}
                </p>
              );
            })()}
        </div>
      ))}

      <div className="gatebar" style={{ marginTop: 14 }}>
        <div className="tally">
          {undecided > 0 ? (
            <span className="pill low">{undecided} still to answer</span>
          ) : (
            <span className="pill">all answered</span>
          )}
        </div>
        {/* Answering some and not others is allowed: an unanswered document keeps no
            kind, governs nothing, and is asked about again on the next run. That is a
            better outcome than forcing a guess to get the run moving. */}
        <button className="primary" onClick={submit} disabled={submitting}>
          {submitting ? "Continuing…" : "Continue the run"}
        </button>
      </div>
    </section>
  );
}
