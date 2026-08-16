import { useEffect, useState } from "react";
import * as api from "../api";

/**
 * Where a value came from.
 *
 * This is the panel that makes the register checkable rather than merely readable. It
 * shows the claim, then every fact behind it, then the actual passage each fact was
 * read from — with the value highlighted inside it.
 *
 * The surrounding passage matters as much as the quote. A quote on its own is easy to
 * agree with; the point of provenance is to let a reviewer disagree, and to do that
 * they need to see the sentence in its context.
 */
export default function Provenance({ runId, sectionKey, changes, cost }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!runId || !sectionKey) {
      setData(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .getProvenance(runId, sectionKey)
      .then((result) => {
        // A response that arrives after the user has clicked elsewhere must not
        // overwrite what they are now looking at.
        if (!cancelled) setData(result);
      })
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoading(false));

    return () => {
      cancelled = true;
    };
  }, [runId, sectionKey]);

  if (!sectionKey) {
    return (
      <>
        <h2>Provenance</h2>
        <p className="empty">Select a register row to see the documents and passages it was read from.</p>
        <RunSummary changes={changes} cost={cost} />
      </>
    );
  }

  return (
    <>
      <h2>Provenance</h2>
      {loading && <p className="empty">Loading…</p>}
      {error && <div className="error">{error}</div>}

      {data && (
        <>
          {data.claims.map((claim, i) => (
            <p key={i} style={{ fontSize: 13, marginTop: 0 }}>
              {claim.text}{" "}
              <span className={`pill ${claim.status === "supported" ? "agreed" : "unsupported"}`}>
                {claim.status}
              </span>
            </p>
          ))}

          <div className="hash" style={{ marginBottom: 14 }}>
            {data.content_hash.slice(0, 16)}…
            {data.carried_forward && " · carried forward unchanged"}
          </div>

          <h3 style={{ marginTop: 0 }}>
            {data.citations.length} citation{data.citations.length === 1 ? "" : "s"}
          </h3>

          {data.citations.length === 0 && (
            <p className="empty">
              No citations. The register reports this row as unsupported rather than
              stating a value it cannot evidence.
            </p>
          )}

          {data.citations.map((citation) => (
            <div key={citation.fact_id} className="citation">
              <div style={{ marginBottom: 4 }}>
                <span className="value">{citation.value}</span>{" "}
                <span className="pill">{citation.document_kind ?? "document"}</span>
              </div>
              <div className="hash" style={{ marginBottom: 6 }}>
                {citation.document}
                {citation.char_start != null &&
                  ` · chars ${citation.char_start}–${citation.char_end}`}
                {citation.effective_date && ` · effective ${citation.effective_date}`}
              </div>
              {citation.passage ? (
                <div className="passage">{highlight(citation.passage, citation.value)}</div>
              ) : (
                <div className="empty">
                  Passage unavailable — the quote spanned a chunk boundary, so the exact
                  span cannot be re-shown.
                </div>
              )}
            </div>
          ))}
        </>
      )}
    </>
  );
}

/** Highlight the cited value inside its passage, tolerating formatting differences. */
function highlight(passage, value) {
  if (!value) return passage;

  const index = passage.indexOf(value);
  if (index === -1) {
    // The stored value and the source text can differ in presentation — "$195" against
    // "$195.00". Failing to highlight is the right outcome; inventing a match is not.
    return passage;
  }

  return (
    <>
      {passage.slice(0, index)}
      <mark>{passage.slice(index, index + value.length)}</mark>
      {passage.slice(index + value.length)}
    </>
  );
}

function RunSummary({ changes, cost }) {
  if (!changes && !cost) return null;

  return (
    <div style={{ marginTop: 24 }}>
      {changes && (
        <>
          <h3>This run changed</h3>
          <div className="stat"><span>Sections</span><span>{changes.sections_total}</span></div>
          <div className="stat"><span>Added</span><span>{changes.added}</span></div>
          <div className="stat"><span>Changed</span><span>{changes.changed}</span></div>
          <div className="stat"><span>Untouched</span><span>{changes.unchanged}</span></div>
          {changes.sections_total > 0 && (
            <div className="stat">
              <span>Proven untouched</span>
              <span>{Math.round(changes.untouched_fraction * 100)}%</span>
            </div>
          )}
        </>
      )}

      {cost && (
        <>
          <h3>Cost by stage</h3>
          {cost.by_stage.map((stage) => (
            <div className="stat" key={stage.stage}>
              <span>
                {stage.stage}
                {stage.skipped && <span className="pill" style={{ marginLeft: 6 }}>skipped</span>}
              </span>
              <span>
                {stage.cache_misses} call{stage.cache_misses === 1 ? "" : "s"}
                {stage.cost_usd > 0 && ` · $${stage.cost_usd.toFixed(4)}`}
              </span>
            </div>
          ))}
          <div className="stat" style={{ borderTop: "1px solid var(--line)", marginTop: 6, paddingTop: 6 }}>
            <span>Cache hits</span><span>{cost.total_cache_hits}</span>
          </div>
        </>
      )}
    </div>
  );
}
