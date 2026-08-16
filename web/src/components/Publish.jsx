import { useState } from "react";
import * as api from "../api";

/**
 * Publish the register to SuperDocs as an editable document.
 *
 * Deliberately a button rather than something that happens on its own. Publishing
 * spends someone's operations budget, and a system that spends it unasked is one
 * people learn to distrust.
 *
 * The result reports what was actually sent — how many sections were edited against
 * how many were left alone — because that count *is* the claim. "Published" on its own
 * would hide the only interesting thing about it.
 */
export default function Publish({ runId }) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const publish = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.publishRegister(runId));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="publish">
      <button onClick={publish} disabled={busy || !runId}>
        {busy ? "Publishing…" : "Publish to SuperDocs"}
      </button>

      {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}

      {result && (
        <div className={result.published ? "note" : "error"} style={{ marginTop: 8 }}>
          {result.published ? (
            result.mode === "incremental-edit" ? (
              <>
                <strong>
                  {result.sections_edited} edited, {result.sections_untouched} untouched
                </strong>
                {result.verified && " · verified against the exported document"}
              </>
            ) : result.mode === "no-op" ? (
              <>Nothing to publish — this run carried every section forward.</>
            ) : (
              <>Uploaded {result.sections_total} sections.</>
            )
          ) : (
            <>
              {/* Not published is never rendered as a success, and the reason is shown
                  rather than summarised — a partial write is the case where the detail
                  is the whole message. */}
              <strong>Not published.</strong> {result.reason}
              {result.mismatches?.length > 0 && (
                <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
                  {result.mismatches.map((m) => (
                    <li key={m.section_key}>{m.error}</li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
