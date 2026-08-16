/**
 * The register itself, one row per obligation.
 *
 * Every row is clickable, because the point of the register is that any value in it
 * can be traced back to the sentence it was read from. A row you cannot interrogate is
 * asking to be trusted.
 *
 * Rows carried forward unchanged are marked as such — that is the visible face of the
 * incrementality claim, and it is worth seeing beside the values rather than only in a
 * summary.
 */
export default function Register({ deliverable, changes, selectedKey, onSelect }) {
  if (!deliverable) return null;

  const sections = deliverable.sections ?? [];
  if (!sections.length) {
    return <p className="empty">This run produced no register rows.</p>;
  }

  // status by section, from the change ledger, so "what moved in this run" is visible
  // in the same table as the values themselves.
  const changeByKey = Object.fromEntries(
    (changes?.sections ?? []).map((s) => [s.section_key, s])
  );

  return (
    <section>
      <h2>Register</h2>
      {deliverable.carried_forward > 0 && (
        <p className="empty" style={{ marginTop: -8, marginBottom: 12 }}>
          {deliverable.carried_forward} rows carried forward untouched ·{" "}
          {deliverable.rederived} re-derived
        </p>
      )}

      <div className="tablewrap">
      <table>
        <thead>
          <tr>
            <th>Vendor</th>
            <th>Term</th>
            <th>Value</th>
            <th>Effective</th>
            <th>Source</th>
            <th>Status</th>
            <th>This run</th>
          </tr>
        </thead>
        <tbody>
          {sections.map((section) => {
            let row = {};
            try {
              row = JSON.parse(section.content);
            } catch {
              // Content is stored as canonical JSON; if that ever changes, show the
              // key rather than crashing the whole table over one row.
              row = { vendor: section.section_key, term: "", value: null };
            }

            const change = changeByKey[section.section_key];
            const selected = section.section_key === selectedKey;

            return (
              <tr
                key={section.section_key}
                className={`clickable ${selected ? "selected" : ""}`}
                onClick={() => onSelect(section.section_key)}
              >
                <td>{row.vendor}</td>
                <td>{row.term}</td>
                <td>
                  {row.value ?? <span style={{ color: "var(--faint)" }}>—</span>}
                  {row.superseded?.length > 0 && (
                    <div className="was">was {row.superseded.join(", ")}</div>
                  )}
                </td>
                <td>{row.effective_date ?? ""}</td>
                <td>{row.governing_source ?? ""}</td>
                <td>
                  <span className={`pill ${row.status ?? ""}`}>{row.status ?? "—"}</span>
                </td>
                <td>
                  {section.carried_forward ? (
                    <span className="pill carried">carried</span>
                  ) : (
                    <span className="pill">{change?.status ?? "re-derived"}</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
    </section>
  );
}
