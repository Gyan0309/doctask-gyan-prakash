// Corpus names carry a random suffix so tests and repeat drops never collide
// ("changes-40a908ec"), which is exactly the part a reviewer doesn't need to read.
// The timestamp already tells runs on the same corpus apart, so it's dropped rather
// than shortened.
function humanize(corpus) {
  const withoutSuffix = corpus.replace(/-[0-9a-f]{6,}$/i, "");
  const words = withoutSuffix.replace(/[-_]/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export default function RunList({ runs, activeId, onSelect }) {
  if (!runs.length) {
    // The first thing a new arrival reads, and `inbox/` is gitignored — so on a fresh
    // clone they have no documents and nowhere obvious to get one. Naming the seed
    // corpus is the difference between a working system and a working system nobody
    // can start.
    return (
      <div className="empty">
        <p style={{ margin: "0 0 8px" }}>No runs yet.</p>
        <p style={{ margin: 0 }}>
          Drop a contract above — or try the eight synthetic ones in{" "}
          <code>corpus/seed/</code>, which contain a deliberate overcharge and a
          liability breach to find.
        </p>
      </div>
    );
  }

  return (
    <div>
      {runs.map((run) => (
        <div
          key={run.run_id}
          className={`run ${run.run_id === activeId ? "active" : ""}`}
          onClick={() => onSelect(run.run_id)}
        >
          <div className="name">{humanize(run.corpus)}</div>
          <div className="meta">
            {/* Status first: a blocked or awaiting run is the reason someone opened
                this page, so it should not need hunting for. */}
            {run.status.replace(/_/g, " ")} · {new Date(run.started_at).toLocaleString()}
          </div>
        </div>
      ))}
    </div>
  );
}
