export default function RunList({ runs, activeId, onSelect }) {
  if (!runs.length) {
    return <p className="empty">No runs yet. Start one with POST /runs, or drop a document into the watched folder.</p>;
  }

  return (
    <div>
      {runs.map((run) => (
        <div
          key={run.run_id}
          className={`run ${run.run_id === activeId ? "active" : ""}`}
          onClick={() => onSelect(run.run_id)}
        >
          <div className="name">{run.corpus}</div>
          <div className="meta">
            {/* Status first: a blocked or awaiting run is the reason someone opened
                this page, so it should not need hunting for. */}
            {run.status} · {new Date(run.started_at).toLocaleString()}
          </div>
        </div>
      ))}
    </div>
  );
}
