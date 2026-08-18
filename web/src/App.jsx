import { useCallback, useEffect, useState } from "react";
import * as api from "./api";
import ClassifyGate from "./components/ClassifyGate.jsx";
import Register from "./components/Register.jsx";
import ReviewGate from "./components/ReviewGate.jsx";
import Provenance from "./components/Provenance.jsx";
import RunList from "./components/RunList.jsx";
import Upload from "./components/Upload.jsx";

// Stage names in the words a reviewer would use. The graph's own node names are
// accurate and read like internals — "detect_conflicts" is not what a person waiting on
// a run wants to be told they are waiting for.
const STAGE_LABELS = {
  ingest: "reading documents",
  classify: "identifying documents",
  escalate: "waiting on a classification",
  extract: "extracting terms",
  detect_conflicts: "comparing documents",
  adjudicate: "judging conflicts",
  compose: "building the register",
  examine: "applying the playbook",
  verify: "verifying citations",
  gate: "waiting for review",
  commit: "committing",
  blocked: "blocked",
};

export default function App() {
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState(null);
  const [run, setRun] = useState(null);
  const [deliverable, setDeliverable] = useState(null);
  const [changes, setChanges] = useState(null);
  const [cost, setCost] = useState(null);
  const [selectedSection, setSelectedSection] = useState(null);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("review");

  const refreshRuns = useCallback(async () => {
    try {
      const rows = await api.listRuns();
      setRuns(rows);
      // Land on the newest run rather than an empty page.
      setRunId((current) => current ?? rows[0]?.run_id ?? null);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refreshRuns();
  }, [refreshRuns]);

  const loadRun = useCallback(async (id) => {
    if (!id) return;
    setError(null);
    try {
      // Fetched together so the register, the ledger and the cost panel always
      // describe the same run. Loading them independently lets the page show a
      // register from one run beside costs from another during a refresh.
      const [runData, deliverableData, changesData, costData] = await Promise.all([
        api.getRun(id),
        api.getDeliverable(id),
        api.getChanges(id).catch(() => null),
        api.getCost(id).catch(() => null),
      ]);
      setRun(runData);
      setDeliverable(deliverableData);
      setChanges(changesData);
      setCost(costData);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    // Only on a genuine run change — a poll refreshing the same run must not clear a
    // row the reviewer is reading.
    setSelectedSection(null);
    loadRun(runId);
  }, [runId, loadRun]);

  // A run takes as long as it takes, and a page that never revisits it shows "running"
  // until someone thinks to click. Poll only while there is something to wait for, so
  // a settled run costs nothing.
  useEffect(() => {
    if (run?.status !== "running") return;
    const timer = setInterval(() => {
      loadRun(runId);
      refreshRuns();
    }, 3000);
    return () => clearInterval(timer);
  }, [run?.status, runId, loadRun, refreshRuns]);

  const afterDecisions = useCallback(async () => {
    await Promise.all([refreshRuns(), loadRun(runId)]);
  }, [refreshRuns, loadRun, runId]);

  // Jump to the run the upload started, so the page shows its consequence rather than
  // leaving the reviewer on the previous run wondering whether anything happened.
  const afterUpload = useCallback(
    async (newRunId) => {
      await refreshRuns();
      if (newRunId) setRunId(newRunId);
    },
    [refreshRuns],
  );

  const pendingCount =
    (run?.pending_findings?.length ?? 0) + (run?.escalations?.length ?? 0);
  const rowCount = deliverable?.sections?.length ?? 0;

  // Land on whichever pane has something to do. A completed run opening on an empty
  // review pane hides the deliverable behind a click for no reason.
  useEffect(() => {
    setTab(
      run?.awaiting_review ||
        run?.awaiting_classification ||
        run?.status === "interrupted"
        ? "review"
        : "register",
    );
  }, [runId, run?.awaiting_review, run?.awaiting_classification, run?.status]);

  // Evidence lives in the right-hand panel, which is visible from either pane, so
  // showing it must not silently leave the reviewer on a different tab.
  const showEvidence = useCallback((sectionKey) => setSelectedSection(sectionKey), []);

  return (
    <>
      <header>
        <span className="mark">Ledger</span>
        <span className="sub">Obligation register</span>
        <span className="spacer" />
        {run && (
          <span className="status">
            {run.status.replace(/_/g, " ")}
            {/* What it is doing, not just that it is doing something. Stage metrics
                only land when a stage finishes, so a two-minute run used to show
                "running · 0 rows" for ninety seconds with no way to tell progress
                from a hang. */}
            {run.current_stage && (
              <>
                {" · "}
                <span className="stage">
                  {STAGE_LABELS[run.current_stage] ?? run.current_stage}
                </span>
                {run.stage_detail ? ` — ${run.stage_detail}` : ""}
              </>
            )}
            {deliverable ? ` · ${deliverable.sections.length} rows` : ""}
          </span>
        )}
      </header>

      <div className="layout">
        <aside>
          <h2>Add documents</h2>
          <Upload onUploaded={afterUpload} />

          <h2 style={{ marginTop: 26 }}>Runs</h2>
          <RunList runs={runs} activeId={runId} onSelect={setRunId} />
        </aside>

        <main>
          {error && <div className="error">{error}</div>}

          {/* Two panes, one at a time. Stacked, a run with 13 findings and 19 register
              rows is a single column metres long, and the register — the actual
              deliverable — sits below a fold nobody reaches. They are also two
              different jobs: deciding, and reading what was decided about. */}
          <nav className="tabs">
            <button
              className={tab === "review" ? "on" : ""}
              onClick={() => setTab("review")}
            >
              Review
              {pendingCount > 0 && <span className="count">{pendingCount}</span>}
            </button>
            <button
              className={tab === "register" ? "on" : ""}
              onClick={() => setTab("register")}
            >
              Register
              {rowCount > 0 && <span className="count">{rowCount}</span>}
            </button>
          </nav>

          {tab === "review" ? (
            /* The classification gate takes the pane when it is open. It arrives before
               extraction, so there is nothing else to review yet — showing an empty
               review queue beside an unanswered question would hide the only thing the
               run is actually waiting on. */
            run?.awaiting_classification ? (
              <ClassifyGate runId={runId} run={run} onDecided={afterDecisions} />
            ) : (
              <ReviewGate
                runId={runId}
                run={run}
                onDecided={afterDecisions}
                onShowEvidence={showEvidence}
              />
            )
          ) : (
            <Register
              runId={runId}
              deliverable={deliverable}
              changes={changes}
              selectedKey={selectedSection}
              onSelect={setSelectedSection}
            />
          )}
        </main>

        <div className="detail">
          <Provenance
            runId={runId}
            sectionKey={selectedSection}
            changes={changes}
            cost={cost}
          />
        </div>
      </div>
    </>
  );
}
