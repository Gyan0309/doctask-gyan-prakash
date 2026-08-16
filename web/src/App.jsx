import { useCallback, useEffect, useState } from "react";
import * as api from "./api";
import Register from "./components/Register.jsx";
import ReviewGate from "./components/ReviewGate.jsx";
import Provenance from "./components/Provenance.jsx";
import RunList from "./components/RunList.jsx";

export default function App() {
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState(null);
  const [run, setRun] = useState(null);
  const [deliverable, setDeliverable] = useState(null);
  const [changes, setChanges] = useState(null);
  const [cost, setCost] = useState(null);
  const [selectedSection, setSelectedSection] = useState(null);
  const [error, setError] = useState(null);

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
    setSelectedSection(null);
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
    loadRun(runId);
  }, [runId, loadRun]);

  const afterDecisions = useCallback(async () => {
    await Promise.all([refreshRuns(), loadRun(runId)]);
  }, [refreshRuns, loadRun, runId]);

  return (
    <>
      <header>
        <span className="mark">Ledger</span>
        <span className="sub">Obligation register</span>
        <span className="spacer" />
        {run && (
          <span className="status">
            {run.status.replace(/_/g, " ")}
            {deliverable ? ` · ${deliverable.sections.length} rows` : ""}
          </span>
        )}
      </header>

      <div className="layout">
        <aside>
          <h2>Runs</h2>
          <RunList runs={runs} activeId={runId} onSelect={setRunId} />
        </aside>

        <main>
          {error && <div className="error">{error}</div>}

          <ReviewGate runId={runId} run={run} onDecided={afterDecisions} />

          <Register
            deliverable={deliverable}
            changes={changes}
            selectedKey={selectedSection}
            onSelect={setSelectedSection}
          />
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
