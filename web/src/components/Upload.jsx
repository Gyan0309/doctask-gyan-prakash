import { useRef, useState } from "react";
import * as api from "../api";

/**
 * Add documents from the browser.
 *
 * The upload lands in the same watched folder the filesystem drop uses and triggers
 * the same poll, so this is a second doorway onto one path rather than a second path.
 * Drag-and-drop and the file picker are both here because a reviewer holding a PDF
 * should not have to know where the server keeps its inbox.
 */
export default function Upload({ onUploaded }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const send = async (files) => {
    if (!files?.length) return;
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const result = await api.uploadDocuments(files);
      // Says what the run covers, not just what was dropped. A register is over a
      // corpus, so three new files start a run across the whole folder — the other
      // documents are reused without a model call, but a reviewer watching fourteen
      // classifications go by after adding three files should not have to work that out.
      const scope =
        result.corpus_documents > result.saved.length
          ? ` — run started over all ${result.corpus_documents} documents in the corpus`
          : " — run started";
      setMessage(
        result.run_id
          ? `${result.saved.length} added${scope}`
          : `${result.saved.length} uploaded; nothing changed, so no run started`,
      );
      await onUploaded(result.run_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      // Clearing lets the same file be chosen twice in a row; without it the picker
      // fires no change event the second time and the click looks broken.
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <div className="upload">
      <div
        className={`dropzone ${dragging ? "over" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          send(e.dataTransfer.files);
        }}
      >
        {busy ? "Uploading…" : "Drop a document, or click to choose"}
        <div className="hint">
          {/* The upload now returns as soon as the run has an id; the run itself is
              watched by the polling in App. So this is a file-transfer wait measured
              in moments, not the whole run. */}
          {busy ? "handing off to the pipeline" : "PDF, Markdown or text"}
        </div>
      </div>

      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".pdf,.md,.txt"
        style={{ display: "none" }}
        onChange={(e) => send(e.target.files)}
      />

      {message && <div className="note">{message}</div>}
      {error && <div className="error">{error}</div>}
    </div>
  );
}
