// Every call the review page makes. One place, so a change to the API surface has one
// place to land rather than being scattered through components.

async function request(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    // Carry the server's own explanation through. A generic "request failed" hides the
    // one piece of information the user needs, and this API is careful to say what
    // went wrong.
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* body was not JSON; the status line is what we have */
    }
    throw new Error(detail);
  }

  return response.json();
}

export const listRuns = (limit = 25) => request(`/runs?limit=${limit}`);

export const getRun = (runId) => request(`/runs/${runId}`);

export const getDeliverable = (runId) => request(`/runs/${runId}/deliverable`);

export const getChanges = (runId) => request(`/runs/${runId}/changes`);

export const getCost = (runId) => request(`/runs/${runId}/cost`);

export const getDecisions = (runId) => request(`/runs/${runId}/decisions`);

export const getProvenance = (runId, sectionKey) =>
  request(`/runs/${runId}/provenance?section_key=${encodeURIComponent(sectionKey)}`);

// decisions maps finding index -> "approved" | "rejected".
export const submitDecisions = (runId, decisions) =>
  request(`/runs/${runId}/decisions`, {
    method: "POST",
    body: JSON.stringify({ decisions }),
  });

export const pollWatch = () => request("/watch/poll", { method: "POST" });
