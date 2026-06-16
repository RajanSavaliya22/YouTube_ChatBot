/**
 * Background Service Worker
 * ===========================
 * Stage: Step 4 — Indexing trigger + job status polling.
 *
 * Flow:
 *   1. content.js sends VIDEO_CHANGED message with {videoId, url}
 *   2. Check chrome.storage.sync — skip if autoIndex is OFF
 *   3. POST /index to the configured API → get job_id
 *   4. Poll GET /index/status/{job_id} every 4s until status is "done" or "failed"
 *   5. Broadcast status updates to the originating tab via chrome.tabs.sendMessage
 *      so content.js can update the chat panel UI (added in Step 5)
 *
 * Per-tab state is tracked in memory (Map) since indexing status only
 * matters while the tab is open. Re-indexing the same video in another
 * tab is independent — each tab polls its own job.
 */

const DEFAULT_SETTINGS = {
  apiUrl: "https://youtube-rag-api-6xhe.onrender.com",
  apiKey: "",
  autoIndex: true,
};

const POLL_INTERVAL_MS = 4000;
const POLL_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes — long videos take a while

// tabId -> { videoId, jobId, status, pollTimer }
const tabState = new Map();

console.log("[YT-RAG][background] Service worker loaded.");

chrome.runtime.onInstalled.addListener(() => {
  console.log("[YT-RAG][background] Extension installed.");
});

// ── Settings helper ───────────────────────────────────────────

async function getSettings() {
  return chrome.storage.sync.get(DEFAULT_SETTINGS);
}

function buildHeaders(apiKey) {
  const headers = { "Content-Type": "application/json" };
  if (apiKey) headers["X-Api-Key"] = apiKey;
  return headers;
}

// ── Messaging to content script ───────────────────────────────

function notifyTab(tabId, payload) {
  chrome.tabs.sendMessage(tabId, { type: "INDEX_STATUS", ...payload }).catch((err) => {
    // Tab may have navigated away or closed — safe to ignore
    console.debug("[YT-RAG][background] notifyTab failed (tab likely gone):", err.message);
  });
}

// ── Core indexing flow ───────────────────────────────────────

async function startIndexing(tabId, videoId, videoUrl) {
  const settings = await getSettings();

  if (!settings.autoIndex) {
    console.log("[YT-RAG][background] Auto-index disabled — skipping.");
    notifyTab(tabId, { status: "disabled", videoId });
    return;
  }

  if (!settings.apiUrl) {
    console.warn("[YT-RAG][background] No API URL configured.");
    notifyTab(tabId, { status: "error", videoId, error: "No API URL configured" });
    return;
  }

  // Clear any existing poll loop for this tab (e.g. user navigated to a new video mid-poll)
  clearTabPoll(tabId);

  notifyTab(tabId, { status: "starting", videoId });

  try {
    const res = await fetch(`${settings.apiUrl}/index`, {
      method: "POST",
      headers: buildHeaders(settings.apiKey),
      body: JSON.stringify({ url: videoUrl, force_reindex: false }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
    }

    const data = await res.json();

    // /index can return either a queued job or an already-indexed result
    // depending on how the API was called (sync vs async). Handle both.
    if (data.job_id) {
      console.log(`[YT-RAG][background] Indexing job queued: ${data.job_id} for ${videoId}`);
      tabState.set(tabId, { videoId, jobId: data.job_id, status: "queued" });
      notifyTab(tabId, { status: "queued", videoId, jobId: data.job_id });
      pollJobStatus(tabId, settings, data.job_id, videoId, Date.now());
    } else if (data.already_existed !== undefined) {
      // Sync response shape from IndexResponse
      const status = data.already_existed ? "already_indexed" : "done";
      console.log(`[YT-RAG][background] Index result for ${videoId}: ${status}`);
      notifyTab(tabId, { status, videoId, chunksIndexed: data.chunks_indexed });
    } else {
      notifyTab(tabId, { status: "done", videoId });
    }
  } catch (err) {
    console.error("[YT-RAG][background] Indexing request failed:", err);
    notifyTab(tabId, { status: "error", videoId, error: err.message });
  }
}

async function pollJobStatus(tabId, settings, jobId, videoId, startTime) {
  // Stop polling if the tab moved on to a different video in the meantime
  const current = tabState.get(tabId);
  if (!current || current.jobId !== jobId) {
    return;
  }

  if (Date.now() - startTime > POLL_TIMEOUT_MS) {
    console.warn(`[YT-RAG][background] Polling timed out for job ${jobId}`);
    notifyTab(tabId, { status: "error", videoId, error: "Indexing timed out" });
    tabState.delete(tabId);
    return;
  }

  try {
    const res = await fetch(`${settings.apiUrl}/index/status/${jobId}`, {
      method: "GET",
      headers: buildHeaders(settings.apiKey),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const job = await res.json();

    if (job.status === "done") {
      console.log(`[YT-RAG][background] Indexing complete for ${videoId}:`, job);
      notifyTab(tabId, {
        status: "done",
        videoId,
        chunksIndexed: job.chunks_indexed,
        alreadyExisted: job.already_existed,
      });
      tabState.delete(tabId);
      return;
    }

    if (job.status === "failed") {
      console.error(`[YT-RAG][background] Indexing failed for ${videoId}:`, job.error);
      notifyTab(tabId, { status: "error", videoId, error: job.error || "Indexing failed" });
      tabState.delete(tabId);
      return;
    }

    // Still running — report current step and poll again
    notifyTab(tabId, { status: "running", videoId, step: job.step || "processing" });

    const timer = setTimeout(
      () => pollJobStatus(tabId, settings, jobId, videoId, startTime),
      POLL_INTERVAL_MS
    );
    tabState.set(tabId, { videoId, jobId, status: "running", pollTimer: timer });
  } catch (err) {
    console.error("[YT-RAG][background] Poll request failed:", err);
    // Transient network errors shouldn't kill the whole flow — retry once more
    const timer = setTimeout(
      () => pollJobStatus(tabId, settings, jobId, videoId, startTime),
      POLL_INTERVAL_MS
    );
    tabState.set(tabId, { videoId, jobId, status: "running", pollTimer: timer });
  }
}

function clearTabPoll(tabId) {
  const state = tabState.get(tabId);
  if (state?.pollTimer) {
    clearTimeout(state.pollTimer);
  }
  tabState.delete(tabId);
}

// ── Message listener ───────────────────────────────────────────

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "VIDEO_CHANGED") {
    const tabId = sender.tab?.id;
    if (tabId === undefined) return;

    console.log(
      "[YT-RAG][background] VIDEO_CHANGED received:",
      message.videoId, "| tab:", tabId
    );

    startIndexing(tabId, message.videoId, message.url);
  }
});

// Clean up state when a tab closes — prevents orphaned poll loops
chrome.tabs.onRemoved.addListener((tabId) => {
  clearTabPoll(tabId);
});