/**
 * Content Script
 * ===============
 * Stage: Step 5 — Wires the chat panel (panel.js) to video detection,
 * the background worker's indexing status, and the RAG query API.
 *
 * Responsibilities carried over from earlier steps:
 *   - Detect YouTube SPA navigation and extract video_id (Step 2)
 *   - Relay VIDEO_CHANGED to background.js, which triggers indexing (Step 4)
 *   - Relay INDEX_STATUS updates from background.js into the panel UI
 *
 * New in this step:
 *   - Inject the chat panel on script load
 *   - Reset the panel's message thread whenever the video changes
 *   - Wire panel.onSend() to call POST /query/stream on the API and
 *     stream tokens into the assistant bubble in real time
 *   - Wire panel.trySeek() to seek the YouTube <video> element when a
 *     citation timestamp is clicked, instead of opening a new tab
 */

console.log("[YT-RAG] Content script injected on:", window.location.href);

let currentVideoId = null;

const DEFAULT_SETTINGS = {
  apiUrl: "https://youtube-rag-api-6xhe.onrender.com",
  apiKey: "",
  autoIndex: true,
};

async function getSettings() {
  return chrome.storage.sync.get(DEFAULT_SETTINGS);
}

/**
 * Extract the YouTube video ID from a URL.
 * Handles: /watch?v=ID, youtu.be/ID, /embed/ID, /shorts/ID
 */
function extractVideoId(url) {
  const patterns = [
    /(?:v=)([A-Za-z0-9_-]{11})/,
    /youtu\.be\/([A-Za-z0-9_-]{11})/,
    /\/embed\/([A-Za-z0-9_-]{11})/,
    /\/shorts\/([A-Za-z0-9_-]{11})/,
  ];
  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match) return match[1];
  }
  return null;
}

/**
 * Called whenever the video actually changes (not on every navigation event).
 */
function onVideoChanged(videoId, url) {
  console.log("[YT-RAG] Video changed →", videoId, "|", url);

  // Reset the chat panel for the new video before indexing starts,
  // so old messages/citations from the previous video don't linger.
  if (window.YTRagPanel) {
    window.YTRagPanel.resetForNewVideo();
  }

  chrome.runtime.sendMessage({
    type: "VIDEO_CHANGED",
    videoId: videoId,
    url: url,
  }).catch((err) => {
    console.debug("[YT-RAG] sendMessage failed:", err.message);
  });
}

/**
 * Check the current URL and fire onVideoChanged if the video_id changed.
 */
function checkForVideoChange() {
  const url = window.location.href;
  const videoId = extractVideoId(url);

  if (!videoId) return; // Not a watch page

  if (videoId !== currentVideoId) {
    currentVideoId = videoId;
    onVideoChanged(videoId, url);
  }
}

// ── SPA navigation detection ──────────────────────────────────

document.addEventListener("yt-navigate-finish", () => {
  checkForVideoChange();
});

const titleObserver = new MutationObserver(() => {
  checkForVideoChange();
});

const titleElement = document.querySelector("title");
if (titleElement) {
  titleObserver.observe(titleElement, {
    childList: true,
    characterData: true,
    subtree: true,
  });
}

// ── Indexing status relay → panel UI ───────────────────────────

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "INDEX_STATUS") {
    console.log(`[YT-RAG] Index status: ${message.status}`, message);
    if (window.YTRagPanel) {
      window.YTRagPanel.setIndexStatus(message.status, message);
    }
  }
});

// ── Query streaming ─────────────────────────────────────────────

/**
 * Send a question to the API and stream the answer into the panel.
 * Uses fetch + ReadableStream to consume Server-Sent Events manually
 * (EventSource doesn't support POST bodies, so we parse SSE by hand).
 */
async function sendQuestion(question) {
  const settings = await getSettings();
  const panel = window.YTRagPanel;

  panel.addUserMessage(question);
  panel.clearInput();
  panel.setInputEnabled(false);

  const assistantEl = panel.addStreamingAssistantMessage();

  try {
    const headers = { "Content-Type": "application/json" };
    if (settings.apiKey) headers["X-Api-Key"] = settings.apiKey;

    const response = await fetch(`${settings.apiUrl}/query/stream`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        question: question,
        video_id: currentVideoId,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by double newlines: "data: {...}\n\n"
      const events = buffer.split("\n\n");
      buffer = events.pop(); // last chunk may be incomplete — keep for next read

      for (const rawEvent of events) {
        const line = rawEvent.trim();
        if (!line.startsWith("data:")) continue;

        const jsonStr = line.slice(5).trim();
        let evt;
        try {
          evt = JSON.parse(jsonStr);
        } catch (e) {
          console.warn("[YT-RAG] Failed to parse SSE event:", jsonStr);
          continue;
        }

        handleStreamEvent(evt, assistantEl, panel);
      }
    }

    panel.finalizeAssistantMessage(assistantEl);
  } catch (err) {
    console.error("[YT-RAG] Query failed:", err);
    panel.finalizeAssistantMessage(assistantEl);
    if (!assistantEl.textContent) {
      assistantEl.remove();
    }
    panel.addErrorMessage("Something went wrong — please try again.");
  } finally {
    panel.setInputEnabled(true);
  }
}

function handleStreamEvent(evt, assistantEl, panel) {
  switch (evt.type) {
    case "token":
      panel.appendToken(assistantEl, evt.content);
      break;
    case "citation":
      panel.addCitations(evt.citations);
      break;
    case "done":
      // Nothing extra needed — finalize happens after the stream ends
      break;
    case "error":
      panel.addErrorMessage(evt.detail || "An error occurred.");
      break;
    default:
      console.debug("[YT-RAG] Unknown SSE event type:", evt.type);
  }
}

// ── Video seek for citations ────────────────────────────────────

/**
 * Attempt to seek the currently-playing YouTube video to the timestamp
 * encoded in a citation URL, instead of opening a new tab.
 *
 * Returns true if the seek was handled (caller should preventDefault),
 * false if the link should be opened normally (e.g. citation refers to
 * a different video than the one currently playing).
 */
function trySeekToCitation(url) {
  try {
    const citedVideoId = extractVideoId(url);
    if (citedVideoId !== currentVideoId) {
      return false; // Different video — let the link open normally
    }

    const tMatch = url.match(/[?&]t=(\d+)/);
    if (!tMatch) return false;

    const seconds = parseInt(tMatch[1], 10);
    const video = document.querySelector("video");
    if (!video) return false;

    video.currentTime = seconds;
    if (video.paused) video.play().catch(() => {});

    console.log("[YT-RAG] Seeked to", seconds, "seconds");
    return true;
  } catch (err) {
    console.warn("[YT-RAG] Seek failed:", err);
    return false;
  }
}

// ── Initialization ──────────────────────────────────────────────

function init() {
  if (!window.YTRagPanel) {
    // panel.js failed to load or hasn't run yet — retry shortly
    setTimeout(init, 200);
    return;
  }

  window.YTRagPanel.inject();
  window.YTRagPanel.onSend = sendQuestion;
  window.YTRagPanel.trySeek = trySeekToCitation;

  // Initial check in case the script loads directly on a watch page
  checkForVideoChange();
}

init();