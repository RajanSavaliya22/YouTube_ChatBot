/**
 * Content Script
 * ===============
 * Stage: Step 2 — Video detection + SPA navigation handling.
 *
 * YouTube is a Single Page App — clicking a new video does NOT trigger a
 * full page reload. The URL changes via the History API, and YouTube fires
 * a custom event `yt-navigate-finish` when navigation completes.
 *
 * Strategy:
 *   1. Listen for `yt-navigate-finish` (YouTube's own event — most reliable)
 *   2. Fallback: MutationObserver on document.title (catches edge cases)
 *   3. Extract video_id from the URL on every navigation
 *   4. Only fire onVideoChanged() when the video_id actually changes
 *      (avoids re-triggering on query param changes like `&t=123s`)
 */

console.log("[YT-RAG] Content script injected on:", window.location.href);

let currentVideoId = null;

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
 * This is the hook point for Step 4 (triggering indexing) and Step 5 (panel injection).
 */
function onVideoChanged(videoId, url) {
  console.log("[YT-RAG] Video changed →", videoId, "|", url);

  // Notify the background service worker so it can trigger indexing (Step 4)
  chrome.runtime.sendMessage({
    type: "VIDEO_CHANGED",
    videoId: videoId,
    url: url,
  }).catch((err) => {
    // Background worker may not be listening yet — safe to ignore during early steps
    console.debug("[YT-RAG] sendMessage failed (expected until Step 4):", err.message);
  });

  // Panel injection/update will be added in Step 5
}

/**
 * Check the current URL and fire onVideoChanged if the video_id changed.
 */
function checkForVideoChange() {
  const url = window.location.href;
  const videoId = extractVideoId(url);

  if (!videoId) {
    // Not a watch page (e.g. homepage, search results) — nothing to do
    return;
  }

  if (videoId !== currentVideoId) {
    currentVideoId = videoId;
    onVideoChanged(videoId, url);
  }
}

// ── Primary detection: YouTube's own SPA navigation event ─────
// Fired by YouTube's router after a client-side navigation completes.
// This is the most reliable signal — fires exactly once per navigation.
document.addEventListener("yt-navigate-finish", () => {
  checkForVideoChange();
});

// ── Fallback detection: MutationObserver on <title> ────────────
// YouTube updates document.title almost immediately after a video change,
// even slightly before yt-navigate-finish in some cases. Acts as a safety
// net in case the custom event doesn't fire (e.g. very fast back/forward nav).
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

// ── Initial check on script injection ──────────────────────────
// Handles the case where the content script loads directly on a watch page
// (e.g. user pastes a YouTube URL or opens a link in a new tab).
checkForVideoChange();

// ── Temporary: log indexing status updates from background.js ──
// Step 5 will replace this with actual chat panel UI updates.
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "INDEX_STATUS") {
    console.log(
      `[YT-RAG] Index status: ${message.status}`,
      message.step ? `(step: ${message.step})` : "",
      message
    );
  }
});