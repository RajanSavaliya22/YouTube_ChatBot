/**
 * Popup Settings Logic
 * =====================
 * Stage: Step 3 — Settings UI.
 *
 * Persists API URL, API key, and auto-index toggle to chrome.storage.sync
 * (syncs across the user's Chrome installs if they're signed in).
 *
 * Default API URL points to the deployed Render instance so the extension
 * works out of the box without any setup.
 */

const DEFAULT_API_URL = "https://youtube-rag-api-6xhe.onrender.com";

const apiUrlInput = document.getElementById("apiUrl");
const apiKeyInput = document.getElementById("apiKey");
const autoIndexToggle = document.getElementById("autoIndex");
const checkHealthBtn = document.getElementById("checkHealthBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const savedMsg = document.getElementById("savedMsg");

/**
 * Load saved settings into the form on popup open.
 */
async function loadSettings() {
  const settings = await chrome.storage.sync.get({
    apiUrl: DEFAULT_API_URL,
    apiKey: "",
    autoIndex: true,
  });

  apiUrlInput.value = settings.apiUrl;
  apiKeyInput.value = settings.apiKey;
  autoIndexToggle.checked = settings.autoIndex;
}

/**
 * Persist current form values to chrome.storage.sync.
 * Called on every input change (debounced) so settings are never lost.
 */
let saveTimeout = null;
function saveSettings() {
  clearTimeout(saveTimeout);
  saveTimeout = setTimeout(async () => {
    const apiUrl = apiUrlInput.value.trim().replace(/\/$/, ""); // strip trailing slash
    const apiKey = apiKeyInput.value.trim();
    const autoIndex = autoIndexToggle.checked;

    await chrome.storage.sync.set({ apiUrl, apiKey, autoIndex });

    savedMsg.textContent = "Saved";
    setTimeout(() => { savedMsg.textContent = ""; }, 1200);
  }, 300);
}

/**
 * Ping the API's /health endpoint to verify connectivity.
 * Uses the currently-typed URL (not yet-saved value) so users can test
 * before committing to a URL.
 */
async function checkHealth() {
  const apiUrl = apiUrlInput.value.trim().replace(/\/$/, "");
  if (!apiUrl) {
    setStatus("err", "Enter an API URL first");
    return;
  }

  setStatus("", "Checking...");

  try {
    const res = await fetch(`${apiUrl}/health`, { method: "GET" });
    if (!res.ok) {
      setStatus("err", `HTTP ${res.status}`);
      return;
    }
    const data = await res.json();
    if (data.status === "healthy") {
      setStatus("ok", "Connected — healthy");
    } else if (data.status === "degraded") {
      setStatus("ok", "Connected — degraded (some services down)");
    } else {
      setStatus("err", `Status: ${data.status}`);
    }
  } catch (err) {
    setStatus("err", "Connection failed");
    console.error("[YT-RAG][popup] Health check failed:", err);
  }
}

function setStatus(state, text) {
  statusDot.className = "dot" + (state ? ` ${state}` : "");
  statusText.textContent = text;
}

// ── Event listeners ─────────────────────────────────────────

document.addEventListener("DOMContentLoaded", loadSettings);
apiUrlInput.addEventListener("input", saveSettings);
apiKeyInput.addEventListener("input", saveSettings);
autoIndexToggle.addEventListener("change", saveSettings);
checkHealthBtn.addEventListener("click", checkHealth);