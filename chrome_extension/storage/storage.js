/**
 * Shared Storage Helper
 * =======================
 * Used by background.js and content.js to read the same settings
 * the popup writes to chrome.storage.sync.
 *
 * Centralizing this avoids duplicating the default values in multiple files.
 */

const DEFAULT_SETTINGS = {
  apiUrl: "https://youtube-rag-api-6xhe.onrender.com",
  apiKey: "",
  autoIndex: true,
};

/**
 * Get current settings, falling back to defaults for any missing keys.
 * Safe to call from background or content script contexts.
 */
async function getSettings() {
  return chrome.storage.sync.get(DEFAULT_SETTINGS);
}

// Export for use in service worker (importScripts) or content script contexts.
// In MV3 service workers, this file should be loaded via static import or
// importScripts() — see background.js for usage.
if (typeof module !== "undefined") {
  module.exports = { getSettings, DEFAULT_SETTINGS };
}