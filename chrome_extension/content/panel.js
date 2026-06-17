/**
 * Chat Panel
 * ===========
 * Stage: Step 5 — Floating chat UI injected into YouTube watch pages.
 *
 * Responsibilities:
 *   - Inject panel DOM once per page load
 *   - Render indexing status banner (queued/running/done/error)
 *   - Render chat message thread (user + assistant bubbles)
 *   - Render citations as clickable timestamp links that seek the video
 *   - Handle collapse/expand and show/hide (launcher button)
 *   - Reset message thread when the video changes
 *
 * This file only handles UI state and DOM — actual API calls (sending a
 * question, streaming the answer) are wired in from content.js, which
 * calls the functions exposed on `window.YTRagPanel`.
 */

(function () {
  "use strict";

  let panelEl = null;
  let launcherEl = null;
  let messagesEl = null;
  let inputEl = null;
  let sendBtnEl = null;
  let statusTextEl = null;
  let logoDotEl = null;
  let bannerContainerEl = null;

  let isCollapsed = false;
  let isHidden = false;
  let currentIndexStatus = "idle"; // idle | starting | queued | running | done | error | disabled | already_indexed

  // ── DOM construction ─────────────────────────────────────────

  function buildPanelHTML() {
    return `
      <div class="ytrag-header" id="ytrag-header">
        <span class="ytrag-logo-dot" id="ytrag-logo-dot"></span>
        <span class="ytrag-title">Ask this video</span>
        <span class="ytrag-status-text" id="ytrag-status-text"></span>
        <button class="ytrag-collapse-btn" id="ytrag-collapse-btn" title="Collapse">&#x2212;</button>
      </div>
      <div class="ytrag-body" id="ytrag-body">
        <div id="ytrag-banner-container"></div>
        <div class="ytrag-messages" id="ytrag-messages">
          <div class="ytrag-empty-state" id="ytrag-empty-state">
            <strong>Ask anything about this video</strong>
            <span>Try: "What is this video about?" or ask a specific question.</span>
          </div>
        </div>
        <div class="ytrag-input-row">
          <textarea
            class="ytrag-input"
            id="ytrag-input"
            placeholder="Ask a question..."
            rows="1"
            disabled
          ></textarea>
          <button class="ytrag-send-btn" id="ytrag-send-btn" disabled title="Send">&#x27A4;</button>
        </div>
      </div>
    `;
  }

  function injectPanel() {
    if (document.getElementById("ytrag-panel")) {
      return; // Already injected (e.g. SPA navigation re-triggered injection)
    }

    panelEl = document.createElement("div");
    panelEl.id = "ytrag-panel";
    panelEl.innerHTML = buildPanelHTML();
    document.body.appendChild(panelEl);

    launcherEl = document.createElement("div");
    launcherEl.id = "ytrag-launcher";
    launcherEl.className = "ytrag-hidden";
    launcherEl.innerHTML = "&#x1F4AC;";
    launcherEl.title = "Open chat";
    document.body.appendChild(launcherEl);

    // Cache references
    messagesEl = document.getElementById("ytrag-messages");
    inputEl = document.getElementById("ytrag-input");
    sendBtnEl = document.getElementById("ytrag-send-btn");
    statusTextEl = document.getElementById("ytrag-status-text");
    logoDotEl = document.getElementById("ytrag-logo-dot");
    bannerContainerEl = document.getElementById("ytrag-banner-container");

    attachEventListeners();
    console.log("[YT-RAG] Chat panel injected.");
  }

  function attachEventListeners() {
    document.getElementById("ytrag-header").addEventListener("click", (e) => {
      // Don't toggle collapse if the click was on the collapse button itself
      // (it has its own handler below, this prevents double-firing)
      if (e.target.id === "ytrag-collapse-btn") return;
      toggleCollapse();
    });

    document.getElementById("ytrag-collapse-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      toggleCollapse();
    });

    launcherEl.addEventListener("click", () => {
      setHidden(false);
    });

    sendBtnEl.addEventListener("click", () => {
      handleSendClick();
    });

    inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSendClick();
      }
    });

    // Auto-grow textarea up to CSS max-height
    inputEl.addEventListener("input", () => {
      inputEl.style.height = "auto";
      inputEl.style.height = Math.min(inputEl.scrollHeight, 80) + "px";
    });
  }

  function handleSendClick() {
    const text = inputEl.value.trim();
    if (!text) return;
    if (typeof window.YTRagPanel.onSend === "function") {
      window.YTRagPanel.onSend(text);
    }
  }

  // ── Collapse / hide ──────────────────────────────────────────

  function toggleCollapse() {
    isCollapsed = !isCollapsed;
    panelEl.classList.toggle("ytrag-collapsed", isCollapsed);
    document.getElementById("ytrag-collapse-btn").innerHTML = isCollapsed ? "&#x25A1;" : "&#x2212;";
  }

  function setHidden(hidden) {
    isHidden = hidden;
    panelEl.classList.toggle("ytrag-hidden", hidden);
    launcherEl.classList.toggle("ytrag-hidden", !hidden);
  }

  // ── Status / indexing banner ─────────────────────────────────

  const STATUS_LABELS = {
    idle:             "",
    starting:         "Starting…",
    queued:           "Queued…",
    running:          "Indexing…",
    done:             "Ready",
    already_indexed:  "Ready",
    error:            "Error",
    disabled:         "Auto-index off",
  };

  function setIndexStatus(status, extra = {}) {
    currentIndexStatus = status;

    logoDotEl.className = "ytrag-logo-dot";
    if (status === "done" || status === "already_indexed") {
      logoDotEl.classList.add("ytrag-ready");
    } else if (status === "running" || status === "starting" || status === "queued") {
      logoDotEl.classList.add("ytrag-indexing");
    } else if (status === "error") {
      logoDotEl.classList.add("ytrag-error");
    }

    let label = STATUS_LABELS[status] !== undefined ? STATUS_LABELS[status] : status;
    if (status === "running" && extra.step) {
      label = `Indexing (${extra.step})…`;
    }
    statusTextEl.textContent = label;

    renderBanner(status, extra);

    const isReady = status === "done" || status === "already_indexed";
    inputEl.disabled = !isReady;
    sendBtnEl.disabled = !isReady;
    inputEl.placeholder = isReady
      ? "Ask a question..."
      : status === "error"
      ? "Indexing failed — try reloading the page"
      : "Waiting for indexing to finish...";
  }

  function renderBanner(status, extra) {
    if (status === "running" || status === "starting" || status === "queued") {
      bannerContainerEl.innerHTML = `
        <div class="ytrag-indexing-banner">
          <span class="ytrag-spinner"></span>
          <span>${STATUS_LABELS[status]}${extra.step ? ` (${extra.step})` : ""} This video is being indexed for the first time.</span>
        </div>
      `;
    } else if (status === "error") {
      bannerContainerEl.innerHTML = `
        <div class="ytrag-error-banner">
          <span>&#9888;</span>
          <span>${extra.error || "Indexing failed. Please try reloading the page."}</span>
        </div>
      `;
    } else {
      bannerContainerEl.innerHTML = "";
    }
  }

  // ── Messages ──────────────────────────────────────────────────

  function clearMessages() {
    messagesEl.innerHTML = `
      <div class="ytrag-empty-state" id="ytrag-empty-state">
        <strong>Ask anything about this video</strong>
        <span>Try: "What is this video about?" or ask a specific question.</span>
      </div>
    `;
  }

  function removeEmptyState() {
    const el = document.getElementById("ytrag-empty-state");
    if (el) el.remove();
  }

  function addUserMessage(text) {
    removeEmptyState();
    const div = document.createElement("div");
    div.className = "ytrag-msg ytrag-msg-user";
    div.textContent = text;
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  /**
   * Add an empty assistant message bubble that will be filled token-by-token.
   * Returns the DOM element so the caller can append tokens to it.
   */
  function addStreamingAssistantMessage() {
    removeEmptyState();
    const div = document.createElement("div");
    div.className = "ytrag-msg ytrag-msg-assistant ytrag-streaming";
    messagesEl.appendChild(div);
    scrollToBottom();
    return div;
  }

  function appendToken(msgEl, token) {
    msgEl.textContent += token;
    scrollToBottom();
  }

  function finalizeAssistantMessage(msgEl) {
    msgEl.classList.remove("ytrag-streaming");
  }

  function addCitations(citations) {
    if (!citations || citations.length === 0) return;

    const container = document.createElement("div");
    container.className = "ytrag-citations";

    for (const c of citations) {
      const link = document.createElement("a");
      link.className = "ytrag-citation";
      link.href = c.url_with_timestamp || "#";
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.innerHTML = `
        <span class="ytrag-citation-badge">${c.index}</span>
        <span class="ytrag-citation-time">${c.timestamp_label || ""}</span>
        <span>${truncate(c.video_title, 30)}</span>
      `;

      // Clicking seeks the current video instead of opening a new tab,
      // when the citation refers to the video currently playing.
      link.addEventListener("click", (e) => {
        const seeked = window.YTRagPanel.trySeek(c.url_with_timestamp);
        if (seeked) e.preventDefault();
      });

      container.appendChild(link);
    }

    messagesEl.appendChild(container);
    scrollToBottom();
  }

  function addErrorMessage(text) {
    removeEmptyState();
    const div = document.createElement("div");
    div.className = "ytrag-msg ytrag-msg-assistant";
    div.style.color = "#ff5c5c";
    div.textContent = text;
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function truncate(text, max) {
    if (!text) return "";
    return text.length > max ? text.slice(0, max) + "…" : text;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setInputEnabled(enabled) {
    inputEl.disabled = !enabled;
    sendBtnEl.disabled = !enabled;
  }

  function clearInput() {
    inputEl.value = "";
    inputEl.style.height = "auto";
  }

  // ── Reset on video change ───────────────────────────────────

  function resetForNewVideo() {
    clearMessages();
    clearInput();
    setIndexStatus("idle");
  }

  // ── Public API ────────────────────────────────────────────────

  window.YTRagPanel = {
    inject: injectPanel,
    setIndexStatus: setIndexStatus,
    addUserMessage: addUserMessage,
    addStreamingAssistantMessage: addStreamingAssistantMessage,
    appendToken: appendToken,
    finalizeAssistantMessage: finalizeAssistantMessage,
    addCitations: addCitations,
    addErrorMessage: addErrorMessage,
    setInputEnabled: setInputEnabled,
    clearInput: clearInput,
    resetForNewVideo: resetForNewVideo,
    getInputValue: () => (inputEl ? inputEl.value.trim() : ""),

    // Set by content.js — called when the user sends a message
    onSend: null,
    // Set by content.js — attempts to seek the YouTube player, returns true/false
    trySeek: () => false,
  };
})();