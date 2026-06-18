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
  let charCounterEl = null;

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
          <div class="ytrag-input-wrap">
            <textarea
              class="ytrag-input"
              id="ytrag-input"
              placeholder="Ask a question..."
              rows="1"
              maxlength="500"
              disabled
            ></textarea>
            <span class="ytrag-char-counter" id="ytrag-char-counter"></span>
          </div>
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
    charCounterEl = document.getElementById("ytrag-char-counter");

    attachEventListeners();
    loadUIState();
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

    // Auto-grow textarea, update char counter, and validate length on every keystroke
    inputEl.addEventListener("input", () => {
      inputEl.style.height = "auto";
      inputEl.style.height = Math.min(inputEl.scrollHeight, 80) + "px";
      updateInputValidation();
    });
  }

  // Matches the backend's QueryRequest validation (min_length=3, max_length=500)
  // so the user gets instant feedback instead of a round-trip error.
  const MIN_QUESTION_LENGTH = 3;
  const MAX_QUESTION_LENGTH = 500;
  const COUNTER_WARNING_THRESHOLD = 400; // start showing the counter once getting close to the cap

  /**
   * Re-evaluate the current input length against MIN/MAX and update:
   *  - the character counter (hidden unless near the limit or too short)
   *  - the send button's disabled state (in addition to the indexing-ready gate)
   * Called on every keystroke.
   */
  function updateInputValidation() {
    const length = inputEl.value.trim().length;
    const tooShort = length > 0 && length < MIN_QUESTION_LENGTH;
    const tooLong = length > MAX_QUESTION_LENGTH; // maxlength attr should prevent this, but double-check
    const nearLimit = length >= COUNTER_WARNING_THRESHOLD;

    if (nearLimit || tooShort) {
      charCounterEl.textContent = tooShort
        ? `${length}/${MIN_QUESTION_LENGTH} min`
        : `${length}/${MAX_QUESTION_LENGTH}`;
      charCounterEl.classList.toggle("ytrag-char-counter-warn", tooLong || (nearLimit && length >= MAX_QUESTION_LENGTH - 20));
      charCounterEl.classList.add("ytrag-char-counter-visible");
    } else {
      charCounterEl.classList.remove("ytrag-char-counter-visible");
    }

    // Only gate the send button on length here if indexing is already done —
    // setIndexStatus() owns the disabled state while indexing is in progress,
    // and we don't want to fight it by enabling the button mid-index just
    // because the text is valid length.
    if (currentIndexStatus === "done" || currentIndexStatus === "already_indexed") {
      sendBtnEl.disabled = length === 0 || tooShort || tooLong;
    }
  }

  function handleSendClick() {
    const text = inputEl.value.trim();
    if (text.length < MIN_QUESTION_LENGTH) {
      // Belt-and-suspenders: button should already be disabled in this case,
      // but guard directly here too (e.g. programmatic Enter-key submission).
      return;
    }
    if (text.length > MAX_QUESTION_LENGTH) {
      return;
    }
    if (typeof window.YTRagPanel.onSend === "function") {
      window.YTRagPanel.onSend(text);
    }
  }

  // ── Collapse / hide ──────────────────────────────────────────
  //
  // State is persisted to chrome.storage.local (device-local, not synced —
  // this is a UI preference, not data the user would want to roam across
  // machines) so the panel stays collapsed/hidden across video changes
  // and page reloads, instead of resetting every time content.js re-injects.

  const UI_STATE_KEY = "ytrag_ui_state";

  function saveUIState() {
    chrome.storage.local.set({
      [UI_STATE_KEY]: { isCollapsed, isHidden },
    });
  }

  async function loadUIState() {
    const result = await chrome.storage.local.get(UI_STATE_KEY);
    const saved = result[UI_STATE_KEY];
    if (!saved) return;

    isCollapsed = !!saved.isCollapsed;
    isHidden = !!saved.isHidden;

    panelEl.classList.toggle("ytrag-collapsed", isCollapsed);
    document.getElementById("ytrag-collapse-btn").innerHTML = isCollapsed ? "&#x25A1;" : "&#x2212;";
    panelEl.classList.toggle("ytrag-hidden", isHidden);
    launcherEl.classList.toggle("ytrag-hidden", !isHidden);
  }

  function toggleCollapse() {
    isCollapsed = !isCollapsed;
    panelEl.classList.toggle("ytrag-collapsed", isCollapsed);
    document.getElementById("ytrag-collapse-btn").innerHTML = isCollapsed ? "&#x25A1;" : "&#x2212;";
    saveUIState();
  }

  function setHidden(hidden) {
    isHidden = hidden;
    panelEl.classList.toggle("ytrag-hidden", hidden);
    launcherEl.classList.toggle("ytrag-hidden", !hidden);
    saveUIState();
  }

  function toggleHidden() {
    setHidden(!isHidden);
  }

  // ── Status / indexing banner ─────────────────────────────────

  const STATUS_LABELS = {
    idle:             "",
    starting:         "Preparing…",
    queued:           "Preparing…",
    running:          "Preparing…",
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
    statusTextEl.textContent = label;

    renderBanner(status, extra);

    const isReady = status === "done" || status === "already_indexed";
    inputEl.disabled = !isReady;
    // When indexing finishes, the send button should only enable if the
    // current input is also a valid length — updateInputValidation() is
    // the single source of truth for that once isReady is true.
    sendBtnEl.disabled = !isReady;
    if (isReady) {
      updateInputValidation();
    }
    inputEl.placeholder = isReady
      ? "Ask a question..."
      : status === "error"
      ? "Something went wrong — click Retry above"
      : "Getting video ready…";
  }

  function renderBanner(status, extra) {
    if (status === "error") {
      bannerContainerEl.innerHTML = `
        <div class="ytrag-error-banner">
          <span>&#9888;</span>
          <span>${extra.error || "Indexing failed. Please try reloading the page."}</span>
          <a href="#" class="ytrag-retry-link ytrag-retry-index-link">Retry</a>
        </div>
      `;
      const retryLink = bannerContainerEl.querySelector(".ytrag-retry-index-link");
      if (retryLink) {
        retryLink.addEventListener("click", (e) => {
          e.preventDefault();
          if (typeof window.YTRagPanel.onRetryIndex === "function") {
            window.YTRagPanel.onRetryIndex();
          }
        });
      }
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
   * Add a "thinking" placeholder bubble shown immediately after the user
   * sends a question, before the first token arrives from the server.
   * Replaced in-place once the first real token shows up (see appendToken).
   */
  function addThinkingMessage() {
    removeEmptyState();
    const div = document.createElement("div");
    div.className = "ytrag-msg ytrag-msg-assistant ytrag-thinking";
    div.innerHTML = `
      <span class="ytrag-thinking-dot"></span>
      <span class="ytrag-thinking-dot"></span>
      <span class="ytrag-thinking-dot"></span>
    `;
    messagesEl.appendChild(div);
    scrollToBottom();
    return div;
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
    // First token arriving — strip the "thinking" placeholder markup and
    // start rendering real content in its place, without inserting a new bubble.
    if (msgEl.classList.contains("ytrag-thinking")) {
      msgEl.classList.remove("ytrag-thinking");
      msgEl.classList.add("ytrag-streaming");
      msgEl.innerHTML = "";
    }
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

  /**
   * Show an error bubble in the message thread.
   * If onRetry is provided, shows a "Retry" link that re-sends the
   * original question when clicked.
   */
  function addErrorMessage(text, onRetry = null) {
    removeEmptyState();
    const div = document.createElement("div");
    div.className = "ytrag-msg ytrag-msg-assistant ytrag-msg-error";

    const textSpan = document.createElement("span");
    textSpan.textContent = text;
    div.appendChild(textSpan);

    if (typeof onRetry === "function") {
      const retryLink = document.createElement("a");
      retryLink.className = "ytrag-retry-link";
      retryLink.textContent = "Retry";
      retryLink.href = "#";
      retryLink.addEventListener("click", (e) => {
        e.preventDefault();
        div.remove();
        onRetry();
      });
      div.appendChild(document.createElement("br"));
      div.appendChild(retryLink);
    }

    messagesEl.appendChild(div);
    scrollToBottom();
  }

  /**
   * Remove a specific message bubble from the thread (e.g. an empty
   * "thinking" placeholder that never received any tokens before erroring).
   */
  function removeMessage(msgEl) {
    if (msgEl && msgEl.parentNode) {
      msgEl.remove();
    }
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
    charCounterEl.classList.remove("ytrag-char-counter-visible");
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
    addThinkingMessage: addThinkingMessage,
    addStreamingAssistantMessage: addStreamingAssistantMessage,
    appendToken: appendToken,
    finalizeAssistantMessage: finalizeAssistantMessage,
    addCitations: addCitations,
    addErrorMessage: addErrorMessage,
    removeMessage: removeMessage,
    setInputEnabled: setInputEnabled,
    clearInput: clearInput,
    resetForNewVideo: resetForNewVideo,
    toggleHidden: toggleHidden,
    getInputValue: () => (inputEl ? inputEl.value.trim() : ""),

    // Set by content.js — called when the user sends a message
    onSend: null,
    // Set by content.js — attempts to seek the YouTube player, returns true/false
    trySeek: () => false,
    // Set by content.js — called when the user clicks "Retry" on an indexing error
    onRetryIndex: null,
  };
})();