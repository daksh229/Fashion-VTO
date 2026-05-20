// Poll the Django status proxy and update the page in real time:
//   - render Groq's streaming prompt token-by-token in the right pane
//   - swap the spinner for the generated image once Gemini finishes
//   - flag failure / timeout clearly
//
// Polling cadence: 700 ms early on (catches Groq tokens fast), 2.5 s after
// the streaming phase ends (Gemini is opaque, no point burning requests).

(function () {
  "use strict";

  const STATUS_URL = window.TRYON_STATUS_URL;
  if (!STATUS_URL) {
    console.error("poll.js: TRYON_STATUS_URL not set");
    return;
  }

  const heading       = document.getElementById("result-heading");
  const statusLine    = document.getElementById("result-status");
  const pane          = document.getElementById("result-pane");
  const spinner       = document.getElementById("result-spinner");
  const meta          = document.getElementById("result-meta");
  const fitEl         = document.getElementById("result-fit");
  const promptPane    = document.getElementById("prompt-pane");
  const promptTitle   = document.getElementById("prompt-pane-title");
  const promptStage   = document.getElementById("prompt-stage");
  const promptTextEl  = document.getElementById("prompt-text");

  const start  = Date.now();
  const MAX_MS = 5 * 60 * 1000;

  // Track what's already rendered so we don't reflow the DOM on every poll.
  let lastPromptLen   = 0;
  let promptDoneFlag  = false;
  let imageShown      = false;
  const cursorEl      = document.createElement("span");
  cursorEl.className  = "cursor";

  function setStage(text, paneState) {
    promptStage.textContent = text;
    if (paneState) promptPane.dataset.state = paneState;
  }

  function setStatusText(text) { statusLine.textContent = text; }

  function ensureCursor(present) {
    if (present && !cursorEl.isConnected) {
      promptTextEl.appendChild(cursorEl);
    } else if (!present && cursorEl.isConnected) {
      cursorEl.remove();
    }
  }

  function renderPromptText(text, streaming) {
    if (text.length !== lastPromptLen) {
      // Replace text content (cursor is removed by replacing); cheap enough.
      promptTextEl.textContent = text;
      lastPromptLen = text.length;
      // Auto-scroll to keep the latest tokens visible.
      promptTextEl.scrollTop = promptTextEl.scrollHeight;
    }
    ensureCursor(streaming);
  }

  function showImage(url) {
    if (imageShown) return;
    spinner.remove();
    const img = document.createElement("img");
    img.src = url;
    img.alt = "Try-on result";
    img.className = "result-image";
    pane.appendChild(img);
    pane.classList.add("has-image");
    imageShown = true;
  }

  function showError(message) {
    if (spinner.isConnected) spinner.remove();
    heading.textContent = "Try-on failed";
    statusLine.classList.add("error-line");
    setStatusText(message);
    setStage("Aborted.", "failed");
    ensureCursor(false);
  }

  function showFit(data) {
    if (data.fit) {
      fitEl.textContent = JSON.stringify(data.fit, null, 2);
      meta.hidden = false;
    }
  }

  // 6 fast polls catch the streaming phase cleanly; back off afterwards.
  let polls = 0;
  function nextDelay() {
    if (!promptDoneFlag && polls < 30) return 700;
    return 2500;
  }

  async function poll() {
    polls += 1;
    let resp;
    try {
      resp = await fetch(STATUS_URL, { credentials: "same-origin" });
    } catch (e) {
      showError("Network error talking to the server. Please retry.");
      return;
    }

    let data;
    try {
      data = await resp.json();
    } catch (e) {
      showError(`Bad response (HTTP ${resp.status}).`);
      return;
    }

    // --- Streaming prompt rendering ---
    const streaming = !!data.prompt_streaming;
    const promptText = data.prompt_text || "";
    if (streaming || promptText.length > 0 || data.status === "running") {
      if (streaming) {
        setStage("Groq is writing the prompt…", "streaming");
        promptTitle.textContent = "Live prompt (streaming)";
      } else if (data.status === "running" && promptText) {
        setStage("Prompt ready. Gemini is rendering the image…", "image");
        promptTitle.textContent = "Final prompt";
        promptDoneFlag = true;
      } else if (data.status === "running") {
        setStage("Working…", "queued");
      }
      renderPromptText(promptText, streaming);
    } else if (data.status === "queued") {
      setStage("Queued, waiting for a worker…", "queued");
    }

    // --- Terminal states ---
    if (data.status === "done") {
      heading.innerHTML = "Your <span class=\"accent\">try-on</span>";
      setStatusText(`Done in ${Math.round((Date.now() - start) / 1000)}s.`);
      setStage("Done.", "done");
      promptTitle.textContent = "Final prompt";
      ensureCursor(false);
      if (data.image_url) showImage(data.image_url);
      showFit(data);
      return;
    }

    if (data.status === "failed") {
      showError(data.error || "Job failed without a message.");
      return;
    }

    if (Date.now() - start > MAX_MS) {
      showError("Timed out after 5 minutes. The job may still finish in the background.");
      return;
    }

    if (!streaming && data.status === "running") {
      setStatusText(`Rendering image… (${Math.round((Date.now() - start) / 1000)}s)`);
    } else if (streaming) {
      setStatusText(`Writing prompt… (${Math.round((Date.now() - start) / 1000)}s)`);
    }

    setTimeout(poll, nextDelay());
  }

  poll();
})();
