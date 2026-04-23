/**
 * plmMSA web UI — pure client.
 *
 * State model:
 *   localStorage.plmmsa.jobs = [{id, submittedAt, label, lastStatus}]
 *   URL params:
 *     ?job=<id>         → select + poll this job
 *     ?seq=<seq>        → prefill submit form
 *     ?jobs=<id1,id2>   → merge these ids into the cache (shareable list)
 *
 * No build step. No framework. Just fetch, setInterval, localStorage.
 */
console.log("plmMSA UI: app.js loaded");
(() => {
  "use strict";

  const LS_KEY = "plmmsa.jobs";
  const POLL_VISIBLE_MS = 5000;
  const POLL_HIDDEN_MS = 60000;
  const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);
  // Only these modes are universal. glocal/q2t/t2q are OTalign-only —
  // we hide them when PLMAlign is the selected aligner.
  const ALIGNER_MODES = {
    plmalign: ["local", "global"],
    otalign: ["local", "global", "glocal", "q2t", "t2q"],
  };
  const DEFAULT_MODE = {
    plmalign: "global",
    otalign: "glocal",
  };

  /* --- localStorage-backed job cache --- */

  function loadJobs() {
    try {
      return JSON.parse(localStorage.getItem(LS_KEY) || "[]");
    } catch {
      return [];
    }
  }

  function saveJobs(jobs) {
    localStorage.setItem(LS_KEY, JSON.stringify(jobs));
  }

  function upsertJob(entry) {
    const jobs = loadJobs();
    const idx = jobs.findIndex((j) => j.id === entry.id);
    if (idx === -1) jobs.unshift(entry);
    else jobs[idx] = { ...jobs[idx], ...entry };
    saveJobs(jobs);
  }

  function clearCache() {
    localStorage.removeItem(LS_KEY);
    // Also drop the current selection + strip ?job=<id> and ?jobs=<...>
    // from the URL so the auto-repopulate on next load doesn't bring
    // the just-cleared ids back.
    currentJobId = null;
    const url = new URL(window.location.href);
    url.searchParams.delete("job");
    url.searchParams.delete("jobs");
    window.history.replaceState({}, "", url);
    resultSection.hidden = true;
    renderJobs();
  }

  /* --- DOM refs --- */

  const $ = (id) => document.getElementById(id);
  const form = $("submit-form");
  const seqInput = $("seq");
  const alignerSel = $("aligner");
  const modeSel = $("mode");
  const kInput = $("k");
  const filterCb = $("filter_by_score");
  const submitBtn = $("submit-btn");
  const jobsList = $("jobs-list");
  const clearBtn = $("clear-btn");
  const resultSection = $("result-section");
  const resultMeta = $("result-meta");
  const resultStats = $("result-stats");
  const resultA3m = $("result-a3m");
  const downloadBtn = $("download-btn");
  const copyBtn = $("copy-btn");
  const versionEl = $("version");

  let currentJobId = null;
  let renderedMsaJobId = null;
  let renderedMsaPayload = null;

  /* --- URL query-param state --- */

  function readURL() {
    const params = new URLSearchParams(window.location.search);
    return {
      job: params.get("job"),
      seq: params.get("seq"),
      jobs: (params.get("jobs") || "").split(",").filter(Boolean),
    };
  }

  function setURLJob(id) {
    const url = new URL(window.location.href);
    if (id) url.searchParams.set("job", id);
    else url.searchParams.delete("job");
    window.history.replaceState({}, "", url);
  }

  /* --- Mode picker auto-hide --- */

  function refreshModeOptions({ applyDefault = false } = {}) {
    const aligner = alignerSel.value;
    const allowed = new Set(ALIGNER_MODES[aligner] || ["local", "global"]);
    for (const opt of modeSel.options) {
      const only = opt.dataset.only;
      opt.hidden = !!only && only !== aligner;
      opt.disabled = opt.hidden;
    }
    if (applyDefault) {
      modeSel.value = DEFAULT_MODE[aligner] || "global";
    } else if (!allowed.has(modeSel.value) || modeSel.options[modeSel.selectedIndex]?.disabled) {
      modeSel.value = DEFAULT_MODE[aligner] || "global";
    }
  }

  /* --- Submit --- */

  async function onSubmit(ev) {
    ev.preventDefault();
    submitBtn.disabled = true;
    submitBtn.textContent = "Submitting…";
    try {
      const seq = parseSeqInput(seqInput.value);
      const body = {
        sequences: [seq],
        aligner: alignerSel.value,
        mode: modeSel.value,
        k: parseInt(kInput.value || "1000", 10),
        filter_by_score: filterCb.checked,
      };
      const resp = await fetch("/v2/msa", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.message || `HTTP ${resp.status}`);
      }
      const j = await resp.json();
      upsertJob({
        id: j.job_id,
        submittedAt: Date.now(),
        label: seq.slice(0, 20) + (seq.length > 20 ? "…" : ""),
        sequence: seq,
        lastStatus: j.status,
      });
      selectJob(j.job_id);
      renderJobs();
      kickPoll();
    } catch (e) {
      alert(`Submit failed: ${e.message}`);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Submit";
    }
  }

  function parseSeqInput(raw) {
    const s = raw.trim();
    if (!s) throw new Error("sequence is empty");
    if (s.startsWith(">")) {
      return s
        .split("\n")
        .filter((l) => !l.startsWith(">"))
        .map((l) => l.trim())
        .join("");
    }
    return s.replace(/\s+/g, "");
  }

  /* --- Polling loop --- */

  let pollTimer = null;

  function kickPoll() {
    if (pollTimer) clearTimeout(pollTimer);
    pollTimer = setTimeout(tick, 0);
  }

  async function tick() {
    const jobs = loadJobs();
    const active = jobs.filter((j) => !TERMINAL.has(j.lastStatus));
    await Promise.all(active.map(pollOne));
    renderJobs();
    if (currentJobId) await renderResult();
    const interval = document.hidden ? POLL_HIDDEN_MS : POLL_VISIBLE_MS;
    pollTimer = setTimeout(tick, interval);
  }

  async function pollOne(j) {
    try {
      const r = await fetch(`/v2/msa/${encodeURIComponent(j.id)}`);
      if (!r.ok) return;
      const body = await r.json();
      hydrateFromBody(j, body);
    } catch {
      /* transient network; try again next tick */
    }
  }

  /** Update the cached entry from a /v2/msa/<id> response. Idempotent;
   *  callers invoke this both from the poll loop (active jobs) and
   *  from a one-shot on-load hydration pass (terminal jobs pulled via
   *  `?job=<id>` need their sequence populated even though they'll
   *  never be polled again). */
  function hydrateFromBody(j, body) {
    const update = { id: j.id, lastStatus: body.status };
    if (!j.sequence) {
      const req = body.request || {};
      const seq = (req.sequences || [])[0] || "";
      if (seq) {
        update.sequence = seq;
        if (!j.label || j.label === "shared" || j.label === "imported") {
          update.label = seq.slice(0, 20) + (seq.length > 20 ? "…" : "");
        }
      }
    }
    // Shared / imported jobs start with Date.now() as a placeholder;
    // replace with the actual server-side creation time so the list
    // shows when the job was really submitted, not when the URL was
    // opened. body.created_at is epoch seconds → multiply to ms.
    const serverCreated = body.created_at;
    if (typeof serverCreated === "number" && serverCreated > 0) {
      update.submittedAt = Math.floor(serverCreated * 1000);
    }
    upsertJob(update);
  }

  /** Fire a one-shot fetch for every cached job that's missing its
   *  sequence — runs on page load so terminal jobs loaded via
   *  `?job=<id>` get hydrated without waiting for the next poll
   *  (which skips terminal jobs). */
  async function hydrateMissingSequences() {
    const jobs = loadJobs();
    const missing = jobs.filter((j) => !j.sequence);
    if (missing.length === 0) return;
    await Promise.all(
      missing.map(async (j) => {
        try {
          const r = await fetch(`/v2/msa/${encodeURIComponent(j.id)}`);
          if (!r.ok) return;
          const body = await r.json();
          hydrateFromBody(j, body);
        } catch {
          /* transient; next page load tries again */
        }
      })
    );
    renderJobs();
  }

  /* --- Render --- */

  function renderJobs() {
    const jobs = loadJobs();
    jobsList.innerHTML = "";
    if (jobs.length === 0) {
      jobsList.innerHTML =
        '<li class="empty">No jobs yet. Submit a sequence above.</li>';
      return;
    }
    for (const j of jobs) {
      const li = document.createElement("li");
      if (j.id === currentJobId) li.classList.add("selected");
      const when = new Date(j.submittedAt).toLocaleString();
      const fullSeq = j.sequence || "";
      // Always prefer a fresh slice of the cached sequence over a stale
      // "shared" / "imported" label. Hydration fills `sequence` on the
      // first poll tick, but the renderer may run before that — if we
      // already have the sequence, use its prefix.
      const display = fullSeq
        ? fullSeq.slice(0, 20) + (fullSeq.length > 20 ? "…" : "")
        : (j.label || "job");
      // `title` triggers the browser's native tooltip (OS-rendered
      // floating element, not a DOM child), so full sequences of any
      // length display without tripping layout. Clicking the label
      // copies the full sequence to clipboard.
      li.innerHTML = `
        <div>
          <strong class="seq-label" title="${escapeAttr(fullSeq || j.id)}">
            ${escapeHtml(display)}
          </strong>
          <div class="job-id">${escapeHtml(j.id)}</div>
          <div class="job-id">submitted ${when}</div>
        </div>
        <span class="status-pill status-${j.lastStatus || "queued"}">
          ${escapeHtml(j.lastStatus || "queued")}
        </span>
      `;
      const seqLabel = li.querySelector(".seq-label");
      if (seqLabel) {
        seqLabel.addEventListener("click", (ev) => {
          ev.stopPropagation();
          if (!fullSeq) return;
          navigator.clipboard.writeText(fullSeq).then(() => {
            const original = seqLabel.textContent;
            seqLabel.textContent = "copied ✓";
            setTimeout(() => (seqLabel.textContent = original), 900);
          });
        });
      }
      li.addEventListener("click", () => {
        selectJob(j.id);
        renderResult();
      });
      jobsList.appendChild(li);
    }
  }

  function selectJob(id) {
    currentJobId = id;
    setURLJob(id);
    renderJobs();
  }

  async function renderResult() {
    if (!currentJobId) {
      resultSection.hidden = true;
      return;
    }
    try {
      const r = await fetch(`/v2/msa/${encodeURIComponent(currentJobId)}`);
      if (!r.ok) return;
      const body = await r.json();
      resultSection.hidden = false;
      resultMeta.innerHTML = `
        <strong>${escapeHtml(currentJobId)}</strong> ·
        status <code>${escapeHtml(body.status)}</code>
      `;
      resultStats.textContent = JSON.stringify(body.result?.stats || {}, null, 2);
      const a3m = body.result?.payload || "";
      const preview = a3m.split("\n").slice(0, 80).join("\n");

      if (a3m) {
        const container = document.getElementById("msa-viewer-container");
        if (renderedMsaJobId !== currentJobId || renderedMsaPayload !== a3m) {
          const seqs = parseA3mForViewer(a3m);
          renderMsaViewer(container, seqs);
          renderedMsaJobId = currentJobId;
          renderedMsaPayload = a3m;
        }
      } else {
        const container = document.getElementById("msa-viewer-container");
        if (container) container.innerHTML = "";
        renderedMsaJobId = currentJobId;
        renderedMsaPayload = "";
      }

      resultA3m.textContent = preview || "(no A3M yet)";
      if (a3m) {
        downloadBtn.href = `data:text/plain;charset=utf-8,${encodeURIComponent(a3m)}`;
        downloadBtn.download = `${currentJobId}.a3m`;
        downloadBtn.hidden = false;
      } else {
        downloadBtn.hidden = true;
      }
    } catch {
      /* ignore transient */
    }
  }

  function parseA3mForViewer(text) {
    const seqs = [];
    let current = null;
    for (const rawLine of String(text).split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line) continue;
      if (line.startsWith(">")) {
        const label = line.slice(1).trim() || `seq_${seqs.length + 1}`;
        current = {
          id: label.split(/\s+/)[0] || `seq_${seqs.length + 1}`,
          name: label,
          seq: "",
        };
        seqs.push(current);
      } else if (current) {
        current.seq += line.replace(/[a-z.]/g, "");
      }
    }
    return seqs;
  }

  function renderMsaViewer(container, seqs) {
    container.innerHTML = "";
    if (seqs.length === 0) return;

    const pageSize = 199;
    let page = 0;
    const query = seqs[0];
    const hits = seqs.slice(1);
    const pageCount = Math.max(1, Math.ceil(hits.length / pageSize));

    const wrapper = document.createElement("div");
    wrapper.className = "static-msa-viewer";
    const controls = document.createElement("div");
    controls.className = "static-msa-controls";
    const prevBtn = document.createElement("button");
    prevBtn.type = "button";
    prevBtn.className = "msa-page-btn";
    prevBtn.textContent = "<";
    const pageLabel = document.createElement("span");
    pageLabel.className = "msa-page-label";
    const nextBtn = document.createElement("button");
    nextBtn.type = "button";
    nextBtn.className = "msa-page-btn";
    nextBtn.textContent = ">";
    const fullBtn = document.createElement("button");
    fullBtn.type = "button";
    fullBtn.className = "msa-fullscreen-btn";
    fullBtn.title = "Full screen";
    fullBtn.setAttribute("aria-label", "Full screen");
    setFullscreenButtonIcon(fullBtn, false);
    controls.appendChild(prevBtn);
    controls.appendChild(pageLabel);
    controls.appendChild(nextBtn);
    controls.appendChild(fullBtn);

    const table = document.createElement("div");
    table.className = "static-msa-table";
    wrapper.appendChild(controls);
    wrapper.appendChild(table);
    container.appendChild(wrapper);

    function drawPage() {
      table.innerHTML = "";
      renderMsaRow(table, query, "static-msa-query");
      const start = page * pageSize;
      const rows = hits.slice(start, start + pageSize);
      for (const seq of rows) renderMsaRow(table, seq);

      prevBtn.disabled = page === 0;
      nextBtn.disabled = page >= pageCount - 1;
      const firstHit = hits.length === 0 ? 0 : start + 1;
      const lastHit = Math.min(start + rows.length, hits.length);
      pageLabel.textContent =
        hits.length === 0
          ? "query only"
          : `${firstHit}-${lastHit} of ${hits.length}`;
    }

    prevBtn.addEventListener("click", () => {
      if (page > 0) {
        page--;
        drawPage();
      }
    });
    nextBtn.addEventListener("click", () => {
      if (page < pageCount - 1) {
        page++;
        drawPage();
      }
    });
    fullBtn.addEventListener("click", () => toggleMsaFullscreen(wrapper));
    document.addEventListener("fullscreenchange", () => {
      const isFullScreen = document.fullscreenElement === wrapper;
      fullBtn.title = isFullScreen ? "Exit full screen" : "Full screen";
      fullBtn.setAttribute("aria-label", fullBtn.title);
      setFullscreenButtonIcon(fullBtn, isFullScreen);
    });

    drawPage();
  }

  async function toggleMsaFullscreen(wrapper) {
    if (document.fullscreenElement === wrapper) {
      await document.exitFullscreen();
    } else if (wrapper.requestFullscreen) {
      await wrapper.requestFullscreen();
    }
  }

  function setFullscreenButtonIcon(button, isFullScreen) {
    button.innerHTML = isFullScreen
      ? '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 5V3h2M11 3h2v2M13 11v2h-2M5 13H3v-2" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>'
      : '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3H3v3M10 3h3v3M13 10v3h-3M3 10v3h3" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
  }

  function renderMsaRow(parent, seq, extraClass = "") {
    const row = document.createElement("div");
    row.className = "static-msa-row";
    if (extraClass) row.classList.add(extraClass);

    const label = document.createElement("span");
    label.className = "static-msa-label";
    label.textContent = seq.name || seq.id || "";
    label.title = label.textContent;

    const body = document.createElement("span");
    body.className = "static-msa-seq";
    renderColoredSequence(body, seq.seq || "");

    row.appendChild(label);
    row.appendChild(body);
    parent.appendChild(row);
  }

  function renderColoredSequence(container, seq) {
    const fragment = document.createDocumentFragment();
    for (const residue of seq) {
      const cell = document.createElement("span");
      cell.className = `msa-residue msa-aa-${residueClass(residue)}`;
      cell.textContent = residue;
      fragment.appendChild(cell);
    }
    container.appendChild(fragment);
  }

  function residueClass(residue) {
    switch (String(residue).toUpperCase()) {
      case "A":
      case "I":
      case "L":
      case "M":
      case "F":
      case "W":
      case "V":
        return "hydrophobic";
      case "D":
      case "E":
        return "acidic";
      case "K":
      case "R":
        return "basic";
      case "H":
      case "Y":
        return "aromatic";
      case "S":
      case "T":
      case "N":
      case "Q":
        return "polar";
      case "C":
        return "cysteine";
      case "G":
        return "glycine";
      case "P":
        return "proline";
      case "-":
        return "gap";
      default:
        return "other";
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  /** Same escape, but keep newlines out of attribute values. The
   *  `title` attribute shows the browser's native tooltip; newlines
   *  collapse so a 1000-residue sequence renders on one (very long)
   *  line rather than breaking out of the tooltip. */
  function escapeAttr(s) {
    return escapeHtml(String(s).replace(/\s+/g, ""));
  }

  /* --- Bootstrap --- */

  async function loadVersion() {
    try {
      const r = await fetch("/v2/version");
      if (!r.ok) return;
      const v = await r.json();
      versionEl.textContent = `v${v.plmmsa}`;
    } catch {
      versionEl.textContent = "";
    }
  }

  /** Fill the footer CF-compat URLs with the fully-qualified host so
   *  users can copy them directly (instead of pasting only the path
   *  and wondering what host prefix belongs in front). */
  function fillCfUrls() {
    const origin = window.location.origin;
    const plmmsa = document.getElementById("cf-url-plmmsa");
    const otalign = document.getElementById("cf-url-otalign");
    if (plmmsa) plmmsa.textContent = `${origin}/v2/colabfold/plmmsa`;
    if (otalign) otalign.textContent = `${origin}/v2/colabfold/otalign`;
  }

  function init() {
    console.log("plmMSA UI: init() running");
    const q = readURL();

    if (q.seq) seqInput.value = q.seq;

    // Merge any ?jobs= ids into the cache so a shared dashboard URL
    // repopulates in the new browser.
    for (const id of q.jobs) {
      upsertJob({
        id,
        submittedAt: Date.now(),
        label: "imported",
        lastStatus: "queued",
      });
    }

    if (q.job) {
      upsertJob({
        id: q.job,
        submittedAt: Date.now(),
        label: "shared",
        lastStatus: "queued",
      });
      selectJob(q.job);
    }

    refreshModeOptions({ applyDefault: true });
    alignerSel.addEventListener("change", () => refreshModeOptions({ applyDefault: true }));
    form.addEventListener("submit", onSubmit);
    clearBtn.addEventListener("click", clearCache);
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(window.location.href);
      copyBtn.textContent = "Copied";
      setTimeout(() => (copyBtn.textContent = "Copy URL"), 1200);
    });
    document.addEventListener("visibilitychange", kickPoll);

    loadVersion();
    fillCfUrls();
    renderJobs();
    // One-shot hydration for jobs loaded from the cache or a shared URL
    // — terminal jobs would otherwise never be fetched and stay tagged
    // "shared"/"imported" with no sequence.
    hydrateMissingSequences();
    if (currentJobId) renderResult();
    kickPoll();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
