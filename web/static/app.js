/* DataNarrate web UI. Vanilla JS: upload -> configure -> run with live progress -> results tabs. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 4) => (x === null || x === undefined || Number.isNaN(x) ? "n/a" : Number(x).toFixed(d));
const fmtP = (p) => (p === null || p === undefined || Number.isNaN(p) ? "n/a" : (p < 0.0001 ? "< 0.0001" : Number(p).toFixed(4)));
const signed = (x, d = 4) => (x === null || x === undefined ? "n/a" : (Math.abs(x) < 0.5 * 10 ** -d ? (0).toFixed(d) : (x > 0 ? "+" : "") + Number(x).toFixed(d)));
const pretty = (name) => ({
  logistic_regression: "Logistic regression", random_forest: "Random forest", gradient_boosting: "Gradient boosting",
  svm: "SVM", knn: "KNN", naive_bayes: "Naive Bayes", mlp: "Neural network (MLP)", voting_ensemble: "Voting ensemble",
}[name] || name);

const state = { dataset: null, job: null, stages: [], result: null, rendered: {} };

/* ---------------- theme ---------------- */
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem("dn-theme", t); } catch (e) {}
  if (state.result) { state.rendered = {}; showTab(currentTab()); }
}
(function initTheme() {
  let t = null;
  try { t = localStorage.getItem("dn-theme"); } catch (e) {}
  if (!t) t = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", t);
})();
$("#themeToggle").addEventListener("click", () =>
  applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"));

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function series() {   // fixed categorical order, validated palette
  return document.documentElement.getAttribute("data-theme") === "dark"
    ? ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
    : ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
}
function layout(extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: css("--f-ui"), color: css("--fg"), size: 12.5 },
    margin: { l: 56, r: 16, t: 16, b: 48 },
    xaxis: { gridcolor: css("--line"), zerolinecolor: css("--line-strong"), automargin: true },
    yaxis: { gridcolor: css("--line"), zerolinecolor: css("--line-strong"), automargin: true },
    legend: { orientation: "h", y: -0.22 },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--line-strong"), font: { color: css("--fg") } },
  }, extra);
}
function plot(el, data, lay) {
  if (!window.Plotly) { el.innerHTML = "<p class='muted small'>Charts could not load.</p>"; return; }
  if (lay && lay.height) el.style.height = `${lay.height}px`;   // the container must grow with tall charts
  Plotly.newPlot(el, data, layout(lay), { displayModeBar: false, responsive: true });
}

/* ---------------- navigation ---------------- */
$$("[data-view]").forEach((b) => b.addEventListener("click", (e) => {
  e.preventDefault();
  const v = b.dataset.view;
  $$(".navbtn").forEach((n) => n.classList.toggle("active", n.dataset.view === v));
  $("#view-analyze").hidden = v !== "analyze";
  $("#view-history").hidden = v !== "history";
  if (v === "history") loadHistory();
}));

/* ---------------- upload ---------------- */
const dz = $("#dropzone");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadFile(f); });
$("#fileInput").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) uploadFile(f); });
$$("[data-sample]").forEach((b) => b.addEventListener("click", () => loadSample(b.dataset.sample)));

async function api(url, opts) {
  const r = await fetch(url, opts);
  let body = null;
  try { body = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error((body && body.detail) || `Request failed (${r.status})`);
  return body;
}
function busy(on, label) {
  dz.classList.toggle("busy", on);
  $("strong", dz).textContent = on ? (label || "Reading the file...") : "Drop a CSV here or click to browse";
}
async function uploadFile(file) {
  $("#uploadError").hidden = true;
  busy(true, `Reading ${file.name}...`);
  const fd = new FormData(); fd.append("file", file);
  try { showDataset(await api("/api/datasets", { method: "POST", body: fd })); }
  catch (e) { showUploadError(e.message); }
  finally { busy(false); $("#fileInput").value = ""; }
}
async function loadSample(name) {
  $("#uploadError").hidden = true;
  busy(true, "Loading the sample dataset...");
  try { showDataset(await api(`/api/datasets/sample/${name}`, { method: "POST" })); }
  catch (e) { showUploadError(e.message); }
  finally { busy(false); }
}
function showUploadError(msg) { const el = $("#uploadError"); el.textContent = msg; el.hidden = false; }

function kpi(v, l) { return `<div class="kpi"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`; }
function table(cols, rows, opts = {}) {
  const head = cols.map((c) => `<th class="${c.num ? "num" : ""}">${esc(c.label)}</th>`).join("");
  const body = rows.map((r, i) => `<tr class="${opts.rowClass ? opts.rowClass(r, i) : ""}">` + cols.map((c) => {
    const v = typeof c.value === "function" ? c.value(r) : r[c.key];
    return `<td class="${c.num ? "num" : ""}">${c.html ? v : esc(v)}</td>`;
  }).join("") + "</tr>").join("");
  return `<div class="tablewrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function showDataset(d) {
  state.dataset = d;
  $("#datasetSection").hidden = false;
  $("#results").hidden = true; $("#progressCard").hidden = true; $("#runError").hidden = true;
  $("#datasetTitle").textContent = d.name;
  $("#datasetKpis").innerHTML = kpi(d.n_rows.toLocaleString(), "Rows") + kpi(d.n_cols, "Columns") +
    kpi(d.missing.pct_cells + "%", "Missing cells") + kpi(d.missing.pct_rows + "%", "Rows with a missing value");

  const m = d.missing;
  let html = `<ul class="plain">${m.summary.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>`;
  if (!m.table.length) html += `<div class="notice good">No missing values found, not even hidden placeholders such as "?".</div>`;
  else {
    html += `<div class="grid2"><div>${table([
      { label: "Column", key: "Column" }, { label: "Type", key: "Type" },
      { label: "Empty", key: "Empty cells", num: true }, { label: "Placeholders", key: "Hidden placeholders", num: true },
      { label: "Missing %", key: "Missing %", num: true }, { label: "Severity", key: "Severity" },
    ], m.table)}</div><div class="chart" id="missingChart"></div></div>`;
  }
  m.warnings.forEach((w) => { html += `<div class="notice warn">${esc(w)}</div>`; });
  $("#missingBody").innerHTML = html;
  $("#missingFold").open = m.n_missing > 0;
  if (m.table.length) {
    const rows = [...m.table].reverse();
    plot($("#missingChart"), [{ type: "bar", orientation: "h", x: rows.map((r) => r["Missing %"]), y: rows.map((r) => r.Column),
      marker: { color: series()[0] }, text: rows.map((r) => r["Missing %"] + "%"), textposition: "outside", cliponaxis: false,
      hovertemplate: "%{y}: %{x}% missing<extra></extra>" }],
      { xaxis: { range: [0, 100], title: { text: "Missing %" }, gridcolor: css("--line") }, margin: { l: 120, r: 30, t: 10, b: 40 },
        height: Math.max(200, 34 * rows.length + 70),
        shapes: [{ type: "line", x0: m.high_threshold, x1: m.high_threshold, yref: "paper", y0: 0, y1: 1, line: { dash: "dash", color: css("--muted") } }] });
  }

  const cols = Object.keys(d.preview[0] || {});
  $("#previewTable").innerHTML = table(cols.map((c) => ({ label: c, key: c })), d.preview.map((r) => {
    const o = {}; cols.forEach((c) => { o[c] = r[c] === null ? "—" : r[c]; }); return o;
  }));

  const sel = $("#targetSelect");
  sel.innerHTML = d.columns.map((c) => `<option value="${esc(c.name)}" ${c.name === d.suggested_target ? "selected" : ""}>${esc(c.name)}</option>`).join("");
  const hint = () => { const c = d.columns.find((x) => x.name === sel.value); $("#targetHint").textContent = c ? `${c.dtype}, ${c.n_unique.toLocaleString()} distinct values` : ""; };
  sel.onchange = hint; hint();
  $("#fastWrap").hidden = !d.fast_mode_available;
  $("#fastHint").textContent = `Use a random ${d.fast_mode_rows.toLocaleString()}-row sample instead of all ${d.n_rows.toLocaleString()} rows. Faster, and SVM is included (it is skipped above 15,000 rows).`;
  $("#datasetSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------------- run + progress ---------------- */
$("#runBtn").addEventListener("click", startRun);
let pollTimer = null;
async function startRun() {
  if (!state.dataset) return;
  $("#runBtn").disabled = true; $("#runError").hidden = true; $("#results").hidden = true;
  try {
    const r = await api("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: state.dataset.dataset_id, target: $("#targetSelect").value,
        fast_mode: state.dataset.fast_mode_available ? $("#fastMode").checked : false }) });
    state.job = r.job_id; state.stages = r.stages;
    $("#stepper").innerHTML = r.stages.map((s) => `<li data-k="${s.key}"><span class="dot"></span><span>${esc(s.label)}</span><span class="t"></span></li>`).join("");
    $("#progressCard").hidden = false;
    $("#progressCard").scrollIntoView({ behavior: "smooth", block: "center" });
    poll();
  } catch (e) { runError(e.message); }
}
function runError(msg) { const el = $("#runError"); el.textContent = msg; el.hidden = false; $("#runBtn").disabled = false; $("#progressCard").hidden = true; }
async function poll() {
  clearTimeout(pollTimer);
  let s;
  try { s = await api(`/api/jobs/${state.job}`); } catch (e) { runError(e.message); return; }
  const order = state.stages.map((x) => x.key);
  const cur = s.status === "done" ? order.length : order.indexOf(s.stage);
  const times = {}; (s.log || []).forEach((l, i) => { const next = s.log[i + 1]; times[l.stage] = next ? next.t - l.t : null; });
  $$("#stepper li").forEach((li, i) => {
    li.classList.toggle("done", i < cur); li.classList.toggle("active", i === cur);
    const t = times[li.dataset.k]; $(".t", li).textContent = i < cur && t !== null && t !== undefined ? `${t.toFixed(1)} s` : "";
  });
  $("#elapsed").textContent = `${Math.round(s.elapsed)} s`;
  if (s.status === "running") { pollTimer = setTimeout(poll, 900); return; }
  $("#runBtn").disabled = false;
  if (s.status === "error") { runError(s.error); return; }
  state.result = s.result; state.rendered = {};
  setTimeout(() => { $("#progressCard").hidden = true; renderResults(); }, 400);
}

/* ---------------- results ---------------- */
function currentTab() { const b = $("#tabs [aria-selected=true]"); return b ? b.dataset.tab : "overview"; }
$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(name) {
  $$("#tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".tabpanel").forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
  if (!state.rendered[name]) { RENDER[name](state.result); state.rendered[name] = true; }
  else window.Plotly && $$(`#tab-${name} .js-plotly-plot`).forEach((el) => Plotly.Plots.resize(el));
}

function renderResults() {
  const R = state.result, res = R.result;
  const best = res.candidate_results.find((c) => c.model_name === res.best_model_name);
  const d = res.improvement_over_baseline;
  const sig = R.significance;
  $("#winnerBanner").innerHTML = `
    <div><div class="title">Selected model</div><div class="name">${esc(pretty(res.best_model_name))}</div><div class="fam">${esc(best.family)}</div></div>
    <div><div class="title">${esc(res.metric_used.toUpperCase())} (5-fold CV)</div><div class="score">${fmt(res.best_score)}</div></div>
    <div><div class="title">vs fixed baseline ${fmt(res.baseline_score)}</div><div><span class="delta ${d >= 0 ? "up" : "down"}">${signed(d)}</span></div></div>
    <div><div class="title">Lead over runner-up</div><div>${sig ? (sig.significant
      ? `<span class="badge good">Significant (p = ${fmt(sig.p, 4)})</span>`
      : `<span class="badge warn">Could be noise (p = ${sig.p === null ? "n/a" : fmt(sig.p, 3)})</span>`) : "<span class='badge'>Only one model</span>"}</div></div>
    <div class="muted small" style="flex-basis:100%">${esc(R.dataset_name)} → <b>${esc(R.target)}</b> · ${R.n_rows_used.toLocaleString()} rows${R.sampled ? " (fast-mode sample)" : ""} · run #${R.run_id ?? "not saved"}</div>`;
  $("#results").hidden = false;
  showTab("overview");
  $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

const RENDER = {
  overview(R) {
    const res = R.result, p = R.profile;
    const clipped = R.plan.column_decisions.filter((d) => d.outlier_action === "clip").length;
    let h = `<div class="kpis">${kpi(p.n_classes, "Classes")}${kpi(p.is_imbalanced ? "Yes" : "No", "Imbalanced")}${kpi(res.metric_used.toUpperCase(), "Selection metric")}${kpi(res.dropped_columns.length, "Columns dropped")}${kpi(clipped, "Columns clipped (R8)")}${kpi(res.cv_folds_used, "CV folds")}</div>`;
    if (res.cv_folds_used < 5) h += `<div class="notice warn">Used ${res.cv_folds_used}-fold cross-validation because the smallest class has only ${res.cv_folds_used} rows.</div>`;
    if (R.kept_high.length) h += `<div class="notice info"><b>${R.kept_high.length} column(s) with ${R.thresholds.high}% or more missing were kept, not dropped:</b> ${esc(R.kept_high.join(", "))}. Only columns with ${R.thresholds.drop}% or more missing are dropped.</div>`;
    h += `<div class="grid2"><div class="panel"><h3>How the target was prepared</h3><ul class="plain">${R.target_notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></div>
      <div class="panel"><h3>Class distribution</h3><div class="chart" id="classChart" style="min-height:240px"></div></div></div>
      <div class="panel" style="margin-top:14px"><h3>Final justification</h3><p>${esc(res.final_justification)}</p></div>`;
    $("#tab-overview").innerHTML = h;
    const cd = Object.entries(p.class_distribution).sort((a, b) => b[1] - a[1]);
    plot($("#classChart"), [{ type: "bar", x: cd.map((c) => c[0]), y: cd.map((c) => c[1] * 100), marker: { color: series()[0] },
      text: cd.map((c) => (c[1] * 100).toFixed(1) + "%"), textposition: "outside", cliponaxis: false, hovertemplate: "%{x}: %{y:.1f}%<extra></extra>" }],
      { yaxis: { title: { text: "% of rows" }, gridcolor: css("--line"), rangemode: "tozero" }, height: 240, margin: { l: 50, r: 10, t: 20, b: 60 } });
  },

  prep(R) {
    const pf = R.proof;
    let h = "";
    if (pf) {
      h += `<h3>Proof that each step did what it claims</h3><div class="kpis">${kpi(pf.columns.length, "Columns examined")}${kpi(`${pf.n_features_before} → ${pf.n_features_after}`, "Features before → after")}${kpi(pf.nan_cells_after, "Missing cells left")}${kpi(`${pf.n_passed}/${pf.n_checks}`, "Checks passed")}</div>`;
      pf.notes.forEach((n) => { h += `<p class="muted small">${esc(n)}</p>`; });
      pf.columns.forEach((c) => {
        const passed = c.checks.filter((x) => x.ok).length;
        const keys = [...new Set([...Object.keys(c.before), ...Object.keys(c.after)])];
        h += `<details class="fold"><summary><span>${esc(c.column)}</span><span class="muted small" style="font-weight:400">${esc(c.actions.join(", "))}</span><span class="badge ${c.all_passed ? "good" : "bad"}">${passed}/${c.checks.length} checks</span></summary><div class="fold-body">
          <ul class="plain">${c.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
          ${keys.length ? table([{ label: "Statistic", key: "k" }, { label: "Before", key: "b" }, { label: "After", key: "a" }], keys.map((k) => ({ k, b: c.before[k] ?? "—", a: c.after[k] ?? "—" }))) : ""}
          <ul class="checklist">${c.checks.map((x) => `<li><span class="${x.ok ? "ok" : "no"}">${x.ok ? "✓" : "✗"}</span> ${esc(x.name)}${x.detail ? ` <span class="muted">· ${esc(x.detail)}</span>` : ""}</li>`).join("")}</ul>
          ${c.warnings.map((w) => `<div class="notice warn">${esc(w)}</div>`).join("")}
          ${c.examples.length ? `<p class="small muted">Example rows</p>` + table(Object.keys(c.examples[0]).map((k) => ({ label: k, key: k })), c.examples) : ""}
        </div></details>`;
      });
    } else if (R.proof_error) h += `<div class="notice warn">The preprocessing proof could not be built (${esc(R.proof_error)}). The rest of the run is unaffected.</div>`;

    h += `<h3>How missing values were handled</h3>`;
    h += R.missing_handling.length ? table([{ label: "Column", key: "column" }, { label: "Missing %", key: "missing_pct", num: true },
      { label: "Decision", key: "decision" }, { label: "'Was missing' flag", value: (r) => (r.flag ? "Yes" : "No") }, { label: "Rule", key: "rule" }], R.missing_handling)
      : `<p class="muted">No column has missing values, so nothing needed filling.</p>`;

    h += `<h3>Outlier detection (rule R8)</h3><div class="notice info">A value is an outlier if it lies below Q1 − 1.5×IQR or above Q3 + 1.5×IQR. Outliers are clipped to these limits instead of deleting rows, and the limits are learned from the training part of each cross-validation fold only. If the IQR is 0, the 1st and 99th percentiles are used instead.</div>`;
    if (R.outliers.length) {
      h += table([{ label: "Column", key: "column" }, { label: "Outlier %", key: "outlier_pct", num: true },
        { label: "Fences", value: (r) => (r.lower_fence === null ? "—" : `[${fmt(r.lower_fence, 2)}, ${fmt(r.upper_fence, 2)}]`) },
        { label: "Action", html: true, value: (r) => (r.action === "clip" ? "<span class='badge accent'>clip</span>" : "<span class='badge'>none</span>") },
        { label: "Clipped to", value: (r) => (r.action === "clip" ? `[${fmt(r.clip_lower, 2)}, ${fmt(r.clip_upper, 2)}]` : "") },
        { label: "Reason", key: "reason" }], R.outliers);
      const cols = Object.keys(R.outlier_values || {});
      if (cols.length) h += `<p class="small muted" style="margin-top:12px">Box plots of clipped columns (dashed lines are the clip limits):</p><div class="grid2">${cols.map((c, i) => `<div class="panel"><div class="chart" id="box${i}" style="min-height:260px"></div></div>`).join("")}</div>`;
    } else h += `<p class="muted">This dataset has no numeric feature columns, so outlier handling does not apply.</p>`;

    h += `<h3>Step-by-step decision trace</h3><div class="trace">${R.plan.step_trace.map((s) => {
      const m = s.match(/^\[(.*?)\]\s*(R\d+[ab]?:)?(.*)$/);
      return m ? `<div><b>[${esc(m[1])}]</b> ${m[2] ? `<span class="badge accent">${esc(m[2].replace(":", ""))}</span>` : ""}${esc(m[3])}</div>` : `<div>${esc(s)}</div>`;
    }).join("")}</div>`;
    h += `<h3>Summary</h3><ul class="plain">${R.plan.summary.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>`;
    $("#tab-prep").innerHTML = h;
    const rowsBy = Object.fromEntries(R.outliers.map((r) => [r.column, r]));
    Object.keys(R.outlier_values || {}).forEach((c, i) => {
      const r = rowsBy[c];
      plot($(`#box${i}`), [{ type: "box", y: R.outlier_values[c], name: c, marker: { color: series()[0], size: 3 }, boxpoints: "outliers", line: { width: 1.5 } }], {
        height: 260, showlegend: false, margin: { l: 60, r: 10, t: 30, b: 30 }, title: { text: c, font: { size: 13 } },
        shapes: [r.clip_lower, r.clip_upper].map((y) => ({ type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y, line: { dash: "dash", color: series()[1], width: 1.5 } })) });
    });
  },

  baseline(R) {
    const res = R.result, info = res.baseline_info;
    let h = `<div class="grid2"><div class="panel"><h3>What the baseline is</h3><p>${esc(info.summary)}</p><p><span class="badge">Baseline ${esc(res.metric_used.toUpperCase())}: ${fmt(info.score)}</span></p></div>
      <div class="panel"><h3>Why this baseline</h3><ul class="plain">${info.why_chosen.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div></div>`;
    const keys = Object.keys(info.steps[0] || {});
    h += `<h3>Baseline vs DataNarrate, stage by stage</h3>` + table(keys.map((k) => ({ label: k, key: k })), info.steps);
    h += `<h3>Where the improvement comes from</h3><div class="grid2"><div>${table([{ label: "Step", key: "label" }, { label: res.metric_used + " score", value: (s) => fmt(s.score), num: true },
      { label: "Change", value: (s) => s._d, num: true }, { label: "What changed", key: "what_changed" }], res.ladder.map((s, i) => Object.assign({}, s, { _d: i ? signed(s.score - res.ladder[i - 1].score) : "—" })))}</div>
      <div class="panel"><div class="chart" id="ladderChart"></div></div></div>`;
    if (res.ladder.length === 4) {
      const [a, b, c, d] = res.ladder.map((s) => s.score);
      h += `<div class="notice info">Total change over the baseline: <b>${signed(d - a)}</b>. Class balancing accounts for ${signed(b - a)}, DataNarrate's preprocessing for ${signed(c - b)}, and choosing a different model for ${signed(d - c)}.</div>`;
    }
    h += `<h3>Hyperparameter tuning (nested cross-validation)</h3><p class="muted">Tunes the selected model with a randomized search inside each training fold (3 inner folds) and scores it on the same 5 outer folds, so the test folds never influence the chosen settings. The result is the next step of the ladder.</p>
      <button class="secondary" id="tuneBtn" type="button">Tune the selected model</button><div id="tuneOut" style="margin-top:12px"></div>`;
    h += `<h3>Ablation study</h3><p class="muted">Re-scores the selected model with one preprocessing decision switched off at a time, on the same folds. A negative change means the decision helped.</p>
      <button class="secondary" id="ablBtn" type="button">Run ablation study</button><div id="ablOut" style="margin-top:12px"></div>`;
    h += `<h3>Caveats</h3><ul class="plain">${info.caveats.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>`;
    $("#tab-baseline").innerHTML = h;
    const sc = res.ladder.map((s) => s.score), lo = Math.min(...sc), hi = Math.max(...sc);
    plot($("#ladderChart"), [{ type: "bar", x: res.ladder.map((s) => s.label), y: sc, marker: { color: series()[0] },
      text: sc.map((v) => fmt(v)), textposition: "outside", cliponaxis: false, hovertemplate: "%{x}: %{y:.4f}<extra></extra>" }],
      { yaxis: { range: [Math.max(0, lo - 0.05), hi + 0.04], title: { text: res.metric_used }, gridcolor: css("--line") }, height: 320, margin: { l: 56, r: 10, t: 30, b: 90 } });
    $("#ablBtn").addEventListener("click", runAblation);
    $("#tuneBtn").addEventListener("click", () => runTask("tuning", $("#tuneBtn"), $("#tuneOut"), renderTuning, "Tuning the selected model..."));
  },

  models(R) {
    const res = R.result;
    const cr = [...res.candidate_results].sort((a, b) => b[res.metric_used] - a[res.metric_used]);
    let h = table([{ label: "Model", value: (r) => pretty(r.model_name) }, { label: "Family", key: "family" },
      { label: "Accuracy", value: (r) => fmt(r.accuracy), num: true }, { label: "Precision", value: (r) => fmt(r.precision), num: true },
      { label: "Recall", value: (r) => fmt(r.recall), num: true }, { label: "F1", value: (r) => fmt(r.f1), num: true },
      { label: "ROC-AUC", value: (r) => fmt(r.roc_auc), num: true }], cr, { rowClass: (r) => (r.model_name === res.best_model_name ? "best" : "") });
    Object.entries(res.skipped_models || {}).forEach(([k, v]) => { h += `<div class="notice warn"><b>${esc(pretty(k))}:</b> ${esc(v)}</div>`; });
    h += `<div class="grid2" style="margin-top:14px"><div class="panel"><h3>All metrics (darker = higher)</h3><div class="chart" id="metricsChart"></div></div>
      <div class="panel"><h3>Per-fold ${esc(res.metric_used)} spread</h3><div class="chart" id="foldChart"></div></div></div>`;
    h += `<h3>Rigorous significance test</h3><p class="muted">The quick test above uses one 5-fold split, and its training sets overlap, so it can overstate significance. This re-scores the winner, the runner-up and the baseline with 5-fold cross-validation repeated 3 times (15 scores each) and applies the Nadeau–Bengio corrected t-test.</p>
      <button class="secondary" id="statsBtn" type="button">Run repeated-CV significance test</button><div id="statsOut" style="margin-top:12px"></div>`;
    $("#tab-models").innerHTML = h;
    $("#statsBtn").addEventListener("click", () => runTask("stats", $("#statsBtn"), $("#statsOut"), renderStats, "Running repeated cross-validation..."));
    const metrics = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"], ["roc_auc", "ROC-AUC"]];
    const names = res.candidate_results.map((r) => pretty(r.model_name));
    const z = res.candidate_results.map((r) => metrics.map(([k]) => r[k]));
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    const flat = z.flat().filter((v) => v !== null), zMid = (Math.max(0, Math.min(...flat) - 0.05) + 1) / 2;
    plot($("#metricsChart"), [{ type: "heatmap", z, x: metrics.map((m) => m[1]), y: names, showscale: false, zmin: Math.max(0, Math.min(...z.flat().filter((v) => v !== null)) - 0.05), zmax: 1,
      colorscale: dark ? [[0, "#1b2524"], [1, "#3987e5"]] : [[0, "#eef4fc"], [1, "#184f95"]],
      hovertemplate: "%{y}<br>%{x}: %{z:.4f}<extra></extra>" }],
      { height: 60 + 36 * names.length, margin: { l: 150, r: 10, t: 10, b: 40 }, yaxis: { autorange: "reversed", automargin: true },
        annotations: z.flatMap((row, i) => row.map((v, j) => ({ x: metrics[j][1], y: names[i], text: v === null ? "n/a" : v.toFixed(3), showarrow: false,
          font: { size: 12, color: v !== null && v > zMid ? "#ffffff" : css("--fg") } }))) });
    plot($("#foldChart"), res.candidate_results.map((r, i) => ({ type: "box", name: pretty(r.model_name), y: res.fold_scores[r.model_name], boxpoints: "all", jitter: 0.3, pointpos: 0,
      marker: { color: series()[i % 8], size: 6 }, line: { width: 1.5 } })), { height: 340, showlegend: false, margin: { l: 50, r: 10, t: 10, b: 110 } });
  },

  cm(R) {
    const res = R.result;
    const classOrder = (R.target_notes.find((n) => n.startsWith("Class order")) || "").replace(/^.*?:\s*/, "");
    const classNames = {};
    classOrder.split(/,\s*(?=\d+\s=\s)/).forEach((part) => { const m = part.match(/^(\d+)\s=\s(.*)$/); if (m) classNames[+m[1]] = m[2]; });
    let h = `<p class="muted">Rows are the actual class, columns the predicted class. ${classOrder ? "Class order: " + esc(classOrder) : ""}</p><div class="grid2">`;
    res.candidate_results.forEach((r, i) => { h += `<div class="panel"><h3>${esc(pretty(r.model_name))} <span class="muted small" style="font-weight:400">· ${esc(r.family)}</span></h3><div class="chart" id="cm${i}" style="min-height:280px"></div></div>`; });
    $("#tab-cm").innerHTML = h + "</div>";
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    res.candidate_results.forEach((r, i) => {
      const z = r.confusion_matrix, n = z.length;
      const labels = [...Array(n).keys()].map((k) => (classNames[k] !== undefined ? classNames[k] : String(k)));
      const max = Math.max(...z.flat());
      plot($(`#cm${i}`), [{ type: "heatmap", z, x: labels, y: labels, showscale: false, hovertemplate: "Actual %{y}, predicted %{x}: %{z}<extra></extra>",
        colorscale: dark ? [[0, "#1b2524"], [1, "#3987e5"]] : [[0, "#eef4fc"], [1, "#184f95"]] }],
        { height: 280, margin: { l: 70, r: 10, t: 10, b: 50 }, xaxis: { title: { text: "Predicted" }, type: "category" }, yaxis: { title: { text: "Actual" }, type: "category", autorange: "reversed" },
          annotations: z.flatMap((row, a) => row.map((v, p) => ({ x: labels[p], y: labels[a], text: v.toLocaleString(), showarrow: false,
            font: { color: v > max * 0.55 ? "#ffffff" : css("--fg"), size: n > 6 ? 10 : 13 } }))) });
    });
  },

  explain(R) {
    $("#tab-explain").innerHTML = `<p class="muted">Explains the selected model, <b>${esc(pretty(R.result.best_model_name))}</b>. It is fitted on 75% of the rows and explained on the other 25%, which it never saw.</p>
      <button class="primary" id="explBtn" type="button">Compute feature importance</button><div id="explOut" style="margin-top:14px"></div>`;
    $("#explBtn").addEventListener("click", () => runTask("explain", $("#explBtn"), $("#explOut"), renderExplain, "Computing permutation importance..."));
  },

  why(R) {
    const res = R.result;
    let h = `<div class="notice good">${esc(res.final_justification)}</div>`;
    [...res.candidate_results].sort((a, b) => b[res.metric_used] - a[res.metric_used]).forEach((r) => {
      h += `<details class="fold" ${r.model_name === res.best_model_name ? "open" : ""}><summary><span>${esc(pretty(r.model_name))}</span><span class="muted small" style="font-weight:400">${esc(r.family)}</span><span class="badge ${r.model_name === res.best_model_name ? "accent" : ""}">${esc(res.metric_used)} ${fmt(r[res.metric_used])}</span></summary>
        <div class="fold-body grid2"><div><b>Gains</b><ul class="plain">${r.gains.map((g) => `<li>${esc(g)}</li>`).join("")}</ul></div>
        <div><b>Limitations</b><ul class="plain">${r.limitations.map((g) => `<li>${esc(g)}</li>`).join("")}</ul></div></div></details>`;
    });
    $("#tab-why").innerHTML = h;
  },

  narration(R) {
    let h;
    if (R.narration) h = `<div class="panel narr">${R.narration.split(/\n\s*\n/).map((p) => `<p>${esc(p)}</p>`).join("")}</div><p class="muted small">Written by an LLM from the computed results only. The tables in the other tabs are the source of truth.</p>`;
    else h = `<div class="notice warn">Narration is unavailable for this run (${esc(R.narration_error || "no API key")}). Add GROQ_API_KEY to a .env file to enable it. Every other result is unaffected.</div>`;
    $("#tab-narration").innerHTML = h;
  },
};

async function runAblation() {
  const btn = $("#ablBtn"), out = $("#ablOut");
  btn.disabled = true; btn.textContent = "Running the ablation study...";
  out.innerHTML = `<p class="muted small">Re-fitting the selected model five times on the same folds.</p>`;
  try {
    const a = await api(`/api/jobs/${state.job}/ablation`, { method: "POST" });
    const full = a.variants.find((v) => v.variant_name === "full_pipeline").score;
    const label = { full_pipeline: "Full pipeline", no_scaling: "No scaling (R7 off)", no_smart_encoding_forced_onehot: "Forced one-hot (R5 off)",
      no_imputation_zero_fill: "Constant imputation (R3/R4 off)", no_outlier_clipping_R8: "No outlier clipping (R8 off)" };
    out.innerHTML = table([{ label: "Variant", value: (v) => label[v.variant_name] || v.variant_name }, { label: a.metric, value: (v) => fmt(v.score), num: true },
      { label: "Change vs full", value: (v) => (v.variant_name === "full_pipeline" ? "—" : signed(v.score - full)), num: true }], a.variants)
      + `<p class="muted small">${esc(pretty(a.model))}, ${a.seconds} s.</p>`;
    btn.textContent = "Run again";
  } catch (e) { out.innerHTML = `<p class="error">${esc(e.message)}</p>`; btn.textContent = "Run ablation study"; }
  finally { btn.disabled = false; }
}

/* ---------------- history ---------------- */
async function loadHistory() {
  const el = $("#historyTable");
  el.innerHTML = `<p class="empty">Loading...</p>`;
  try {
    const runs = await api("/api/history");
    el.innerHTML = runs.length ? table([{ label: "#", key: "id", num: true }, { label: "Dataset → target", key: "dataset_name" },
      { label: "Best model", value: (r) => pretty(r.best_model_name) }, { label: "Metric", key: "metric_used" },
      { label: "Score", value: (r) => fmt(r.best_score), num: true }, { label: "Baseline", value: (r) => fmt(r.baseline_score), num: true },
      { label: "Change", value: (r) => signed(r.improvement_over_baseline), num: true }, { label: "Rows", value: (r) => (r.n_rows || 0).toLocaleString(), num: true },
      { label: "When", value: (r) => (r.timestamp || "").replace("T", " ").slice(0, 16) }], runs)
      : `<p class="empty">No runs saved yet.</p>`;
  } catch (e) { el.innerHTML = `<p class="error">${esc(e.message)}</p>`; }
}

/* ---------------- extra analyses (background tasks) ---------------- */
async function runTask(kind, btn, out, render, label) {
  const original = btn.textContent;
  btn.disabled = true; btn.textContent = label;
  out.innerHTML = `<p class="muted small">Running in the background. This can take a minute or two. <span class="tick">0 s</span></p>`;
  try {
    await api(`/api/jobs/${state.job}/tasks/${kind}`, { method: "POST" });
    for (;;) {
      await new Promise((r) => setTimeout(r, 1500));
      const t = await api(`/api/jobs/${state.job}/tasks/${kind}`);
      const tick = $(".tick", out); if (tick) tick.textContent = `${Math.round(t.elapsed)} s`;
      if (t.status === "running") continue;
      if (t.status === "error") throw new Error(t.error);
      render(t.result, out, t.elapsed);
      btn.textContent = "Run again";
      break;
    }
  } catch (e) { out.innerHTML = `<p class="error">${esc(e.message)}</p>`; btn.textContent = original; }
  finally { btn.disabled = false; }
}

function renderExplain(r, out, secs) {
  const g = r.global, imps = g.importances.slice(0, 15);
  out.innerHTML = `<div class="kpis">${kpi(fmt(g.holdout_score), `Held-out ${g.metric}`)}${kpi(g.n_train.toLocaleString(), "Training rows")}${kpi(g.n_test.toLocaleString(), "Explained rows")}${kpi(g.n_repeats, "Shuffles per column")}</div>
    <div class="grid2"><div class="panel"><h3>Which columns matter most (permutation importance)</h3><p class="muted small">Each column is shuffled in turn; the bar is the drop in ${esc(g.metric)} on the held-out rows. A bar near zero means the model barely uses that column.</p><div class="chart" id="impChart"></div></div>
    <div class="panel"><h3>Why the model predicted this row</h3><p class="muted small">Each column is replaced by a typical value; the bar is how much the ${esc("score of the predicted class")} falls. Positive bars pushed the model towards its prediction.</p>
      <label class="small muted" for="rowSel">Held-out row</label> <select id="rowSel">${r.rows.map((x) => `<option value="${x.row}">Row ${x.row}: actual ${esc(x.actual)}, predicted ${esc(x.predicted)}</option>`).join("")}</select>
      <div id="localOut" style="margin-top:10px"></div></div></div>
    <p class="muted small">${secs ? `${secs} s.` : ""}</p>`;
  const rev = [...imps].reverse();
  plot($("#impChart"), [{ type: "bar", orientation: "h", x: rev.map((f) => f.importance), y: rev.map((f) => f.feature),
    error_x: { type: "data", array: rev.map((f) => f.std), visible: true, color: css("--muted") }, marker: { color: series()[0] },
    hovertemplate: "%{y}: %{x:.4f}<extra></extra>" }],
    { height: 40 + 26 * rev.length + 40, margin: { l: 140, r: 20, t: 10, b: 40 }, xaxis: { title: { text: `Drop in ${g.metric}` }, gridcolor: css("--line") } });
  renderLocal(r.local);
  $("#rowSel").addEventListener("change", async (e) => {
    $("#localOut").innerHTML = `<p class="muted small">Explaining row ${e.target.value}...</p>`;
    try { renderLocal(await api(`/api/jobs/${state.job}/explain/row`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ row: +e.target.value }) })); }
    catch (err) { $("#localOut").innerHTML = `<p class="error">${esc(err.message)}</p>`; }
  });
}

function renderLocal(l) {
  const c = [...l.contributions].reverse();
  $("#localOut").innerHTML = `<p><span class="badge">Actual: ${esc(l.actual)}</span> <span class="badge accent">Predicted: ${esc(l.predicted)}</span> ${l.confidence !== null ? `<span class="badge">${(l.confidence * 100).toFixed(1)}% confident</span>` : ""}</p><div class="chart" id="localChart"></div>`
    + table([{ label: "Column", key: "feature" }, { label: "Value", value: (x) => x.value ?? "missing" }, { label: "Typical value", value: (x) => x.reference ?? "missing" }, { label: "Effect", value: (x) => signed(x.effect, 3), num: true }], l.contributions);
  plot($("#localChart"), [{ type: "bar", orientation: "h", x: c.map((x) => x.effect), y: c.map((x) => x.feature),
    marker: { color: c.map((x) => (x.effect >= 0 ? series()[0] : series()[1])) }, hovertemplate: "%{y}: %{x:.3f}<extra></extra>" }],
    { height: 40 + 26 * c.length + 40, margin: { l: 140, r: 20, t: 10, b: 40 }, xaxis: { title: { text: `Change in ${l.score_kind}` }, gridcolor: css("--line"), zeroline: true } });
}

function renderTuning(t, out, secs) {
  const rows = t.default_fold_scores.map((d, i) => ({ fold: i + 1, d, t: t.tuned_fold_scores[i] }));
  out.innerHTML = `<div class="kpis">${kpi(fmt(t.default_score), `Default ${t.metric}`)}${kpi(fmt(t.tuned_score), `Tuned ${t.metric}`)}${kpi(signed(t.gain), "Gain from tuning")}${kpi(fmtP(t.p_value), "p (tuned vs default)")}</div>
    <div class="notice ${t.gain > 0 && t.significant ? "good" : "info"}">Ladder step <b>+ tuning</b>: ${fmt(t.default_score)} → ${fmt(t.tuned_score)} (${signed(t.gain)}). ${t.significant ? "The gain is statistically significant on the outer folds." : "The difference is not statistically significant on the outer folds."}</div>
    <div class="grid2"><div>${table([{ label: "Outer fold", key: "fold", num: true }, { label: "Default", value: (r) => fmt(r.d), num: true }, { label: "Tuned", value: (r) => fmt(r.t), num: true }, { label: "Change", value: (r) => signed(r.t - r.d), num: true }], rows)}</div>
    <div>${table([{ label: "Setting", key: "k" }, { label: "Most often chosen", key: "v" }], Object.entries(t.most_common_params).map(([k, v]) => ({ k, v })))}</div></div>
    <p class="muted small">${esc(pretty(t.model_name))}: ${t.n_iter} random settings per fold, ${t.inner_folds} inner folds, ${t.outer_folds} outer folds. ${secs ? `${secs} s.` : ""}</p>`;
}

function renderStats(s, out, secs) {
  const names = Object.keys(s.mean);
  out.innerHTML = table([{ label: "Model", value: (n) => (n === "baseline" ? "Fixed baseline" : pretty(n)) }, { label: `Mean ${s.metric}`, value: (n) => fmt(s.mean[n]), num: true }, { label: "Std", value: (n) => fmt(s.std[n]), num: true }, { label: "Scores", value: (n) => s.scores[n].length, num: true }], names)
    + table([{ label: "Comparison", value: (t) => `${pretty(t.a)} vs ${t.b === "baseline" ? "baseline" : pretty(t.b)}` }, { label: "Mean difference", value: (t) => signed(t.mean_difference), num: true },
      { label: "Plain p", value: (t) => fmtP(t.plain_p), num: true }, { label: "Corrected p", value: (t) => fmtP(t.corrected_p), num: true },
      { label: "Significant (corrected)", html: true, value: (t) => (t.significant_corrected ? "<span class='badge good'>Yes</span>" : "<span class='badge warn'>No</span>") }], s.tests)
    + `<p class="muted small">${s.n_splits}-fold cross-validation repeated ${s.n_repeats} times. The corrected test inflates the variance by (1/J + n_test/n_train) to account for overlapping training sets. ${secs ? `${secs} s.` : ""}</p>`;
}
