from __future__ import annotations

import json
import os
import pathlib
import urllib.error
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


DEFAULT_MODAL_ENDPOINT = "https://edisonscientific--reasyn-routes.modal.run"
DEFAULT_MODAL_STREAM_ENDPOINT = "https://edisonscientific--reasyn-routes-stream.modal.run"
DEFAULT_RENDERER = "http://mol2txt.app/"
DEFAULT_RENDERER_FORMAT = "png"
TOKEN_FILE = pathlib.Path("/tmp/reasyn_web_auth_token")

app = FastAPI(title="ReaSyn Local UI")


def _auth_token() -> str | None:
    token = os.environ.get("REASYN_AUTH_TOKEN")
    if token:
        return token
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    return None


def _modal_endpoint() -> str:
    return os.environ.get("REASYN_MODAL_ENDPOINT", DEFAULT_MODAL_ENDPOINT)


def _modal_stream_endpoint() -> str:
    return os.environ.get("REASYN_MODAL_STREAM_ENDPOINT", DEFAULT_MODAL_STREAM_ENDPOINT)


def _renderer_endpoint() -> str:
    return os.environ.get("REASYN_RENDERER_ENDPOINT", DEFAULT_RENDERER)


def _renderer_format() -> str:
    renderer_format = os.environ.get("REASYN_RENDERER_FORMAT", DEFAULT_RENDERER_FORMAT).lower()
    if renderer_format not in {"png", "svg"}:
        return DEFAULT_RENDERER_FORMAT
    return renderer_format


def _modal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    smiles = payload.get("smiles", [])
    if isinstance(smiles, str):
        smiles = [line.strip() for line in smiles.splitlines() if line.strip()]
    if not smiles:
        raise HTTPException(status_code=400, detail="Provide at least one SMILES string.")

    modal_payload = {
        "smiles": smiles,
        "k": int(payload.get("k", 3)),
        "effort": payload.get("effort", "low"),
        "verbose": True,
    }
    overrides = payload.get("overrides")
    if overrides:
        modal_payload["overrides"] = {
            key: value
            for key, value in overrides.items()
            if value not in (None, "")
        }
    return modal_payload


def _authorized_request(url: str, payload: dict[str, Any]) -> urllib.request.Request:
    token = _auth_token()
    if not token:
        raise HTTPException(
            status_code=500,
            detail=f"Set REASYN_AUTH_TOKEN or write the token to {TOKEN_FILE}.",
        )

    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )


def _sse(event: dict[str, Any]) -> str:
    event_name = str(event.get("event", "message"))
    data = json.dumps(event, separators=(",", ":"))
    return f"event: {event_name}\ndata: {data}\n\n"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "modal_endpoint": _modal_endpoint(),
        "modal_stream_endpoint": _modal_stream_endpoint(),
        "has_token": _auth_token() is not None,
        "renderer_endpoint": _renderer_endpoint(),
        "renderer_format": _renderer_format(),
    }


@app.post("/api/routes")
def proxy_routes(payload: dict[str, Any]) -> JSONResponse:
    request = _authorized_request(_modal_endpoint(), _modal_payload(payload))
    try:
        with urllib.request.urlopen(request, timeout=payload.get("timeout", 1800)) as response:
            data = json.loads(response.read().decode("utf-8"))
            return JSONResponse(data)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body)
        except json.JSONDecodeError:
            detail = body
        return JSONResponse({"detail": detail}, status_code=exc.code)
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/routes/stream")
def proxy_routes_stream(payload: dict[str, Any]) -> StreamingResponse:
    request = _authorized_request(_modal_stream_endpoint(), _modal_payload(payload))

    def events():
        try:
            with urllib.request.urlopen(request, timeout=payload.get("timeout", 1800)) as response:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    yield line
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body)
            except json.JSONDecodeError:
                detail = body
            yield _sse({"event": "stream_error", "detail": detail})
        except urllib.error.URLError as exc:
            yield _sse({"event": "stream_error", "detail": str(exc)})

    return StreamingResponse(events(), media_type="text/event-stream")


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ReaSyn Route Explorer</title>
  <style>
    :root {
      --ink: #17201b;
      --muted: #5d6f66;
      --paper: #fbf7ed;
      --panel: rgba(255, 252, 243, 0.88);
      --line: rgba(23, 32, 27, 0.14);
      --sage: #66815f;
      --moss: #213d30;
      --clay: #b85f42;
      --gold: #d8a24a;
      --blue: #4f7e99;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Trebuchet MS", Verdana, sans-serif;
      background:
        radial-gradient(circle at 12% 18%, rgba(216, 162, 74, 0.28), transparent 28rem),
        radial-gradient(circle at 88% 8%, rgba(79, 126, 153, 0.24), transparent 24rem),
        linear-gradient(135deg, #f4ead7 0%, #d7dfcf 52%, #f7efe2 100%);
      min-height: 100vh;
    }
    header {
      padding: 42px min(6vw, 72px) 24px;
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 28px;
      align-items: end;
    }
    h1 {
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(44px, 7vw, 96px);
      line-height: 0.88;
      margin: 0;
      letter-spacing: -0.065em;
      color: var(--moss);
    }
    .subtitle {
      max-width: 720px;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.55;
    }
    .shell {
      padding: 0 min(6vw, 72px) 64px;
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 24px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: 0 28px 80px rgba(33, 61, 48, 0.13);
      backdrop-filter: blur(14px);
    }
    form.panel { padding: 24px; position: sticky; top: 18px; }
    label {
      display: block;
      color: var(--moss);
      font-weight: 750;
      font-size: 13px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin: 16px 0 8px;
    }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.72);
      color: var(--ink);
      padding: 12px 13px;
      font: inherit;
      outline: none;
    }
    textarea { min-height: 138px; resize: vertical; font-family: "SF Mono", "Menlo", monospace; }
    textarea:focus, input:focus, select:focus { border-color: var(--sage); box-shadow: 0 0 0 4px rgba(102,129,95,0.14); }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .checkbox {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: 16px;
      color: var(--muted);
      font-size: 14px;
    }
    .checkbox input { width: auto; }
    .advanced {
      display: none;
      padding: 14px;
      margin-top: 12px;
      border-radius: 18px;
      background: rgba(33, 61, 48, 0.06);
      border: 1px dashed rgba(33, 61, 48, 0.2);
    }
    .advanced.on { display: block; }
    button {
      width: 100%;
      border: 0;
      margin-top: 20px;
      border-radius: 18px;
      padding: 14px 18px;
      background: linear-gradient(135deg, var(--moss), var(--sage));
      color: #fffaf0;
      font-weight: 850;
      letter-spacing: 0.02em;
      cursor: pointer;
      box-shadow: 0 16px 36px rgba(33, 61, 48, 0.25);
    }
    button:disabled { opacity: 0.58; cursor: wait; }
    .status {
      margin-top: 14px;
      padding: 12px;
      border-radius: 16px;
      color: var(--muted);
      background: rgba(255,255,255,0.45);
      font-size: 13px;
      line-height: 1.45;
    }
    .progress-log {
      display: grid;
      gap: 8px;
      max-height: 260px;
      overflow: auto;
      margin-top: 12px;
      padding-right: 4px;
    }
    .progress-item {
      border-left: 3px solid rgba(102,129,95,0.45);
      padding: 8px 10px;
      border-radius: 12px;
      background: rgba(255,255,255,0.42);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .progress-item strong { color: var(--moss); }
    #results { display: grid; gap: 18px; }
    .empty {
      padding: 44px;
      text-align: center;
      color: var(--muted);
    }
    .mol-card { padding: 22px; overflow: hidden; }
    .mol-title {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 16px;
    }
    .mol-title h2 {
      font-family: Georgia, "Times New Roman", serif;
      font-size: 28px;
      letter-spacing: -0.03em;
      margin: 0;
      overflow-wrap: anywhere;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 800;
      color: var(--moss);
      background: rgba(216, 162, 74, 0.22);
      border: 1px solid rgba(216, 162, 74, 0.36);
      white-space: nowrap;
    }
    .routes { display: grid; gap: 16px; }
    .route {
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(255,255,255,0.56);
      padding: 18px;
    }
    .route-head {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 12px;
    }
    .metric {
      border-radius: 999px;
      padding: 6px 9px;
      background: rgba(79,126,153,0.12);
      color: #29495a;
      font-size: 12px;
      font-weight: 800;
    }
    .metric.good { background: rgba(102,129,95,0.16); color: var(--moss); }
    .metric.warn { background: rgba(184,95,66,0.15); color: #813e2c; }
    .viz {
      width: 100%;
      min-height: 260px;
      max-height: 560px;
      background: white;
      border: 1px solid var(--line);
      border-radius: 18px;
      object-fit: contain;
    }
    .viz-link {
      display: inline-block;
      margin-top: 6px;
      color: var(--blue);
      font-size: 12px;
      font-weight: 800;
      text-decoration: none;
    }
    .step-list { display: grid; gap: 12px; margin-top: 14px; }
    .step {
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }
    .step-body { min-width: 0; }
    .step-index {
      font-weight: 900;
      color: var(--clay);
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 12px;
      padding-top: 12px;
    }
    .step-summary {
      display: grid;
      gap: 8px;
      margin: 0 0 10px;
      padding: 12px;
      border-radius: 16px;
      background: rgba(33, 61, 48, 0.06);
      color: var(--muted);
      font-size: 13px;
    }
    .step-summary code, .template-list code {
      overflow-wrap: anywhere;
      white-space: normal;
    }
    .template-list {
      display: grid;
      gap: 8px;
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }
    .template-list b { color: var(--moss); }
    .annotation {
      display: grid;
      gap: 6px;
      padding: 10px;
      border-radius: 14px;
      background: rgba(216, 162, 74, 0.14);
      border: 1px solid rgba(216, 162, 74, 0.24);
    }
    .note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      margin: 8px 0 0;
    }
    details {
      margin-top: 14px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }
    summary { cursor: pointer; color: var(--moss); font-weight: 850; }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: rgba(23,32,27,0.06);
      padding: 12px;
      border-radius: 14px;
      font-size: 12px;
      line-height: 1.5;
    }
    .bb-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    .bb {
      background: white;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 8px;
      min-height: 120px;
    }
    .bb img { width: 100%; height: 96px; object-fit: contain; }
    .bb code { font-size: 10px; overflow-wrap: anywhere; }
    @media (max-width: 960px) {
      header, .shell { grid-template-columns: 1fr; }
      form.panel { position: static; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>ReaSyn Route Explorer</h1>
    </div>
    <p class="subtitle">
      Local UI for the Modal-hosted ReaSyn retrosynthesis worker. Requests run in verbose mode,
      are rendered through mol2txt.app, and expose the model's search settings, cache hits,
      stack route, and RDKit-forward validation.
    </p>
  </header>

  <main class="shell">
    <form id="route-form" class="panel">
      <label for="smiles">Input molecules</label>
      <textarea id="smiles" spellcheck="false">c1ccccc1
CC(=O)Oc1ccccc1C(=O)O</textarea>

      <div class="grid2">
        <div>
          <label for="k">Routes per molecule</label>
          <input id="k" type="number" min="1" max="20" value="3" />
        </div>
        <div>
          <label for="effort">Effort</label>
          <select id="effort">
            <option value="low">low</option>
            <option value="medium">medium</option>
            <option value="high">high</option>
          </select>
        </div>
      </div>

      <label class="checkbox">
        <input id="advanced-toggle" type="checkbox" />
        Demonstration overrides
      </label>
      <div id="advanced" class="advanced">
        <div class="grid2">
          <div><label>Search width</label><input data-override="search_width" type="number" min="1" max="64" placeholder="preset" /></div>
          <div><label>Exhaustiveness</label><input data-override="exhaustiveness" type="number" min="1" max="256" placeholder="preset" /></div>
          <div><label>Cycles</label><input data-override="num_cycles" type="number" min="1" max="12" placeholder="preset" /></div>
          <div><label>Edit samples</label><input data-override="num_editflow_samples" type="number" min="1" max="100" placeholder="preset" /></div>
          <div><label>Edit steps</label><input data-override="num_editflow_steps" type="number" min="1" max="200" placeholder="preset" /></div>
          <div><label>Time limit</label><input data-override="time_limit" type="number" min="0" max="1800" placeholder="preset" /></div>
        </div>
      </div>

      <button id="submit" type="submit">Find Routes</button>
      <div id="status" class="status">Checking local configuration...</div>
      <div id="progress-log" class="progress-log"></div>
    </form>

    <section id="results">
      <div class="panel empty">Submit one or more SMILES strings to inspect validated synthesis routes.</div>
    </section>
  </main>

  <script>
    const form = document.getElementById("route-form");
    const results = document.getElementById("results");
    const statusBox = document.getElementById("status");
    const progressLog = document.getElementById("progress-log");
    const submit = document.getElementById("submit");
    const advancedToggle = document.getElementById("advanced-toggle");
    const advanced = document.getElementById("advanced");
    let renderer = "http://mol2txt.app/";
    let rendererFormat = "png";

    const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
    }[ch]));

    const rendererUrl = (params) => {
      const url = new URL(renderer);
      Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value));
      url.searchParams.set("format", rendererFormat);
      return url.toString();
    };
    const molUrl = (smi) => rendererUrl({ smi });
    const rxnUrl = (rxn) => rendererUrl({ rxn });

    advancedToggle.addEventListener("change", () => {
      advanced.classList.toggle("on", advancedToggle.checked);
    });

    async function loadConfig() {
      const response = await fetch("/api/config");
      const config = await response.json();
      renderer = config.renderer_endpoint;
      rendererFormat = config.renderer_format || "png";
      statusBox.innerHTML = `
        Modal endpoint: <strong>${escapeHtml(config.modal_endpoint)}</strong><br>
        Stream endpoint: <strong>${escapeHtml(config.modal_stream_endpoint)}</strong><br>
        Auth token: <strong>${config.has_token ? "available" : "missing"}</strong><br>
        Renderer: <strong>${escapeHtml(config.renderer_endpoint)}</strong> (${escapeHtml(rendererFormat)})
      `;
    }

    function payloadFromForm() {
      const smiles = document.getElementById("smiles").value
        .split(/\n|,/)
        .map((value) => value.trim())
        .filter(Boolean);
      const payload = {
        smiles,
        k: Number(document.getElementById("k").value || 3),
        effort: document.getElementById("effort").value,
      };
      if (advancedToggle.checked) {
        const overrides = {};
        document.querySelectorAll("[data-override]").forEach((input) => {
          if (input.value !== "") overrides[input.dataset.override] = Number(input.value);
        });
        if (Object.keys(overrides).length) payload.overrides = overrides;
      }
      return payload;
    }

    function routeMetrics(route) {
      const search = route.search || {};
      return [
        `<span class="metric">rank ${route.rank}</span>`,
        `<span class="metric ${route.forward_valid ? "good" : "warn"}">${route.forward_valid ? "RDKit valid" : "not validated"}</span>`,
        `<span class="metric">score ${Number(route.score ?? 0).toFixed(3)}</span>`,
        `<span class="metric">${route.num_forward_steps ?? 0} rxn steps</span>`,
        `<span class="metric">${route.effort} effort</span>`,
        `<span class="metric ${search.cache_hit ? "good" : ""}">${search.cache_hit ? "cache hit" : "fresh search"}</span>`,
      ].join("");
    }

    function renderBuildingBlocks(route) {
      const blocks = route.building_blocks || [];
      if (!blocks.length) return "";
      return `
        <details>
          <summary>Building blocks (${blocks.length})</summary>
          <div class="bb-grid">
            ${blocks.map((smi) => `
              <div class="bb">
                <img alt="${escapeHtml(smi)}" src="${molUrl(smi)}" />
                <code>${escapeHtml(smi)}</code>
              </div>
            `).join("")}
          </div>
        </details>`;
    }

    function renderTemplateList(label, values) {
      if (!values || !values.length) return `<div><b>${escapeHtml(label)}:</b> none in template</div>`;
      return `<div><b>${escapeHtml(label)}:</b> ${values.map((value) => `<code>${escapeHtml(value)}</code>`).join(" | ")}</div>`;
    }

    function renderAnnotation(annotation) {
      if (!annotation) return "";
      const confidence = annotation.annotation_confidence === null || annotation.annotation_confidence === undefined
        ? "n/a"
        : Number(annotation.annotation_confidence).toFixed(3);
      return `
        <div class="annotation">
          <div><b>Annotated reaction:</b> ${escapeHtml(annotation.reaction_name)} <code>${escapeHtml(annotation.reaction_class_id)}</code></div>
          <div><b>Method/confidence:</b> ${escapeHtml(annotation.annotation_method)} / ${escapeHtml(confidence)}</div>
          <div><b>Annotation example:</b> <code>${escapeHtml((annotation.example_reactants || []).join(" . "))}</code> -> <code>${escapeHtml(annotation.example_product)}</code></div>
        </div>`;
    }

    function renderStepDetails(step) {
      const reaction = step.reaction || {};
      const annotation = reaction.annotation || null;
      return `
        <div class="step-summary">
          ${renderAnnotation(annotation)}
          <div><b>Actual reactants:</b> <code>${escapeHtml((step.reactants || []).join(" . "))}</code></div>
          <div><b>Actual product:</b> <code>${escapeHtml(step.product)}</code></div>
        </div>
        <details>
          <summary>Reaction template details</summary>
          <p class="note">${escapeHtml(reaction.metadata_note || "This model exposes reaction SMARTS templates, not named conditions.")}</p>
          <div class="template-list">
            <div><b>Template ID:</b> R${escapeHtml(step.rxn_id)}</div>
            <div><b>Full SMARTS:</b> <code>${escapeHtml(step.reaction_smarts || reaction.smarts)}</code></div>
            ${renderTemplateList("Reactant template SMARTS", reaction.reactant_templates)}
            ${renderTemplateList("Agent/reagent SMARTS", reaction.agent_templates)}
            ${renderTemplateList("Product template SMARTS", reaction.product_templates)}
          </div>
        </details>`;
    }

    function renderRoute(route) {
      if (Array.isArray(route)) {
        return `<article class="route"><pre>${escapeHtml(route.join("\n"))}</pre></article>`;
      }

      const multistep = route.multistep_reaction_smiles || (route.reaction_smiles || []).join(".");
      const steps = route.forward_steps || [];
      return `
        <article class="route">
          <div class="route-head">${routeMetrics(route)}</div>
          ${multistep ? `<img class="viz" alt="multistep reaction" src="${rxnUrl(multistep)}" /><a class="viz-link" href="${rxnUrl(multistep)}" target="_blank" rel="noreferrer">Open full route image</a>` : ""}
          <div class="step-list">
            ${steps.map((step, index) => `
              <div class="step">
                <div class="step-index">Step ${index + 1}<br>R${step.rxn_id}</div>
                <div class="step-body">
                  ${renderStepDetails(step)}
                  <img class="viz" alt="reaction step ${index + 1}" src="${rxnUrl(step.reaction_smiles)}" />
                  <a class="viz-link" href="${rxnUrl(step.reaction_smiles)}" target="_blank" rel="noreferrer">Open step image</a>
                </div>
              </div>
            `).join("")}
          </div>
          ${renderBuildingBlocks(route)}
          <details>
            <summary>Search and route details</summary>
            <pre>${escapeHtml(JSON.stringify({
              generated_product: route.smiles,
              synthesis_stack: route.synthesis_tokens,
              search: route.search,
              forward_candidate_count: route.forward_candidate_count,
              forward_error: route.forward_error,
              multistep_reaction_smiles: route.multistep_reaction_smiles,
            }, null, 2))}</pre>
          </details>
        </article>`;
    }

    function renderResults(data) {
      const groups = data.results || [];
      if (!groups.length) {
        results.innerHTML = `<div class="panel empty">No results returned.</div>`;
        return;
      }
      results.innerHTML = groups.map((group) => `
        <section class="panel mol-card">
          <div class="mol-title">
            <h2>${escapeHtml(group.input)}</h2>
            <span class="badge">${(group.routes || []).length} routes</span>
          </div>
          ${group.error ? `<pre>${escapeHtml(group.error)}</pre>` : ""}
          <div class="routes">${(group.routes || []).map(renderRoute).join("")}</div>
        </section>
      `).join("");
    }

    function progressText(event) {
      const prefix = event.smiles ? `${event.smiles}: ` : "";
      if (event.event === "stream_connected") return "Connected to Modal stream.";
      if (event.event === "request_started") return `Request started for ${event.input_count} molecule(s).`;
      if (event.event === "molecule_started") return `${prefix}search started.`;
      if (event.event === "cache_hit") return `${prefix}served from warm worker cache.`;
      if (event.event === "cache_miss") return `${prefix}cache miss; running sampler.`;
      if (event.event === "phase_started") return `${prefix}${event.phase} phase started.`;
      if (event.event === "active_states_resampled") return `${prefix}resampled active states from finished routes.`;
      if (event.event === "ar_step_completed") return `${prefix}${event.phase} step ${event.step} completed.`;
      if (event.event === "phase_completed") return `${prefix}${event.phase} phase completed.`;
      if (event.event === "exact_match_found") return `${prefix}exact fingerprint match found; stopping early.`;
      if (event.event === "time_limit_exceeded") return `${prefix}time limit reached.`;
      if (event.event === "search_completed") return `${prefix}sampler completed.`;
      if (event.event === "molecule_completed") return `${prefix}routes validated and rendered.`;
      if (event.event === "molecule_failed") return `${prefix}failed: ${event.error || "unknown error"}`;
      if (event.event === "request_completed") return `Request completed in ${Number(event.elapsed_seconds || 0).toFixed(1)} seconds.`;
      if (event.event === "stream_error") return `Stream error: ${JSON.stringify(event.detail)}`;
      return `${prefix}${event.event || "event"}`;
    }

    function appendProgress(event) {
      const metrics = [];
      if (event.elapsed_seconds !== undefined) metrics.push(`${Number(event.elapsed_seconds).toFixed(1)}s`);
      if (event.finished_states !== undefined) metrics.push(`${event.finished_states} finished`);
      if (event.active_states !== undefined) metrics.push(`${event.active_states} active`);
      if (event.best_score !== null && event.best_score !== undefined) metrics.push(`best ${Number(event.best_score).toFixed(3)}`);
      const item = document.createElement("div");
      item.className = "progress-item";
      item.innerHTML = `<strong>${escapeHtml(progressText(event))}</strong>${metrics.length ? `<br>${escapeHtml(metrics.join(" | "))}` : ""}`;
      progressLog.prepend(item);
      while (progressLog.children.length > 80) progressLog.removeChild(progressLog.lastChild);
    }

    function parseSseBlock(block) {
      const parsed = { event: "message", data: "" };
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith("event:")) parsed.event = line.slice(6).trim();
        if (line.startsWith("data:")) parsed.data += line.slice(5).trim();
      }
      if (!parsed.data) return { event: parsed.event };
      const payload = JSON.parse(parsed.data);
      if (!payload.event) payload.event = parsed.event;
      return payload;
    }

    async function runFallbackRequest(payload) {
      appendProgress({ event: "stream_error", detail: "Falling back to non-streaming JSON endpoint." });
      const response = await fetch("/api/routes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(data));
      renderResults(data);
      return data;
    }

    async function runStreamingRequest(payload) {
      const response = await fetch("/api/routes/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok || !response.body) return runFallbackRequest(payload);

      const partialResults = [];
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalResult = null;
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const blocks = buffer.split(/\n\n/);
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          if (!block.trim()) continue;
          const event = parseSseBlock(block);
          appendProgress(event);
          if (event.event === "molecule_completed" || event.event === "molecule_failed") {
            partialResults.push(event.result);
            renderResults({ results: partialResults });
          }
          if (event.event === "request_completed") {
            finalResult = event.result;
            renderResults(finalResult);
          }
          if (event.event === "stream_error") {
            throw new Error(JSON.stringify(event.detail));
          }
        }
        if (done) break;
      }
      return finalResult;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      submit.disabled = true;
      const started = performance.now();
      progressLog.innerHTML = "";
      statusBox.textContent = "Running verbose Modal stream...";
      results.innerHTML = `<div class="panel empty">ReaSyn is searching. Low effort is usually quick; high effort can take several minutes.</div>`;

      try {
        await runStreamingRequest(payloadFromForm());
        statusBox.textContent = `Done in ${((performance.now() - started) / 1000).toFixed(1)} seconds.`;
      } catch (error) {
        results.innerHTML = `<div class="panel empty"><pre>${escapeHtml(error.message)}</pre></div>`;
        statusBox.textContent = "Request failed.";
      } finally {
        submit.disabled = false;
      }
    });

    loadConfig().catch((error) => {
      statusBox.textContent = `Could not load config: ${error.message}`;
    });
  </script>
</body>
</html>
"""
