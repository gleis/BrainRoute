const state = {
  config: null,
  lastDecision: null,
  sessionId: globalThis.crypto.randomUUID(),
  controller: null,
};

const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(errorMessage(data) || "Request failed");
  }
  return data;
}

// Errors arrive either as a plain string or as an OpenAI-style
// { error: { message, type } } object, depending on the endpoint.
function errorMessage(data) {
  if (!data) return "";
  if (typeof data.error === "string") return data.error;
  if (data.error?.message) return data.error.message;
  return data.fallback_error || data.decision?.reason || "";
}

async function loadConfig() {
  state.config = await request("/api/config");
  $("profile").innerHTML = state.config.profiles
    .map((name) => `<option value="${escapeHtml(name)}" ${name === state.config.default_profile ? "selected" : ""}>${escapeHtml(name)}</option>`)
    .join("");
  $("model").innerHTML = `<option value="">Auto</option>` + state.config.models
    .filter((model) => model.enabled)
    .map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.name)} (${escapeHtml(model.provider)})</option>`)
    .join("");
  renderModels(state.config.models);
  renderClassifier();
  renderPolicy();
  renderRemoteSources();
}

async function routePrompt(event) {
  event.preventDefault();
  $("status").textContent = "Routing...";
  $("output").textContent = "";
  const payload = {
    prompt: $("prompt").value,
    profile: $("profile").value,
    model: $("model").value,
    execute: $("execute").checked,
    session_id: state.sessionId,
  };
  try {
    addMessage("user", payload.prompt);
    if (payload.execute) {
      await streamPrompt(payload);
      await loadTelemetry();
      return;
    }
    const data = await request("/api/ask", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.lastDecision = data.decision;
    renderDecision(data.decision);
    $("output").textContent = data.executed
      ? data.output || data.error || "No output returned."
      : "Dry run only. Enable provider execution to call the selected model.";
    $("status").textContent = data.executed ? "Executed" : "Dry run";
    await loadTelemetry();
  } catch (error) {
    $("status").textContent = "Error";
    $("output").textContent = error.message;
  }
}

async function sendFeedback(rating) {
  if (!state.lastDecision) {
    $("status").textContent = "Route a prompt first";
    return;
  }
  await request("/api/feedback", {
    method: "POST",
    body: JSON.stringify({
      rating,
      model: state.lastDecision.recommended_model,
      profile: $("profile").value,
      task_type: state.lastDecision.task_type,
    }),
  });
  $("status").textContent = rating === "good" ? "Feedback saved" : "Feedback saved";
  await loadTelemetry();
}

function renderDecision(decision) {
  $("summary").innerHTML = [
    ["Recommended", decision.recommended_model],
    ["Fallback", decision.fallback_model || "None"],
    ["Task", `${decision.task_type} / ${decision.complexity}`],
    ["Classifier", `${decision.classifier} / ${Math.round((decision.confidence || 0) * 100)}%`],
    ["Destination", decision.local ? "local" : "cloud"],
    ["Posture", decision.policy?.posture || "-"],
    ["Reason", decision.reason],
  ].map(([label, value]) => `<div class="metric"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(value)}</span></div>`).join("");

  // A privacy trade should never be silent.
  $("warnings").innerHTML = (decision.warnings || []).map((item) => `
    <div class="row warning">
      <strong>${escapeHtml(item.code)}</strong>
      <span>${escapeHtml(item.message)}</span>
    </div>
  `).join("");

  $("ranking").innerHTML = decision.ranked.map((item) => `
    <div class="rank">
      <div class="score">${Number(item.score).toFixed(4)}</div>
      <div>
        <strong>${escapeHtml(item.id)}</strong>
        <span>${escapeHtml(item.reasons.length ? item.reasons.join(", ") : "weighted score")}</span>
      </div>
      <span class="pill">${escapeHtml(item.provider)}</span>
    </div>
  `).join("");
  const rejected = decision.policy?.rejected || [];
  if (rejected.length) {
    $("ranking").innerHTML += `<div class="row"><strong>Policy rejected</strong><span>${escapeHtml(rejected.map((item) => `${item.id}: ${item.reason}`).join(" / "))}</span></div>`;
  }
}

async function loadHealth() {
  $("health").innerHTML = `<div class="row"><span>Checking providers...</span></div>`;
  const data = await request("/api/health");
  $("health").innerHTML = data.providers.map((provider) => `
    <div class="row">
      <strong class="${provider.ok ? "ok" : "not-ok"}">${escapeHtml(provider.id)}</strong>
      <span>${escapeHtml(provider.provider)}: ${escapeHtml(provider.detail)}</span>
    </div>
  `).join("");
}

async function loadTelemetry() {
  const data = await request("/api/telemetry");
  $("telemetry").innerHTML = data.events.length
    ? data.events.reverse().map((event) => `
      <div class="row">
        <strong>${escapeHtml(event.event || "event")}</strong>
        <span>${escapeHtml(event.created_at || "")}</span>
        <p>${escapeHtml(event.selected_model || event.decision?.recommended_model || event.profile || "")}</p>
      </div>
    `).join("")
    : `<div class="row"><span>No telemetry yet.</span></div>`;
  const dashboard = await request("/api/dashboard");
  $("dashboard").innerHTML = dashboard.models.length
    ? dashboard.models.map((item) => `<span>${escapeHtml(item.model_id)}: ${item.runs} runs, ${item.ok_rate * 100}% ok, ${item.avg_latency_ms} ms avg</span>`).join("")
    : `<span>No model runs yet.</span>`;
}

async function streamPrompt(payload) {
  $("status").textContent = "Streaming...";
  state.controller = new AbortController();
  const assistant = addMessage("assistant", "");
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: state.controller.signal,
  });
  if (!response.ok || !response.body) {
    throw new Error("Stream could not start.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop();
    frames.forEach((frame) => applyFrame(frame, assistant));
  }
}

function applyFrame(frame, assistant) {
  const event = frame.match(/^event: (.+)$/m)?.[1];
  const data = JSON.parse(frame.match(/^data: (.+)$/m)?.[1] || "{}");
  if (event === "route") {
    state.lastDecision = data.decision;
    renderDecision(data.decision);
  }
  if (event === "chunk") {
    assistant.textContent += data.text;
    $("output").textContent = assistant.textContent;
  }
  if (event === "done") $("status").textContent = "Executed";
  if (event === "fallback") $("status").textContent = `Fallback ${data.from} to ${data.to}`;
  if (event === "error") {
    $("status").textContent = "Provider error";
    assistant.textContent = data.error;
  }
}

function addMessage(role, content) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = content;
  $("messages").appendChild(message);
  message.scrollIntoView({ block: "end" });
  return message;
}

function stopStream() {
  state.controller?.abort();
  $("status").textContent = "Stopped";
}

function newChat() {
  state.sessionId = globalThis.crypto.randomUUID();
  $("messages").innerHTML = "";
  $("output").textContent = "";
  $("status").textContent = "New chat";
}

// An estimated window is worth distinguishing: it is the one that can send a
// long prompt to the cloud unnecessarily, or get it truncated.
function formatContext(model) {
  if (!model.max_context) return "context unknown";
  const size = model.max_context >= 1024
    ? `${Math.round(model.max_context / 1024)}k context`
    : `${model.max_context} context`;
  return model.context_source === "estimated" ? `${size} (estimated)` : size;
}

function renderModels(models) {
  $("models").innerHTML = models.map((model) => `
    <article class="model">
      <strong>${escapeHtml(model.name)}</strong>
      <span>${escapeHtml(model.provider)} / ${escapeHtml(model.model)}</span>
      <span>${model.local ? "local" : "external"}${model.discovered ? " / discovered" : ""}</span>
      <span>${formatContext(model)}</span>
      <label>Route with model
        <input data-model-id="${escapeHtml(model.id)}" type="checkbox" ${model.enabled ? "checked" : ""}>
      </label>
    </article>
  `).join("");
  $("models").querySelectorAll("input[data-model-id]").forEach((input) => {
    input.addEventListener("change", () => toggleModel(input.dataset.modelId, input.checked));
  });
}

async function toggleModel(id, enabled) {
  await request("/api/models", { method: "POST", body: JSON.stringify({ id, enabled }) });
  $("status").textContent = `${enabled ? "Enabled" : "Disabled"} ${id}`;
  await loadConfig();
}

// Remote sources are declared in settings, not hard-coded, so the UI only
// toggles whichever gateways the operator has configured.
function renderRemoteSources() {
  const sources = state.config.settings.catalog.remote_sources || [];
  if (!sources.length) {
    $("remoteSources").innerHTML =
      `<p class="sources">Local models are discovered automatically. To add an
       OpenAI-compatible gateway, define <code>catalog.remote_sources</code> in
       settings.</p>`;
    return;
  }
  $("remoteSources").innerHTML = sources.map((source, index) => `
    <label class="checkbox">
      <input data-source-index="${index}" type="checkbox" ${source.enabled ? "checked" : ""}>
      Include ${escapeHtml(source.id || "remote")} catalog
    </label>
  `).join("");
  for (const input of $("remoteSources").querySelectorAll("input")) {
    input.addEventListener("change", () => toggleRemoteSource(Number(input.dataset.sourceIndex), input.checked));
  }
}

async function toggleRemoteSource(index, enabled) {
  const sources = (state.config.settings.catalog.remote_sources || []).map((item, i) =>
    i === index ? { ...item, enabled } : item);
  await request("/api/settings", {
    method: "POST",
    body: JSON.stringify({ catalog: { remote_sources: sources } }),
  });
  await loadConfig();
}

async function discoverCatalog() {
  $("catalogSources").textContent = "Refreshing model sources...";
  const catalog = await request("/api/catalog");
  $("catalogSources").textContent = catalog.sources
    .map((source) => `${source.id}: ${source.ok ? `${source.count} models` : source.detail}`)
    .join(" / ");
  await loadConfig();
}

async function compareModels() {
  $("status").textContent = "Comparing...";
  const data = await request("/api/compare", {
    method: "POST",
    body: JSON.stringify({
      prompt: $("prompt").value,
      profile: $("profile").value,
      execute: $("execute").checked,
    }),
  });
  renderDecision(data.decision);
  $("compareResult").innerHTML = data.comparisons.map((item) => `
    <div class="row">
      <strong>${escapeHtml(item.model)}</strong>
      <span>${escapeHtml(item.provider)}${item.fallback_attempted ? ` / fallback ${escapeHtml(item.fallback_attempted)}` : ""}</span>
      <pre>${escapeHtml(item.error || item.output || "Dry compare route. Enable execution to compare responses.")}</pre>
    </div>
  `).join("");
  $("status").textContent = "Compared";
}

async function runEvals() {
  $("evalResults").innerHTML = `<div class="row"><span>Running routing evals...</span></div>`;
  const data = await request("/api/evals");
  $("evalResults").innerHTML = data.results.map((item) => `
    <div class="row">
      <strong class="${item.passed ? "ok" : "not-ok"}">${item.passed ? "PASS" : "FAIL"} ${escapeHtml(item.name)}</strong>
      <span>${escapeHtml(item.profile)}: expected ${escapeHtml(item.expected_model)}, got ${escapeHtml(item.actual_model)}</span>
    </div>
  `).join("");
}

function renderClassifier() {
  const classifier = state.config.settings.classifier;
  const localModels = state.config.models.filter((model) => model.provider === "ollama");
  $("classifierEnabled").checked = classifier.enabled;
  $("classifierModel").innerHTML = `<option value="">Select local model</option>` + localModels
    .map((model) => `<option value="${model.id}" ${model.id === classifier.model_id ? "selected" : ""}>${model.name}</option>`)
    .join("");
}

async function saveClassifier() {
  await request("/api/settings", {
    method: "POST",
    body: JSON.stringify({
      classifier: {
        enabled: $("classifierEnabled").checked,
        model_id: $("classifierModel").value,
      },
    }),
  });
  $("status").textContent = "Router classifier saved";
  await loadConfig();
}

function renderPolicy() {
  const policy = state.config.settings.policy;
  const routing = state.config.settings.routing;
  const pinned = state.config.pinned_policy || [];

  $("posture").innerHTML = (state.config.postures || [])
    .map((item) => `<option value="${item.id}" ${item.id === policy.posture ? "selected" : ""}>${escapeHtml(item.label)}</option>`)
    .join("");
  renderPostureDescription();

  $("preferLocal").checked = policy.prefer_local;
  $("allowCloud").checked = policy.allow_cloud;
  $("allowPrivateCloud").checked = policy.allow_cloud_for_private;
  $("localBonus").value = policy.local_bonus;
  $("floorHigh").value = routing.capability_floor.high;
  $("requestBudget").value = policy.max_estimated_cost_usd;
  $("monthlyBudget").value = policy.monthly_budget_usd;

  // Show which flags the posture no longer controls because they were pinned.
  for (const [id, key] of [
    ["preferLocal", "prefer_local"],
    ["allowCloud", "allow_cloud"],
    ["allowPrivateCloud", "allow_cloud_for_private"],
  ]) {
    $(id).closest("label").classList.toggle("pinned", pinned.includes(key));
  }
}

function renderPostureDescription() {
  const selected = (state.config.postures || []).find((item) => item.id === $("posture").value);
  $("postureDescription").textContent = selected ? selected.description : "";
}

async function setPosture() {
  await request("/api/postures", {
    method: "POST",
    body: JSON.stringify({ posture: $("posture").value }),
  });
  $("status").textContent = `Privacy posture set to ${$("posture").value}`;
  await loadConfig();
}

async function savePolicy() {
  await request("/api/settings", {
    method: "POST",
    body: JSON.stringify({
      policy: {
        prefer_local: $("preferLocal").checked,
        allow_cloud: $("allowCloud").checked,
        allow_cloud_for_private: $("allowPrivateCloud").checked,
        local_bonus: Number($("localBonus").value),
        max_estimated_cost_usd: Number($("requestBudget").value),
        monthly_budget_usd: Number($("monthlyBudget").value),
      },
      routing: {
        capability_floor: { high: Number($("floorHigh").value) },
      },
    }),
  });
  $("status").textContent = "Routing policy saved";
  await loadConfig();
}

async function resetPolicyOverrides() {
  await request("/api/postures", {
    method: "POST",
    body: JSON.stringify({ posture: $("posture").value, reset_overrides: true }),
  });
  $("status").textContent = "Policy flags reset to the posture defaults";
  await loadConfig();
}

async function boot() {
  await loadConfig();
  await loadHealth();
  await loadTelemetry();
  $("promptForm").addEventListener("submit", routePrompt);
  $("refreshHealth").addEventListener("click", loadHealth);
  $("refreshTelemetry").addEventListener("click", loadTelemetry);
  $("refreshModels").addEventListener("click", discoverCatalog);
  $("saveClassifier").addEventListener("click", saveClassifier);
  $("savePolicy").addEventListener("click", savePolicy);
  $("posture").addEventListener("change", setPosture);
  $("resetPolicyOverrides").addEventListener("click", resetPolicyOverrides);
  $("goodFeedback").addEventListener("click", () => sendFeedback("good"));
  $("badFeedback").addEventListener("click", () => sendFeedback("bad"));
  $("stop").addEventListener("click", stopStream);
  $("newChat").addEventListener("click", newChat);
  $("compare").addEventListener("click", compareModels);
  $("runEvals").addEventListener("click", runEvals);
}

boot().catch((error) => {
  $("status").textContent = "Startup error";
  $("output").textContent = error.message;
});
