const state = {
  config: null,
  lastDecision: null,
  sessionId: globalThis.crypto.randomUUID(),
  controller: null,
};

const $ = (id) => document.getElementById(id);

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || data.fallback_error || data.decision?.reason || "Request failed");
  }
  return data;
}

async function loadConfig() {
  state.config = await request("/api/config");
  $("profile").innerHTML = state.config.profiles
    .map((name) => `<option value="${name}" ${name === state.config.default_profile ? "selected" : ""}>${name}</option>`)
    .join("");
  $("model").innerHTML = `<option value="">Auto</option>` + state.config.models
    .filter((model) => model.enabled)
    .map((model) => `<option value="${model.id}">${model.name} (${model.provider})</option>`)
    .join("");
  renderModels(state.config.models);
  renderClassifier();
  renderPolicy();
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
    ["Reason", decision.reason],
  ].map(([label, value]) => `<div class="metric"><strong>${label}</strong><span>${value}</span></div>`).join("");

  $("ranking").innerHTML = decision.ranked.map((item) => `
    <div class="rank">
      <div class="score">${Number(item.score).toFixed(4)}</div>
      <div>
        <strong>${item.id}</strong>
        <span>${item.reasons.length ? item.reasons.join(", ") : "weighted score"}</span>
      </div>
      <span class="pill">${item.provider}</span>
    </div>
  `).join("");
  const rejected = decision.policy?.rejected || [];
  if (rejected.length) {
    $("ranking").innerHTML += `<div class="row"><strong>Policy rejected</strong><span>${rejected.map((item) => `${item.id}: ${item.reason}`).join(" / ")}</span></div>`;
  }
}

async function loadHealth() {
  $("health").innerHTML = `<div class="row"><span>Checking providers...</span></div>`;
  const data = await request("/api/health");
  $("health").innerHTML = data.providers.map((provider) => `
    <div class="row">
      <strong class="${provider.ok ? "ok" : "not-ok"}">${provider.id}</strong>
      <span>${provider.provider}: ${provider.detail}</span>
    </div>
  `).join("");
}

async function loadTelemetry() {
  const data = await request("/api/telemetry");
  $("telemetry").innerHTML = data.events.length
    ? data.events.reverse().map((event) => `
      <div class="row">
        <strong>${event.event || "event"}</strong>
        <span>${event.created_at || ""}</span>
        <p>${event.selected_model || event.decision?.recommended_model || event.profile || ""}</p>
      </div>
    `).join("")
    : `<div class="row"><span>No telemetry yet.</span></div>`;
  const dashboard = await request("/api/dashboard");
  $("dashboard").innerHTML = dashboard.models.length
    ? dashboard.models.map((item) => `<span>${item.model_id}: ${item.runs} runs, ${item.ok_rate * 100}% ok, ${item.avg_latency_ms} ms avg</span>`).join("")
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

function renderModels(models) {
  $("models").innerHTML = models.map((model) => `
    <article class="model">
      <strong>${model.name}</strong>
      <span>${model.provider} / ${model.model}</span>
      <span>${model.local ? "local" : "external"}${model.discovered ? " / discovered" : ""}</span>
      <label>Route with model
        <input data-model-id="${model.id}" type="checkbox" ${model.enabled ? "checked" : ""}>
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
      <strong>${item.model}</strong>
      <span>${item.provider}${item.fallback_attempted ? ` / fallback ${item.fallback_attempted}` : ""}</span>
      <pre>${item.error || item.output || "Dry compare route. Enable execution to compare responses."}</pre>
    </div>
  `).join("");
  $("status").textContent = "Compared";
}

async function runEvals() {
  $("evalResults").innerHTML = `<div class="row"><span>Running routing evals...</span></div>`;
  const data = await request("/api/evals");
  $("evalResults").innerHTML = data.results.map((item) => `
    <div class="row">
      <strong class="${item.passed ? "ok" : "not-ok"}">${item.passed ? "PASS" : "FAIL"} ${item.name}</strong>
      <span>${item.profile}: expected ${item.expected_model}, got ${item.actual_model}</span>
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
  $("preferLocal").checked = policy.prefer_local;
  $("allowCloud").checked = policy.allow_cloud;
  $("allowPrivateCloud").checked = policy.allow_cloud_for_private;
  $("requestBudget").value = policy.max_estimated_cost_usd;
  $("monthlyBudget").value = policy.monthly_budget_usd;
}

async function savePolicy() {
  await request("/api/settings", {
    method: "POST",
    body: JSON.stringify({
      policy: {
        prefer_local: $("preferLocal").checked,
        allow_cloud: $("allowCloud").checked,
        allow_cloud_for_private: $("allowPrivateCloud").checked,
        max_estimated_cost_usd: Number($("requestBudget").value),
        monthly_budget_usd: Number($("monthlyBudget").value),
      },
    }),
  });
  $("status").textContent = "Routing policy saved";
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
