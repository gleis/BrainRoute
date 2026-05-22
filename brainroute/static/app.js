const state = {
  config: null,
  lastDecision: null,
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
  };
  try {
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

async function boot() {
  await loadConfig();
  await loadHealth();
  await loadTelemetry();
  $("promptForm").addEventListener("submit", routePrompt);
  $("refreshHealth").addEventListener("click", loadHealth);
  $("refreshTelemetry").addEventListener("click", loadTelemetry);
  $("refreshModels").addEventListener("click", discoverCatalog);
  $("goodFeedback").addEventListener("click", () => sendFeedback("good"));
  $("badFeedback").addEventListener("click", () => sendFeedback("bad"));
}

boot().catch((error) => {
  $("status").textContent = "Startup error";
  $("output").textContent = error.message;
});
