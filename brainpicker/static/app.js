const state = {
  config: null,
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

async function boot() {
  await loadConfig();
  await loadHealth();
  await loadTelemetry();
  $("promptForm").addEventListener("submit", routePrompt);
  $("refreshHealth").addEventListener("click", loadHealth);
  $("refreshTelemetry").addEventListener("click", loadTelemetry);
}

boot().catch((error) => {
  $("status").textContent = "Startup error";
  $("output").textContent = error.message;
});

