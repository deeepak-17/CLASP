// Shared helpers for the 3 demo pages. No business logic here — only fetch
// wrappers and DOM rendering. All real data/decisions come from the demo_ui
// backend (demo_ui/server.py), which itself only reads real files or proxies
// to the real Cluster/Registry FastAPI apps.

async function apiGet(path) {
  const res = await fetch(path);
  const body = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, body };
}

async function apiPost(path, json) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: json ? JSON.stringify(json) : undefined,
  });
  const body = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, body };
}

function fmt(n, digits = 3) {
  if (n === null || n === undefined) return "—";
  if (typeof n !== "number") return String(n);
  return n.toFixed(digits);
}

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstChild;
}

function log(container, text) {
  const line = document.createElement("div");
  const ts = new Date().toLocaleTimeString();
  line.textContent = `[${ts}] ${text}`;
  container.appendChild(line);
  container.scrollTop = container.scrollHeight;
}

async function refreshHealth() {
  const bar = document.getElementById("healthbar");
  if (!bar) return;
  const { body } = await apiGet("/api/health");
  const dot = (s) => (s === "ok" ? "ok" : s === "unreachable" ? "bad" : "unknown");
  bar.innerHTML = `
    <span><span class="dot ${dot(body.cluster)}"></span>cluster :8002</span>
    <span><span class="dot ${dot(body.registry)}"></span>registry :8004</span>
  `;
}

document.addEventListener("DOMContentLoaded", () => {
  refreshHealth();
  setInterval(refreshHealth, 5000);
});
