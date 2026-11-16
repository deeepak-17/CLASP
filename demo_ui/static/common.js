// Shared helpers for the 3 demo pages. No business logic here — only fetch
// wrappers and DOM rendering. All real data/decisions come from the demo_ui
// backend (demo_ui/server.py), which itself only reads real files or proxies
// to the real Cluster/Registry FastAPI apps.

// The login token lives in sessionStorage: it goes with the tab, is sent on
// every call, and any 401 (no token, an expired one) asks the page to show its
// login form ("clasp-login-required"). The server keeps the round either way.
const TOKEN_KEY = "clasp-token";
const authHeaders = () => {
  const token = sessionStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
};
function loginRequired(res, path) {
  if (res.status === 401 && path !== "/api/login") {
    sessionStorage.removeItem(TOKEN_KEY);
    window.dispatchEvent(new Event("clasp-login-required"));
  }
}

async function apiGet(path) {
  const res = await fetch(path, { headers: authHeaders() });
  loginRequired(res, path);
  const body = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, body };
}

async function apiPost(path, json) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: json ? JSON.stringify(json) : undefined,
  });
  loginRequired(res, path);
  const body = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, body };
}

async function apiLogin(username, password) {
  const r = await apiPost("/api/login", { username, password });
  if (r.ok) sessionStorage.setItem(TOKEN_KEY, r.body.token);
  return r;
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
