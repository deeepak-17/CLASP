import React from "react";
import ReactDOM from "react-dom/client";
import { HashRouter } from "react-router-dom";
import App from "./App";
import "./index.css";

// HashRouter, not BrowserRouter: this is a static build with no server-side
// routing (no backend at all, per the Week-5 brief) — hash routes work
// correctly from a plain `vite preview` / any static file host with zero
// server configuration.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </React.StrictMode>,
);
