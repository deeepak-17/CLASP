import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Overview", end: true },
  { to: "/benchmarks", label: "Benchmarks" },
  { to: "/pass-at-k", label: "Pass@k" },
  { to: "/pipeline", label: "Evaluation Pipeline" },
];

interface AppShellProps {
  children: ReactNode;
  topbarRight?: ReactNode;
}

/** Compact sidebar + topbar shell — the "application shell" the Week-5
 * brief asks for, not a marketing layout. Navigation is fixed to the four
 * pages the brief names; nothing else is added. */
export function AppShell({ children, topbarRight }: AppShellProps) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="name">CLASP Evaluation</div>
          <div className="role">P5 — Eval &amp; Data</div>
        </div>
        <nav>
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          Team 102 · CLASP
          <br />
          Phase II · Week 5
        </div>
      </aside>
      <header className="topbar">
        <span className="topbar-title">Evaluation &amp; Data Dashboard</span>
        <div>{topbarRight}</div>
      </header>
      <main className="content">{children}</main>
    </div>
  );
}
