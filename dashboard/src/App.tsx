import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Benchmarks } from "./pages/Benchmarks";
import { InProject } from "./pages/InProject";
import { Overview } from "./pages/Overview";
import { PassAtK } from "./pages/PassAtK";
import { Pipeline } from "./pages/Pipeline";
import { Rounds } from "./pages/Rounds";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/in-project" element={<InProject />} />
        <Route path="/rounds" element={<Rounds />} />
        <Route path="/benchmarks" element={<Benchmarks />} />
        <Route path="/pass-at-k" element={<PassAtK />} />
        <Route path="/pipeline" element={<Pipeline />} />
      </Routes>
    </AppShell>
  );
}
