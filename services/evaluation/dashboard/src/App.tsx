import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Benchmarks } from "./pages/Benchmarks";
import { InProject } from "./pages/InProject";
import { Lineage } from "./pages/Lineage";
import { Noise } from "./pages/Noise";
import { Overview } from "./pages/Overview";
import { PassAtK } from "./pages/PassAtK";
import { Personalization } from "./pages/Personalization";
import { Pipeline } from "./pages/Pipeline";
import { Rounds } from "./pages/Rounds";

export default function App() {
  return (
    <AppShell>
      <Routes>
        {/* Each page reads one JSON feed from public/data/ (copied from services/evaluation/results/ by npm run sync-data). */}
        <Route path="/" element={<Overview />} />
        <Route path="/in-project" element={<InProject />} />
        <Route path="/personalization" element={<Personalization />} />
        <Route path="/rounds" element={<Rounds />} />
        <Route path="/lineage" element={<Lineage />} />
        <Route path="/noise" element={<Noise />} />
        <Route path="/benchmarks" element={<Benchmarks />} />
        <Route path="/pass-at-k" element={<PassAtK />} />
        <Route path="/pipeline" element={<Pipeline />} />
      </Routes>
    </AppShell>
  );
}
