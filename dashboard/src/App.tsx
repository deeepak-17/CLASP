import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Benchmarks } from "./pages/Benchmarks";
import { Overview } from "./pages/Overview";
import { PassAtK } from "./pages/PassAtK";
import { Pipeline } from "./pages/Pipeline";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/benchmarks" element={<Benchmarks />} />
        <Route path="/pass-at-k" element={<PassAtK />} />
        <Route path="/pipeline" element={<Pipeline />} />
      </Routes>
    </AppShell>
  );
}
