# Evaluation & Dashboard — HumanEval/MBPP, Pass@k, React/Recharts

**Owner:** Aditya (P5)

Depends on the shared `contracts` package; keep cross-module interaction
interface-driven. Everything P5 owns lives in this directory — see
[`docs/p5_integration_handoff.md`](docs/p5_integration_handoff.md) for the layout
and seams.

```bash
pip install -e contracts -e "services/evaluation[test]"
pytest services/evaluation/tests

cd services/evaluation
python scripts/rebuild_results.py        # every result, report, figure + the archive, no GPU
cd dashboard && npm install && npm run dev   # dashboard (or: docker compose up -d dashboard → :8005)
```

| Read | For |
|---|---|
| [`docs/report/eval_section.md`](docs/report/eval_section.md) | the evaluation section of the Phase II report |
| [`docs/demo_scenario.md`](docs/demo_scenario.md) | the demo walkthrough and its dataset |
| [`docs/debt_list.md`](docs/debt_list.md) | review feedback and what is still open |
| [`docs/phase3_ablation_reuse.md`](docs/phase3_ablation_reuse.md) | reusing the harness for Phase III sweeps |
