# Evaluation & Dashboard — HumanEval/MBPP, Pass@k, React/Recharts

**Owner:** Aditya (P5)

Depends on the shared `contracts` package; keep cross-module interaction
interface-driven. Everything P5 owns lives in this directory — see
[`docs/p5_integration_handoff.md`](docs/p5_integration_handoff.md) for the layout
and seams.

```bash
pip install -e contracts -e "services/evaluation[test]"
pytest services/evaluation/tests

cd services/evaluation && python scripts/run_baseline_eval.py --help   # CLIs run from here
cd dashboard && npm install && npm run dev                              # dashboard
```
