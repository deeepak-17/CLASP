# State Registry — safetensors versioning, two-sided promotion, FastAPI

**Owner:** Deepak (P4)

Versioned safetensors adapter storage with save/load/list endpoints (W3);
two-sided promotion/rollback (D5) lands in W5. Depends on the shared `contracts`
package; keep cross-module interaction interface-driven.

- API + schemas: `docs/registry_api.md`
- Architecture + data flow: `docs/architecture.md`

```bash
pip install -e contracts -e "services/registry[test]"
pytest services/registry/tests -q                 # storage + endpoints + manifest
CLASP_REGISTRY_DATA=./_data uvicorn registry.app:app --port 8004   # docs at /docs
```

Data persists under `CLASP_REGISTRY_DATA` (compose named volume `registry-data`).
Versions are immutable — never overwritten in place (D9).
