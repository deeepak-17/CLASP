# Security (P3)

**Owner:** Kapilan (P3)

> **Library, not a service.** This module is imported by the Edge and Cluster
> layers to apply Opacus DP-SGD (model-weight privacy) and mTLS (transport
> security). It has **no Dockerfile** and is never deployed on its own.

```bash
pip install -e contracts -e security
pytest security/tests
```
