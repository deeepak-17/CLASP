# contracts — CLASP shared interfaces

The integration seam of the monorepo. Every service installs this package and
communicates through the types defined here, rather than importing each other
directly. Changes here are high-impact: flag them and keep them
backward-compatible where possible.

```bash
pip install -e contracts
```
