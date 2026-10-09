# CLASP State Registry — safetensors versioning, two-sided promotion/rollback (D5),
# pre-merged composites (D6), retention/GC, optional mTLS (D7). FastAPI (P4).
FROM python:3.11-slim
WORKDIR /app
COPY contracts /app/contracts
COPY services/registry /app/service
RUN pip install --no-cache-dir -e /app/contracts -e /app/service
ENV CLASP_REGISTRY_DATA=/data/registry \
    CLASP_REGISTRY_PORT=8004
EXPOSE 8004
# Plain HTTP unless CLASP_TLS_CERT/KEY/CA are all set, then mTLS + TLS 1.3.
HEALTHCHECK --interval=10s --timeout=5s --retries=5 CMD ["python", "-m", "registry.healthcheck"]
CMD ["python", "-m", "registry.serve"]
