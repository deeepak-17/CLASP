# CLASP Cluster Layer — Flower + FedProx aggregation, FastAPI (P2)
FROM python:3.11-slim
WORKDIR /app
COPY contracts /app/contracts
COPY services/cluster /app/service
RUN pip install --no-cache-dir -e /app/contracts -e /app/service
EXPOSE 8002
# Plain HTTP unless CLASP_TLS_CERT/KEY/CA are all set, then mTLS + TLS 1.3 with
# uploads bound to the client certificate's CN (cluster.serve).
HEALTHCHECK --interval=10s --timeout=5s --retries=5 CMD ["python", "-m", "cluster.healthcheck"]
CMD ["python", "-m", "cluster.serve"]
