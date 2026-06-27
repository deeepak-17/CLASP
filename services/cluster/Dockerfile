# CLASP Cluster Layer — Flower + FedProx aggregation, FastAPI (P2)
FROM python:3.11-slim
WORKDIR /app
COPY contracts /app/contracts
COPY services/cluster /app/service
RUN pip install --no-cache-dir -e /app/contracts -e /app/service
CMD ["uvicorn", "cluster.server:app", "--host", "0.0.0.0", "--port", "8002"]
