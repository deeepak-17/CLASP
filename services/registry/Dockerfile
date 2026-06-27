# CLASP State Registry — safetensors versioning, Pass@k rollback, FastAPI (P4)
FROM python:3.11-slim
WORKDIR /app
COPY contracts /app/contracts
COPY services/registry /app/service
RUN pip install --no-cache-dir -e /app/contracts -e /app/service
CMD ["uvicorn", "registry.app:app", "--host", "0.0.0.0", "--port", "8004"]
