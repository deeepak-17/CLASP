# CLASP Edge Layer — PEFT training, 4-bit inference, dynamic merger (P1)
FROM python:3.11-slim
WORKDIR /app
COPY contracts /app/contracts
COPY services/edge /app/service
RUN pip install --no-cache-dir -e /app/contracts -e /app/service
CMD ["python", "-m", "edge"]
