# CLASP Evaluation & Dashboard — HumanEval/MBPP, Pass@k, React/Recharts (P5)
FROM python:3.11-slim
WORKDIR /app
COPY contracts /app/contracts
COPY services/evaluation /app/service
RUN pip install --no-cache-dir -e /app/contracts -e /app/service
CMD ["python", "-m", "evaluation"]
