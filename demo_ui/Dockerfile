# CLASP panel demo UI — presentation layer only, not a CLASP service (P4).
# Started by `docker compose --profile demo up`; never part of the default stack.
FROM python:3.11-slim
WORKDIR /app
COPY demo_ui/requirements.txt /app/demo_ui/requirements.txt
RUN pip install --no-cache-dir -r /app/demo_ui/requirements.txt
COPY demo_ui /app/demo_ui
EXPOSE 8010
CMD ["uvicorn", "demo_ui.server:app", "--host", "0.0.0.0", "--port", "8010"]
