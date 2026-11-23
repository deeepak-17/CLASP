# CLASP panel demo UI — presentation layer only, not a CLASP service (P4).
# Started by `docker compose --profile demo up`; never part of the default stack.
FROM python:3.11-slim
WORKDIR /app
COPY demo_ui/requirements.txt /app/demo_ui/requirements.txt
# P3's library (logins, signatures), without its DP extra: no torch in this image
COPY contracts /app/contracts
COPY security /app/security
RUN pip install --no-cache-dir -r /app/demo_ui/requirements.txt -e /app/contracts -e /app/security
COPY demo_ui /app/demo_ui
EXPOSE 8010
# HTTPS when CLASP_PAGE_TLS_CERT / CLASP_PAGE_TLS_KEY are set, else plain HTTP (demo_ui.serve).
CMD ["python", "-m", "demo_ui.serve"]
