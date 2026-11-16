# CLASP evaluation dashboard — static build served by nginx.
# Build context is the repository root (docker-compose.yml), so the build can
# read services/evaluation/results/, which `npm run build` copies into the app.
FROM node:20.17-alpine AS build
WORKDIR /src/services/evaluation
COPY services/evaluation/dashboard/package.json services/evaluation/dashboard/package-lock.json dashboard/
RUN cd dashboard && npm ci --no-audit --no-fund
COPY services/evaluation/results results
COPY services/evaluation/dashboard dashboard
RUN cd dashboard && npm run build

FROM nginx:1.27-alpine
COPY --from=build /src/services/evaluation/dashboard/dist /usr/share/nginx/html
# HTTPS when the pages certificate is mounted at /certs (see the script).
COPY services/evaluation/dashboard/nginx-tls.sh /docker-entrypoint.d/30-clasp-tls.sh
RUN sed -i 's/$//' /docker-entrypoint.d/30-clasp-tls.sh && chmod +x /docker-entrypoint.d/30-clasp-tls.sh
EXPOSE 80
# plain HTTP, or HTTPS on the same port (the certificate is the dev CA's: not verified here)
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD wget -q -O /dev/null http://localhost/ || wget -q --no-check-certificate -O /dev/null https://localhost:80/ || exit 1
