#!/bin/sh
# Runs from /docker-entrypoint.d/ before nginx starts. When the pages
# certificate is mounted at /certs (docker-compose.yml), the dashboard is served
# over HTTPS on the same port; otherwise (or with CLASP_DASHBOARD_PLAIN_HTTP=1,
# the insecure overlay) the image's plain-HTTP default is left as it is.
set -eu

if [ "${CLASP_DASHBOARD_PLAIN_HTTP:-}" = "1" ] || [ ! -f /certs/pages.pem ] || [ ! -f /certs/pages.key ]; then
  echo "dashboard: serving plain HTTP (no pages certificate mounted at /certs)"
  exit 0
fi

cat > /etc/nginx/conf.d/default.conf <<'EOF'
server {
    listen 80 ssl;
    server_name _;
    ssl_certificate     /certs/pages.pem;
    ssl_certificate_key /certs/pages.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    # a browser that asks in plain HTTP on this port is sent to HTTPS
    error_page 497 =301 https://$host:8005$request_uri;

    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header Strict-Transport-Security "max-age=31536000" always;

    location / {
        root   /usr/share/nginx/html;
        index  index.html;
        try_files $uri $uri/ /index.html;
    }
}
EOF
echo "dashboard: serving HTTPS with /certs/pages.pem"
