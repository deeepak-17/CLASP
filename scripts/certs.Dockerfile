# The one-shot `certs` service of docker-compose.yml: bash + openssl, which is
# all scripts/make_dev_certs.sh needs. The repository is mounted at /work.
FROM alpine:3.20
RUN apk add --no-cache bash openssl
