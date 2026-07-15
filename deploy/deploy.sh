#!/usr/bin/env bash
# Runs on the production VM. The workflow streams this file over SSH.
set -Eeuo pipefail

image_ref=${1:?immutable image reference is required}
deploy_dir=${2:-/opt/uniche-media-editor}

cd "$deploy_dir"

if [[ ! -f .env ]]; then
  echo "Deployment stopped: $deploy_dir/.env does not exist." >&2
  exit 1
fi

if [[ -n $(find .env -maxdepth 0 -perm /077 -print -quit) ]]; then
  echo "Deployment stopped: $deploy_dir/.env must not be accessible by group or others (use chmod 600)." >&2
  exit 1
fi

if [[ ! -f compose.prod.yml.new ]]; then
  echo "Deployment stopped: uploaded Compose file is missing." >&2
  exit 1
fi

printf 'APP_IMAGE=%s\n' "$image_ref" > .image.env.new
chmod 600 .image.env.new

# Validate the uploaded file before replacing the last known Compose definition.
docker compose --env-file .image.env.new -f compose.prod.yml.new config --quiet
mv compose.prod.yml.new compose.prod.yml

compose=(docker compose --env-file .image.env.new -f compose.prod.yml)

on_error() {
  echo "Deployment failed. Container status on the VM:" >&2
  "${compose[@]}" ps >&2 || true
}
trap on_error ERR

"${compose[@]}" pull
"${compose[@]}" up -d postgres redis
"${compose[@]}" run --rm migrate
"${compose[@]}" up -d --remove-orphans api worker

for attempt in {1..30}; do
  if curl --fail --silent --show-error http://127.0.0.1:8000/health >/dev/null; then
    mv .image.env.new .image.env
    trap - ERR
    echo "Deployment health check passed."
    exit 0
  fi
  sleep 2
done

echo "Deployment stopped: API health check did not pass within 60 seconds." >&2
exit 1
