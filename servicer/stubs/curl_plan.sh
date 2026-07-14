#!/usr/bin/env bash
set -euo pipefail

URL="${HANWAM_SERVICE_URL:-http://127.0.0.1:8080/v1/plan}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

curl -sS \
  -X POST "${URL}" \
  -H 'Content-Type: application/json' \
  --data-binary "@${SCRIPT_DIR}/sample_plan_request.json"
