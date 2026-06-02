#!/usr/bin/env bash
# Generate Python stubs for hyw_sim protos (used by waymo_to_scenario / split scripts).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKBENCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HYW_ROOT="$(cd "${WORKBENCH_ROOT}/.." && pwd)"

PROTO_ROOT="${HYW_ROOT}/hyw-proto"
OUT="${SCRIPT_DIR}/gen"

if ! command -v protoc >/dev/null 2>&1; then
  echo "protoc not found; install protobuf-compiler (e.g. apt install protobuf-compiler)" >&2
  exit 1
fi

mkdir -p "${OUT}"
protoc -I "${PROTO_ROOT}" \
  --python_out="${OUT}" \
  "${PROTO_ROOT}/proto/sim/common.proto" \
  "${PROTO_ROOT}/proto/sim/map.proto" \
  "${PROTO_ROOT}/proto/sim/scenario.proto"

mkdir -p "${OUT}/proto/sim"
touch "${OUT}/__init__.py" "${OUT}/proto/__init__.py" "${OUT}/proto/sim/__init__.py" 2>/dev/null || true

echo "[gen_sim_protos] wrote Python stubs under ${OUT}/proto/sim/"
