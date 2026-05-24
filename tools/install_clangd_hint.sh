#!/usr/bin/env bash
# Verify clangd for Cursor / VS Code (WSL).
set -euo pipefail

LOCAL_CLANGD="${HOME}/.local/bin/clangd"
if [[ -x "$LOCAL_CLANGD" ]]; then
  echo "OK: clangd found at $LOCAL_CLANGD"
  "$LOCAL_CLANGD" --version | head -1
  echo ""
  echo "Workspace settings use: clangd.path -> $LOCAL_CLANGD"
  exit 0
fi

if command -v clangd-18 >/dev/null 2>&1; then
  echo "OK: clangd-18 at $(command -v clangd-18)"
  clangd-18 --version | head -1
  exit 0
fi

echo "clangd not found. Install one of:"
echo "  sudo apt install clangd-18"
echo "  # or download from https://github.com/clangd/clangd/releases"
exit 1
