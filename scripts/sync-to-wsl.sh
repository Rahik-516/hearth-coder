#!/bin/bash
# Mirrors the Windows-side working tree into the WSL-native checkout used for testing.
# Run from WSL: bash "/mnt/e/AI-Projects/Local offline Coding agent/scripts/sync-to-wsl.sh"
set -euo pipefail

rsync -a --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '.git' --exclude '.pytest_cache' \
  --exclude '.mypy_cache' --exclude '.ruff_cache' \
  "/mnt/e/AI-Projects/Local offline Coding agent/" \
  "/home/hearth/code/hearth/"

echo "synced"
