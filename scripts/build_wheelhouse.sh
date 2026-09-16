#!/usr/bin/env bash
# Build an offline install bundle.
#
# Hearth's offline guarantee covers runtime, not first install. This script downloads every
# wheel on a connected machine so the target machine never needs an index.
#
#   Connected machine:   ./scripts/build_wheelhouse.sh
#   Target machine:      uv pip install --no-index --find-links wheelhouse hearth
#
# Build on the same platform and Python version as the target: grammar wheels and any other
# binary wheels are platform-specific.
set -euo pipefail

DEST="${1:-wheelhouse}"
EXTRAS="${EXTRAS:-evals}"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

mkdir -p "$DEST"

echo "Exporting locked requirements (extras: ${EXTRAS})..."
uv export --frozen --no-dev --extra "$EXTRAS" --format requirements-txt >"$DEST/requirements.txt"

echo "Downloading wheels into ${DEST}/ ..."
uv pip download --requirements "$DEST/requirements.txt" --dest "$DEST"

echo
echo "Wheelhouse ready: $(find "$DEST" -name '*.whl' | wc -l) wheels in ${DEST}/"
echo "Copy the directory to the target machine, then:"
echo "  uv pip install --no-index --find-links ${DEST} -r ${DEST}/requirements.txt"
echo
echo "Models are separate: 'ollama pull' on a connected machine, or copy ~/.ollama/models."
