#!/usr/bin/env bash
# Upload this script + requirements.txt to your server, then run:
#   chmod +x prepare_remote_server.sh && ./prepare_remote_server.sh
#
# Creates .venv, installs requirements.txt, downloads The Well dataset.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-venv}"
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" &>/dev/null; then
  echo "error: $PYTHON not found. Install Python 3.10+ or set PYTHON=/path/to/python3" >&2
  exit 1
fi

if [[ ! -f requirements.txt ]]; then
  echo "error: requirements.txt not found in $ROOT" >&2
  exit 1
fi

echo "==> Virtualenv: $ROOT/$VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing requirements.txt (this may take a while)"
pip install -r requirements.txt

if ! command -v the-well-download &>/dev/null; then
  echo "error: the-well-download not on PATH after install (check the_well package)" >&2
  exit 1
fi

echo "==> Downloading dataset (blocks until finished)"
the-well-download --base-path ./data --dataset turbulent_radiative_layer_2D

echo "==> Done. Activate later with: source $VENV_DIR/bin/activate"
