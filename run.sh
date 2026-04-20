#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
#  Ho Jaayega v2 — Launcher
# ─────────────────────────────────────────────────────────

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

VENV="../.venv"

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║     Ho Jaayega v2 — AI Text Detector      ║"
echo "  ║   with Selective Feature Group Analysis   ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

if [ ! -d "$VENV" ]; then
    echo "  ✖  Virtual environment not found at $VENV"
    exit 1
fi

echo "  ✔  Virtual environment found"
echo "  ✔  Starting server on http://127.0.0.1:8000"
echo "  ✔  Press Ctrl+C to stop"
echo ""

"$VENV/bin/uvicorn" app:app --host 127.0.0.1 --port 8000 --reload
