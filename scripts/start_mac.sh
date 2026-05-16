#!/usr/bin/env bash
# One-time setup + instructions (macOS)
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt

echo ""
echo "=== Python OK. Start in TWO terminals: ==="
echo ""
echo "  Terminal 1 — tracker + AI:"
echo "    cd $ROOT && source .venv/bin/activate && python main.py"
echo ""
echo "  Terminal 2 — React HUD:"
echo "    cd $ROOT/hud && npm install && npm run dev"
echo ""
echo "  Browser: http://localhost:5173"
echo "  Keys: ESC quit | T push-to-talk (press twice)"
echo ""
