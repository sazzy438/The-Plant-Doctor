#!/usr/bin/env bash
# Plant Doctor - setup and launch for macOS / Linux
set -e
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "Creating a private Python environment (first run only)..."
  python3 -m venv .venv
fi
source .venv/bin/activate
if [ ! -f ".venv/.deps_installed" ]; then
  echo "Installing dependencies (first run only)..."
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r requirements.txt
  echo ok > .venv/.deps_installed
fi
if [ ! -f "artifacts/plant_doctor_efficientnet_b0.pt" ]; then
  echo "The trained model artifacts/plant_doctor_efficientnet_b0.pt is missing. Copy it into artifacts/ or train one with notebooks/training_colab.ipynb."
  exit 1
fi
echo "Starting Plant Doctor at http://127.0.0.1:7860 (Ctrl+C to stop)"
PLANT_DOCTOR_OPEN_BROWSER=1 python app.py
