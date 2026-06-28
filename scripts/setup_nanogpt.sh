#!/bin/bash
set -euo pipefail
exec > ~/nanogpt_setup.log 2>&1
echo "=== setup start: $(date) ==="
mkdir -p ~/research && cd ~/research
[ -d modded-nanogpt ] || git clone https://github.com/KellerJordan/modded-nanogpt.git
cd modded-nanogpt
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "=== downloading fineweb tokens: $(date) ==="
python data/cached_fineweb10B.py 9
echo "=== baseline speedrun: $(date) ==="
./run.sh
echo "=== done: $(date) ==="
