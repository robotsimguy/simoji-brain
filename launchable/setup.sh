#!/usr/bin/env bash
# Simoji robot brain: one-time setup of a fresh Brev GPU instance (Ubuntu + NVIDIA driver).
# Installs NVIDIA Isaac GR00T + the Simoji connector, then starts three services:
#   simoji-groot   GR00T policy server (ZMQ, localhost:5555)
#   simoji-brain   simoji-policy: pairs with the Simoji relay, bridges the phone <-> GR00T
#   simoji-page    the ready page on port 8080 (code, Open in Simoji, QR, Stop now)
# plus simoji-idle.timer (powers off after SIMOJI_IDLE_MINUTES without a phone).
# Safe to re-run.
set -euo pipefail

REPO_URL="${SIMOJI_BRAIN_REPO:-https://github.com/robotsimguy/simoji-brain}"
REPO_REF="${SIMOJI_BRAIN_REF:-main}"
GROOT_URL="https://github.com/NVIDIA/Isaac-GR00T"
GROOT_REF="${SIMOJI_GROOT_REF:-7aa0bca}"   # Isaac-GR00T main, 2026-11-07 (tested)
MODEL="${SIMOJI_MODEL:-nvidia/GR00T-N1.7-DROID}"
EMBODIMENT="${SIMOJI_EMBODIMENT:-OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT}"
IDLE_MIN="${SIMOJI_IDLE_MINUTES:-30}"
ROOT=/opt/simoji
STATE=/var/lib/simoji
RUN_USER="$(id -un)"

log() { echo "[simoji] $*"; }
status() { sudo mkdir -p "$STATE"; echo "{\"state\":\"loading\",\"message\":\"$1\",\"updated\":$(date +%s)}" | sudo tee "$STATE/status.json" >/dev/null; }

sudo mkdir -p "$ROOT" "$STATE" && sudo chown -R "$RUN_USER" "$ROOT" "$STATE"
cd "$ROOT"

# 1. The ready page first, so the user sees progress from the first minute.
if [ ! -d brain/.git ]; then git clone -q "$REPO_URL" brain; fi
git -C brain fetch -q origin "$REPO_REF" && git -C brain checkout -q FETCH_HEAD
status "installing (step 1 of 3)…"
sudo tee /etc/systemd/system/simoji-page.service >/dev/null <<EOF
[Unit]
Description=Simoji ready page
After=network-online.target
[Service]
User=$RUN_USER
Environment=SIMOJI_IDLE_MINUTES=$IDLE_MIN
ExecStart=/usr/bin/python3 $ROOT/brain/launchable/ready_page.py --status $STATE/status.json --port 8080
Restart=always
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now simoji-page
# Stop now on the page needs a password-less shutdown for this user only.
echo "$RUN_USER ALL=(root) NOPASSWD: /usr/sbin/shutdown" | sudo tee /etc/sudoers.d/simoji-shutdown >/dev/null
sudo chmod 440 /etc/sudoers.d/simoji-shutdown

# 2. GR00T + the connector in one environment (uv). GR00T needs git-lfs (its wheels) and FFmpeg 4-7 (torchcodec).
status "installing system packages (step 2 of 3)…"
sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git-lfs ffmpeg >/dev/null
git lfs install --skip-repo >/dev/null
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -d Isaac-GR00T/.git ]; then GIT_LFS_SKIP_SMUDGE=1 git clone -q "$GROOT_URL" Isaac-GR00T; fi
cd Isaac-GR00T
GIT_LFS_SKIP_SMUDGE=1 git fetch -q origin && GIT_LFS_SKIP_SMUDGE=1 git checkout -q "$GROOT_REF"
git submodule update --init --recursive -q
git lfs pull --include "scripts/deployment/*/wheels/*.whl"
status "installing NVIDIA GR00T (step 2 of 3, a few minutes)…"
uv sync -q --python 3.12
uv pip install -q -e . -e "$ROOT/brain/simoji-policy"
uv pip uninstall -q deepspeed 2>/dev/null || true   # imports need a CUDA toolkit we don't use for inference
PY="$ROOT/Isaac-GR00T/.venv/bin/python"

# 3. Model weights (public, no token). The gated Cosmos-Reason2 backbone comes from Simoji's
#    redistributed copy (NVIDIA Open Model License); see the repo README.
status "downloading the model (step 3 of 3, about 10 GB)…"
"$PY" "$ROOT/brain/launchable/fetch_models.py" --model "$MODEL"

sudo tee /etc/systemd/system/simoji-groot.service >/dev/null <<EOF
[Unit]
Description=NVIDIA GR00T policy server
After=network-online.target
[Service]
User=$RUN_USER
WorkingDirectory=$ROOT/Isaac-GR00T
Environment=HF_HUB_OFFLINE=1
ExecStart=$PY gr00t/eval/run_gr00t_server.py --model-path $MODEL --embodiment-tag $EMBODIMENT --device cuda:0 --port 5555
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
sudo tee /etc/systemd/system/simoji-brain.service >/dev/null <<EOF
[Unit]
Description=Simoji connector (pairs with the Simoji app)
After=simoji-groot.service
Requires=simoji-groot.service
[Service]
User=$RUN_USER
WorkingDirectory=$ROOT
ExecStartPre=/bin/sleep 20
ExecStart=$PY -m simoji_policy.cli serve $ROOT/brain/bridges/droid_policy.py --port 0 --chunk 16 --hz 15 --obs-hz 2 \
  --name "GR00T N1.7 on Brev" --status-file $STATE/status.json \
  -x backend=groot -x url=localhost:5555 -x log=$STATE/activity.jsonl -x save_every=0
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF
sudo tee /etc/systemd/system/simoji-idle.service >/dev/null <<EOF
[Unit]
Description=Power off when no phone uses the robot brain
[Service]
Type=oneshot
Environment=SIMOJI_IDLE_MINUTES=$IDLE_MIN
ExecStart=/usr/bin/python3 $ROOT/brain/launchable/idle_shutdown.py
EOF
sudo tee /etc/systemd/system/simoji-idle.timer >/dev/null <<EOF
[Unit]
Description=Check for an idle robot brain every minute
[Timer]
OnBootSec=5min
OnUnitActiveSec=1min
[Install]
WantedBy=timers.target
EOF
status "starting the model…"
sudo systemctl daemon-reload
sudo systemctl enable --now simoji-groot simoji-brain simoji-idle.timer
log "done: open port 8080 for the ready page"
