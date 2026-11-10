"""Power the instance off when no phone has used the robot brain for SIMOJI_IDLE_MINUTES (default 30).

Activity = the bridge's timing log being written (one line per model call while a phone is connected).
The clock starts when the model is ready, so a slow first download never triggers it. Run every minute
(systemd timer).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

STATUS = os.environ.get("SIMOJI_STATUS", "/var/lib/simoji/status.json")
ACTIVITY = os.environ.get("SIMOJI_ACTIVITY", "/var/lib/simoji/activity.jsonl")
IDLE_MIN = float(os.environ.get("SIMOJI_IDLE_MINUTES", "30"))


def mtime(p: str) -> float:
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def main() -> int:
    if IDLE_MIN <= 0:
        return 0
    try:
        with open(STATUS) as f:
            st = json.load(f)
    except (OSError, ValueError):
        return 0
    ready_since = st.get("ready_since")
    if st.get("state") != "ready" and not ready_since:
        return 0                                   # still loading: never stop here
    if not ready_since:
        st["ready_since"] = ready_since = time.time()
        with open(STATUS + ".tmp", "w") as f:
            json.dump(st, f)
        os.replace(STATUS + ".tmp", STATUS)
    last = max(float(ready_since), mtime(ACTIVITY))
    idle = (time.time() - last) / 60
    print(f"idle {idle:.1f} / {IDLE_MIN:.0f} min", flush=True)
    if idle >= IDLE_MIN:
        print("no phone activity: powering off", flush=True)
        subprocess.call(["shutdown", "-h", "+0"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
