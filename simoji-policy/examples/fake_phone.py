"""Act as the Simoji phone from a terminal: join a pairing code (or a direct URL), send hello + obs, print actions.

    python examples/fake_phone.py 483912                       # relay, by code
    python examples/fake_phone.py ws://localhost:8765/ws [KEY]  # direct
"""

import asyncio
import json
import sys

import numpy as np

from simoji_policy.protocol import Hello, Camera, encode_image, encode_message, decode_message
from simoji_policy.server import DEFAULT_RELAY, relay_ws_url

JOINTS = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf"]


def hello():
    return Hello("Go2", JOINTS, len(JOINTS), 50, [Camera("front", 224, 224)], "walk").to_dict()


def obs(t, q):
    img = np.zeros((64, 64, 3), np.uint8)
    return {
        "type": "obs", "t": t, "time": t / 5,
        "proprio": {"q": q, "qd": [0.0] * len(q), "base_pos": [0, 0, 0.3], "base_quat": [0, 0, 0, 1],
                    "base_lin_vel": [0, 0, 0], "base_ang_vel": [0, 0, 0], "gravity_body": [0, 0, -1]},
        "images": {"front": encode_image(img)}, "instruction": "walk",
    }


async def main(target: str, key: str = None, n: int = 5):
    from websockets.asyncio.client import connect

    headers = {}
    if target.startswith(("ws://", "wss://")):
        url = target
        if key:
            headers["Authorization"] = f"Bearer {key}"
    else:
        import requests

        r = requests.post(DEFAULT_RELAY + "/v1/relay/join", json={"code": target.replace(" ", "")}, timeout=10)
        print("join:", r.status_code, r.text[:160])
        r.raise_for_status()
        d = r.json()
        url = relay_ws_url(DEFAULT_RELAY, d["session"], "phone", d["token"])
    async with connect(url, additional_headers=headers) as ws:
        await ws.send(encode_message(hello()))
        q = [0.0] * len(JOINTS)
        t = 0
        while t < n:
            m = decode_message(await asyncio.wait_for(ws.recv(), 15))
            if m["type"] == "ready":
                print("ready:", json.dumps(m["policy"]))
                t += 1
                await ws.send(encode_message(obs(t, q)))
            elif m["type"] == "action":
                a = np.asarray(m["actions"])
                print(f"action t={m['t']} shape={a.shape} first={np.round(a[0], 3).tolist()}")
                q = a[-1].tolist()
                await asyncio.sleep(0.2)
                t += 1
                await ws.send(encode_message(obs(t, q)))
            else:
                print(m)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
