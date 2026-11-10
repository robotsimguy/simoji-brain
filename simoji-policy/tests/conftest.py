"""Shared helpers: a fake phone client and a fake relay, no network, no models."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from simoji_policy.protocol import Hello, Camera, encode_image, encode_message, decode_message

JOINTS = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf"]


def make_hello(joints: List[str] = JOINTS, cameras=("front",), instruction="walk") -> Dict[str, Any]:
    return Hello(
        robot_name="Go2",
        joint_names=list(joints),
        dofs=len(joints),
        control_hz=50,
        cameras=[Camera(name=c, w=224, h=224) for c in cameras],
        instruction=instruction,
    ).to_dict()


def make_obs(t: int, q: List[float], time: float = 0.0, cameras=("front",), img_size: int = 32) -> Dict[str, Any]:
    rng = np.random.default_rng(t)
    img = rng.integers(0, 255, size=(img_size, img_size, 3), dtype=np.uint8)
    return {
        "type": "obs",
        "t": t,
        "time": time,
        "proprio": {
            "q": q,
            "qd": [0.0] * len(q),
            "base_pos": [0, 0, 0.3],
            "base_quat": [0, 0, 0, 1],
            "base_lin_vel": [0, 0, 0],
            "base_ang_vel": [0, 0, 0],
            "gravity_body": [0, 0, -1],
        },
        "images": {c: encode_image(img) for c in cameras},
        "instruction": "walk",
    }


class FakePhone:
    """Drives a websockets client connection like the Simoji app would."""

    def __init__(self, ws):
        self.ws = ws

    async def send(self, msg: Dict[str, Any]) -> None:
        await self.ws.send(encode_message(msg))

    async def recv(self, timeout: float = 5.0) -> Dict[str, Any]:
        return decode_message(await asyncio.wait_for(self.ws.recv(), timeout))

    async def recv_type(self, kind: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Skip frames (e.g. status) until one of ``kind`` arrives."""
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"no {kind} frame")
            m = await self.recv(remaining)
            if m["type"] == kind:
                return m
            if m["type"] == "error" and kind != "error":
                raise AssertionError(f"got error frame: {m['message']}")


def phone_connect(url: str, headers: Optional[Dict[str, str]] = None):
    """``async with phone_connect(url) as ws:`` then wrap with FakePhone(ws)."""
    from websockets.asyncio.client import connect

    return connect(url, additional_headers=headers or {})


@pytest.fixture
def echo_runner():
    from simoji_policy.adapters.base import PolicyConfig
    from simoji_policy.adapters.echo import EchoAdapter
    from simoji_policy.server import PolicyRunner

    cfg = PolicyConfig(model="", chunk=10, hz=10.0, obs_hz=5.0, image=64)
    return PolicyRunner(EchoAdapter(cfg))
