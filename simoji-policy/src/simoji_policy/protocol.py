"""Simoji Remote Policy protocol v1: message framing (JSON text frames).

Mirrors docs/remote-policy-protocol.md exactly. Everything the phone sends is
parsed into small dataclasses; everything the server sends is produced by the
``*_message`` helpers so the key names live in exactly one place.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

PROTOCOL_VERSION = 1

ACTION_SPACES = ("joint_targets", "joint_deltas", "velocity_command", "ee_delta")


class ProtocolError(ValueError):
    """A frame that does not follow the protocol."""


# --------------------------------------------------------------------------- #
# phone -> server
# --------------------------------------------------------------------------- #


@dataclass
class Camera:
    name: str
    w: int = 224
    h: int = 224
    mount: str = ""  # "base" | "exterior" | "wrist" (phones since Run 2; "" = not told)


@dataclass
class Hello:
    robot_name: str
    joint_names: List[str]
    dofs: int
    control_hz: float
    cameras: List[Camera] = field(default_factory=list)
    instruction: str = ""
    protocol: int = PROTOCOL_VERSION
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Hello":
        robot = d.get("robot") or {}
        joint_names = [str(j) for j in (robot.get("joint_names") or [])]
        dofs = int(robot.get("dofs", len(joint_names)))
        cams = [
            Camera(name=str(c.get("name", "front")), w=int(c.get("w", 224)), h=int(c.get("h", 224)),
                   mount=str(c.get("mount") or ""))
            for c in (d.get("cameras") or [])
        ]
        return cls(
            robot_name=str(robot.get("name", "robot")),
            joint_names=joint_names,
            dofs=dofs,
            control_hz=float(robot.get("control_hz", 50)),
            cameras=cams,
            instruction=str(d.get("instruction") or ""),
            protocol=int(d.get("protocol", PROTOCOL_VERSION)),
            raw=d,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "hello",
            "protocol": self.protocol,
            "robot": {
                "name": self.robot_name,
                "joint_names": list(self.joint_names),
                "dofs": self.dofs,
                "control_hz": self.control_hz,
            },
            "cameras": [dict({"name": c.name, "w": c.w, "h": c.h}, **({"mount": c.mount} if c.mount else {}))
                        for c in self.cameras],
            "instruction": self.instruction,
        }


@dataclass
class Proprio:
    q: np.ndarray
    qd: np.ndarray
    base_pos: np.ndarray
    base_quat: np.ndarray  # x, y, z, w
    base_lin_vel: np.ndarray
    base_ang_vel: np.ndarray
    gravity_body: np.ndarray

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Proprio":
        def arr(key: str, default: Optional[List[float]] = None) -> np.ndarray:
            v = d.get(key)
            if v is None:
                v = default if default is not None else []
            return np.asarray(v, dtype=np.float32)

        return cls(
            q=arr("q"),
            qd=arr("qd"),
            base_pos=arr("base_pos", [0.0, 0.0, 0.0]),
            base_quat=arr("base_quat", [0.0, 0.0, 0.0, 1.0]),
            base_lin_vel=arr("base_lin_vel", [0.0, 0.0, 0.0]),
            base_ang_vel=arr("base_ang_vel", [0.0, 0.0, 0.0]),
            gravity_body=arr("gravity_body", [0.0, 0.0, -1.0]),
        )

    def state_vector(self) -> np.ndarray:
        """Flat proprio vector in a fixed order (q, qd, base_pos, base_quat, lin, ang, gravity)."""
        return np.concatenate(
            [self.q, self.qd, self.base_pos, self.base_quat, self.base_lin_vel, self.base_ang_vel, self.gravity_body]
        ).astype(np.float32)


@dataclass
class Obs:
    t: int
    time: float
    proprio: Proprio
    images_b64: Dict[str, str]
    instruction: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)
    _decoded: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Obs":
        return cls(
            t=int(d.get("t", 0)),
            time=float(d.get("time", 0.0)),
            proprio=Proprio.from_dict(d.get("proprio") or {}),
            images_b64=dict(d.get("images") or {}),
            instruction=str(d.get("instruction") or ""),
            raw=d,
        )

    def image(self, name: str, size: Optional[int] = None) -> np.ndarray:
        """Decode camera ``name`` to an HxWx3 uint8 RGB array (optionally resized to size x size).

        Decoded arrays are cached per Obs so adapters can call this repeatedly.
        """
        key = f"{name}:{size}"
        if key not in self._decoded:
            b64 = self.images_b64.get(name)
            if b64 is None:
                raise KeyError(f"camera {name!r} not in obs (have {list(self.images_b64)})")
            self._decoded[key] = decode_image(b64, size)
        return self._decoded[key]

    def images(self, size: Optional[int] = None) -> Dict[str, np.ndarray]:
        return {name: self.image(name, size) for name in self.images_b64}

    def first_image(self, size: Optional[int] = None) -> Optional[np.ndarray]:
        for name in self.images_b64:
            return self.image(name, size)
        return None


def decode_image(b64: str, size: Optional[int] = None) -> np.ndarray:
    """base64 JPEG/PNG -> uint8 RGB array."""
    from PIL import Image

    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if size is not None and (img.width != size or img.height != size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def encode_image(arr: np.ndarray, quality: int = 85) -> str:
    """uint8 RGB array -> base64 JPEG (used by tests / the fake phone)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# server -> phone
# --------------------------------------------------------------------------- #


@dataclass
class Ready:
    name: str
    action_space: str
    joint_names: List[str]
    chunk: int = 10
    hz: float = 10.0
    obs_hz: float = 5.0
    image_w: int = 224
    image_h: int = 224
    cameras: List[str] = field(default_factory=lambda: ["front"])
    # per-camera size {name: (w, h)} (optional; others use image_w x image_h)
    sizes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        if self.action_space not in ACTION_SPACES:
            raise ProtocolError(f"unknown action_space {self.action_space!r}")
        return {
            "type": "ready",
            "policy": {
                "name": self.name,
                "action_space": self.action_space,
                "joint_names": list(self.joint_names),
                "chunk": int(self.chunk),
                "hz": float(self.hz),
                "obs_hz": float(self.obs_hz),
                "image": dict({"w": int(self.image_w), "h": int(self.image_h), "cameras": list(self.cameras)},
                              **({"sizes": {k: [int(v[0]), int(v[1])] for k, v in self.sizes.items()}} if self.sizes else {})),
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Ready":
        p = d.get("policy") or {}
        img = p.get("image") or {}
        return cls(
            name=str(p.get("name", "")),
            action_space=str(p.get("action_space", "joint_targets")),
            joint_names=list(p.get("joint_names") or []),
            chunk=int(p.get("chunk", 10)),
            hz=float(p.get("hz", 10)),
            obs_hz=float(p.get("obs_hz", 5)),
            image_w=int(img.get("w", 224)),
            image_h=int(img.get("h", 224)),
            cameras=list(img.get("cameras") or []),
            sizes={k: (int(v[0]), int(v[1])) for k, v in (img.get("sizes") or {}).items()},
        )


def action_message(t: int, actions: Any) -> Dict[str, Any]:
    """Build an ``action`` frame. ``actions`` is (chunk, dof) array-like; NaN/inf are rejected."""
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ProtocolError(f"actions must be 2-D (chunk, dof), got shape {arr.shape}")
    if arr.size and not np.all(np.isfinite(arr)):
        raise ProtocolError("actions contain NaN or inf")
    return {"type": "action", "t": int(t), "actions": [[float(x) for x in row] for row in arr]}


def status_message(message: str) -> Dict[str, Any]:
    return {"type": "status", "message": str(message)}


def error_message(message: str) -> Dict[str, Any]:
    return {"type": "error", "message": str(message)}


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #


def encode_message(msg: Dict[str, Any]) -> str:
    return json.dumps(msg, separators=(",", ":"))


def decode_message(frame: Any) -> Dict[str, Any]:
    """Parse a text frame into a dict with a ``type`` key. Raises ProtocolError on bad input."""
    if isinstance(frame, (bytes, bytearray)):
        try:
            frame = bytes(frame).decode("utf-8")
        except UnicodeDecodeError as e:
            raise ProtocolError("binary frame is not UTF-8 JSON") from e
    try:
        d = json.loads(frame)
    except (TypeError, ValueError) as e:
        raise ProtocolError(f"invalid JSON frame: {e}") from e
    if not isinstance(d, dict) or "type" not in d:
        raise ProtocolError("frame must be a JSON object with a 'type' field")
    return d


def parse_phone_message(frame: Any):
    """Return ``(type, payload)``: ('hello', Hello) | ('obs', Obs) | (<other type>, dict)."""
    d = decode_message(frame)
    t = d["type"]
    if t == "hello":
        return t, Hello.from_dict(d)
    if t == "obs":
        return t, Obs.from_dict(d)
    return t, d
