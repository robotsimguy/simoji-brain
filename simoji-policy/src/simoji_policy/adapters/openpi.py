"""``openpi`` adapter: client to an openpi policy server (pi0 / pi0-FAST / pi05).

Start the model side with openpi's own server, e.g.::

    uv run scripts/serve_policy.py --env ALOHA_SIM   # listens on ws://0.0.0.0:8000

then ``simoji-policy serve ws://localhost:8000 --adapter openpi``.

The openpi server speaks msgpack over WebSocket: send one obs dict, receive ``{"actions": (chunk, dof)}``.
Install: ``pip install msgpack`` (``pip install simoji-policy[openpi]``); openpi's ``openpi_client`` is used if
importable (it adds numpy-aware msgpack packing), otherwise a minimal compatible packer is used.

Obs mapping (openpi examples convention)::

    {"observation/state": q, "observation/image": <first camera HxWx3 uint8>,
     "observation/wrist_image": <second camera if any>, "prompt": instruction}

Override keys with ``--extra state_key=... image_key=... wrist_key=...``.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..protocol import Obs
from .base import Adapter, MissingDependency, PolicyConfig

HINT = "pip install msgpack  (or: pip install 'simoji-policy[openpi]'); run openpi's scripts/serve_policy.py"


def _pack(obj: Any) -> bytes:
    import msgpack

    def default(o):
        if isinstance(o, np.ndarray):
            return {"__ndarray__": True, "data": o.tobytes(), "dtype": str(o.dtype), "shape": list(o.shape)}
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"cannot pack {type(o)}")

    return msgpack.packb(obj, default=default, use_bin_type=True)


def _unpack(data: bytes) -> Any:
    import msgpack

    def hook(o):
        if o.get("__ndarray__"):
            return np.frombuffer(o["data"], dtype=o["dtype"]).reshape(o["shape"])
        return o

    return msgpack.unpackb(data, object_hook=hook, raw=False)


class OpenPiAdapter(Adapter):
    name = "openpi"
    action_space = "joint_targets"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.url = config.model or "ws://localhost:8000"
        if not self.url.startswith(("ws://", "wss://")):
            self.url = "ws://" + self.url
        self.ws = None
        self.state_key = config.extra.get("state_key", "observation/state")
        self.image_key = config.extra.get("image_key", "observation/image")
        self.wrist_key = config.extra.get("wrist_key", "observation/wrist_image")
        self.action_space = str(config.extra.get("action_space") or "joint_targets")
        self._pack = _pack
        self._unpack = _unpack

    def load(self, status) -> None:
        try:
            import msgpack  # noqa: F401
        except ImportError as e:
            raise MissingDependency("msgpack", HINT) from e
        try:
            from openpi_client import msgpack_numpy  # type: ignore

            self._pack = msgpack_numpy.packb
            self._unpack = msgpack_numpy.unpackb
        except Exception:
            pass
        status(f"connecting to openpi server {self.url}…")
        self._connect()
        status("connected to openpi server")

    def _connect(self) -> None:
        from websockets.sync.client import connect

        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        self.ws = connect(self.url, max_size=None, compression=None)
        # openpi servers send their metadata as the first message
        try:
            meta = self._unpack(self.ws.recv(timeout=5))
            if isinstance(meta, dict) and meta:
                self.name = f"openpi:{meta.get('name', meta.get('type', 'policy'))}"
        except Exception:
            pass

    def _obs_dict(self, obs: Obs) -> Dict[str, Any]:
        d: Dict[str, Any] = {self.state_key: np.asarray(obs.proprio.q, dtype=np.float32)}
        cams = list(obs.images_b64)
        if cams:
            d[self.image_key] = obs.image(cams[0], self.config.image)
        if len(cams) > 1:
            d[self.wrist_key] = obs.image(cams[1], self.config.image)
        instr = self.instruction(obs)
        if instr:
            d["prompt"] = instr
        return d

    def act(self, obs: Obs) -> Any:
        if self.ws is None:
            self._connect()
        try:
            self.ws.send(self._pack(self._obs_dict(obs)))
            resp = self._unpack(self.ws.recv(timeout=10))
        except Exception:
            self.ws = None
            raise
        actions = resp["actions"] if isinstance(resp, dict) else resp
        arr = np.asarray(actions, dtype=np.float32)
        dof = len(self.joint_names) if self.action_space in ("joint_targets", "joint_deltas") else None
        if dof is not None and arr.ndim == 2 and arr.shape[1] > dof:
            arr = arr[:, :dof]
        return self.pad_chunk(arr, dof)

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            finally:
                self.ws = None
