"""``groot`` adapter: client to NVIDIA Isaac GR00T's inference service.

Start the model side with GR00T's own server (from the Isaac-GR00T checkout)::

    python scripts/inference_service.py --server --model_path nvidia/GR00T-N1.5-3B \
        --embodiment_tag new_embodiment --data_config <your-config>   # ZMQ on tcp://0.0.0.0:5555

then ``simoji-policy serve localhost:5555 --adapter groot``.

Requires GR00T's Python client (``from gr00t.eval.robot import RobotInferenceClient``), i.e. the
Isaac-GR00T repo installed in this environment (``pip install -e Isaac-GR00T``). Imported lazily.

Obs mapping (``--extra`` keys override the defaults)::

    video.<cam>          <- each phone camera, (1, H, W, 3) uint8              (prefix: video_prefix=video.)
    state.<state_key>    <- obs.proprio.q as (1, dof) float64                  (state_key=joints)
    annotation.human.action.task_description <- [instruction]

Outputs: every key starting with ``action.`` is concatenated in sorted order (or the single key named by
``--extra action_key=...``) into a ``(chunk, dof)`` array of ``joint_targets``. If the concatenated dim
differs from the robot dof the first ``dof`` joints are controlled.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..protocol import Hello, Obs
from .base import Adapter, MissingDependency, PolicyConfig

HINT = "install Isaac-GR00T (git clone https://github.com/NVIDIA/Isaac-GR00T && pip install -e Isaac-GR00T) and run scripts/inference_service.py --server"


class GrootAdapter(Adapter):
    name = "groot"
    action_space = "joint_targets"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        hostport = config.model or "localhost:5555"
        hostport = hostport.replace("tcp://", "")
        host, _, port = hostport.partition(":")
        self.host = host or "localhost"
        self.port = int(port or 5555)
        self.client = None
        self.video_prefix = config.extra.get("video_prefix", "video.")
        self.state_key = config.extra.get("state_key", "joints")
        self.action_key = config.extra.get("action_key")
        self.action_space = str(config.extra.get("action_space") or "joint_targets")
        self.modality_config = None

    def load(self, status) -> None:
        try:
            from gr00t.eval.robot import RobotInferenceClient  # type: ignore
        except ImportError as e:
            raise MissingDependency("gr00t", HINT) from e
        status(f"connecting to GR00T inference service {self.host}:{self.port}…")
        self.client = RobotInferenceClient(host=self.host, port=self.port)
        try:
            self.modality_config = self.client.get_modality_config()
        except Exception:
            self.modality_config = None
        status("connected to GR00T")

    def reset(self, hello: Hello) -> Dict[str, Any]:
        return super().reset(hello)

    def _obs_dict(self, obs: Obs) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for cam in obs.images_b64:
            d[f"{self.video_prefix}{cam}"] = obs.image(cam, self.config.image)[None]
        d[f"state.{self.state_key}"] = np.asarray(obs.proprio.q, dtype=np.float64)[None]
        d["annotation.human.action.task_description"] = [self.instruction(obs)]
        return d

    def act(self, obs: Obs) -> Any:
        if self.client is None:
            raise RuntimeError("GR00T client not loaded")
        resp = self.client.get_action(self._obs_dict(obs))
        if self.action_key:
            arr = np.asarray(resp[self.action_key], dtype=np.float32)
        else:
            keys = sorted(k for k in resp if str(k).startswith("action."))
            if not keys:
                raise ValueError(f"GR00T returned no action.* keys: {list(resp)}")
            parts = [np.asarray(resp[k], dtype=np.float32).reshape(np.asarray(resp[k]).shape[0], -1) for k in keys]
            arr = np.concatenate(parts, axis=1)
        if arr.ndim == 3:  # (batch, chunk, dof)
            arr = arr[0]
        dof = len(self.joint_names)
        if self.action_space in ("joint_targets", "joint_deltas"):
            if arr.shape[1] > dof:
                arr = arr[:, :dof]
            elif arr.shape[1] < dof:
                self.joint_names = self.joint_names[: arr.shape[1]]
                dof = arr.shape[1]
            return self.pad_chunk(arr, dof)
        return self.pad_chunk(arr)
