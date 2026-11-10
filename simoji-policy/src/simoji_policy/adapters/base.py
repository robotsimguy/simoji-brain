"""Adapter interface: obs -> model inputs, model outputs -> action chunk."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..protocol import Hello, Obs


@dataclass
class PolicyConfig:
    """Everything from the CLI that an adapter needs (``simoji-policy serve <model> ...``)."""

    model: str = ""
    chunk: int = 10
    hz: float = 10.0
    obs_hz: float = 5.0
    image: int = 224
    # cameras to request (names from the phone's hello, or DROID / LIBERO names the phone maps to its
    # exterior / wrist / front cameras); {name: (w, h)} sizes per camera. Empty: the phone's first camera.
    cameras: List[str] = field(default_factory=list)
    camera_sizes: Dict[str, Any] = field(default_factory=dict)
    instruction: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Adapter:
    """Base class for all adapters.

    Lifecycle (driven by :class:`simoji_policy.server.Session`):

    1. ``load(status)`` — once, when the first ``hello`` arrives (heavy imports / model download go here;
       runs in a worker thread, call ``status("...")`` to push progress to the phone).
    2. ``reset(hello)`` — each time a phone sends ``hello``. Returns the ``policy`` block for ``ready``:
       at minimum ``action_space`` and ``joint_names`` (order of every action vector).
    3. ``act(obs)`` — for each ``obs``: return ``(chunk, dof)`` array-like of actions in ``joint_names`` order.
       Runs in a worker thread, so blocking inference is fine.
    """

    name: str = "adapter"
    action_space: str = "joint_targets"

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.hello: Optional[Hello] = None
        self.joint_names: List[str] = []

    # -- optional hooks -------------------------------------------------------
    def load(self, status) -> None:  # noqa: D401 - simple hook
        """Load weights. ``status(msg)`` sends a protocol ``status`` frame to the phone."""

    def reset(self, hello: Hello) -> Dict[str, Any]:
        """Configure for this robot. Default: control every joint the phone reports."""
        self.hello = hello
        self.joint_names = list(hello.joint_names)
        return self.policy_block()

    def act(self, obs: Obs) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (connections, GPU memory)."""

    # -- helpers --------------------------------------------------------------
    def instruction(self, obs: Optional[Obs] = None) -> str:
        """CLI --instruction wins, else per-obs instruction, else the hello instruction."""
        if self.config.instruction:
            return self.config.instruction
        if obs is not None and obs.instruction:
            return obs.instruction
        return self.hello.instruction if self.hello else ""

    def camera_names(self) -> List[str]:
        """Cameras requested in ``ready``: --cameras, else the phone's first (legacy: its one "front" camera).
        Every camera is an extra render on the phone, so nothing more by default."""
        if self.config.cameras:
            return list(self.config.cameras)
        if self.hello and self.hello.cameras:
            return [self.hello.cameras[0].name]
        return ["front"]

    def image_block(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"w": int(self.config.image), "h": int(self.config.image), "cameras": self.camera_names()}
        if self.config.camera_sizes:
            d["sizes"] = {k: [int(v[0]), int(v[1])] for k, v in self.config.camera_sizes.items()}
        return d

    def policy_block(self, **overrides) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "action_space": self.action_space,
            "joint_names": list(self.joint_names),
            "chunk": int(self.config.chunk),
            "hz": float(self.config.hz),
            "obs_hz": float(self.config.obs_hz),
            "image": self.image_block(),
        }
        d.update(overrides)
        return d

    def pad_chunk(self, actions: Any, dof: Optional[int] = None) -> np.ndarray:
        """Coerce model output to exactly ``(config.chunk, dof)``: repeat last row if short, truncate if long."""
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"actions must be (chunk, dof), got shape {arr.shape}")
        if dof is not None and arr.shape[1] != dof:
            raise ValueError(f"expected {dof} action dims, model gave {arr.shape[1]}")
        n = int(self.config.chunk)
        if arr.shape[0] >= n:
            return arr[:n]
        pad = np.repeat(arr[-1:], n - arr.shape[0], axis=0)
        return np.concatenate([arr, pad], axis=0)


class MissingDependency(RuntimeError):
    """Raised by heavy adapters when their optional dependency is not installed."""

    def __init__(self, package: str, hint: str):
        super().__init__(f"{package} is not installed. {hint}")
        self.package = package
        self.hint = hint
