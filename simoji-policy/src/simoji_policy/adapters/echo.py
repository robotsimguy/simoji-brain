"""Test adapters that need no model: ``echo`` (sinusoid around the observed pose) and ``zero`` (hold pose)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..protocol import Obs
from .base import Adapter, PolicyConfig


class EchoAdapter(Adapter):
    """Returns ``chunk`` joint targets oscillating around the current ``obs.q``.

    Each vector i (played at ``hz``) is ``q + amp * sin(2*pi*freq*(time + i/hz) + phase_j)`` so the
    robot visibly wiggles and the phone's chunk playback / interpolation can be verified end to end.
    """

    name = "echo"
    action_space = "joint_targets"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.amp = float(config.extra.get("amp", 0.15))
        self.freq = float(config.extra.get("freq", 0.5))

    def act(self, obs: Obs) -> Any:
        q = np.asarray(obs.proprio.q, dtype=np.float32)
        if q.size == 0:
            q = np.zeros(len(self.joint_names), dtype=np.float32)
        dof = q.shape[0]
        n = int(self.config.chunk)
        hz = float(self.config.hz)
        phase = np.arange(dof, dtype=np.float32) * (2 * math.pi / max(dof, 1))
        out = np.empty((n, dof), dtype=np.float32)
        for i in range(n):
            t = obs.time + i / hz
            out[i] = q + self.amp * np.sin(2 * math.pi * self.freq * t + phase)
        return out


class ZeroAdapter(Adapter):
    """Holds the current pose: every action vector equals ``obs.q``."""

    name = "zero"
    action_space = "joint_targets"

    def act(self, obs: Obs) -> Any:
        q = np.asarray(obs.proprio.q, dtype=np.float32)
        if q.size == 0:
            q = np.zeros(len(self.joint_names), dtype=np.float32)
        return np.repeat(q[None, :], int(self.config.chunk), axis=0)
