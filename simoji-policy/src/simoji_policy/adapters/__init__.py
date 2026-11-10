"""Adapter registry. Heavy adapters are imported lazily so the base package stays light."""

from __future__ import annotations

import os
from typing import Callable, Dict, Type

from .base import Adapter, MissingDependency, PolicyConfig

ADAPTER_NAMES = ("auto", "echo", "zero", "simoji", "lerobot", "openpi", "openvla", "groot")


def _echo():
    from .echo import EchoAdapter

    return EchoAdapter


def _zero():
    from .echo import ZeroAdapter

    return ZeroAdapter


def _simoji():
    from .simoji import SimojiAdapter

    return SimojiAdapter


def _lerobot():
    from .lerobot import LeRobotAdapter

    return LeRobotAdapter


def _openpi():
    from .openpi import OpenPiAdapter

    return OpenPiAdapter


def _openvla():
    from .openvla import OpenVLAAdapter

    return OpenVLAAdapter


def _groot():
    from .groot import GrootAdapter

    return GrootAdapter


_REGISTRY: Dict[str, Callable[[], Type[Adapter]]] = {
    "echo": _echo,
    "zero": _zero,
    "simoji": _simoji,
    "lerobot": _lerobot,
    "openpi": _openpi,
    "openvla": _openvla,
    "groot": _groot,
}


def detect_adapter(model: str) -> str:
    """Guess the adapter from ``<model>`` for ``--adapter auto``."""
    m = (model or "").strip()
    if not m or m in ("echo", "zero"):
        return m or "echo"
    if m.endswith(".py") or ":" in m and os.path.isfile(m.split(":")[0]):
        return "simoji"
    if m.startswith(("ws://", "wss://")):
        return "openpi"
    if m.startswith("tcp://") or (m.split(":")[-1].isdigit() and "/" not in m):
        return "groot"
    low = m.lower()
    if "openvla" in low:
        return "openvla"
    if "gr00t" in low or "groot" in low:
        return "groot"
    if os.path.isdir(m):
        if os.path.exists(os.path.join(m, "config.json")):
            try:
                import json

                with open(os.path.join(m, "config.json")) as f:
                    cfg = json.load(f)
                arch = " ".join(cfg.get("architectures", []) or []).lower()
                if "openvla" in arch or "prismatic" in arch:
                    return "openvla"
            except Exception:
                pass
        return "lerobot"
    return "lerobot"


def resolve_adapter_name(name: str, model: str) -> str:
    name = (name or "auto").lower()
    if name == "auto":
        name = detect_adapter(model)
    if name not in _REGISTRY:
        raise ValueError(f"unknown adapter {name!r}; choose from {', '.join(ADAPTER_NAMES)}")
    return name


def make_adapter(name: str, config: PolicyConfig) -> Adapter:
    """Instantiate an adapter by name (``auto`` resolves from ``config.model``)."""
    name = resolve_adapter_name(name, config.model)
    cls = _REGISTRY[name]()
    return cls(config)


__all__ = ["Adapter", "MissingDependency", "PolicyConfig", "ADAPTER_NAMES", "make_adapter", "detect_adapter", "resolve_adapter_name"]
