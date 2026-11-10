"""``simoji`` adapter: load a user Python file that exposes ``class Policy``.

The simplest way to plug any model in. The file must define::

    class Policy:
        def reset(self, hello) -> dict | None: ...   # called on each phone `hello`
        def act(self, obs) -> actions: ...            # (chunk, dof) array-like, or a single vector

Optional attributes / methods on ``Policy``:

* ``action_space`` (str, default ``"joint_targets"``): ``joint_targets`` | ``joint_deltas`` |
  ``velocity_command`` | ``ee_delta``.
* ``joint_names`` (list[str]): order of every action vector. Defaults to the robot's joint names from ``hello``.
* ``name`` (str): shown on the phone.
* ``load(status)``: heavy one-time setup; ``status("msg")`` pushes a progress message to the phone.
* ``close()``.
* ``reset`` may return a dict that overrides any key of the ``ready.policy`` block
  (``action_space``, ``joint_names``, ``chunk``, ``hz``, ``obs_hz``, ``image``).

``Policy.__init__`` may take zero args or a single ``config`` (:class:`PolicyConfig`).
See ``examples/simple_policy.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from typing import Any, Dict

from ..protocol import Hello, Obs
from .base import Adapter, PolicyConfig


def load_policy_class(spec: str):
    """``spec`` is a path to a ``.py`` file, a ``path.py:ClassName``, or a ``module:ClassName`` / ``module``."""
    target, _, cls_name = spec.partition(":")
    cls_name = cls_name or "Policy"
    if target.endswith(".py") or os.path.isfile(target):
        path = os.path.abspath(target)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"policy file not found: {path}")
        mod_name = "simoji_user_policy_" + os.path.splitext(os.path.basename(path))[0]
        modspec = importlib.util.spec_from_file_location(mod_name, path)
        if modspec is None or modspec.loader is None:
            raise ImportError(f"cannot import {path}")
        module = importlib.util.module_from_spec(modspec)
        sys.modules[mod_name] = module
        # let the file import siblings
        d = os.path.dirname(path)
        if d not in sys.path:
            sys.path.insert(0, d)
        modspec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    if not hasattr(module, cls_name):
        raise AttributeError(f"{spec!r} has no class {cls_name!r} (define `class Policy:` with reset/act)")
    cls = getattr(module, cls_name)
    for required in ("act",):
        if not callable(getattr(cls, required, None)):
            raise TypeError(f"{cls_name} in {spec!r} must define {required}(self, obs)")
    return cls


def _construct(cls, config: PolicyConfig):
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        params = {}
    if len(params) >= 1:
        return cls(config)
    return cls()


class SimojiAdapter(Adapter):
    name = "simoji"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        if not config.model:
            raise ValueError("--adapter simoji needs the path to a Python file: simoji-policy serve my_policy.py")
        self.policy = None

    def load(self, status) -> None:
        cls = load_policy_class(self.config.model)
        self.policy = _construct(cls, self.config)
        self.name = str(getattr(self.policy, "name", None) or os.path.basename(self.config.model))
        self.action_space = str(getattr(self.policy, "action_space", None) or "joint_targets")
        loader = getattr(self.policy, "load", None)
        if callable(loader):
            loader(status)

    def reset(self, hello: Hello) -> Dict[str, Any]:
        if self.policy is None:
            self.load(lambda _m: None)
        self.hello = hello
        names = getattr(self.policy, "joint_names", None)
        self.joint_names = list(names) if names else list(hello.joint_names)
        overrides: Dict[str, Any] = {}
        resetter = getattr(self.policy, "reset", None)
        if callable(resetter):
            r = resetter(hello)
            if isinstance(r, dict):
                overrides = r
        if "joint_names" in overrides:
            self.joint_names = list(overrides["joint_names"])
        if "action_space" in overrides:
            self.action_space = str(overrides["action_space"])
        block = self.policy_block()
        block.update(overrides)
        return block

    def act(self, obs: Obs) -> Any:
        return self.pad_chunk(self.policy.act(obs))

    def close(self) -> None:
        closer = getattr(self.policy, "close", None)
        if callable(closer):
            closer()
