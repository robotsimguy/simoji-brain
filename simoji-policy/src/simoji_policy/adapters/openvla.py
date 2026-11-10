"""``openvla`` adapter: OpenVLA (``openvla/openvla-7b``) via transformers ``AutoModelForVision2Seq``.

Install: ``pip install 'simoji-policy[openvla]'`` (torch, transformers, timm, tokenizers); a GPU with
~16 GB (bf16) is needed for the 7B model.

OpenVLA predicts a **7-D end-effector action** ``[dx, dy, dz, droll, dpitch, dyaw, gripper]`` per step,
un-normalised with ``unnorm_key`` (a dataset name from its training mix, default ``bridge_orig``;
``--extra unnorm_key=...``).

Limitation: Simoji robots are joint-space. Two modes (``--extra mode=ee_delta|joint_deltas``):

* ``ee_delta`` (default): pass the 7-D vector through with ``action_space="ee_delta"`` and
  ``joint_names=["dx","dy","dz","droll","dpitch","dyaw","gripper"]``. The phone needs an IK bridge for
  the selected arm; without one it only shows the predicted deltas.
* ``joint_deltas``: map the first 6 EE dims onto the first 6 robot joints (and gripper onto the 7th if
  present) scaled by ``--extra gain=1.0`` as ``joint_deltas``. This is a crude proxy (no IK) and only
  makes sense for quick demos.

OpenVLA is single-step: ``act`` returns one action repeated ``chunk`` times (set ``--chunk 1 --hz 5`` for
the model's native ~5 Hz behaviour, or ``--extra steps=N`` to run N sequential forward passes per obs,
which is slow).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..protocol import Hello, Obs
from .base import Adapter, MissingDependency, PolicyConfig

HINT = "pip install 'simoji-policy[openvla]'  (torch, transformers>=4.40, timm, tokenizers)"
EE_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


class OpenVLAAdapter(Adapter):
    name = "openvla"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.model_id = config.model or "openvla/openvla-7b"
        self.unnorm_key = config.extra.get("unnorm_key", "bridge_orig")
        self.mode = config.extra.get("mode", "ee_delta")
        self.gain = float(config.extra.get("gain", 1.0))
        self.steps = int(config.extra.get("steps", 1))
        self.action_space = "ee_delta" if self.mode == "ee_delta" else "joint_deltas"
        self.model = None
        self.processor = None
        self.torch = None
        self.device = config.extra.get("device")

    def load(self, status) -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as e:
            raise MissingDependency("transformers/torch", HINT) from e
        self.torch = torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        status(f"loading {self.model_id} (this can take minutes)…")
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        kwargs: Dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if self.device == "cuda":
            kwargs["torch_dtype"] = torch.bfloat16
            try:
                kwargs["attn_implementation"] = "flash_attention_2"
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **kwargs)
            except Exception:
                kwargs.pop("attn_implementation", None)
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **kwargs)
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.name = f"openvla:{self.model_id.split('/')[-1]}"
        status("model loaded")

    def reset(self, hello: Hello) -> Dict[str, Any]:
        self.hello = hello
        if self.mode == "ee_delta":
            self.joint_names = list(EE_NAMES)
        else:
            self.joint_names = list(hello.joint_names)
        return self.policy_block()

    def _predict(self, img: np.ndarray, instr: str) -> np.ndarray:
        from PIL import Image

        torch = self.torch
        prompt = f"In: What action should the robot take to {instr or 'complete the task'}?\nOut:"
        inputs = self.processor(prompt, Image.fromarray(img)).to(self.device)
        if self.device == "cuda":
            inputs = inputs.to(torch.bfloat16)
        with torch.inference_mode():
            a = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return np.asarray(a, dtype=np.float32).reshape(-1)

    def act(self, obs: Obs) -> Any:
        img = obs.first_image(self.config.image)
        if img is None:
            raise ValueError("openvla needs a camera image in obs")
        instr = self.instruction(obs)
        rows = [self._predict(img, instr) for _ in range(max(1, self.steps))]
        ee = np.stack(rows, axis=0)  # (steps, 7)
        if self.mode == "ee_delta":
            return self.pad_chunk(ee, 7)
        dof = len(self.joint_names)
        out = np.zeros((ee.shape[0], dof), dtype=np.float32)
        n = min(dof, 6)
        out[:, :n] = ee[:, :n] * self.gain
        if dof >= 7 and ee.shape[1] >= 7:
            out[:, 6] = ee[:, 6] * self.gain
        return self.pad_chunk(out, dof)
