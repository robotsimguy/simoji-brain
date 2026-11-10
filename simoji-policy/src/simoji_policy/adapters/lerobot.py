"""``lerobot`` adapter: run a LeRobot ``PreTrainedPolicy`` (ACT, Diffusion, pi0, SmolVLA, ...) in-process.

Install: ``pip install lerobot`` (plus torch). Model = HF repo id or local checkpoint dir.

Mapping
* phone cameras -> every ``observation.images.*`` input feature of the policy, in order (first camera
  fills the first image feature, ...; a camera called like the feature wins). Images are resized to the
  feature's (H, W) and normalised to float [0, 1], CHW.
* ``observation.state`` <- ``obs.proprio.q`` (padded/truncated to the feature's size).
* ``task`` <- instruction (for language-conditioned policies).
* action chunking: LeRobot policies keep an internal action queue; we call ``select_action`` ``chunk`` times
  on the same observation to drain a chunk (``n_action_steps``), which is what the phone plays at ``hz``.
  Action dim must equal the robot's dof (``joint_targets``) unless ``--action-space`` says otherwise.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..protocol import Hello, Obs
from .base import Adapter, MissingDependency, PolicyConfig

HINT = "pip install lerobot  (see https://github.com/huggingface/lerobot)"


class LeRobotAdapter(Adapter):
    name = "lerobot"
    action_space = "joint_targets"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.policy = None
        self.torch = None
        self.device = config.extra.get("device")
        self.image_features: Dict[str, Any] = {}
        self.state_dim = None
        self.action_dim = None
        self.action_space = str(config.extra.get("action_space") or "joint_targets")

    def load(self, status) -> None:
        try:
            import torch
            from lerobot.policies.factory import get_policy_class  # lerobot >= 0.2
            from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: F401
        except ImportError:
            try:  # older layout
                import torch
                from lerobot.common.policies.factory import get_policy_class  # type: ignore
            except ImportError as e:
                raise MissingDependency("lerobot", HINT) from e
        self.torch = torch
        status(f"loading LeRobot policy {self.config.model}…")
        from lerobot.configs.policies import PreTrainedConfig  # type: ignore

        try:
            cfg = PreTrainedConfig.from_pretrained(self.config.model)
            cls = get_policy_class(cfg.type)
            policy = cls.from_pretrained(self.config.model)
        except Exception:
            # fallback: try common classes directly
            from lerobot.policies.act.modeling_act import ACTPolicy  # type: ignore

            policy = ACTPolicy.from_pretrained(self.config.model)
            cfg = policy.config
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        policy.to(self.device)
        policy.eval()
        self.policy = policy
        feats = getattr(cfg, "input_features", {}) or {}
        self.image_features = {k: v for k, v in feats.items() if k.startswith("observation.image")}
        st = feats.get("observation.state")
        self.state_dim = int(st.shape[0]) if st is not None else None
        out = (getattr(cfg, "output_features", {}) or {}).get("action")
        self.action_dim = int(out.shape[0]) if out is not None else None
        self.name = f"lerobot:{getattr(cfg, 'type', 'policy')}"
        status("model loaded")

    def reset(self, hello: Hello) -> Dict[str, Any]:
        block = super().reset(hello)
        if self.policy is not None:
            self.policy.reset()
        if self.action_dim is not None and self.action_space == "joint_targets" and self.action_dim != len(self.joint_names):
            # policy controls fewer joints: expose the first action_dim joint names
            self.joint_names = self.joint_names[: self.action_dim]
            block = self.policy_block()
        return block

    def _batch(self, obs: Obs):
        torch = self.torch
        batch: Dict[str, Any] = {}
        cams = list(obs.images_b64)
        for i, (feat, spec) in enumerate(self.image_features.items()):
            shape = tuple(spec.shape)  # (C, H, W)
            h, w = (int(shape[1]), int(shape[2])) if len(shape) == 3 else (self.config.image, self.config.image)
            cam = next((c for c in cams if c in feat), None) or (cams[i] if i < len(cams) else (cams[0] if cams else None))
            if cam is None:
                raise ValueError(f"policy needs image feature {feat} but obs has no images")
            img = obs.image(cam, None)
            if img.shape[0] != h or img.shape[1] != w:
                from PIL import Image

                img = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            batch[feat] = t.unsqueeze(0).to(self.device)
        if self.state_dim is not None:
            q = np.asarray(obs.proprio.q, dtype=np.float32)
            if q.shape[0] < self.state_dim:
                q = np.concatenate([q, np.zeros(self.state_dim - q.shape[0], np.float32)])
            q = q[: self.state_dim]
            batch["observation.state"] = torch.from_numpy(q).unsqueeze(0).to(self.device)
        instr = self.instruction(obs)
        if instr:
            batch["task"] = [instr]
        return batch

    def act(self, obs: Obs) -> Any:
        torch = self.torch
        batch = self._batch(obs)
        n = int(self.config.chunk)
        rows = []
        with torch.inference_mode():
            for _ in range(n):
                a = self.policy.select_action(batch)
                rows.append(a[0].detach().float().cpu().numpy())
        arr = np.stack(rows, axis=0)
        return self.pad_chunk(arr, len(self.joint_names) if self.action_space in ("joint_targets", "joint_deltas") else None)
