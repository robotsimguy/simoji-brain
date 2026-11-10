"""DROID-style Franka bridge: Simoji's Franka Panda <-> openpi pi0.5-DROID, Cosmos3-*-Policy-DROID or GR00T N1.6/N1.7-DROID.

Run on the GPU box next to the model's own server (see docs/remote-vla-on-brev.md):

    simoji-policy serve tools/vla_brev/droid_policy.py --chunk 8 --hz 15 --obs-hz 2 \
        -x backend=openpi -x url=ws://localhost:8000         # pi05_droid (joint VELOCITY chunk, 15 x 8)
    simoji-policy serve tools/vla_brev/droid_policy.py --chunk 16 --hz 15 --obs-hz 1 \
        -x backend=cosmos -x url=ws://localhost:8001         # Cosmos3 DROID (joint POSITION chunk, 32 x 8)
    simoji-policy serve tools/vla_brev/droid_policy.py --chunk 16 --hz 15 --obs-hz 2 \
        -x backend=groot -x url=localhost:5555                # GR00T N1.6-DROID (joint POSITION chunk, 32 x 8)

Why a Policy file and not the built-in adapters: the built-in ``openpi`` adapter sends ``observation/state`` and reads
``actions`` (LIBERO/ALOHA style), but the DROID policies need ``observation/joint_position`` (7) +
``observation/gripper_position`` (1) + exterior/wrist images; Cosmos3 answers ``action`` (singular); the built-in
``groot`` adapter targets the N1.5 ``RobotInferenceClient`` API that N1.6/N1.7 replaced with ``PolicyClient``.

Mapping (Simoji Franka Panda from the robot catalogue, 9 joints: panda_joint1..7 + 2 finger joints):
* state: joint_position = the 7 arm joints; gripper_position (DROID: 0 open .. 1 closed) = 1 - finger/0.04.
* images: phones that list exterior / wrist cameras in their hello (Run 2, docs/remote-policy-protocol.md) are asked
  for ``exterior_image_1_left`` (+ ``exterior_image_2_left`` for Cosmos3) and ``wrist_image_left`` at 320x180 (16:9 like
  DROID's ZED cameras; the bridge letterboxes them to the model's size). Older phones send one 224x224 "front" camera
  (robot base, looking forward), used for every view (``-x cameras=legacy`` forces that).
* actions -> Simoji ``joint_targets`` for all 9 joints (fingers = 0.04 * (1 - gripper)).
  pi05_droid outputs joint velocities v in [-1, 1] at 15 Hz; DROID's controller turns them into a per-step joint
  delta of v * 0.2 rad (droid/robot_ik/robot_ik_solver.py), so targets = q + cumsum(v * 0.2 * vel_scale).
  Cosmos3 / GR00T N1.6 / N1.7 output absolute joint positions (N1.7 predicts them relative to the state and its
  server returns them absolute). N1.7 also needs the hand pose (eef_9d), computed here with Panda forward kinematics.

Timing of every model call is appended as JSON lines to ``--extra log=/path`` (default /tmp/droid_policy_timing.jsonl).
"""

from __future__ import annotations

import json
import os
import re
import time

import numpy as np
from PIL import Image

FINGER_OPEN = 0.04  # Franka Panda finger joint travel (m), open
# Franka Panda joint limits (rad). Simoji does not clamp remote joint targets to the asset's limits, so the bridge does.
PANDA_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], np.float32)
PANDA_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], np.float32)
DROID_HOME = [0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0]


def _resize_pad(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize keeping aspect ratio, pad with black to (h, w) (openpi/GR00T resize_with_pad convention)."""
    ih, iw = img.shape[:2]
    s = min(h / ih, w / iw)
    nh, nw = max(1, int(round(ih * s))), max(1, int(round(iw * s)))
    r = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - nh) // 2, (w - nw) // 2
    out[y : y + nh, x : x + nw] = r
    return out


# Franka Panda forward kinematics (modified DH, Craig convention; Franka's published parameters), link0 -> flange.
_PANDA_DH = [  # (a, d, alpha) per joint
    (0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2),
]
_FLANGE_D = 0.107          # link7 -> flange (link8). DROID's cartesian_position is exactly this frame: checked against
                           # GR00T's demo_data/droid_sample (844 frames, 0 mm / 0 deg error), no hand / TCP offset.
# GR00T N1.7 DROID: euler(robot) -> matrix, then this egocentric correction, then the first two rows (rot6d).
# Same matrix as Isaac-GR00T gr00t/data/state_action/droid_frame.py (DROID_EEF_ROTATION_CORRECT).
_DROID_EEF_ROTATION_CORRECT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], np.float64)


def _mdh(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])


def panda_fk(q7: np.ndarray) -> np.ndarray:
    """4x4 pose of the Panda flange (link8) in the robot base (link0) frame = DROID's cartesian_position."""
    T = np.eye(4)
    for (a, d, al), th in zip(_PANDA_DH, q7):
        T = T @ _mdh(a, d, al, float(th))
    return T @ _mdh(0.0, _FLANGE_D, 0.0, 0.0)


def droid_eef_9d(q7: np.ndarray) -> np.ndarray:
    """DROID cartesian_position (xyz + extrinsic xyz euler, as DROID records it) -> GR00T N1.7 eef_9d."""
    from scipy.spatial.transform import Rotation

    T = panda_fk(q7)
    euler = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    rot = Rotation.from_euler("XYZ", euler).as_matrix() @ _DROID_EEF_ROTATION_CORRECT   # = GR00T compute_eef_9d
    return np.concatenate([T[:3, 3], rot[:2, :].reshape(6)]).astype(np.float32)


class Policy:
    name = "droid-bridge"
    action_space = "joint_targets"

    def __init__(self, config):
        self.cfg = config
        ex = config.extra
        self.backend = str(ex.get("backend", "openpi"))
        self.url = str(ex.get("url", "ws://localhost:8000" if self.backend != "groot" else "localhost:5555"))
        self.vel_scale = float(ex.get("vel_scale", 1.0))
        self.binarize = str(ex.get("binarize_gripper", "1")) not in ("0", "false", "no")
        self.log_path = str(ex.get("log", "/tmp/droid_policy_timing.jsonl"))
        # DROID episodes start at this pose (droid RobotEnv reset); Simoji's Franka spawns at q = 0 (arm straight up,
        # far out of distribution), so the bridge first drives the arm home (no model call) unless home=none.
        self.home = None if str(ex.get("home", "droid")) == "none" else np.array(DROID_HOME, np.float32)
        self.home_seconds = float(ex.get("home_seconds", 2.0))
        self.warmup = int(ex.get("warmup", 2))
        self.homed = False
        self.clip = str(ex.get("clip_limits", "1")) not in ("0", "false", "no")
        self.save_every = int(ex.get("save_every", 25))   # save the camera images the model sees every N calls
        self.camera_mode = str(ex.get("cameras", "droid"))   # droid | legacy
        self.cam_w, self.cam_h = (int(v) for v in str(ex.get("camera_size", "320x180")).lower().split("x"))
        self.cams: dict = {}                                 # DROID view -> phone image name
        self.ws = None
        self.client = None
        self.arm_idx: list[int] = []
        self.finger_idx: list[int] = []
        self.joint_names: list[str] = []
        self.groot_api = "n16"                               # n16: flat keys; n17: nested video/state/language
        self.name = {"openpi": "pi05-droid", "cosmos": "cosmos3-droid", "groot": "gr00t-droid"}.get(self.backend, self.backend)

    # -- lifecycle ------------------------------------------------------------------------------------------
    def load(self, status):
        status(f"connecting to {self.backend} at {self.url}…")
        if self.backend in ("openpi", "cosmos"):
            from openpi_client import msgpack_numpy  # noqa: F401  (pip install openpi-client)
            from websockets.sync.client import connect

            t0 = time.time()
            self.ws = connect(self.url, max_size=None, compression=None, open_timeout=60)
            meta = msgpack_numpy.unpackb(self.ws.recv(timeout=60))
            self._log({"event": "connected", "backend": self.backend, "meta": str(meta)[:200], "s": time.time() - t0})
        elif self.backend == "groot":
            from gr00t.policy.server_client import PolicyClient  # Isaac-GR00T n1.6 checkout

            host, _, port = self.url.replace("tcp://", "").partition(":")
            self.client = PolicyClient(host=host or "localhost", port=int(port or 5555), timeout_ms=30000)
            # N1.7 checkpoints declare eef_9d in their state; their requests are nested by modality.
            try:
                mc = self.client.get_modality_config()
                st = mc["state"] if isinstance(mc, dict) else None
                keys = getattr(st, "modality_keys", None) or (st or {}).get("modality_keys", [])
                if "eef_9d" in keys:
                    self.groot_api = "n17"
                    self.name = "gr00t-n1.7-droid"
            except Exception as e:  # older servers: keep the N1.6 format
                self._log({"event": "modality_config_failed", "error": repr(e)[:200]})
            self._log({"event": "groot_api", "api": self.groot_api})
        else:
            raise ValueError(f"unknown backend {self.backend!r}")
        # Warm up: the first call compiles (openpi/JAX ~20 s) -- longer than the phone's 15 s action timeout, and a
        # timed-out phone cannot re-join the same pairing code. So pay it here, before the ready frame.
        dummy = np.zeros((224, 224, 3), np.uint8)
        for k in range(self.warmup):
            status(f"warming up {self.backend} ({k + 1}/{self.warmup})…")
            views = {"exterior_image_1_left": dummy, "exterior_image_2_left": dummy, "wrist_image_left": dummy}
            _, ms, sms = self._infer(views, np.array(DROID_HOME, np.float64), np.array([0.0]), "pick up the salt shaker")
            self._log({"event": "warmup", "k": k, "call_ms": round(ms, 1), "server_ms": sms})
        status(f"connected to {self.backend}")

    def reset(self, hello):
        names = list(hello.joint_names)
        self.joint_names = names
        self.homed = False
        arm = [i for i, n in enumerate(names) if re.search(r"joint[1-7]$", n) and "finger" not in n]
        if len(arm) != 7:
            arm = [i for i, n in enumerate(names) if "finger" not in n][:7]
        self.arm_idx = arm
        self.finger_idx = [i for i, n in enumerate(names) if "finger" in n]
        if len(arm) != 7:
            raise ValueError(f"{hello.robot_name} has {len(arm)} non-finger joints; the DROID policies need a 7-joint Franka")
        mounts = {(c.mount or c.name).lower() for c in hello.cameras}
        out = {"joint_names": names, "action_space": "joint_targets"}
        if self.camera_mode != "legacy" and any("exterior" in m for m in mounts) and any("wrist" in m for m in mounts):
            views = ["exterior_image_1_left"] + (["exterior_image_2_left"] if self.backend == "cosmos" else []) + ["wrist_image_left"]
            self.cams = {v: v for v in views}
            out["image"] = {"w": self.cam_w, "h": self.cam_h, "cameras": views,
                            "sizes": {v: [self.cam_w, self.cam_h] for v in views}}
        else:
            first = hello.cameras[0].name if hello.cameras else "front"
            self.cams = {"exterior_image_1_left": first, "exterior_image_2_left": first, "wrist_image_left": first}
        self._log({"event": "hello", "robot": hello.robot_name, "joint_names": names, "arm": arm, "fingers": self.finger_idx,
                   "phone_cameras": [[c.name, c.mount, c.w, c.h] for c in hello.cameras], "requested": self.cams})
        return out

    # -- helpers --------------------------------------------------------------------------------------------
    def _log(self, d):
        d["wall"] = time.time()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(d) + "\n")

    def _state(self, q):
        jp = q[self.arm_idx].astype(np.float64)
        if self.finger_idx:
            fing = float(np.mean(q[self.finger_idx]))
            g = float(np.clip(1.0 - fing / FINGER_OPEN, 0.0, 1.0))
        else:
            g = 0.0
        return jp, np.array([g])

    def _to_targets(self, q, arm_targets, grip):
        """(T,7) arm targets + (T,) gripper [0 open..1 closed] -> (T, dof) Simoji joint targets."""
        T = arm_targets.shape[0]
        out = np.repeat(q[None].astype(np.float32), T, axis=0)
        out[:, self.arm_idx] = np.clip(arm_targets, PANDA_LO, PANDA_HI) if self.clip else arm_targets
        if self.finger_idx:
            g = (grip > 0.5).astype(np.float32) if self.binarize else np.clip(grip, 0, 1)
            for i in self.finger_idx:
                out[:, i] = FINGER_OPEN * (1.0 - g)
        return out

    def _infer(self, imgs, jp, g, prompt):
        """One model call -> (raw (T, 8) actions, round-trip ms to the model server, server-reported ms).
        imgs: DROID view name -> HxWx3 uint8 (exterior_image_1_left, wrist_image_left[, exterior_image_2_left])."""
        t0 = time.perf_counter()
        ext1, wrist = imgs["exterior_image_1_left"], imgs["wrist_image_left"]
        ext2 = imgs.get("exterior_image_2_left", ext1)
        if self.backend in ("openpi", "cosmos"):
            from openpi_client import msgpack_numpy

            im = _resize_pad(ext1, 224, 224)
            req = {
                "observation/exterior_image_1_left": im,
                "observation/wrist_image_left": _resize_pad(wrist, 224, 224),
                "observation/joint_position": jp,
                "observation/gripper_position": g,
                "prompt": prompt,
            }
            if self.backend == "cosmos":
                req["observation/exterior_image_2_left"] = _resize_pad(ext2, 224, 224)
            self.ws.send(msgpack_numpy.packb(req))
            resp = msgpack_numpy.unpackb(self.ws.recv(timeout=120))
            if isinstance(resp, str):
                raise RuntimeError(f"model server error: {resp[:300]}")
            raw = np.asarray(resp.get("actions", resp.get("action")), dtype=np.float32)
            server_ms = (resp.get("server_timing") or {}).get("infer_ms") if isinstance(resp, dict) else None
        elif self.groot_api == "n17":
            req = {
                "video": {"exterior_image_1_left": _resize_pad(ext1, 180, 320)[None, None],
                          "wrist_image_left": _resize_pad(wrist, 180, 320)[None, None]},
                "state": {"eef_9d": droid_eef_9d(jp)[None, None],
                          "gripper_position": g.astype(np.float32)[None, None],
                          "joint_position": jp.astype(np.float32)[None, None]},
                "language": {"annotation.language.language_instruction": [[prompt]]},
            }
            resp = self.client.get_action(req)
            a = resp[0] if isinstance(resp, (tuple, list)) else resp
            raw = np.concatenate([np.asarray(a["joint_position"])[0], np.asarray(a["gripper_position"])[0]], axis=1).astype(np.float32)
            server_ms = None
        else:
            req = {
                "video.exterior_image_1_left": _resize_pad(ext1, 180, 320)[None, None],
                "video.wrist_image_left": _resize_pad(wrist, 180, 320)[None, None],
                "state.joint_position": jp.astype(np.float32)[None, None],
                "state.gripper_position": g.astype(np.float32)[None, None],
                "annotation.language.language_instruction": [prompt],
            }
            resp = self.client.get_action(req)
            a = resp[0] if isinstance(resp, (tuple, list)) else resp
            raw = np.concatenate([np.asarray(a["action.joint_position"])[0], np.asarray(a["action.gripper_position"])[0]], axis=1).astype(np.float32)
            server_ms = None
        return raw, (time.perf_counter() - t0) * 1000, server_ms

    # -- act ------------------------------------------------------------------------------------------------
    def act(self, obs):
        q = np.asarray(obs.proprio.q, dtype=np.float32)
        have = list(obs.images_b64)
        blank = np.zeros((self.cam_h, self.cam_w, 3), np.uint8)

        def view(v):
            n = self.cams.get(v)
            if n in obs.images_b64:
                return obs.image(n)
            return obs.image(have[0]) if have else blank
        imgs = {v: view(v) for v in ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")}
        jp, g = self._state(q)
        if self.home is not None and not self.homed and np.abs(jp - self.home).max() < 0.1:
            self.homed = True                                  # home once per phone connection, then the model drives
        if self.home is not None and not self.homed:
            n = max(2, int(self.home_seconds * self.cfg.hz))   # go home first (open gripper), no model call
            arm = jp[None] + (self.home - jp)[None] * np.linspace(1.0 / n, 1.0, n)[:, None]
            self._log({"event": "homing", "t": obs.t, "err": round(float(np.abs(jp - self.home).max()), 3)})
            return self._to_targets(q, arm.astype(np.float32), np.zeros(n, np.float32))
        prompt = obs.instruction or self.cfg.instruction or "pick up the salt shaker"
        raw, call_ms, server_ms = self._infer(imgs, jp, g, prompt)
        if self.save_every and obs.t % self.save_every == 0:
            for v, im in imgs.items():
                if v == "exterior_image_2_left" and self.backend != "cosmos":
                    continue
                Image.fromarray(im).save(self.log_path.replace(".jsonl", f"_{v}_{obs.t:04d}.jpg"))
        if self.backend == "openpi":  # joint velocities -> integrated joint position targets (DROID: 0.2 rad / step)
            v = np.clip(raw[:, :7], -1, 1)
            m = np.abs(v).max(axis=1, keepdims=True)
            v = np.where(m > 1, v / m, v)
            arm = jp[None].astype(np.float32) + np.cumsum(v * 0.2 * self.vel_scale, axis=0)
        else:
            arm = raw[:, :7]
        grip = raw[:, 7] if raw.shape[1] > 7 else np.zeros(raw.shape[0], np.float32)
        out = self._to_targets(q, arm, grip)
        self._log({
            "event": "act", "t": obs.t, "call_ms": round(call_ms, 1), "server_ms": server_ms, "raw_shape": list(raw.shape),
            "prompt": prompt, "images": {k: list(v.shape[:2]) for k, v in imgs.items()}, "q_arm": np.round(jp, 3).tolist(), "grip_in": round(float(g[0]), 3),
            "first": np.round(raw[0], 3).tolist(), "last": np.round(raw[-1], 3).tolist(),
            "arm_delta_end": np.round(arm[-1] - jp, 3).tolist(), "grip_out": np.round(grip[[0, -1]], 2).tolist(),
        })
        return out

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
