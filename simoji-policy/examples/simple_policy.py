"""Template for the ``simoji`` adapter: the simplest way to plug any model into Simoji.

Run:   simoji-policy serve examples/simple_policy.py
       (adapter auto-detected from the .py extension; or add --adapter simoji)

Simoji calls:
    Policy()              once (or Policy(config) if __init__ takes an argument)
    policy.load(status)   optional, once: load weights; status("msg") shows progress on the phone
    policy.reset(hello)   every time a phone connects; may return a dict overriding ready.policy keys
    policy.act(obs)       for every observation -> (chunk, dof) array in joint_names order
                          (a single vector is also accepted and repeated to fill the chunk)

``hello`` (simoji_policy.protocol.Hello): robot_name, joint_names, dofs, control_hz, cameras, instruction
``obs``   (simoji_policy.protocol.Obs):   t, time, instruction, proprio.{q, qd, base_pos, base_quat,
          base_lin_vel, base_ang_vel, gravity_body} (numpy), obs.image("front", 224) -> HxWx3 uint8,
          obs.images() -> {name: array}
"""

import math

import numpy as np


class Policy:
    name = "simple-sine"            # shown on the phone
    action_space = "joint_targets"  # joint_targets | joint_deltas | velocity_command | ee_delta
    # joint_names = [...]           # optional: control a subset / different order than the robot's joints

    def __init__(self, config=None):
        # config is simoji_policy.PolicyConfig (chunk, hz, obs_hz, image, instruction, extra) or None
        self.chunk = config.chunk if config else 10
        self.hz = config.hz if config else 10.0
        self.model = None

    def load(self, status):
        status("loading my model…")
        # self.model = torch.load(...)
        status("ready")

    def reset(self, hello):
        self.joint_names = list(hello.joint_names)
        self.q0 = None
        print(f"robot {hello.robot_name}: {hello.dofs} joints, cameras {[c.name for c in hello.cameras]}")
        # return {"chunk": 20, "hz": 20} to override what the CLI set

    def act(self, obs):
        q = obs.proprio.q
        if self.q0 is None:
            self.q0 = q.copy()
        # img = obs.image("front", 224)   # HxWx3 uint8, if the robot has a camera
        # text = obs.instruction
        dof = q.shape[0]
        out = np.zeros((self.chunk, dof), dtype=np.float32)
        for i in range(self.chunk):
            t = obs.time + i / self.hz
            out[i] = self.q0 + 0.2 * math.sin(2 * math.pi * 0.5 * t)
        return out
