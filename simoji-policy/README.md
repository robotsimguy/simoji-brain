# simoji-policy

Serve a robot policy to the **Simoji** iPhone simulator. The phone runs the physics and streams
observations (proprio + camera JPEGs); your GPU box runs the model and streams back action chunks.
Protocol: [`docs/remote-policy-protocol.md`](../docs/remote-policy-protocol.md) (Simoji Remote Policy protocol v1).

## Quickstart (one command + scan the code)

```bash
pip install simoji-policy
simoji-policy serve --adapter echo        # test policy, no model needed
```

The terminal prints a 6-digit pairing code and a QR. In Simoji open **Policy > Pair by code**, scan the
QR (or type the digits). The robot starts wiggling (the `echo` adapter sends sinusoidal joint targets
around the current pose). Nothing to port-forward, no URLs, no keys: the server dials *out* to the
Simoji relay (`https://simoji-proxy.parzival2108.workers.dev`).

Then plug in a real model:

```bash
simoji-policy serve my_policy.py                               # your own Python file (see below)
simoji-policy serve lerobot/act_aloha_sim_transfer_cube_human  # LeRobot checkpoint (HF id or dir)
simoji-policy serve ws://localhost:8000 --adapter openpi        # openpi policy server
simoji-policy serve openvla/openvla-7b                          # OpenVLA via transformers
simoji-policy serve localhost:5555 --adapter groot              # GR00T inference service
```

## CLI

```
simoji-policy serve <model> [--adapter auto|echo|zero|simoji|lerobot|openpi|openvla|groot]
                    [--relay URL] [--no-relay] [--port 8765] [--key KEY]
                    [--chunk 10] [--hz 10] [--obs-hz 5] [--image 224]
                    [--instruction TEXT] [--extra KEY=VAL ...] [-v]
```

* `--adapter auto` guesses from `<model>`: `.py` -> simoji, `ws://` -> openpi, `host:port` -> groot,
  name containing `openvla` -> openvla, otherwise lerobot.
* `--chunk/--hz`: each `action` carries `chunk` vectors the phone plays at `hz` (interpolating at its
  control rate). `--obs-hz`: how often the phone sends observations (at most 1 in flight).
* `--image`: square camera size requested from the phone.
* Both transports run at once: relay (pair by code) **and** Direct mode on `ws://0.0.0.0:8765/ws`.
  `--no-relay` disables pairing; `--port 0` disables direct mode.

### Direct mode (URL + key)

For a lab box behind a tunnel, a HF Inference Endpoint, etc.:

```bash
simoji-policy serve my_policy.py --no-relay --port 8765 --key s3cret
```

In Simoji choose **Policy > Direct** and enter `ws://<host>:8765/ws` and the key. The phone sends
`Authorization: Bearer s3cret` (a `?key=s3cret` query parameter also works). Without `--key` any client
is accepted; put a TLS terminator / tunnel in front for `wss://`.

## Writing your own policy (`simoji` adapter)

Create a file with a `Policy` class (template: `examples/simple_policy.py`):

```python
import numpy as np

class Policy:
    name = "my-policy"
    action_space = "joint_targets"       # joint_targets | joint_deltas | velocity_command | ee_delta

    def load(self, status):              # optional, once
        status("loading weights…")
        self.model = ...

    def reset(self, hello):              # each time a phone connects
        self.joint_names = hello.joint_names
        # optional: return {"chunk": 20, "hz": 20, "joint_names": [...]} to override ready.policy

    def act(self, obs):                  # every observation -> (chunk, dof) in joint_names order
        img = obs.image("front", 224)    # HxWx3 uint8 RGB
        q = obs.proprio.q                # np.float32 (dof,)
        return np.repeat(q[None], 10, axis=0)
```

```bash
simoji-policy serve my_policy.py        # or my_policy.py:ClassName, or some.module:Policy
```

`act` runs in a worker thread, so blocking inference is fine. `__init__` may accept a
`simoji_policy.PolicyConfig` (`chunk`, `hz`, `obs_hz`, `image`, `instruction`, `extra` from
`--extra k=v`). Exceptions in `act` become `error` frames on the phone; the server never crashes.

### Observation fields

`obs.t`, `obs.time` (sim seconds), `obs.instruction`, `obs.proprio.q / qd / base_pos / base_quat (xyzw) /
base_lin_vel / base_ang_vel / gravity_body` (numpy), `obs.image(name, size)`, `obs.images(size)`,
`obs.proprio.state_vector()` (everything concatenated).

## Adapters

| adapter | `<model>` | install | notes |
|---|---|---|---|
| `echo` | - | base | sinusoidal joint targets around `obs.q` (`--extra amp=0.15 freq=0.5`) |
| `zero` | - | base | holds the current pose |
| `simoji` | `file.py[:Class]` | base | your Python `Policy` class (above) |
| `lerobot` | HF id / checkpoint dir | `pip install lerobot` | `PreTrainedPolicy`; cameras -> `observation.images.*`, `q` -> `observation.state`, instruction -> `task`; chunk drained via `select_action` |
| `openpi` | `ws://host:8000` | `pip install 'simoji-policy[openpi]'` | client to openpi's `serve_policy.py` (msgpack over WS); keys via `--extra state_key= image_key= wrist_key=` |
| `openvla` | `openvla/openvla-7b` | `pip install 'simoji-policy[openvla]'` | `AutoModelForVision2Seq.predict_action(unnorm_key=…)`; see limitation below |
| `groot` | `host:5555` | Isaac-GR00T checkout | `RobotInferenceClient` to `inference_service.py --server` |

Heavy dependencies are imported lazily; the package itself needs only `websockets numpy pillow qrcode requests`.
Missing deps produce a clear install hint instead of a traceback.

### OpenVLA limitation

OpenVLA outputs a 7-D **end-effector delta** `[dx, dy, dz, droll, dpitch, dyaw, gripper]`, not joint
targets. Default `--extra mode=ee_delta` passes it through with `action_space="ee_delta"` and
`joint_names=[dx, dy, dz, droll, dpitch, dyaw, gripper]`; the phone needs an IK bridge for the selected
arm to turn that into joint motion. `--extra mode=joint_deltas gain=1.0` instead maps the 6 EE dims onto
the first 6 joints (and gripper onto the 7th) as `joint_deltas`: a crude demo proxy, not real IK.
OpenVLA is single-step (one forward pass -> one action, ~5 Hz on a good GPU): the action is repeated to
fill the chunk; use `--chunk 1 --hz 5 --obs-hz 5` for its native cadence.

## Protocol in 10 lines

```
phone  -> hello  {protocol:1, robot:{name, joint_names, dofs, control_hz}, cameras:[{name,w,h}], instruction}
server -> status {message}                       (while loading)
server -> ready  {policy:{name, action_space, joint_names, chunk, hz, obs_hz, image:{w,h,cameras}}}
phone  -> obs    {t, time, proprio:{q,qd,base_pos,base_quat,...}, images:{front:<b64 jpeg>}, instruction}
server -> action {t, actions:[[...] x chunk]}    (vectors in ready.joint_names order, played at hz)
server -> error  {message}
relay  -> peer   {connected:true|false}          (injected by the relay)
```

Relay: `POST /v1/relay/pair` -> `{code, session, token}`; the server dials
`GET /v1/relay/ws?session=&role=server&token=`; the phone joins with the code. The client reconnects with
backoff and re-pairs (new code) if the relay reports the session gone.

## Development

```bash
pip install -e 'backend/simoji-policy[dev]'
pytest backend/simoji-policy
```

`examples/fake_phone.py <code>` (or `<ws-url> [key]`) acts as the phone from a terminal: joins the relay by
code, sends `hello` + a few `obs`, prints the action chunks. Useful to check a server without the app.

Tests need no network and no models: protocol framing, echo end-to-end over a local ws server with a fake
phone, direct-mode auth, reconnect against a fake relay, and loading a `Policy` from a file.
