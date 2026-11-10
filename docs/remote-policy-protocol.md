# Simoji Remote Policy protocol v1

Big models (VLAs, billions of parameters) run wherever the user has a GPU; the phone runs the physics and
streams observations out / plays action chunks back. Two transports, one message format:

* **Relay (pair by code)** — the policy server dials OUT to the Simoji worker; the phone joins with a 6-digit
  code (or QR). No ports, URLs or keys for the user.
* **Direct (URL + key)** — the phone connects straight to a server (`wss://host/ws`, `Authorization: Bearer <key>`),
  e.g. a lab box behind a tunnel, a Hugging Face Inference Endpoint adapter, openpi / LeRobot / GR00T adapters.

## Relay HTTP (worker, base https://simoji-proxy.parzival2108.workers.dev)
* `POST /v1/relay/pair` (server) → `{ "code": "483 912", "session": "<id>", "token": "<server token>", "expires_in": 600 }`
* `POST /v1/relay/join` body `{ "code": "483912" }` (phone, App-Attest or own-key headers not required) →
  `{ "session": "<id>", "token": "<phone token>", "server_name": "..." }` ; 404 unknown/expired code, 409 already joined.
* `GET /v1/relay/ws?session=<id>&role=server|phone&token=<token>` → WebSocket. The relay forwards every frame
  (text or binary) from one role to the other unchanged; `{ "type": "peer", "connected": true|false }` is
  injected when the other side joins/leaves. Sessions live in a Durable Object; idle 15 min → closed.
  Tokens are HMAC (TOKEN_SECRET), bound to session+role, 24 h. Limits: 1 phone + 1 server per session,
  frames ≤ 1 MB, 60 frames/s per direction.

## Messages (JSON text frames; images = base64 JPEG)
phone → server
* `hello`: `{ "type":"hello", "protocol":1, "robot": { "name":"Go2", "joint_names":[...], "dofs":12,
  "control_hz":50 }, "cameras":[ { "name":"front", "w":224, "h":224 } ], "instruction":"walk to the red cube" }`
* `obs`: `{ "type":"obs", "t":<seq>, "time":<sim s>, "proprio": { "q":[...], "qd":[...], "base_pos":[x,y,z],
  "base_quat":[x,y,z,w], "base_lin_vel":[...], "base_ang_vel":[...], "gravity_body":[...] },
  "images": { "front":"<b64 jpeg>" }, "instruction":"..." }` sent at `obs_hz` (from `ready`), only while a
  previous chunk is being played or none is pending (at most 1 in flight).
server → phone
* `ready`: `{ "type":"ready", "policy": { "name":"openvla-7b", "action_space":"joint_targets"|"joint_deltas"|"velocity_command",
  "joint_names":[...]  (order of each action vector), "chunk":10, "hz":10, "obs_hz":5,
  "image": { "w":224, "h":224, "cameras":["front"] } } }`
* `action`: `{ "type":"action", "t":<obs seq it answers>, "actions":[[...], ...] }` — `chunk` vectors played at
  `hz` (phone interpolates targets between vectors at its control rate; `velocity_command` = [vx, vy, yaw]
  for walking policies; `joint_velocities` = rad/s, integrated from the current joints at `hz`). A newer chunk
  replaces the rest of the old one. Any other `action_space` (e.g. `ee_delta`) is reported to the user and nothing
  is played — the phone never guesses joints. Joint targets are clamped to the robot's joint limits.
* `status`: `{ "type":"status", "message":"loading model…" }` ; `error`: `{ "type":"error", "message":"..." }`.
Timeouts: no `action` within 3 s of an `obs` → hold last target, show "waiting" ("warming up" before the first
action); 15 s → "disconnected" (120 s for the first action: models compile on their first call). On reconnect the
phone reuses its relay join token (same session) instead of redeeming the code again.

Servers that already exist (openpi, GR00T, Triton/KServe, plain HTTP): the phone connects to them directly in their
own protocols — see docs/remote-direct-connect.md.

### Named cameras (additive, phones since 2026-11-08)
* `hello.cameras` lists every camera the robot can have, each with a `mount`: `front` (224², `base`: on the base
  link looking forward — the original single camera), `exterior` and `exterior2` (320×180, `exterior`: fixed beside
  the robot in its base frame, aimed at the workspace; aim presets in ⋯ → Robot cameras) and, for fixed-base arms,
  `wrist` (320×180, `wrist`: on the hand / last arm link, looking past the fingertips). Old servers ignore `mount`.
* `ready.policy.image.cameras` picks the cameras (only those render on the phone: each is an extra render) and the
  optional `ready.policy.image.sizes` `{ "<name>": [w, h] }` their sizes (others: `image.w × image.h`). Names may be
  the phone's (`exterior`, `wrist`, ...) or a model's own: a name containing `wrist`/`hand`/`gripper` means the wrist
  camera, `exterior`/`external`/`side`/`agentview`/`scene`/`third`/`static`/`shoulder` the exterior one (`exterior2`
  when it also contains `2` or `right`, e.g. DROID's `exterior_image_2_left`), anything else the front camera.
* `obs.images` then carries one JPEG per requested name, keyed by the name the server used (DROID:
  `exterior_image_1_left`, `wrist_image_left`). A `ready` without `cameras`, or with just `["front"]`, gets the legacy
  single `front` image, so old servers and old phones keep working unchanged.
* `simoji-policy serve ... --cameras exterior_image_1_left=320x180,wrist_image_left=320x180` requests them (default:
  the phone's first camera, i.e. `front`); a `simoji` Policy's `reset()` may return the `image` block itself.

## Python package `simoji-policy` (open source, pip)
`simoji-policy serve <hf-repo-or-path> [--adapter auto|simoji|lerobot|openpi|openvla|groot] [--relay URL] [--port 8765]
 [--chunk 10 --hz 10 --obs-hz 5 --image 224]` → prints the pairing code + QR, also listens on `ws://0.0.0.0:8765/ws`
 for Direct mode (`--key` sets the bearer key). Adapters map `obs` → model inputs and model outputs → `action`.
 `simoji-policy serve --adapter echo` is a test policy (returns sinusoidal joint targets) for end-to-end checks.
