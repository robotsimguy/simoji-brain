import json

import numpy as np
import pytest

from simoji_policy.protocol import (
    PROTOCOL_VERSION,
    Hello,
    Obs,
    ProtocolError,
    Ready,
    action_message,
    decode_message,
    encode_image,
    encode_message,
    error_message,
    parse_phone_message,
    status_message,
)

from conftest import JOINTS, make_hello, make_obs


def test_hello_roundtrip():
    d = make_hello()
    assert d["type"] == "hello" and d["protocol"] == PROTOCOL_VERSION
    h = Hello.from_dict(d)
    assert h.robot_name == "Go2"
    assert h.joint_names == JOINTS
    assert h.dofs == len(JOINTS)
    assert h.control_hz == 50
    assert [c.name for c in h.cameras] == ["front"]
    assert h.cameras[0].w == 224
    assert h.instruction == "walk"
    assert h.to_dict() == d


def test_obs_parsing_and_image_decode():
    d = make_obs(7, [0.1] * 6, time=1.5, img_size=32)
    kind, obs = parse_phone_message(encode_message(d))
    assert kind == "obs" and isinstance(obs, Obs)
    assert obs.t == 7 and obs.time == 1.5
    np.testing.assert_allclose(obs.proprio.q, np.full(6, 0.1, np.float32))
    assert obs.proprio.base_quat.tolist() == [0, 0, 0, 1]
    img = obs.image("front")
    assert img.shape == (32, 32, 3) and img.dtype == np.uint8
    assert obs.image("front", 16).shape == (16, 16, 3)
    assert obs.image("front", 16) is obs.image("front", 16)  # cached
    with pytest.raises(KeyError):
        obs.image("wrist")
    assert obs.proprio.state_vector().shape == (6 + 6 + 3 + 4 + 3 + 3 + 3,)


def test_encode_image_roundtrip():
    img = np.zeros((8, 8, 3), np.uint8)
    img[..., 0] = 255
    b64 = encode_image(img, quality=95)
    _, obs = parse_phone_message(json.dumps({"type": "obs", "images": {"cam": b64}}))
    out = obs.image("cam")
    assert out.shape == (8, 8, 3)
    assert out[..., 0].mean() > 200 and out[..., 1].mean() < 60


def test_ready_message_shape():
    r = Ready(name="echo", action_space="joint_targets", joint_names=JOINTS, chunk=10, hz=10, obs_hz=5, cameras=["front"])
    d = r.to_dict()
    assert d == {
        "type": "ready",
        "policy": {
            "name": "echo",
            "action_space": "joint_targets",
            "joint_names": JOINTS,
            "chunk": 10,
            "hz": 10.0,
            "obs_hz": 5.0,
            "image": {"w": 224, "h": 224, "cameras": ["front"]},
        },
    }
    assert Ready.from_dict(json.loads(encode_message(d))) == r
    with pytest.raises(ProtocolError):
        Ready(name="x", action_space="bogus", joint_names=[]).to_dict()


def test_action_message():
    m = action_message(3, np.ones((10, 6)))
    assert m["type"] == "action" and m["t"] == 3
    assert len(m["actions"]) == 10 and len(m["actions"][0]) == 6
    assert json.loads(encode_message(m))["actions"][0][0] == 1.0
    assert action_message(1, [1, 2, 3])["actions"] == [[1.0, 2.0, 3.0]]
    with pytest.raises(ProtocolError):
        action_message(1, np.ones((2, 2, 2)))
    with pytest.raises(ProtocolError):
        action_message(1, [[float("nan")]])


def test_status_error_and_framing():
    assert status_message("loading") == {"type": "status", "message": "loading"}
    assert error_message("boom") == {"type": "error", "message": "boom"}
    assert decode_message(b'{"type":"peer","connected":true}') == {"type": "peer", "connected": True}
    for bad in ("not json", "[1,2]", '{"no":"type"}', b"\xff\xfe"):
        with pytest.raises(ProtocolError):
            decode_message(bad)
    kind, d = parse_phone_message('{"type":"peer","connected":false}')
    assert kind == "peer" and d["connected"] is False


def test_named_cameras_additive():
    """Run 2: the hello lists several cameras (with a mount), ready asks for some of them with sizes."""
    h = Hello.from_dict({"type": "hello", "robot": {"name": "Franka", "joint_names": JOINTS},
                         "cameras": [{"name": "front", "w": 224, "h": 224, "mount": "base"},
                                     {"name": "exterior", "w": 320, "h": 180, "mount": "exterior"},
                                     {"name": "wrist", "w": 320, "h": 180, "mount": "wrist"}]})
    assert [c.mount for c in h.cameras] == ["base", "exterior", "wrist"]
    assert h.to_dict()["cameras"][1] == {"name": "exterior", "w": 320, "h": 180, "mount": "exterior"}
    old = Hello.from_dict({"type": "hello", "robot": {"name": "Go2", "joint_names": JOINTS}, "cameras": [{"name": "front"}]})
    assert "mount" not in old.to_dict()["cameras"][0]
    r = Ready(name="droid", action_space="joint_targets", joint_names=JOINTS,
              cameras=["exterior_image_1_left", "wrist_image_left"],
              sizes={"exterior_image_1_left": (320, 180), "wrist_image_left": (320, 180)})
    d = r.to_dict()
    assert d["policy"]["image"]["sizes"]["wrist_image_left"] == [320, 180]
    assert Ready.from_dict(d).sizes["exterior_image_1_left"] == (320, 180)
    assert "sizes" not in Ready(name="e", action_space="joint_targets", joint_names=JOINTS).to_dict()["policy"]["image"]


def test_adapter_requests_first_camera_by_default():
    from simoji_policy.adapters.base import Adapter, PolicyConfig
    from simoji_policy.cli import parse_cameras

    h = Hello.from_dict({"type": "hello", "robot": {"name": "Franka", "joint_names": JOINTS},
                         "cameras": [{"name": "front"}, {"name": "exterior"}, {"name": "wrist"}]})
    a = Adapter(PolicyConfig())
    assert a.reset(h)["image"]["cameras"] == ["front"]          # every camera is a render: only what is asked
    names, sizes = parse_cameras("exterior_image_1_left=320x180,wrist_image_left")
    assert names == ["exterior_image_1_left", "wrist_image_left"] and sizes == {"exterior_image_1_left": (320, 180)}
    b = Adapter(PolicyConfig(cameras=names, camera_sizes=sizes))
    img = b.reset(h)["image"]
    assert img["cameras"] == names and img["sizes"] == {"exterior_image_1_left": [320, 180]}
