"""Echo adapter end to end: fake phone <-> DirectServer over a real local websocket."""

import asyncio
import time

import numpy as np
import pytest

from simoji_policy.server import DirectServer

from conftest import JOINTS, FakePhone, make_hello, make_obs, phone_connect


@pytest.fixture
async def direct(echo_runner):
    srv = DirectServer(echo_runner, host="127.0.0.1", port=0)
    await srv.start()
    yield srv
    await srv.close()


async def test_hello_ready_obs_action(direct):
    async with phone_connect(f"ws://127.0.0.1:{direct.port}/ws") as ws:
        phone = FakePhone(ws)
        await phone.send(make_hello())
        ready = await phone.recv_type("ready")
        p = ready["policy"]
        assert p["name"] == "echo"
        assert p["action_space"] == "joint_targets"
        assert p["joint_names"] == JOINTS
        assert p["chunk"] == 10 and p["hz"] == 10.0 and p["obs_hz"] == 5.0
        assert p["image"] == {"w": 64, "h": 64, "cameras": ["front"]}

        q = [0.0, 0.5, -1.0, 0.0, 0.5, -1.0]
        t0 = time.perf_counter()
        await phone.send(make_obs(1, q, time=0.0))
        act = await phone.recv_type("action")
        dt = time.perf_counter() - t0
        assert dt < 1.0, f"action took {dt:.2f}s"
        assert act["t"] == 1
        arr = np.asarray(act["actions"])
        assert arr.shape == (10, len(JOINTS))  # chunk x dof in joint_names order
        # sinusoid around q: all rows within amp of q, and the rows differ over the chunk
        assert np.all(np.abs(arr - np.asarray(q)) <= 0.15 + 1e-6)
        assert not np.allclose(arr[0], arr[5])

        # several obs in sequence, each answered with matching t
        for t in range(2, 6):
            await phone.send(make_obs(t, q, time=t / 5))
            act = await phone.recv_type("action")
            assert act["t"] == t
            assert np.asarray(act["actions"]).shape == (10, 6)


async def test_obs_before_hello_is_error_not_crash(direct):
    async with phone_connect(f"ws://127.0.0.1:{direct.port}/ws") as ws:
        phone = FakePhone(ws)
        await phone.send(make_obs(1, [0.0] * 6))
        err = await phone.recv_type("error")
        assert "hello" in err["message"]
        # bad frame
        await ws.send("this is not json")
        err = await phone.recv_type("error")
        assert "bad frame" in err["message"]
        # still alive afterwards
        await phone.send(make_hello())
        assert (await phone.recv_type("ready"))["policy"]["name"] == "echo"
        await phone.send(make_obs(2, [0.0] * 6))
        assert (await phone.recv_type("action"))["t"] == 2


async def test_zero_adapter_holds_pose():
    from simoji_policy.adapters.base import PolicyConfig
    from simoji_policy.adapters.echo import ZeroAdapter
    from simoji_policy.server import PolicyRunner

    srv = DirectServer(PolicyRunner(ZeroAdapter(PolicyConfig(chunk=4))), host="127.0.0.1", port=0)
    await srv.start()
    try:
        async with phone_connect(f"ws://127.0.0.1:{srv.port}/ws") as ws:
            phone = FakePhone(ws)
            await phone.send(make_hello())
            await phone.recv_type("ready")
            q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            await phone.send(make_obs(1, q))
            act = await phone.recv_type("action")
            assert np.asarray(act["actions"]).shape == (4, 6)
            assert np.allclose(np.asarray(act["actions"]), np.asarray(q), atol=1e-6)
    finally:
        await srv.close()


async def test_failing_policy_sends_error_and_keeps_running():
    from simoji_policy.adapters.base import Adapter, PolicyConfig
    from simoji_policy.server import PolicyRunner

    class Flaky(Adapter):
        name = "flaky"

        def act(self, obs):
            if obs.t % 2:
                raise RuntimeError("odd obs")
            return np.zeros((self.config.chunk, len(self.joint_names)))

    srv = DirectServer(PolicyRunner(Flaky(PolicyConfig(chunk=3))), host="127.0.0.1", port=0)
    await srv.start()
    try:
        async with phone_connect(f"ws://127.0.0.1:{srv.port}/ws") as ws:
            phone = FakePhone(ws)
            await phone.send(make_hello())
            await phone.recv_type("ready")
            await phone.send(make_obs(1, [0.0] * 6))
            err = await phone.recv_type("error")
            assert "odd obs" in err["message"]
            await phone.send(make_obs(2, [0.0] * 6))
            assert (await phone.recv_type("action"))["t"] == 2
    finally:
        await srv.close()


async def test_status_while_loading():
    from simoji_policy.adapters.base import Adapter, PolicyConfig
    from simoji_policy.server import PolicyRunner

    class Slow(Adapter):
        name = "slow"

        def load(self, status):
            status("downloading weights")
            time.sleep(0.05)
            status("warming up")

        def act(self, obs):
            return np.zeros((self.config.chunk, len(self.joint_names)))

    srv = DirectServer(PolicyRunner(Slow(PolicyConfig(chunk=2))), host="127.0.0.1", port=0)
    await srv.start()
    try:
        async with phone_connect(f"ws://127.0.0.1:{srv.port}/ws") as ws:
            phone = FakePhone(ws)
            await phone.send(make_hello())
            seen = []
            while True:
                m = await phone.recv()
                seen.append(m)
                if m["type"] == "ready":
                    break
            statuses = [m["message"] for m in seen if m["type"] == "status"]
            assert statuses[0] == "loading model…"
            assert "downloading weights" in statuses and "warming up" in statuses
            assert seen[-1]["type"] == "ready"
    finally:
        await srv.close()
