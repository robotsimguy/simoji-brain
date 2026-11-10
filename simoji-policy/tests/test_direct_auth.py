"""Direct mode: bearer key check and path handling."""

import pytest
from websockets.exceptions import InvalidStatus

from simoji_policy.server import DirectServer

from conftest import FakePhone, make_hello, phone_connect


@pytest.fixture
async def keyed(echo_runner):
    srv = DirectServer(echo_runner, host="127.0.0.1", port=0, key="s3cret")
    await srv.start()
    yield srv
    await srv.close()


async def test_missing_key_rejected(keyed):
    with pytest.raises(InvalidStatus) as ei:
        async with phone_connect(f"ws://127.0.0.1:{keyed.port}/ws"):
            pass
    assert ei.value.response.status_code == 401


async def test_wrong_key_rejected(keyed):
    with pytest.raises(InvalidStatus) as ei:
        async with phone_connect(f"ws://127.0.0.1:{keyed.port}/ws", {"Authorization": "Bearer nope"}):
            pass
    assert ei.value.response.status_code == 401


async def test_correct_key_accepted(keyed):
    async with phone_connect(f"ws://127.0.0.1:{keyed.port}/ws", {"Authorization": "Bearer s3cret"}) as ws:
        phone = FakePhone(ws)
        await phone.send(make_hello())
        assert (await phone.recv_type("ready"))["policy"]["name"] == "echo"


async def test_query_key_accepted(keyed):
    async with phone_connect(f"ws://127.0.0.1:{keyed.port}/ws?key=s3cret") as ws:
        phone = FakePhone(ws)
        await phone.send(make_hello())
        assert (await phone.recv_type("ready"))["policy"]["name"] == "echo"


async def test_wrong_path_404(keyed):
    with pytest.raises(InvalidStatus) as ei:
        async with phone_connect(f"ws://127.0.0.1:{keyed.port}/nope", {"Authorization": "Bearer s3cret"}):
            pass
    assert ei.value.response.status_code == 404


async def test_no_key_configured_accepts_anyone(echo_runner):
    srv = DirectServer(echo_runner, host="127.0.0.1", port=0, key=None)
    await srv.start()
    try:
        async with phone_connect(f"ws://127.0.0.1:{srv.port}/ws") as ws:
            phone = FakePhone(ws)
            await phone.send(make_hello())
            assert (await phone.recv_type("ready"))["policy"]["name"] == "echo"
    finally:
        await srv.close()
