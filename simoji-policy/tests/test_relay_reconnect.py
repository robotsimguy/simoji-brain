"""Relay client against a fake relay: pairing, forwarding, peer frames, reconnect and re-pair."""

import asyncio
from http import HTTPStatus
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from websockets.asyncio.server import serve

from simoji_policy.protocol import decode_message, encode_message
from simoji_policy.server import Pairing, RelayClient, relay_ws_url

from conftest import FakePhone, make_hello, make_obs


class FakeRelay:
    """Minimal stand-in for the worker's /v1/relay/ws Durable Object."""

    def __init__(self):
        self.server = None
        self.port = 0
        self.sessions = {}  # session -> {"server": ws|None, "phone": ws|None}
        self.dead = set()  # sessions that answer 404
        self.server_connects = 0
        self.server_joined = asyncio.Event()

    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def _process_request(self, connection, request):
        u = urlsplit(request.path)
        if u.path != "/v1/relay/ws":
            return connection.respond(HTTPStatus.NOT_FOUND, "nope\n")
        qs = parse_qs(u.query)
        session = qs.get("session", [""])[0]
        if session in self.dead:
            return connection.respond(HTTPStatus.NOT_FOUND, "unknown session\n")
        if qs.get("token", [""])[0] != f"tok-{session}-{qs.get('role', [''])[0]}":
            return connection.respond(HTTPStatus.FORBIDDEN, "bad token\n")
        return None

    async def _handler(self, ws):
        qs = parse_qs(urlsplit(ws.request.path).query)
        session, role = qs["session"][0], qs["role"][0]
        other_role = "phone" if role == "server" else "server"
        slot = self.sessions.setdefault(session, {"server": None, "phone": None})
        slot[role] = ws
        if role == "server":
            self.server_connects += 1
            self.server_joined.set()
        other = slot[other_role]
        if other is not None:
            await other.send(encode_message({"type": "peer", "connected": True}))
            await ws.send(encode_message({"type": "peer", "connected": True}))
        try:
            async for frame in ws:
                other = slot[other_role]
                if other is not None:
                    await other.send(frame)
        finally:
            if slot[role] is ws:
                slot[role] = None
            other = slot[other_role]
            if other is not None:
                try:
                    await other.send(encode_message({"type": "peer", "connected": False}))
                except Exception:
                    pass

    async def start(self):
        self.server = await serve(self._handler, "127.0.0.1", 0, process_request=self._process_request)
        self.port = next(iter(self.server.sockets)).getsockname()[1]
        return self

    async def close(self):
        self.server.close()
        await self.server.wait_closed()

    async def kick_server(self, session):
        ws = self.sessions[session]["server"]
        await ws.close(code=1012, reason="relay restart")

    async def wait_server(self, timeout=5.0):
        self.server_joined.clear()
        await asyncio.wait_for(self.server_joined.wait(), timeout)


@pytest.fixture
async def relay():
    r = await FakeRelay().start()
    yield r
    await r.close()


def make_pairer(relay):
    n = {"i": 0}
    paired = []

    def pair_fn():
        n["i"] += 1
        s = f"sess{n['i']}"
        p = Pairing(code=f"{100000 + n['i']:06d}", session=s, token=f"tok-{s}-server")
        paired.append(p)
        return p

    return pair_fn, paired


async def phone_join(relay, session):
    from websockets.asyncio.client import connect

    ws = await connect(relay_ws_url(relay.url(), session, "phone", f"tok-{session}-phone"))
    return ws, FakePhone(ws)


async def test_pair_connect_and_serve_through_relay(relay, echo_runner):
    pair_fn, paired = make_pairer(relay)
    on_paired = []
    client = RelayClient(echo_runner, relay=relay.url(), pair_fn=pair_fn, on_paired=on_paired.append, min_backoff=0.05)
    task = asyncio.create_task(client.run())
    try:
        await asyncio.wait_for(client.connected.wait(), 5)
        assert len(paired) == 1 and on_paired == paired
        assert paired[0].uri == "simoji://pair/100001"
        assert paired[0].pretty == "100 001"

        ws, phone = await phone_join(relay, "sess1")
        try:
            assert (await phone.recv_type("peer"))["connected"] is True
            await phone.send(make_hello())
            assert (await phone.recv_type("ready"))["policy"]["joint_names"]
            await phone.send(make_obs(1, [0.0] * 6))
            act = await phone.recv_type("action")
            assert act["t"] == 1 and np.asarray(act["actions"]).shape == (10, 6)
        finally:
            await ws.close()
    finally:
        client.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_reconnects_after_relay_drops_connection(relay, echo_runner):
    pair_fn, paired = make_pairer(relay)
    client = RelayClient(echo_runner, relay=relay.url(), pair_fn=pair_fn, min_backoff=0.05, max_backoff=0.1)
    task = asyncio.create_task(client.run())
    try:
        await asyncio.wait_for(client.connected.wait(), 5)
        assert relay.server_connects == 1

        await relay.kick_server("sess1")
        await relay.wait_server()
        assert relay.server_connects == 2
        assert len(paired) == 1  # same session, no re-pair
        await asyncio.wait_for(client.connected.wait(), 5)

        # still serves after reconnect
        ws, phone = await phone_join(relay, "sess1")
        try:
            await phone.recv_type("peer")
            await phone.send(make_hello())
            await phone.recv_type("ready")
            await phone.send(make_obs(5, [0.0] * 6))
            assert (await phone.recv_type("action"))["t"] == 5
        finally:
            await ws.close()

        # drop it twice more; backoff must not stall
        for i in range(2):
            await relay.kick_server("sess1")
            await relay.wait_server()
        assert relay.server_connects == 4
    finally:
        client.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_repairs_when_session_is_gone(relay, echo_runner):
    pair_fn, paired = make_pairer(relay)
    client = RelayClient(echo_runner, relay=relay.url(), pair_fn=pair_fn, min_backoff=0.05, max_backoff=0.1)
    task = asyncio.create_task(client.run())
    try:
        await asyncio.wait_for(client.connected.wait(), 5)
        relay.dead.add("sess1")  # relay expired the session
        await relay.kick_server("sess1")
        await relay.wait_server()
        assert len(paired) == 2
        assert paired[1].session == "sess2"
        assert relay.sessions["sess2"]["server"] is not None
    finally:
        client.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_new_code_before_an_unused_one_expires(relay, echo_runner):
    pair_fn, paired = make_pairer(relay)

    def short_lived():
        p = pair_fn()
        p.expires_in = 20.3  # re-paired 20 s before expiry -> after 0.3 s here
        return p

    client = RelayClient(echo_runner, relay=relay.url(), pair_fn=short_lived, min_backoff=0.05, max_backoff=0.1)
    task = asyncio.create_task(client.run())
    try:
        for _ in range(100):
            if len(paired) >= 2 and relay.sessions.get("sess2", {}).get("server") is not None:
                break
            await asyncio.sleep(0.05)
        assert len(paired) >= 2 and paired[1].session == "sess2"
    finally:
        client.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_used_code_is_kept(relay, echo_runner):
    pair_fn, paired = make_pairer(relay)

    def short_lived():
        p = pair_fn()
        p.expires_in = 20.3
        return p

    client = RelayClient(echo_runner, relay=relay.url(), pair_fn=short_lived, min_backoff=0.05, max_backoff=0.1)
    task = asyncio.create_task(client.run())
    try:
        await asyncio.wait_for(client.connected.wait(), 5)
        client._session.joined = True  # a phone joined (peer frame / hello): its code must stay
        await asyncio.sleep(0.8)
        assert len(paired) == 1
    finally:
        client.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_pairing_failure_is_retried(relay, echo_runner):
    calls = {"n": 0}
    pair_fn, _ = make_pairer(relay)

    def flaky_pair():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("relay down")
        return pair_fn()

    client = RelayClient(echo_runner, relay=relay.url(), pair_fn=flaky_pair, min_backoff=0.02, max_backoff=0.05)
    task = asyncio.create_task(client.run())
    try:
        await asyncio.wait_for(client.connected.wait(), 5)
        assert calls["n"] == 3
    finally:
        client.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_relay_ws_url():
    assert (
        relay_ws_url("https://simoji-proxy.parzival2108.workers.dev/", "abc", "server", "tok")
        == "wss://simoji-proxy.parzival2108.workers.dev/v1/relay/ws?session=abc&role=server&token=tok"
    )
    assert relay_ws_url("http://127.0.0.1:1", "s", "phone", "t").startswith("ws://127.0.0.1:1/v1/relay/ws?")


def test_pair_parses_worker_response(monkeypatch):
    import requests

    from simoji_policy.server import pair

    class R:
        status_code = 200
        text = ""

        def json(self):
            return {"code": "483 912", "session": "id1", "token": "tk", "expires_in": 600}

    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen["url"] = url
        return R()

    monkeypatch.setattr(requests, "post", fake_post)
    p = pair("https://relay.example/", "srv")
    assert seen["url"] == "https://relay.example/v1/relay/pair"
    assert p.digits == "483912" and p.pretty == "483 912" and p.uri == "simoji://pair/483912"
    assert p.session == "id1" and p.token == "tk" and p.expires_in == 600
