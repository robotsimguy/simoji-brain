"""Core loop (Session), Direct-mode WebSocket server and Relay client with auto-reconnect."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Awaitable, Callable, Dict, Optional
from urllib.parse import parse_qs, urlsplit

from .adapters.base import Adapter, MissingDependency
from .protocol import (
    Hello,
    Obs,
    ProtocolError,
    action_message,
    encode_message,
    error_message,
    parse_phone_message,
    status_message,
)

log = logging.getLogger("simoji_policy")

Send = Callable[[Dict[str, Any]], Awaitable[None]]


# --------------------------------------------------------------------------- #
# Policy runner: one adapter shared by every session, loaded once, inference serialised
# --------------------------------------------------------------------------- #


class PolicyRunner:
    def __init__(self, adapter: Adapter):
        self.adapter = adapter
        self._loaded = False
        self._load_error: Optional[BaseException] = None
        self._lock = threading.Lock()  # serialises load / reset / act (model is not thread-safe)

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _load_sync(self, status: Callable[[str], None]) -> None:
        with self._lock:
            if self._loaded:
                return
            self.adapter.load(status)
            self._loaded = True

    async def ensure_loaded(self, status: Callable[[str], None]) -> None:
        if self._loaded:
            return
        await asyncio.to_thread(self._load_sync, status)

    async def reset(self, hello: Hello) -> Dict[str, Any]:
        def _reset():
            with self._lock:
                return self.adapter.reset(hello)

        return await asyncio.to_thread(_reset)

    async def act(self, obs: Obs) -> Any:
        def _act():
            with self._lock:
                return self.adapter.act(obs)

        return await asyncio.to_thread(_act)

    def close(self) -> None:
        try:
            self.adapter.close()
        except Exception:  # pragma: no cover - best effort
            log.debug("adapter.close failed", exc_info=True)


# --------------------------------------------------------------------------- #
# Session: the protocol state machine for one phone connection
# --------------------------------------------------------------------------- #


@dataclass
class SessionStats:
    obs_received: int = 0
    actions_sent: int = 0
    errors: int = 0
    last_latency_s: float = 0.0


class Session:
    """Handles frames from one phone. ``send`` delivers a JSON-able dict back to that phone.

    Never raises out of :meth:`handle` — failures become ``error`` frames.
    """

    def __init__(self, runner: PolicyRunner, send: Send, name: str = "session"):
        self.runner = runner
        self._send = send
        self.name = name
        self.hello: Optional[Hello] = None
        self.ready: Optional[Dict[str, Any]] = None
        self.stats = SessionStats()
        self._busy = False
        self._pending: Optional[Obs] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._instruction: Optional[str] = None
        self.joined = False          # a phone has been here (peer connected or hello): its pairing code is used up

    async def send(self, msg: Dict[str, Any]) -> None:
        try:
            await self._send(msg)
        except Exception as e:  # connection gone; the transport loop will notice
            log.debug("[%s] send failed: %s", self.name, e)

    async def error(self, message: str) -> None:
        self.stats.errors += 1
        log.warning("[%s] error: %s", self.name, message)
        await self.send(error_message(message))

    def _status_from_thread(self, message: str) -> None:
        """status() callback usable from the adapter's loader thread."""
        log.info("[%s] status: %s", self.name, message)
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self.send(status_message(message)), self._loop)

    async def handle(self, frame: Any) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            kind, payload = parse_phone_message(frame)
        except ProtocolError as e:
            await self.error(f"bad frame: {e}")
            return
        try:
            if kind == "hello":
                self.joined = True
                await self._on_hello(payload)
            elif kind == "obs":
                await self._on_obs(payload)
            elif kind == "peer":
                connected = bool(payload.get("connected"))
                self.joined = self.joined or connected
                log.info("[%s] peer %s", self.name, "connected" if connected else "disconnected")
                if not connected:
                    self.ready = None
            elif kind in ("ping",):
                await self.send({"type": "pong"})
            else:
                log.debug("[%s] ignoring frame type %r", self.name, kind)
        except Exception as e:  # never crash the loop
            log.debug(traceback.format_exc())
            await self.error(_describe(e))

    async def _on_hello(self, hello: Hello) -> None:
        self.hello = hello
        self.ready = None
        self._instruction = hello.instruction
        log.info(
            "[%s] hello from %s (%d joints, %d cameras, instruction=%r)",
            self.name, hello.robot_name, len(hello.joint_names), len(hello.cameras), hello.instruction,
        )
        if not self.runner.loaded:
            await self.send(status_message("loading model…"))
            try:
                await self.runner.ensure_loaded(self._status_from_thread)
            except MissingDependency as e:
                await self.error(str(e))
                return
            except Exception as e:
                await self.error(f"model load failed: {_describe(e)}")
                return
        policy = await self.runner.reset(hello)
        if not policy.get("joint_names"):
            policy["joint_names"] = list(hello.joint_names)
        self.ready = policy
        await self.send({"type": "ready", "policy": policy})
        log.info("[%s] ready: %s", self.name, {k: v for k, v in policy.items() if k != "joint_names"})

    async def _on_obs(self, obs: Obs) -> None:
        self.stats.obs_received += 1
        if obs.instruction != self._instruction:      # the phone's chat sends a new instruction: say so once
            self._instruction = obs.instruction
            log.info("[%s] instruction: %r (obs %d)", self.name, obs.instruction, obs.t)
        if self.ready is None:
            await self.error("obs before hello/ready")
            return
        if self._busy:
            # protocol allows at most one obs in flight; keep only the newest while inference runs
            self._pending = obs
            return
        self._busy = True
        try:
            while obs is not None:
                await self._act_and_send(obs)
                obs, self._pending = self._pending, None
        finally:
            self._busy = False

    async def _act_and_send(self, obs: Obs) -> None:
        t0 = time.perf_counter()
        try:
            actions = await self.runner.act(obs)
            msg = action_message(obs.t, actions)
        except Exception as e:
            log.debug(traceback.format_exc())
            await self.error(f"policy failed on obs {obs.t}: {_describe(e)}")
            return
        dof = len(self.ready.get("joint_names") or [])
        if dof and msg["actions"] and len(msg["actions"][0]) != dof:
            await self.error(f"policy returned {len(msg['actions'][0])} dims, ready.joint_names has {dof}")
            return
        self.stats.last_latency_s = time.perf_counter() - t0
        self.stats.actions_sent += 1
        await self.send(msg)


def _describe(e: BaseException) -> str:
    s = str(e).strip()
    return f"{type(e).__name__}: {s}" if s else type(e).__name__


# --------------------------------------------------------------------------- #
# Transport helpers
# --------------------------------------------------------------------------- #


async def run_session_over_ws(ws, runner: PolicyRunner, name: str, session: Optional[Session] = None) -> Session:
    """Pump frames from a websockets connection into a Session until it closes."""

    async def send(msg: Dict[str, Any]) -> None:
        await ws.send(encode_message(msg))

    if session is None:
        session = Session(runner, send, name=name)
    else:
        session._send = send
    async for frame in ws:
        await session.handle(frame)
    return session


# --------------------------------------------------------------------------- #
# Direct mode: ws://0.0.0.0:<port>/ws with optional bearer key
# --------------------------------------------------------------------------- #


class DirectServer:
    def __init__(self, runner: PolicyRunner, host: str = "0.0.0.0", port: int = 8765, key: Optional[str] = None):
        self.runner = runner
        self.host = host
        self.port = port
        self.key = key or None
        self._server = None
        self.connections = 0

    def _authorized(self, request) -> bool:
        if not self.key:
            return True
        auth = request.headers.get("Authorization", "") or ""
        if auth.strip() == f"Bearer {self.key}":
            return True
        qs = parse_qs(urlsplit(request.path).query)
        return self.key in qs.get("key", [])

    def _process_request(self, connection, request):
        path = urlsplit(request.path).path
        if path not in ("/ws", "/", "/ws/"):
            return connection.respond(HTTPStatus.NOT_FOUND, "simoji-policy: connect to /ws\n")
        if not self._authorized(request):
            return connection.respond(HTTPStatus.UNAUTHORIZED, "unauthorized: Authorization: Bearer <key> required\n")
        return None

    async def _handler(self, ws) -> None:
        self.connections += 1
        peer = getattr(ws, "remote_address", None)
        name = f"direct:{peer[0]}:{peer[1]}" if peer else "direct"
        log.info("[%s] connected", name)
        try:
            await run_session_over_ws(ws, self.runner, name)
        except Exception as e:
            log.info("[%s] closed: %s", name, _describe(e))
        finally:
            self.connections -= 1
            log.info("[%s] disconnected", name)

    async def start(self):
        from websockets.asyncio.server import serve

        self._server = await serve(
            self._handler,
            self.host,
            self.port,
            process_request=self._process_request,
            max_size=4 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )
        if self.port == 0:  # tests
            self.port = next(iter(self._server.sockets)).getsockname()[1]
        log.info("direct mode listening on ws://%s:%d/ws%s", self.host, self.port, " (bearer key)" if self.key else "")
        return self

    async def serve_forever(self) -> None:
        await self.start()
        await self._server.serve_forever()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


# --------------------------------------------------------------------------- #
# Relay mode: pair, then dial the worker's WebSocket as role=server, reconnecting forever
# --------------------------------------------------------------------------- #

DEFAULT_RELAY = "https://simoji-proxy.parzival2108.workers.dev"


@dataclass
class Pairing:
    code: str
    session: str
    token: str
    expires_in: int = 600

    @property
    def digits(self) -> str:
        return "".join(ch for ch in self.code if ch.isdigit())

    @property
    def pretty(self) -> str:
        d = self.digits
        return f"{d[:3]} {d[3:]}" if len(d) == 6 else self.code

    @property
    def uri(self) -> str:
        return f"simoji://pair/{self.digits}"


def pair(relay: str, server_name: str = "simoji-policy", timeout: float = 15.0) -> Pairing:
    """POST /v1/relay/pair → Pairing."""
    import requests

    r = requests.post(relay.rstrip("/") + "/v1/relay/pair", json={"server_name": server_name}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"relay pair failed: HTTP {r.status_code} {r.text[:200]}")
    d = r.json()
    return Pairing(code=str(d["code"]), session=str(d["session"]), token=str(d["token"]), expires_in=int(d.get("expires_in", 600)))


def relay_ws_url(relay: str, session: str, role: str, token: str) -> str:
    base = relay.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    return f"{base}/v1/relay/ws?session={session}&role={role}&token={token}"


class RelayClient:
    """Keeps a server-role WebSocket to the relay open; re-pairs when the session is gone."""

    def __init__(
        self,
        runner: PolicyRunner,
        relay: str = DEFAULT_RELAY,
        server_name: str = "simoji-policy",
        pair_fn: Optional[Callable[[], Pairing]] = None,
        on_paired: Optional[Callable[[Pairing], None]] = None,
        min_backoff: float = 0.5,
        max_backoff: float = 10.0,
        max_attempts: Optional[int] = None,
    ):
        self.runner = runner
        self.relay = relay
        self.server_name = server_name
        self._pair_fn = pair_fn or (lambda: pair(relay, server_name))
        self.on_paired = on_paired
        self.min_backoff = min_backoff
        self.max_backoff = max_backoff
        self.max_attempts = max_attempts
        self.pairing: Optional[Pairing] = None
        self.connects = 0
        self.attempts = 0
        self.connected = asyncio.Event()
        self._stop = asyncio.Event()
        self._paired_at = 0.0
        self._session: Optional[Session] = None

    async def do_pair(self) -> Pairing:
        self.pairing = await asyncio.to_thread(self._pair_fn)
        self._paired_at = time.monotonic()
        self._session = None
        if self.on_paired:
            self.on_paired(self.pairing)
        return self.pairing

    def stop(self) -> None:
        self._stop.set()

    async def _expire_unused(self, ws) -> None:
        """The relay forgets a code nobody joined after ``expires_in``: get a new one shortly before that,
        so the code shown to the user always works."""
        p = self.pairing
        if p is None:
            return
        left = p.expires_in - 20 - (time.monotonic() - self._paired_at)
        await asyncio.sleep(max(0.0, left))
        if self.pairing is p and not (self._session and self._session.joined):
            log.info("code %s unused for %d s: pairing again", p.pretty, p.expires_in - 20)
            self.pairing = None
            await ws.close()

    async def run(self) -> None:
        from websockets.asyncio.client import connect
        from websockets.exceptions import InvalidStatus

        backoff = self.min_backoff
        while not self._stop.is_set():
            if self.pairing is None:
                try:
                    await self.do_pair()
                except Exception as e:
                    log.warning("pairing failed (%s); retrying in %.1fs", _describe(e), backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.max_backoff)
                    continue
            url = relay_ws_url(self.relay, self.pairing.session, "server", self.pairing.token)
            self.attempts += 1
            if self.max_attempts is not None and self.attempts > self.max_attempts:
                log.error("relay: giving up after %d attempts", self.max_attempts)
                return
            try:
                async with connect(url, max_size=4 * 1024 * 1024, ping_interval=20, ping_timeout=20) as ws:
                    self.connects += 1
                    backoff = self.min_backoff
                    self.connected.set()
                    log.info("relay connected (session %s, code %s)", self.pairing.session, self.pairing.pretty)
                    if self._session is None:
                        self._session = Session(self.runner, lambda m: None, name="relay")
                    expired = asyncio.create_task(self._expire_unused(ws))
                    try:
                        await run_session_over_ws(ws, self.runner, "relay", self._session)
                    finally:
                        expired.cancel()
                    log.info("relay connection closed")
                    if self.pairing is None:     # the code expired unused: pair again at once (new code)
                        continue
            except InvalidStatus as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                log.warning("relay rejected connection (HTTP %s)", code)
                if code in (401, 403, 404, 410):
                    log.info("relay session gone; pairing again")
                    self.pairing = None
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as e:
                rcvd = getattr(e, "rcvd", None)
                code = getattr(rcvd, "code", None)
                log.warning("relay disconnected: %s", _describe(e))
                if code in (4404, 4401, 4410):
                    self.pairing = None
                    continue
            finally:
                self.connected.clear()
            if self._stop.is_set():
                return
            log.info("reconnecting to relay in %.1fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff)
