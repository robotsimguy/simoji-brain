"""``simoji-policy`` command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

from . import __version__
from .adapters import ADAPTER_NAMES, make_adapter, resolve_adapter_name
from .adapters.base import MissingDependency, PolicyConfig
from .server import DEFAULT_RELAY, DirectServer, Pairing, PolicyRunner, RelayClient

BIG_DIGITS = {
    "0": ["█████", "█   █", "█   █", "█   █", "█████"],
    "1": ["   ██", "    █", "    █", "    █", "    █"],
    "2": ["█████", "    █", "█████", "█    ", "█████"],
    "3": ["█████", "    █", "█████", "    █", "█████"],
    "4": ["█   █", "█   █", "█████", "    █", "    █"],
    "5": ["█████", "█    ", "█████", "    █", "█████"],
    "6": ["█████", "█    ", "█████", "█   █", "█████"],
    "7": ["█████", "    █", "    █", "    █", "    █"],
    "8": ["█████", "█   █", "█████", "█   █", "█████"],
    "9": ["█████", "█   █", "█████", "    █", "█████"],
    " ": ["   ", "   ", "   ", "   ", "   "],
}


def big_code(code: str) -> str:
    rows = [""] * 5
    for ch in code:
        glyph = BIG_DIGITS.get(ch, BIG_DIGITS[" "])
        for i in range(5):
            rows[i] += glyph[i] + "  "
    return "\n".join(rows)


def qr_text(data: str) -> str:
    """Render a QR code for the terminal (falls back to the URI if qrcode is missing)."""
    try:
        import qrcode
    except ImportError:  # pragma: no cover
        return f"(install qrcode for a QR)  {data}"
    import io

    qr = qrcode.QRCode(border=1, error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(data)
    qr.make(fit=True)
    buf = io.StringIO()
    try:
        qr.print_ascii(out=buf, invert=True)
    except Exception:  # pragma: no cover
        qr.print_ascii(out=buf)
    return buf.getvalue()


def print_pairing(p: Pairing, relay: str) -> None:
    lines = [
        "",
        "=" * 60,
        "  Simoji pairing code  (Simoji > Policy > Pair by code)",
        "",
        big_code(p.pretty),
        "",
        f"  code: {p.pretty}    expires in {p.expires_in // 60} min if nobody joins",
        f"  relay: {relay}",
        "",
        qr_text(p.uri),
        f"  {p.uri}",
        "=" * 60,
        "",
    ]
    print("\n".join(lines), flush=True)


def write_status(path: Optional[str], **fields) -> None:
    """Merge fields into the JSON status file (read by a launcher's ready page). Atomic replace."""
    if not path:
        return
    try:
        with open(path) as f:
            cur = json.load(f)
    except (OSError, ValueError):
        cur = {}
    cur.update(fields, updated=time.time())
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f)
    os.replace(tmp, path)


def parse_extra(items: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = v.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="simoji-policy", description="Serve a robot policy to the Simoji iPhone simulator.")
    p.add_argument("--version", action="version", version=f"simoji-policy {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="serve a model (pair by code and/or direct ws)")
    s.add_argument("model", nargs="?", default="", help="HF repo, checkpoint dir, policy .py file, or server URL (depends on adapter)")
    s.add_argument("--adapter", default="auto", choices=list(ADAPTER_NAMES), help="which adapter to use (default: auto-detect from <model>)")
    s.add_argument("--relay", default=DEFAULT_RELAY, help="relay base URL for pair-by-code mode")
    s.add_argument("--no-relay", action="store_true", help="do not pair / dial the relay (direct mode only)")
    s.add_argument("--port", type=int, default=8765, help="direct-mode WebSocket port (0 = disabled)")
    s.add_argument("--host", default="0.0.0.0", help="direct-mode bind address")
    s.add_argument("--key", default=None, help="direct-mode bearer key (Authorization: Bearer <key>)")
    s.add_argument("--chunk", type=int, default=10, help="action vectors per chunk")
    s.add_argument("--hz", type=float, default=10.0, help="playback rate of action vectors on the phone")
    s.add_argument("--obs-hz", type=float, default=5.0, help="how often the phone sends obs")
    s.add_argument("--image", type=int, default=224, help="camera image size requested from the phone (square)")
    s.add_argument("--cameras", default="", help="cameras to request, comma separated, each NAME or NAME=WxH "
                   "(e.g. exterior_image_1_left=320x180,wrist_image_left=320x180; default: the phone's first camera)")
    s.add_argument("--instruction", default=None, help="language instruction override")
    s.add_argument("--name", default=None, help="server name shown to the phone when pairing")
    s.add_argument("--extra", "-x", action="append", metavar="KEY=VAL", help="adapter-specific option (repeatable)")
    s.add_argument("--status-file", default=None, help="write the state (loading / ready + pairing code) as JSON here")
    s.add_argument("-v", "--verbose", action="store_true")
    return p


def parse_cameras(text: str):
    """"a=320x180,b" -> (["a", "b"], {"a": (320, 180)})."""
    names, sizes = [], {}
    for part in [p.strip() for p in (text or "").split(",") if p.strip()]:
        name, _, wh = part.partition("=")
        names.append(name)
        if wh:
            w, _, h = wh.lower().partition("x")
            sizes[name] = (int(w), int(h or w))
    return names, sizes


async def serve_async(args) -> int:
    cams, sizes = parse_cameras(getattr(args, "cameras", ""))
    config = PolicyConfig(
        model=args.model,
        chunk=args.chunk,
        hz=args.hz,
        obs_hz=args.obs_hz,
        image=args.image,
        cameras=cams,
        camera_sizes=sizes,
        instruction=args.instruction,
        extra=parse_extra(args.extra),
    )
    name = resolve_adapter_name(args.adapter, args.model)
    try:
        adapter = make_adapter(name, config)
    except MissingDependency as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    runner = PolicyRunner(adapter)
    print(f"simoji-policy {__version__}  adapter={name}  model={args.model or '-'}  chunk={args.chunk} hz={args.hz} obs_hz={args.obs_hz} image={args.image}")

    # Preload the model so the user sees install errors before pairing (and the phone gets ready fast).
    def status(m: str) -> None:
        print(f"  [{name}] {m}", flush=True)
        write_status(args.status_file, state="loading", message=m)

    try:
        await runner.ensure_loaded(status)
    except MissingDependency as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"warning: model preload failed ({type(e).__name__}: {e}); will retry on first hello", file=sys.stderr)

    tasks = []
    direct = None
    if args.port:
        direct = DirectServer(runner, host=args.host, port=args.port, key=args.key)
        await direct.start()
        print(f"direct mode: ws://{args.host}:{direct.port}/ws" + (f"  (key: {args.key})" if args.key else "  (no key)"))
        tasks.append(asyncio.create_task(direct._server.serve_forever()))
    if not args.no_relay:
        relay = RelayClient(
            runner,
            relay=args.relay,
            server_name=args.name or f"{name}:{args.model or 'policy'}",
            on_paired=lambda p: (print_pairing(p, args.relay), write_status(
                args.status_file, state="ready", message="ready", code=p.digits, pretty=p.pretty, uri=p.uri,
                paired_at=time.time(), expires_in=p.expires_in)),
        )
        tasks.append(asyncio.create_task(relay.run()))
    if not tasks:
        print("nothing to do: --no-relay and --port 0", file=sys.stderr)
        return 1
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        for t in tasks:
            t.cancel()
        if direct is not None:
            await direct.close()
        runner.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("websockets").setLevel(logging.WARNING)
    if args.cmd == "serve":
        try:
            return asyncio.run(serve_async(args))
        except KeyboardInterrupt:
            print("\nbye")
            return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
