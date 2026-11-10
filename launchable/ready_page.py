"""Simoji robot brain: the page a Brev user opens when the instance is up.

Shows the live pairing code from simoji-policy's --status-file: an "Open in Simoji" button (for a phone),
a QR (for a second screen) and the 6 digits (type them anywhere). Standard library + qrcode only, no
external resources, no cookies, no analytics.

    python ready_page.py --status /var/lib/simoji/status.json --port 8080
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("SIMOJI_MODEL_TITLE", "NVIDIA GR00T N1.7 (DROID)")
IDLE_MIN = int(os.environ.get("SIMOJI_IDLE_MINUTES", "30"))

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Simoji robot brain</title>
<style>
:root{{--accent:#FF7900;--bg:#FFF2DE;--ink:#2B251D;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1814;--ink:#FFFAF2;--card:#2B251D}}}}
*{{box-sizing:border-box}}
body{{margin:0;font:17px/1.5 system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--ink)}}
main{{max-width:520px;margin:0 auto;padding:28px 20px 40px}}
.mark{{font:700 34px ui-rounded,"SF Pro Rounded",system-ui,sans-serif;letter-spacing:-.04em;position:relative;display:inline-block}}
.mark svg{{position:absolute;left:1.02em;top:.98em}}
.card{{background:var(--card);border-radius:18px;padding:22px;margin-top:20px;text-align:center}}
.btn{{display:block;background:var(--accent);color:#fff;text-decoration:none;font-weight:700;font-size:19px;padding:16px;border-radius:14px}}
.code{{font:700 44px ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.08em;margin:6px 0 2px}}
.muted{{opacity:.7;font-size:15px}}
.qr svg{{width:200px;height:200px;background:#fff;padding:8px;border-radius:10px}}
.step{{text-align:left;font-size:15px}}
button{{font:inherit;background:none;border:1px solid currentColor;color:inherit;border-radius:10px;padding:8px 14px}}
footer{{margin-top:26px;font-size:13px;opacity:.7}}
a{{color:var(--accent)}}
</style>
{refresh}
</head><body><main>
<div class="mark">Simoji<svg width="21" height="7" viewBox="0 0 21 7"><path d="M1.5 1.5 Q10.5 9.6 19.5 1.5" fill="none" stroke="#FF7900" stroke-width="3" stroke-linecap="round"/></svg></div>
<p class="muted">Robot brain · {model}</p>
{body}
<footer>
Stops by itself {idle} minutes after your phone disconnects, so you're not billed for an idle GPU.
You can also stop it now: <form method="post" action="/stop" style="display:inline"><button>Stop now</button></form><br><br>
Model: NVIDIA Isaac GR00T. Built on NVIDIA Cosmos. Licensed by NVIDIA Corporation under the NVIDIA Open Model License.
Simoji is not affiliated with NVIDIA. · <a href="https://www.simoji.ai/support">Help</a>
</footer>
</main></body></html>"""


def qr_svg(data: str) -> str:
    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        return ""
    img = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage, border=1)
    buf = io.BytesIO()
    img.save(buf)
    svg = buf.getvalue().decode()
    return svg[svg.find("<svg"):]


def render(status: dict) -> str:
    state = status.get("state")
    if state == "ready" and status.get("code"):
        code = html.escape(str(status["code"]))
        pretty = html.escape(str(status.get("pretty", code)))
        link = f"https://www.simoji.ai/connect/{code}"
        body = f"""
<div class="card">
  <a class="btn" href="{link}">Open in Simoji</a>
  <p class="muted">On your iPhone: tap the button.</p>
</div>
<div class="card">
  <div class="muted">Or type this code in Simoji → Robots → Policy → Remote policy</div>
  <div class="code">{pretty}</div>
  <div class="muted">A new code appears here if nobody joins within 10 minutes.</div>
</div>
<div class="card qr"><div class="muted">On another screen? Scan with your iPhone camera.</div>{qr_svg(link)}</div>"""
        return PAGE.format(model=html.escape(MODEL), body=body, idle=IDLE_MIN,
                           refresh='<meta http-equiv="refresh" content="30">')
    msg = html.escape(str(status.get("message") or "starting…"))
    body = f"""
<div class="card">
  <p><strong>Getting your robot brain ready</strong></p>
  <p class="muted">{msg}</p>
  <p class="step">The first start downloads the model (about 10 GB) and takes a few minutes.
  This page refreshes by itself; keep it open.</p>
</div>"""
    return PAGE.format(model=html.escape(MODEL), body=body, idle=IDLE_MIN,
                       refresh='<meta http-equiv="refresh" content="5">')


def make_handler(status_path: str):
    class H(BaseHTTPRequestHandler):
        def _status(self) -> dict:
            try:
                with open(status_path) as f:
                    return json.load(f)
            except (OSError, ValueError):
                return {}

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/status.json"):
                s = {k: v for k, v in self._status().items() if k in ("state", "message", "code", "pretty", "updated")}
                return self._send(200, json.dumps(s).encode(), "application/json")
            self._send(200, render(self._status()).encode(), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path == "/stop":
                self._send(200, b"<p style='font:18px system-ui;padding:30px'>Stopping the GPU now. You can close this page.</p>",
                           "text/html; charset=utf-8")
                subprocess.Popen(["sudo", "-n", "shutdown", "-h", "+0"])
                return
            self._send(404, b"not found", "text/plain")

        def log_message(self, *a):
            pass

    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="/var/lib/simoji/status.json")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(a.status)).serve_forever()


if __name__ == "__main__":
    main()
