#!/usr/bin/env python3
"""buoay sub bridge: dashboard -> server -> this script -> ESP32 (UDP).

The ESP32 (Bowie firmware) joins the Pixel hotspot and listens for UDP on port 4210.
The server is on another network, so run this on any computer joined to that same
Wi-Fi (the Pixel hotspot). It connects out to the server and forwards the dashboard's
commands to the ESP32, exactly like Bowie's Control/Drive.py does from the keyboard.

    python sub/bridge.py "<dashboard link from LINKS.txt>"
    python sub/bridge.py "<link>" --esp 10.27.223.132 --port 4210

Standard library only (Python 3.8+), works on Windows, macOS and Linux.
"""
import argparse
import base64
import ipaddress
import json
import os
import socket
import ssl
import struct
import sys
import threading
import time
from urllib.parse import parse_qs, quote, urlsplit

DEFAULT_ESP_IP = "10.27.223.132"  # from Bowie Control/Drive.py; the ESP32 prints its IP on serial at boot
DEFAULT_ESP_PORT = 4210
SEND_RATE = 20       # UDP packets per second (same as Drive.py)
STALE_S = 0.5        # stop the motors if the server has been silent this long
RECONNECT_S = 2


# --------------------------------------------------------------------------
# Minimal WebSocket client (RFC 6455), enough for the buoay server
# --------------------------------------------------------------------------

class WebSocket:
    def __init__(self, host, port, path, tls, insecure):
        raw = socket.create_connection((host, port), timeout=10)
        if tls:
            ctx = ssl.create_default_context()
            if insecure:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            raw = ctx.wrap_socket(raw, server_hostname=host)
        self.sock = raw
        self.buf = b""
        self.lock = threading.Lock()

        key = base64.b64encode(os.urandom(16)).decode()
        host_header = host if port in (80, 443) else f"{host}:{port}"
        req = (f"GET {path} HTTP/1.1\r\nHost: {host_header}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
               f"User-Agent: buoay-sub-bridge\r\n\r\n")
        self.sock.sendall(req.encode())
        while b"\r\n\r\n" not in self.buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("server closed the connection during handshake")
            self.buf += chunk
        head, self.buf = self.buf.split(b"\r\n\r\n", 1)
        status = head.split(b"\r\n", 1)[0].decode(errors="replace")
        if " 101" not in status:
            if " 401" in status:
                raise PermissionError("server refused the access key (use the full link with ?key=...)")
            raise ConnectionError(f"handshake failed: {status}")
        # The server sends a command ~10x per second; silence this long means the link is dead.
        self.sock.settimeout(5)

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("connection closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _send_frame(self, opcode, payload):
        header = bytearray([0x80 | opcode])
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        with self.lock:
            self.sock.sendall(bytes(header) + mask + masked)

    def send_json(self, obj):
        self._send_frame(0x1, json.dumps(obj).encode())

    def recv_text(self):
        """Next text message, or None when the server closes the connection."""
        message = b""
        while True:
            b0, b1 = self._read(2)
            opcode, fin = b0 & 0x0F, b0 & 0x80
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b1 & 0x80 else None
            payload = self._read(n)
            if mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            message += payload
            if fin:
                if opcode in (0x1, 0x0):
                    return message.decode(errors="replace")
                message = b""  # binary: not used, ignore

    def close(self):
        try:
            self._send_frame(0x8, b"")
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Bridge
# --------------------------------------------------------------------------

state = {
    "left": 0.0, "right": 0.0, "servo": 0.5,
    "at": 0.0,             # when the server last sent a command
    "server": "connecting",
    "rtt": None,
    "packets": 0,
    "udp_error": None,
}
state_lock = threading.Lock()


def is_local_host(host):
    if host in ("localhost",):
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return host.endswith(".local")


def server_thread(url, key, target, insecure):
    parts = urlsplit(url)
    tls = parts.scheme in ("https", "wss")
    host = parts.hostname
    port = parts.port or (443 if tls else 80)
    path = f"/ws?role=sub&key={quote(key)}"
    # The desktop's own https://<ip>:8443 links use a self-signed certificate.
    insecure = insecure or is_local_host(host)

    while True:
        ws = None
        try:
            ws = WebSocket(host, port, path, tls, insecure)
            with state_lock:
                state["server"] = "connected"
            ws.send_json({"type": "hello", "target": target, "host": socket.gethostname()})
            last_report = 0.0
            while True:
                text = ws.recv_text()
                if text is None:
                    raise ConnectionError("server closed the connection")
                try:
                    m = json.loads(text)
                except ValueError:
                    continue
                if m.get("type") != "drive":
                    continue
                with state_lock:
                    state["left"] = clamp01(m.get("left"))
                    state["right"] = clamp01(m.get("right"))
                    state["servo"] = clamp01(m.get("servo", 0.5))
                    state["at"] = time.monotonic()
                    packets, err = state["packets"], state["udp_error"]
                # Echo the server's timestamp so the dashboard can show the round trip.
                if isinstance(m.get("t"), (int, float)):
                    ws.send_json({"type": "ack", "t": m["t"]})
                now = time.monotonic()
                if now - last_report > 1:
                    last_report = now
                    ws.send_json({"type": "bridge", "packets": packets, "error": err})
        except PermissionError as e:
            with state_lock:
                state["server"] = str(e)
            print(f"\n{e}")
        except Exception as e:
            with state_lock:
                state["server"] = f"offline ({type(e).__name__}), retrying"
        finally:
            if ws:
                ws.close()
        time.sleep(RECONNECT_S)


def clamp01(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(0.0, min(1.0, v))


def packet(left, right, servo):
    # Bowie firmware TeleopCommand: float left_drive, float right_drive, float servo_pos,
    # uint32_t timestamp (overwritten by the ESP32). Little-endian, 16 bytes.
    return struct.pack("<fffI", left, right, servo, 0)


def main():
    ap = argparse.ArgumentParser(description="Forward dashboard sub controls to the ESP32 over UDP.")
    ap.add_argument("link", nargs="?", help="dashboard (or phone) link from LINKS.txt, including ?key=...")
    ap.add_argument("--esp", default=DEFAULT_ESP_IP, help=f"ESP32 IP address (default {DEFAULT_ESP_IP})")
    ap.add_argument("--port", type=int, default=DEFAULT_ESP_PORT, help=f"ESP32 UDP port (default {DEFAULT_ESP_PORT})")
    ap.add_argument("--key", help="access key, if the link doesn't include ?key=")
    ap.add_argument("--insecure", action="store_true", help="accept a self-signed server certificate")
    args = ap.parse_args()

    link = args.link or input("Paste the dashboard link from LINKS.txt: ").strip().strip('"')
    if "://" not in link:
        link = "https://" + link
    key = args.key or (parse_qs(urlsplit(link).query).get("key") or [None])[0]
    if not key:
        key = input("Access key (the part after ?key= in the link): ").strip()

    target = f"{args.esp}:{args.port}"
    esp = (args.esp, args.port)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print()
    print("================================")
    print("       BUOAY SUB BRIDGE")
    print("================================")
    print(f"Server: {urlsplit(link).hostname}")
    print(f"ESP32:  {target} (UDP)")
    print("Drive the sub from the dashboard. Ctrl+C to stop.")
    print()

    threading.Thread(target=server_thread, args=(link, key, target, args.insecure), daemon=True).start()

    try:
        while True:
            with state_lock:
                fresh = time.monotonic() - state["at"] < STALE_S
                left = state["left"] if fresh else 0.0
                right = state["right"] if fresh else 0.0
                servo = state["servo"]
                server = state["server"]
            try:
                udp.sendto(packet(left, right, servo), esp)
                err = None
            except OSError as e:
                err = f"UDP send failed: {e.strerror or e}"
            with state_lock:
                state["packets"] += 1
                state["udp_error"] = err
                packets = state["packets"]
            status = f"Server: {server:<12} | L {left:.2f} R {right:.2f} SERVO {servo:.2f} | sent {packets}"
            if err:
                status += f" | {err}"
            print("\r" + status.ljust(100)[:120], end="", flush=True)
            time.sleep(1 / SEND_RATE)
    except KeyboardInterrupt:
        pass
    finally:
        # Safety stop, a few times in case one packet is lost.
        for _ in range(5):
            try:
                udp.sendto(packet(0.0, 0.0, state["servo"]), esp)
            except OSError:
                pass
            time.sleep(0.02)
        udp.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
