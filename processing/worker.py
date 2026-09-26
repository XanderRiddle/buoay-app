"""
buoay-app reconstruction worker.

Live mode (default): connects to the Node server, watches the phone's frames,
picks keyframes, runs VGGT in batches and sends the growing 3D model to the dashboard.

    python worker.py                      # server on this machine (https://localhost:8443)
    python worker.py --server wss://192.168.1.20:8443

Offline mode: rebuild a model from a folder of photos (or a saved scan) and
write output/model_preview.html. Handy for testing without the phone.

    python worker.py --images test_images
    python worker.py --images scans/20260926-141500
"""

import argparse
import asyncio
import glob
import json
import os
import queue
import ssl
import sys
import threading
import time

import cv2
import numpy as np

from recon import Reconstructor, VGGTRunner, default_batch, log

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# --------------------------------------------------------------------------- #
# Keyframe selection
# --------------------------------------------------------------------------- #

class KeyframeSelector:
    """
    Picks frames worth reconstructing from the ~10 fps stream:
      - the view has moved enough since the last keyframe (so batches cover new wall)
      - but not so much that neighbouring keyframes stop overlapping
      - and it's the sharpest of the recent frames (skip motion blur)
    """

    SMALL_W = 160

    def __init__(self, min_shift=0.18, max_wait=3.0, min_gap=0.3):
        self.min_shift = min_shift   # fraction of frame width
        self.max_wait = max_wait     # force a keyframe after this many seconds of small motion
        self.min_gap = min_gap       # never faster than this (seconds)
        self.reset()

    def reset(self):
        self.last_small = None
        self.last_time = 0.0
        self.recent = []             # [(sharpness, small, jpeg, rgb)]
        self.sharp_hist = []
        self._window = None

    def _small(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h = int(gray.shape[0] * self.SMALL_W / gray.shape[1])
        return cv2.resize(gray, (self.SMALL_W, h), interpolation=cv2.INTER_AREA).astype(np.float32)

    def offer(self, jpeg, rgb):
        """Returns (jpeg, rgb) of a new keyframe, or None."""
        now = time.time()
        small = self._small(rgb)
        sharp = cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[::2, ::2], cv2.CV_32F).var()
        self.sharp_hist = (self.sharp_hist + [sharp])[-50:]
        self.recent = (self.recent + [(sharp, small, jpeg, rgb)])[-5:]

        if self.last_small is None:
            if len(self.recent) >= 3:           # let autofocus settle for a few frames
                return self._take(now)
            return None
        if now - self.last_time < self.min_gap:
            return None
        if self.last_small.shape != small.shape:  # phone rotated: start fresh
            return self._take(now)

        if self._window is None or self._window.shape != small.shape:
            self._window = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(self.last_small, small, self._window)
        shift = np.hypot(dx, dy) / small.shape[1]
        changed = np.abs(small - self.last_small).mean() > 30  # big appearance change (e.g. moved closer)
        waited = now - self.last_time > self.max_wait

        if shift >= self.min_shift or changed or response < 0.05 or (waited and shift > 0.03):
            return self._take(now)
        return None

    def _take(self, now):
        best = max(self.recent, key=lambda r: r[0])
        # if every recent frame is blurry compared to usual, wait for a sharper one
        if len(self.sharp_hist) >= 10 and best[0] < 0.4 * np.median(self.sharp_hist) \
                and now - self.last_time < self.max_wait * 2:
            return None
        self.last_small = best[1]
        self.last_time = now
        self.recent = []
        return best[2], best[3]


# --------------------------------------------------------------------------- #
# Reconstruction thread
# --------------------------------------------------------------------------- #

class ReconThread(threading.Thread):
    """Runs VGGT off the network thread. Commands arrive through a queue so order is kept."""

    def __init__(self, runner, send, batch_size, overlap):
        super().__init__(daemon=True)
        self.rec = Reconstructor(runner, batch_size=batch_size, overlap=overlap)
        self.runner = runner
        self.send = send              # thread-safe callable(dict)
        self.q = queue.Queue()
        self.scan_id = None
        self.scan_dir = None
        self.state = "ready"

    def status(self, state=None, message=None):
        if state:
            self.state = state
        r = self.rec
        self.send({
            "type": "recon", "state": self.state, "message": message, "scan": self.scan_id,
            "keyframes": len(r.order), "pending": len(r.pending) + self.q.qsize(),
            "batches": r.batches, "rejected": r.rejected, "batchSize": r.batch_size,
            "lastBatchSeconds": r.last_batch_seconds, "peakGb": self.runner.last_peak_gb,
            "device": self.runner.device.type,
        })

    def run(self):
        while True:
            cmd = self.q.get()
            kind = cmd[0]
            if kind == "start":
                self.scan_id = cmd[1]
                self.scan_dir = os.path.join(HERE, "scans", self.scan_id)
                os.makedirs(self.scan_dir, exist_ok=True)
                self.rec.reset()
                self.send({"type": "model", "scan": self.scan_id, "reset": True})
                self.status("recording", "New scan started")
                log(f"scan {self.scan_id} started")
            elif kind == "keyframe":
                kf_id, jpeg, rgb = cmd[1], cmd[2], cmd[3]
                if self.scan_dir:
                    with open(os.path.join(self.scan_dir, f"kf_{kf_id:04d}.jpg"), "wb") as f:
                        f.write(jpeg)
                self.rec.add_keyframe(kf_id, rgb)
                self.status()
                self._process(flush=False)
            elif kind == "stop":
                self._process(flush=True)
                if self.scan_id and self.rec.order:
                    self._save_final()
                self.status("ready", "Scan finished")
                log(f"scan {self.scan_id} finished: {len(self.rec.order)} keyframes")

    def _process(self, flush):
        # take all keyframes that arrived meanwhile first, so batches stay full
        while self.rec.ready(flush):
            self.status("processing", f"Reconstructing batch {self.rec.batches + 1}")
            try:
                changed = self.rec.step(flush)
            except Exception as e:  # keep the worker alive whatever happens
                log(f"ERROR during reconstruction: {e!r}")
                self.rec.pending = []
                self.status("error", str(e))
                return
            if changed:
                msg = self.rec.model_message(self.scan_id)
                self.send(msg)
                if self.scan_dir:
                    with open(os.path.join(self.scan_dir, "model.json"), "w") as f:
                        json.dump(msg, f)
            self.status("recording" if not flush else "processing")

    def _save_final(self):
        pts, cols = self.rec.all_points()
        write_ply(os.path.join(self.scan_dir, "points.ply"), pts, cols)


# --------------------------------------------------------------------------- #
# Live mode
# --------------------------------------------------------------------------- #

async def live(args):
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        sys.exit("Missing package. Run:  pip install websockets")

    loop = asyncio.get_running_loop()
    outbox = asyncio.Queue()

    def send(msg):  # callable from any thread
        loop.call_soon_threadsafe(outbox.put_nowait, msg)

    runner = VGGTRunner("cpu" if args.cpu else None)
    batch, overlap = batch_settings(args, runner)
    recon = ReconThread(runner, send, batch, overlap)
    recon.start()
    selector = KeyframeSelector()

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE   # our own self-signed certificate
    key = args.key or read_key()
    url = args.server.rstrip("/") + "/ws?role=worker" + (f"&key={key}" if key else "")

    recording = False
    kf_count = 0
    while True:
        try:
            async with connect(url, ssl=ssl_ctx, max_size=None, ping_interval=20) as ws:
                log(f"connected to {args.server}")
                recon.status()

                async def pump():
                    while True:
                        await ws.send(json.dumps(await outbox.get()))
                pump_task = asyncio.create_task(pump())
                try:
                    async for msg in ws:
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            if data.get("type") != "status":
                                continue
                            if data.get("recording") and not recording:
                                recording, kf_count = True, 0
                                selector.reset()
                                recon.q.put(("start", time.strftime("%Y%m%d-%H%M%S")))
                            elif not data.get("recording") and recording:
                                recording = False
                                recon.q.put(("stop",))
                            continue
                        if not recording:
                            continue
                        rgb = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
                        if rgb is None:
                            continue
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                        kf = selector.offer(msg, rgb)
                        if kf:
                            kf_count += 1
                            recon.q.put(("keyframe", kf_count, kf[0], kf[1]))
                            send({"type": "keyframe", "n": kf_count})
                finally:
                    pump_task.cancel()
        except (OSError, Exception) as e:
            log(f"server connection lost ({e.__class__.__name__}: {e}); retrying in 2s. Is the Node server running?")
            if recording:
                recording = False
                recon.q.put(("stop",))
            await asyncio.sleep(2)


def batch_settings(args, runner):
    batch, overlap = default_batch(runner.vram_gb)
    if args.batch:
        batch = max(3, args.batch)
        overlap = min(overlap, batch - 1)
    log(f"batches of {batch} keyframes, {overlap} shared with the previous batch")
    return batch, overlap


def read_key():
    path = os.path.join(HERE, "..", "certs", "access-key.txt")
    for _ in range(30):  # the server writes it on first start; give it a moment
        if os.path.exists(path):
            with open(path) as f:
                return f.read().strip()
        time.sleep(1)
    log("no access key file found; connecting without a key")
    return None


# --------------------------------------------------------------------------- #
# Offline mode
# --------------------------------------------------------------------------- #

def offline(args):
    paths = sorted(p for p in glob.glob(os.path.join(args.images, "*")) if p.lower().endswith(IMAGE_EXTS))
    if len(paths) < 2:
        sys.exit(f"Need at least 2 images in {args.images}")
    log(f"{len(paths)} images from {args.images}")
    runner = VGGTRunner("cpu" if args.cpu else None)
    batch, overlap = batch_settings(args, runner)
    rec = Reconstructor(runner, batch_size=batch, overlap=overlap)
    for i, p in enumerate(paths):
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        if w > 1280:  # match the phone stream size, keeps memory down
            bgr = cv2.resize(bgr, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)
        rec.add_keyframe(i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        while rec.step():
            pass
    while rec.step(flush=True):
        pass
    if not rec.order:
        sys.exit("Reconstruction failed.")

    msg = rec.model_message("offline", max_points=80000)
    out = os.path.join(HERE, "output")
    os.makedirs(out, exist_ok=True)
    pts, cols = rec.all_points()
    write_ply(os.path.join(out, "model_points.ply"), pts, cols)
    with open(os.path.join(out, "model.json"), "w") as f:
        json.dump(msg, f)
    write_preview(os.path.join(out, "model_preview.html"), msg)
    log(f"done: {len(rec.order)} keyframes in {rec.batches} batches ({rec.rejected} skipped)")
    log(f"open {os.path.join(out, 'model_preview.html')}")


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #

def write_ply(path, pts, cols):
    cols8 = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
    v = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    v["red"], v["green"], v["blue"] = cols8[:, 0], cols8[:, 1], cols8[:, 2]
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(v.tobytes())


def write_preview(path, msg):
    """Standalone page using the same viewer code as the dashboard (three.js from CDN)."""
    with open(os.path.join(HERE, "..", "public", "model-viewer.js"), encoding="utf-8") as f:
        viewer = f.read().replace("export function", "function")
    html = PREVIEW.replace("__VIEWER__", viewer).replace("__DATA__", json.dumps(msg))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


PREVIEW = """<!doctype html>
<html><head><meta charset="utf-8"><title>buoay model preview</title>
<style>html,body{margin:0;height:100%;background:#0b0e12;color:#e6e9ee;font-family:system-ui,sans-serif;overflow:hidden}
#v{position:fixed;inset:0}#hud{position:fixed;top:12px;left:14px;font-size:13px;opacity:.85}
button{background:#181c22;color:#e6e9ee;border:1px solid #2a3039;border-radius:6px;padding:4px 10px;margin-right:4px;cursor:pointer}</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script></head>
<body><div id="v"></div><div id="hud">
<button data-t="grid">Grid</button><button data-t="points">Points</button><button data-t="cams">Camera path</button>
<button id="reset">Reset view</button> &nbsp; drag to rotate, right-drag to pan, scroll to zoom</div>
<script type="module">
__VIEWER__
const viewer = createModelViewer(document.getElementById('v'));
viewer.update(__DATA__);
document.querySelectorAll('[data-t]').forEach(b => b.onclick = () => viewer.setVisible(b.dataset.t, !viewer.isVisible(b.dataset.t)));
document.getElementById('reset').onclick = () => viewer.resetView();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="wss://localhost:8443", help="Node server address")
    ap.add_argument("--images", help="offline mode: folder of photos to reconstruct")
    ap.add_argument("--batch", type=int, help="photos per VGGT batch (default: picked from GPU memory, 10 on an 8 GB card; lowered automatically if it runs out)")
    ap.add_argument("--key", help="access key (default: read from ../certs/access-key.txt, written by the server)")
    ap.add_argument("--cpu", action="store_true", help="run VGGT on the CPU")
    args = ap.parse_args()
    if args.images:
        offline(args)
    else:
        try:
            asyncio.run(live(args))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
