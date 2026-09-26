"""
VGGT smoke test for a low-VRAM laptop (GTX 1650, 4 GB).

Takes a handful of photos, reconstructs a 3D point cloud with VGGT, and reports
how much GPU memory and time it took. Writes:
  output/points.ply      colored point cloud (open in MeshLab, Blender, or 3dviewer.net)
  output/preview.html    spin-around 3D preview in your browser (needs internet for three.js)

Usage:
  python vggt_smoke_test.py                       # uses ./test_images, first 6 photos, GPU
  python vggt_smoke_test.py --frames 4            # fewer photos if you run out of memory
  python vggt_smoke_test.py --images path\\to\\pics
  python vggt_smoke_test.py --cpu                 # skip the GPU entirely (slow)

How it fits in 4 GB: the big transformer (~0.9B params) runs in float16, the small
output heads stay in float32, and the point-map and tracking heads are dropped
(points come from depth + camera, which Meta says is the more accurate path).
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
except ImportError:
    sys.exit("VGGT is not installed. Run:  pip install git+https://github.com/facebookresearch/vggt")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


LOG_LINES = []


def log(msg):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


def gb(n_bytes):
    return f"{n_bytes / 1024**3:.2f} GB"


def find_images(folder, n):
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMAGE_EXTS))
    if len(paths) < 2:
        sys.exit(f"Need at least 2 images in '{folder}', found {len(paths)}.")
    if len(paths) > n:
        # spread the picks across the whole set instead of taking the first n
        idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
        paths = [paths[i] for i in idx]
    return paths


def load_model(device):
    log("Loading VGGT-1B weights (first run downloads ~5 GB from Hugging Face)...")
    t = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B")  # loads on CPU in float32
    model.track_head = None  # only used for point tracking, not needed
    model.point_head = None  # we build points from depth + camera instead
    model.eval()

    if device.type == "cuda":
        # Big transformer -> float16 (halves its memory). Heads stay float32 for accuracy.
        model.aggregator.half()
        agg_forward = model.aggregator.forward

        def agg_forward_fp16(images):
            tokens, patch_start = agg_forward(images.half())
            return [t.float() if t is not None else None for t in tokens], patch_start

        model.aggregator.forward = agg_forward_fp16

        # The depth head works at full image resolution in float32, which was the biggest
        # memory spike (~1.1 GB extra for 6 photos). Running it one photo at a time fixes that.
        depth_forward = model.depth_head.forward
        model.depth_head.forward = lambda *a, **k: depth_forward(*a, **{**k, "frames_chunk_size": 1})

    model.to(device)
    log(f"  loaded in {time.time() - t:.1f}s")
    if device.type == "cuda":
        log(f"  GPU memory used by weights: {gb(torch.cuda.memory_allocated())}")
    return model


def run(model, images, device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode():
        pred = model(images.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    return pred, elapsed, peak


def to_points(pred, images, conf_percentile):
    H, W = images.shape[-2:]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"].float(), (H, W))
    depth = pred["depth"][0].float().cpu().numpy()          # (S, H, W, 1)
    conf = pred["depth_conf"][0].float().cpu().numpy()      # (S, H, W)
    pts = unproject_depth_map_to_point_map(depth, extrinsic[0], intrinsic[0])  # (S, H, W, 3)
    cols = images.float().permute(0, 2, 3, 1).cpu().numpy()          # (S, H, W, 3) in 0..1

    pts, cols, conf = pts.reshape(-1, 3), cols.reshape(-1, 3), conf.reshape(-1)
    finite = np.isfinite(pts).all(1) & np.isfinite(conf)
    threshold = np.percentile(conf[finite], conf_percentile) if finite.any() else 0
    keep = finite & (conf >= threshold)
    cams = extrinsic[0].float().cpu().numpy()
    return pts[keep], cols[keep], cams, int(finite.sum()), int(pts.shape[0])


def camera_centers(extrinsics):
    # extrinsic is world->camera [R|t]; camera center = -R^T t
    return np.stack([-e[:, :3].T @ e[:, 3] for e in extrinsics])


def plane_fit(pts):
    """Fit a plane to the points. For a flat wall, the residual should be small vs the size."""
    c = pts.mean(0)
    _, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    normal = vt[2]
    resid = np.abs((pts - c) @ normal)
    extent = np.ptp((pts - c) @ vt[:2].T, axis=0)
    return float(np.sqrt((resid**2).mean())), float(extent.max())


def write_ply(path, pts, cols):
    cols8 = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
    vertex = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = cols8[:, 0], cols8[:, 1], cols8[:, 2]
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(vertex.tobytes())


def write_preview(path, pts, cols, cams, max_points=80_000):
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[sel], cols[sel]
    center = pts.mean(0)
    p = np.round(pts - center, 4).astype(np.float32)
    c = np.round(np.clip(cols, 0, 1), 3).astype(np.float32)
    cam = np.round(cams - center, 4).astype(np.float32)
    data = json.dumps({"p": p.ravel().tolist(), "c": c.ravel().tolist(), "cam": cam.ravel().tolist()})
    html = PREVIEW_TEMPLATE.replace("__DATA__", data)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


PREVIEW_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>VGGT smoke test</title>
<style>html,body{margin:0;height:100%;background:#0f1216;color:#e6e9ee;font-family:system-ui,sans-serif;overflow:hidden}
#hud{position:fixed;top:12px;left:14px;font-size:13px;opacity:.8}</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script></head>
<body><div id="hud">Drag to rotate · right-drag to pan · scroll to zoom · red dots = camera positions</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const D = __DATA__;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio); renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0f1216);
const g = new THREE.BufferGeometry();
g.setAttribute('position', new THREE.Float32BufferAttribute(D.p, 3));
g.setAttribute('color', new THREE.Float32BufferAttribute(D.c, 3));
g.computeBoundingSphere();
const r = g.boundingSphere.radius || 1;
scene.add(new THREE.Points(g, new THREE.PointsMaterial({ size: r / 400, vertexColors: true })));
const cg = new THREE.BufferGeometry(); cg.setAttribute('position', new THREE.Float32BufferAttribute(D.cam, 3));
scene.add(new THREE.Points(cg, new THREE.PointsMaterial({ size: r / 40, color: 0xff5449 })));
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, r / 1000, r * 100);
camera.up.set(0, -1, 0);  // VGGT uses OpenCV axes (y points down)
camera.position.set(0, 0, -r * 2.2);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
(function loop() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); })();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="test_images", help="folder of photos (default: test_images)")
    ap.add_argument("--frames", type=int, default=6, help="how many photos to use (default: 6)")
    ap.add_argument("--conf", type=float, default=50, help="drop the least-confident N%% of points (default: 50)")
    ap.add_argument("--cpu", action="store_true", help="run on CPU instead of GPU")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    log("=" * 60)
    log(f"PyTorch {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}  |  {gb(props.total_memory)}  |  compute capability {props.major}.{props.minor}")
    log("=" * 60)

    use_gpu = torch.cuda.is_available() and not args.cpu
    if not use_gpu and not args.cpu:
        log("WARNING: no CUDA GPU visible to PyTorch. You probably installed the CPU-only build of torch.")
        log("         See the README for the CUDA install command. Continuing on CPU (slow).")
    device = torch.device("cuda" if use_gpu else "cpu")

    paths = find_images(args.images, args.frames)
    log(f"Using {len(paths)} images from '{args.images}':")
    for p in paths:
        log(f"  - {os.path.basename(p)}")
    images = load_and_preprocess_images(paths)  # (S, 3, H, W), width 518
    log(f"Model input size: {images.shape[-1]}x{images.shape[-2]} per image")
    if images.shape[-2] > images.shape[-1] * 0.9:
        log("  (tip: landscape photos come out 518x392 and use ~25% less memory than portrait)")

    model = load_model(device)

    log("\nReconstructing...")
    try:
        pred, elapsed, peak = run(model, images, device)
    except torch.cuda.OutOfMemoryError:
        log("\nOUT OF GPU MEMORY.")
        log(f"  Try fewer photos:  python {os.path.basename(__file__)} --frames {max(2, len(paths) - 2)}")
        log("  Close other apps using the GPU (browsers, games), or run with --cpu.")
        sys.exit(1)

    pts, cols, cams, n_finite, n_total = to_points(pred, images, args.conf)
    if n_finite < n_total * 0.9:
        log(f"WARNING: {n_total - n_finite} of {n_total} points are NaN/inf. float16 may be overflowing;"
            " compare against a --cpu run.")
    if len(pts) == 0:
        sys.exit("No valid points produced. Try different photos or --cpu.")

    centers = camera_centers(cams)
    rms, extent = plane_fit(pts)

    os.makedirs(args.out, exist_ok=True)
    write_ply(os.path.join(args.out, "points.ply"), pts, cols)
    write_preview(os.path.join(args.out, "preview.html"), pts, cols, centers)

    log("\n" + "=" * 60)
    log("RESULTS")
    log(f"  Device:            {device.type.upper()}")
    log(f"  Photos:            {len(paths)}")
    log(f"  Inference time:    {elapsed:.1f}s")
    if peak is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        log(f"  Peak GPU memory:   {gb(peak)} of {gb(total)}  ({100 * peak / total:.0f}%)")
        if peak > total * 0.95:
            log("                     Over the limit: Windows spilled into system RAM, which makes it much slower.")
            log("                     Use fewer photos (--frames 4) or take them in landscape.")
    log(f"  Points kept:       {len(pts):,}  (top {100 - args.conf:.0f}% by confidence)")
    log(f"  Flatness check:    RMS distance from best-fit plane = {100 * rms / extent:.1f}% of scene width")
    log("                     (a flat wall should be a few % or less)")
    log(f"  Saved:             {os.path.join(args.out, 'points.ply')}")
    log(f"                     {os.path.join(args.out, 'preview.html')}  <- open this in a browser")
    log(f"                     {os.path.join(args.out, 'results.txt')}   <- this summary")
    log("=" * 60)
    with open(os.path.join(args.out, "results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES) + "\n")


if __name__ == "__main__":
    main()
