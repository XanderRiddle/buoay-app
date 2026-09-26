"""
Incremental 3D reconstruction for buoay-app.

Pipeline:
  keyframes -> VGGT in small overlapping batches -> stitch batches together
  -> fit a grid (heightfield) surface to all points -> send to the dashboard.

VGGT can only look at ~6 photos at once on a 4 GB GPU, so each new batch
repeats the last OVERLAP keyframes of the previous one. Those shared photos
exist in both batches, pixel for pixel, which gives thousands of matching 3D
points. From them we solve for the scale + rotation + shift that lines the
new batch up with everything built so far (the Umeyama method).
"""

import time

import cv2
import numpy as np
import torch
from PIL import Image

VGGT_WIDTH = 518   # VGGT's native input width
PATCH = 14         # image sides must be multiples of this
STORE_STRIDE = 2   # keep every 2nd pixel of each point map (saves memory)


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + str(msg), flush=True)


# --------------------------------------------------------------------------- #
# VGGT model
# --------------------------------------------------------------------------- #

def default_batch(vram_gb):
    """Photos per batch and overlap for the GPU size (weights take ~2.6 GB, ~0.17 GB per photo)."""
    if vram_gb is None:
        return 6, 2          # CPU: keep batches small so updates still come regularly
    if vram_gb >= 11:
        return 16, 4
    if vram_gb >= 7.5:
        return 10, 3         # e.g. RTX 3070 Ti 8 GB (leaves room for the desktop's own display)
    if vram_gb >= 5.5:
        return 8, 2
    return 6, 2              # 4 GB cards


class VGGTRunner:
    """Loads VGGT once and turns a list of RGB images into per-pixel 3D points."""

    def __init__(self, device=None):
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        self._pose_to_cam = pose_encoding_to_extri_intri
        self._unproject = unproject_depth_map_to_point_map

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        log(f"Loading VGGT-1B on {self.device.type.upper()} ...")
        t = time.time()
        model = VGGT.from_pretrained("facebook/VGGT-1B")
        model.track_head = None   # not needed
        model.point_head = None   # points come from depth + camera instead
        model.eval()

        self.vram_gb = None
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            self.vram_gb = props.total_memory / 1024**3
            # RTX 30xx and newer support bfloat16 (more stable); older cards use float16.
            dtype = torch.bfloat16 if props.major >= 8 else torch.float16
            log(f"GPU: {props.name}, {self.vram_gb:.1f} GB, using {str(dtype).split('.')[-1]}")
            # Big transformer in 16-bit (halves its memory), small heads in float32,
            # and the full-resolution depth head one photo at a time.
            model.aggregator.to(dtype)
            agg_forward = model.aggregator.forward

            def agg_forward_16(images):
                tokens, start = agg_forward(images.to(dtype))
                return [x.float() if x is not None else None for x in tokens], start

            model.aggregator.forward = agg_forward_16
            depth_forward = model.depth_head.forward
            model.depth_head.forward = lambda *a, **k: depth_forward(*a, **{**k, "frames_chunk_size": 1})

        self.model = model.to(self.device)
        log(f"VGGT ready in {time.time() - t:.0f}s")
        self.last_peak_gb = None

    @staticmethod
    def preprocess(rgb_list):
        """Same as VGGT's 'crop' mode: width 518, height rounded to 14, center-cropped to <=518."""
        h0, w0 = rgb_list[0].shape[:2]
        new_h = int(round(h0 * VGGT_WIDTH / w0 / PATCH) * PATCH)
        out = []
        for rgb in rgb_list:
            if rgb.shape[:2] != (h0, w0):
                rgb = cv2.resize(rgb, (w0, h0), interpolation=cv2.INTER_AREA)
            # PIL bicubic, exactly like VGGT's own loader (what the model was trained with)
            img = np.asarray(Image.fromarray(rgb).resize((VGGT_WIDTH, new_h), Image.Resampling.BICUBIC))
            if new_h > VGGT_WIDTH:
                top = (new_h - VGGT_WIDTH) // 2
                img = img[top:top + VGGT_WIDTH]
            out.append(img)
        arr = np.stack(out).astype(np.float32) / 255.0          # (S, H, W, 3)
        return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (S, 3, H, W)

    def predict(self, rgb_list):
        """Returns per-frame dicts: points (H,W,3), conf (H,W), color (H,W,3), cam_to_world (4,4)."""
        images = self.preprocess(rgb_list)
        H, W = images.shape[-2:]
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            pred = self.model(images.to(self.device))
        if self.device.type == "cuda":
            self.last_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

        extr, intr = self._pose_to_cam(pred["pose_enc"].float(), (H, W))
        extr = extr[0].cpu().numpy()                              # (S, 3, 4) world->cam
        depth = pred["depth"][0].float().cpu().numpy()            # (S, H, W, 1)
        conf = pred["depth_conf"][0].float().cpu().numpy()        # (S, H, W)
        pts = self._unproject(depth, extr, intr[0].cpu().numpy()) # (S, H, W, 3)
        colors = images.permute(0, 2, 3, 1).numpy()

        del pred
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        frames = []
        for i in range(len(rgb_list)):
            R, t = extr[i][:, :3], extr[i][:, 3]
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
            frames.append({"points": pts[i].astype(np.float32), "conf": conf[i].astype(np.float32),
                           "color": colors[i], "cam_to_world": c2w})
        return frames


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #

def umeyama(src, dst, weights=None):
    """Similarity transform (s, R, t) minimizing |s R src + t - dst|^2."""
    if weights is None:
        weights = np.ones(len(src))
    w = weights / weights.sum()
    mu_s, mu_d = w @ src, w @ dst
    xs, xd = src - mu_s, dst - mu_d
    cov = (xd * w[:, None]).T @ xs
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (w * (xs ** 2).sum(1)).sum()
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def robust_similarity(src, dst, weights, iters=4):
    """Umeyama with outlier trimming. Returns (s, R, t, median_residual, inlier_fraction)."""
    keep = np.ones(len(src), bool)
    for _ in range(iters):
        s, R, t = umeyama(src[keep], dst[keep], weights[keep])
        res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        med = np.median(res[keep])
        keep = res < max(3.0 * med, 1e-9)
    return s, R, t, float(np.median(res[keep])), float(keep.mean())


def apply_sim(sim, pts):
    s, R, t = sim
    return (s * (R @ pts.reshape(-1, 3).T)).T.reshape(pts.shape) + t


# --------------------------------------------------------------------------- #
# Grid surface
# --------------------------------------------------------------------------- #

def fit_grid(points, target_cells=48, min_pts=4):
    """
    Fits a heightfield grid to the points: find the dominant plane, bin points
    over it, and keep the median height per cell. Good for walls and hull sides.
    """
    if len(points) < 100:
        return None
    # dominant plane from the bulk of the points (ignore far-away junk)
    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    core = points[dist < 3 * np.median(dist)]
    center = core.mean(0)
    _, _, vt = np.linalg.svd(core - center, full_matrices=False)
    u_ax, v_ax, n_ax = vt[0], vt[1], vt[2]

    # trim outliers along the surface's own axes, not the raw x/y/z axes
    local = points - center
    u, v, h = local @ u_ax, local @ v_ax, local @ n_ax
    mad = np.median(np.abs(h - np.median(h))) + 1e-9
    keep = np.abs(h - np.median(h)) < 6 * mad
    u, v, h = u[keep], v[keep], h[keep]
    u0, u1 = np.percentile(u, [0.5, 99.5])
    v0, v1 = np.percentile(v, [0.5, 99.5])
    inside = (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
    u, v, h = u[inside], v[inside], h[inside]
    cell = max(u1 - u0, v1 - v0) / target_cells
    if cell <= 0:
        return None
    nu = int(np.ceil((u1 - u0) / cell)) + 1
    nv = int(np.ceil((v1 - v0) / cell)) + 1
    iu = np.clip(np.round((u - u0) / cell).astype(int), 0, nu - 1)
    iv = np.clip(np.round((v - v0) / cell).astype(int), 0, nv - 1)
    flat = iv * nu + iu

    # median height per cell
    order = np.argsort(flat, kind="stable")
    flat_s, h_s = flat[order], h[order]
    cells, starts, counts = np.unique(flat_s, return_index=True, return_counts=True)
    grid = np.full(nu * nv, np.nan, np.float32)
    for c, s0, n in zip(cells, starts, counts):
        if n >= min_pts:
            grid[c] = np.median(h_s[s0:s0 + n])
    grid = grid.reshape(nv, nu)

    grid = _clean_grid(grid)
    origin = center + u0 * u_ax + v0 * v_ax
    return {"origin": origin.tolist(), "u": u_ax.tolist(), "v": v_ax.tolist(), "n": n_ax.tolist(),
            "cell": float(cell), "nu": nu, "nv": nv,
            "h": [None if np.isnan(x) else round(float(x), 5) for x in grid.ravel()]}


def _neighbors(grid):
    p = np.pad(grid, 1, constant_values=np.nan)
    H, W = grid.shape
    return np.stack([p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])


def _clean_grid(grid):
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        # 1. drop spikes that disagree with their neighbors
        nb = _neighbors(grid)
        med = np.nanmedian(nb, axis=0)
        diff = np.abs(grid - med)
        mad = np.nanmedian(diff)
        grid = np.where(diff > 4 * max(mad, 1e-6), np.nan, grid)
        # 2. fill small holes (cells with >= 5 valid neighbors), twice
        for _ in range(2):
            nb = _neighbors(grid)
            count = np.sum(~np.isnan(nb), axis=0)
            fill = np.isnan(grid) & (count >= 5)
            grid = np.where(fill, np.nanmean(nb, axis=0), grid)
        # 3. light smoothing
        nb = _neighbors(grid)
        stacked = np.concatenate([grid[None], nb])
        smooth = np.nanmean(stacked, axis=0)
        grid = np.where(np.isnan(grid), np.nan, 0.5 * grid + 0.5 * smooth)
    return grid


# --------------------------------------------------------------------------- #
# Incremental reconstructor
# --------------------------------------------------------------------------- #

class Reconstructor:
    """
    Feed it keyframes with add_keyframe(); call step() to process a batch when
    enough are waiting. Keeps every keyframe's 3D points in one shared world frame
    (the first batch's frame).
    """

    def __init__(self, runner, batch_size=6, overlap=2, conf_percentile=50):
        self.runner = runner
        self.batch_size = batch_size
        self.overlap = overlap
        self.conf_percentile = conf_percentile
        self.reset()

    def reset(self):
        self.pending = []      # [(kf_id, rgb)] waiting to be processed
        self.frames = {}       # kf_id -> {points, conf, color, cam_to_world} in world frame
        self.order = []        # kf_ids in the order they joined the model
        self.rgb = {}          # kf_id -> rgb, kept only for the overlap frames
        self.batches = 0
        self.rejected = 0
        self.last_batch_seconds = None
        self.last_error = None

    # -- input ------------------------------------------------------------- #
    def add_keyframe(self, kf_id, rgb):
        self.pending.append((kf_id, rgb))

    def ready(self, flush=False):
        new_needed = self.batch_size if not self.order else self.batch_size - self.overlap
        if flush:
            return len(self.pending) >= (2 if not self.order else 1)
        return len(self.pending) >= new_needed

    # -- processing -------------------------------------------------------- #
    def step(self, flush=False):
        """Process one batch if ready. Returns True if the model changed."""
        if not self.ready(flush):
            return False
        first = not self.order
        n_new = self.batch_size if first else self.batch_size - self.overlap
        new = self.pending[:n_new]
        overlap_ids = [] if first else self.order[-self.overlap:]
        batch_ids = overlap_ids + [k for k, _ in new]
        batch_rgb = [self.rgb[k] for k in overlap_ids] + [rgb for _, rgb in new]

        t = time.time()
        try:
            out = self.runner.predict(batch_rgb)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if self.batch_size > 3:
                self.batch_size -= 1
                log(f"GPU out of memory -> batch size lowered to {self.batch_size}")
                return False
            raise
        self.last_batch_seconds = time.time() - t
        self.pending = self.pending[n_new:]

        if first:
            sim = (1.0, np.eye(3), np.zeros(3))
        else:
            sim = self._align(overlap_ids, out[:len(overlap_ids)])
            if sim is None:
                self.rejected += 1
                return False

        for kf_id, f, rgb in zip(batch_ids, out, batch_rgb):
            if kf_id in self.frames:
                continue  # overlap frame: keep the version already in the model
            s, R, tr = sim
            c2w = f["cam_to_world"].copy()
            c2w[:3, :3] = R @ c2w[:3, :3]
            c2w[:3, 3] = s * R @ c2w[:3, 3] + tr
            st = STORE_STRIDE
            self.frames[kf_id] = {
                "points": apply_sim(sim, f["points"][::st, ::st]).astype(np.float32),
                "conf": f["conf"][::st, ::st],
                "color": f["color"][::st, ::st],
                "cam_to_world": c2w,
            }
            self.order.append(kf_id)
            self.rgb[kf_id] = rgb

        # only the last OVERLAP images are needed for the next batch
        for k in list(self.rgb):
            if k not in self.order[-self.overlap:]:
                del self.rgb[k]
        self.batches += 1
        log(f"batch {self.batches}: {len(batch_ids)} frames in {self.last_batch_seconds:.1f}s"
            + (f", peak GPU {self.runner.last_peak_gb:.2f} GB" if self.runner.last_peak_gb else "")
            + f" | model has {len(self.order)} keyframes, {len(self.pending)} waiting")
        return True

    def _align(self, overlap_ids, overlap_out):
        src, dst, w = [], [], []
        st = STORE_STRIDE
        for kf_id, f in zip(overlap_ids, overlap_out):
            old = self.frames[kf_id]
            new_pts, new_conf = f["points"][::st, ::st], f["conf"][::st, ::st]
            good = ((new_conf >= np.percentile(new_conf, self.conf_percentile))
                    & (old["conf"] >= np.percentile(old["conf"], self.conf_percentile))
                    & np.isfinite(new_pts).all(-1) & np.isfinite(old["points"]).all(-1))
            src.append(new_pts[good])
            dst.append(old["points"][good])
            w.append(np.sqrt(new_conf[good] * old["conf"][good]))
        src, dst, w = np.concatenate(src), np.concatenate(dst), np.concatenate(w)
        if len(src) < 200:
            log("alignment failed: too few confident overlapping points, batch skipped")
            return None
        if len(src) > 60000:
            sel = np.random.default_rng(0).choice(len(src), 60000, replace=False)
            src, dst, w = src[sel], dst[sel], w[sel]
        s, R, t, med, inliers = robust_similarity(src, dst, w)
        scene = np.linalg.norm(np.percentile(dst, 95, axis=0) - np.percentile(dst, 5, axis=0))
        rel = med / max(scene, 1e-9)
        if rel > 0.05 or inliers < 0.3:
            log(f"alignment failed: error {100 * rel:.1f}% of scene, {100 * inliers:.0f}% inliers; batch skipped")
            return None
        return s, R, t

    # -- output ------------------------------------------------------------ #
    def all_points(self, max_points=None, conf_percentile=None):
        if not self.order:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
        cp = self.conf_percentile if conf_percentile is None else conf_percentile
        P, C = [], []
        for k in self.order:
            f = self.frames[k]
            good = (f["conf"] >= np.percentile(f["conf"], cp)) & np.isfinite(f["points"]).all(-1)
            P.append(f["points"][good])
            C.append(f["color"][good])
        P, C = np.concatenate(P), np.concatenate(C)
        if max_points and len(P) > max_points:
            sel = np.random.default_rng(0).choice(len(P), max_points, replace=False)
            P, C = P[sel], C[sel]
        return P, C

    def camera_positions(self):
        return np.array([self.frames[k]["cam_to_world"][:3, 3] for k in self.order])

    def model_message(self, scan_id, max_points=25000):
        pts, cols = self.all_points(max_points=max_points)
        grid = fit_grid(self.all_points(max_points=300000)[0])
        return {
            "type": "model",
            "scan": scan_id,
            "keyframes": len(self.order),
            "batches": self.batches,
            "grid": grid,
            "points": {"p": np.round(pts, 4).ravel().tolist(),
                       "c": np.round(cols, 3).ravel().tolist()},
            "cams": np.round(self.camera_positions(), 4).ravel().tolist(),
        }
