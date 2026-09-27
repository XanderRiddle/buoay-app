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

import base64
import collections
import time

import warnings

import cv2
import numpy as np
import torch

import anomaly

# VGGT's own code uses an old PyTorch call that prints a warning on every batch
warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.autocast.*", category=FutureWarning)
from PIL import Image

VGGT_WIDTH = 518   # VGGT's native input width
PATCH = 14         # image sides must be multiples of this
STORE_STRIDE = 2   # keep every 2nd pixel of each point map (saves memory)


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + str(msg), flush=True)


# --------------------------------------------------------------------------- #
# VGGT model
# --------------------------------------------------------------------------- #

def gpu_profile(vram_gb):
    """
    Settings for the GPU size. Weights take ~2.6 GB; each photo in a batch ~0.1-0.2 GB more.
      batch        max photos VGGT sees at once
      min_overlap  photos always repeated from the previous batch (for stitching)
      new          new keyframes that trigger the next batch (fewer = smoother, more frequent updates)
      first        keyframes needed before the very first model appears
      kf_shift     how far the view must move (fraction of frame width) before a new keyframe
      depth_chunk  photos the full-resolution depth step handles at once (1 saves memory on small cards)
    """
    if vram_gb is None:            # CPU: slow, so few and far between
        return dict(batch=6, min_overlap=2, new=4, first=4, kf_shift=0.18, depth_chunk=8)
    if vram_gb >= 11:              # 12 GB+ cards
        return dict(batch=16, min_overlap=4, new=3, first=4, kf_shift=0.10, depth_chunk=8)
    if vram_gb >= 7.5:             # RTX 3070 Ti 8 GB: 10 photos measured 2.1 s / 4.8 GB peak, so 14 fits
        return dict(batch=14, min_overlap=3, new=3, first=4, kf_shift=0.12, depth_chunk=4)
    if vram_gb >= 5.5:
        return dict(batch=8, min_overlap=2, new=4, first=4, kf_shift=0.15, depth_chunk=1)
    return dict(batch=6, min_overlap=2, new=4, first=6, kf_shift=0.18, depth_chunk=1)  # 4 GB cards


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
        self.depth_chunk = 1
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
            model.depth_head.forward = lambda *a, **k: depth_forward(*a, **{**k, "frames_chunk_size": self.depth_chunk})

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

        # Camera intrinsics in the ORIGINAL photo's pixels (undo the resize + crop), so the
        # full-resolution keyframe photos can be projected onto the model for the texture.
        h0, w0 = rgb_list[0].shape[:2]
        new_h = int(round(h0 * VGGT_WIDTH / w0 / PATCH) * PATCH)
        top = (new_h - VGGT_WIDTH) // 2 if new_h > VGGT_WIDTH else 0
        sx, sy = w0 / VGGT_WIDTH, h0 / new_h
        intr = intr[0].cpu().numpy()

        frames = []
        for i in range(len(rgb_list)):
            R, t = extr[i][:, :3], extr[i][:, 3]
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
            k = intr[i]
            K = np.array([[k[0, 0] * sx, 0, k[0, 2] * sx],
                          [0, k[1, 1] * sy, (k[1, 2] + top) * sy],
                          [0, 0, 1]])
            frames.append({"points": pts[i].astype(np.float32), "conf": conf[i].astype(np.float32),
                           "color": colors[i], "cam_to_world": c2w, "K": K})
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


try:
    from scipy.spatial import cKDTree
except ImportError:  # ICP refinement is skipped without scipy
    cKDTree = None


def _rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def icp_refine(src, tgt, anchor_src, anchor_dst, iters=10):
    """
    Loop-closure nudge for a new batch that revisits an area scanned earlier.
      src          new batch points (already placed by the overlap alignment)
      tgt          the OLDER surface it revisits (not the frames it was just chained to)
      anchor_*     exact pixel matches with the shared overlap photos (where it must stay attached)
    Solves for one small rigid correction that pulls src onto the old surface (point-to-plane)
    while the anchors resist, so the revisit meets the old layer halfway instead of doubling,
    without tearing away from its neighbors. Sliding along a flat wall is damped.
    Returns (R, t, before, after) with before/after = median gap to the old surface, or None.
    """
    tree = cKDTree(tgt)
    extent = np.linalg.norm(np.ptp(src, axis=0))
    normals = np.full(tgt.shape, np.nan, np.float64)
    R, t = np.eye(3), np.zeros(3)
    cur = src.astype(np.float64).copy()
    anc = anchor_src.astype(np.float64).copy()
    max_d = 0.05 * extent
    before = None
    for _ in range(iters):
        d, idx = tree.query(cur, k=1, distance_upper_bound=max_d)
        ok = np.isfinite(d)
        if ok.sum() < 500:
            return None
        med = np.median(d[ok])
        if before is None:
            before = med
        keep = ok & (d < max(3 * med, 1e-9))
        p, ti = cur[keep], idx[keep]
        need = np.unique(ti[np.isnan(normals[ti, 0])])
        if len(need):  # surface normals from the 12 nearest points
            _, nb = tree.query(tgt[need], k=12)
            nbp = tgt[nb] - tgt[nb].mean(1, keepdims=True)
            normals[need] = np.linalg.eigh(np.einsum("nki,nkj->nij", nbp, nbp))[1][:, :, 0]
        q, n = tgt[ti], normals[ti]
        r = np.einsum("ij,ij->i", p - q, n)
        w = 1 / np.maximum(1, np.abs(r) / (1.5 * np.median(np.abs(r)) + 1e-12))   # Huber
        A1 = np.hstack([np.cross(p, n), n]) * w[:, None]
        b1 = -r * w
        # anchors: point-to-point, x/y/z rows; total weight equal to the surface term
        wa = np.sqrt(w.sum() / max(len(anc), 1) / 3)
        e = anc - anchor_dst
        rows = []
        for axis in range(3):
            unit = np.zeros(3); unit[axis] = 1
            rows.append(np.hstack([np.cross(anc, unit), np.tile(unit, (len(anc), 1))]))
        A2 = np.vstack(rows) * wa
        b2 = -np.concatenate([e[:, 0], e[:, 1], e[:, 2]]) * wa
        A, b = np.vstack([A1, A2]), np.concatenate([b1, b2])
        H = A.T @ A
        H += np.eye(6) * 1e-3 * np.trace(H) / 6
        x = np.linalg.solve(H, A.T @ b)
        dR = _rodrigues(x[:3])
        cur = cur @ dR.T + x[3:]
        anc = anc @ dR.T + x[3:]
        R, t = dR @ R, dR @ t + x[3:]
        max_d = min(max_d, 5 * med)
        if np.linalg.norm(x[3:]) < 1e-5 * extent and np.linalg.norm(x[:3]) < 1e-5:
            break
    d, _ = tree.query(cur, k=1, distance_upper_bound=0.05 * extent)
    ok = np.isfinite(d)
    after = np.median(d[ok]) if ok.any() else np.inf
    return R, t, before, after


def apply_sim(sim, pts):
    s, R, t = sim
    return (s * (R @ pts.reshape(-1, 3).T)).T.reshape(pts.shape) + t


# --------------------------------------------------------------------------- #
# Grid surface
# --------------------------------------------------------------------------- #

def fit_grid(points, colors=None, views=None, cache=None, target_cells=48, min_pts=3):
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
    if colors is not None:
        colors = colors[keep]
    u0, u1 = np.percentile(u, [0.5, 99.5])
    v0, v1 = np.percentile(v, [0.5, 99.5])
    inside = (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
    u, v, h = u[inside], v[inside], h[inside]
    if colors is not None:
        colors = colors[inside]
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
    # averaged point-color texture: always built, it's what the surface check was tuned on
    tex, img, have = _surface_texture(u, v, colors, u0, v0, cell, nu, nv, grid) if colors is not None else (None, None, None)
    try:  # red / yellow highlights; never let them break the model update
        anomalies = anomaly.analyze(grid, cell, img, have, TEX_SUB)
    except Exception as e:
        log(f"surface check failed: {e!r}")
        anomalies = None
    # what the dashboard shows: the sharp texture painted from the full-resolution keyframe photos
    if views:
        try:
            sharp = _projected_texture(origin, u_ax, v_ax, n_ax, cell, nu, nv, grid, views, cache)
            if sharp is not None:
                tex = sharp
        except Exception as e:  # never let the texture break the model
            log(f"photo texture failed ({e!r}); using point colors")
    return {"origin": origin.tolist(), "u": u_ax.tolist(), "v": v_ax.tolist(), "n": n_ax.tolist(),
            "cell": float(cell), "nu": nu, "nv": nv,
            "h": [None if np.isnan(x) else round(float(x), 5) for x in grid.ravel()],
            "texture": tex, "anomalies": anomalies}


TEX_SUB = 4  # fallback texture (averaged point colors): pixels per grid cell


def _projected_texture(origin, u_ax, v_ax, n_ax, cell, nu, nv, grid, views, cache):
    """
    Sharp photo texture: every part of the surface is painted from the single best keyframe
    photo that saw it (full resolution, near the image center, facing the surface, close by),
    instead of averaging blurry point colors from many frames.
    views: [(kf_id, cam_to_world 4x4, K 3x3 in photo pixels, jpeg bytes, (h, w))]
    """
    if nu < 2 or nv < 2:
        return None
    T = int(max(4, min(40, 2048 // max(nu - 1, nv - 1))))   # texture pixels per grid cell (long side <= 2048)
    W, H = (nu - 1) * T + 1, (nv - 1) * T + 1
    valid = ~np.isnan(grid)

    # 3D position of every texture pixel: bilinear heights between grid nodes
    X, Y = np.meshgrid(np.arange(W) / T, np.arange(H) / T)
    i0 = np.clip(np.floor(X).astype(int), 0, nu - 2)
    j0 = np.clip(np.floor(Y).astype(int), 0, nv - 2)
    fx, fy = X - i0, Y - j0
    acc = np.zeros_like(X)
    wsum = np.zeros_like(X)
    for di, dj, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        hv = grid[j0 + dj, i0 + di]
        ok = ~np.isnan(hv)
        acc += np.where(ok, hv * w, 0)
        wsum += np.where(ok, w, 0)
    node_i = np.clip(np.round(X).astype(int), 0, nu - 1)
    node_j = np.clip(np.round(Y).astype(int), 0, nv - 1)
    inside = valid[node_j, node_i] & (wsum > 1e-6)
    h = np.where(inside, acc / np.maximum(wsum, 1e-9), 0)
    P = (origin[None, None] + (X * cell)[..., None] * u_ax + (Y * cell)[..., None] * v_ax
         + h[..., None] * n_ax).astype(np.float32)

    # best photo for each grid node
    nodes = np.argwhere(valid)                                   # (N, 2) as (j, i)
    Pn = origin + (nodes[:, 1:2] * cell) * u_ax + (nodes[:, 0:1] * cell) * v_ax + grid[valid][:, None] * n_ax
    scores = np.full((len(views), len(nodes)), np.inf)
    for vi, (_, c2w, K, _, (ih, iw)) in enumerate(views):
        Rw, C = c2w[:3, :3], c2w[:3, 3]
        pc = (Pn - C) @ Rw                                       # world -> camera
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            px = K[0, 0] * pc[:, 0] / z + K[0, 2]
            py = K[1, 1] * pc[:, 1] / z + K[1, 2]
        ok = (z > 0) & (px > 0.03 * iw) & (px < 0.97 * iw) & (py > 0.03 * ih) & (py < 0.97 * ih)
        off_center = np.hypot((px - iw / 2) / iw, (py - ih / 2) / ih)
        ray = (Pn - C) / np.maximum(np.linalg.norm(Pn - C, axis=1, keepdims=True), 1e-9)
        facing = np.abs(ray @ n_ax)
        dist = np.linalg.norm(Pn - C, axis=1)
        scores[vi] = np.where(ok, off_center + 0.4 * (1 - facing) + 0.1 * dist / max(np.median(dist), 1e-9), np.inf)
    best_node = np.argmin(scores, axis=0)
    best_node[~np.isfinite(scores.min(axis=0))] = -1
    node_view = np.full((nv, nu), -1)
    node_view[valid] = best_node
    texel_view = np.where(inside, node_view[node_j, node_i], -1)

    # sample each photo for the texels it won
    img = np.zeros((H, W, 3), np.uint8)
    got = np.zeros((H, W), bool)
    flat_view = texel_view.ravel()
    order = np.argsort(flat_view, kind="stable")
    sorted_view = flat_view[order]
    starts = np.searchsorted(sorted_view, np.arange(len(views) + 1))
    Pflat = P.reshape(-1, 3)
    for vi in range(len(views)):
        idx_all = order[starts[vi]:starts[vi + 1]]
        if len(idx_all) == 0:
            continue
        kf_id, c2w, K, jpeg, (ih, iw) = views[vi]
        photo = cache.get(kf_id) if cache is not None else None
        if photo is None:
            photo = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if cache is not None:
                cache.put(kf_id, photo)
        pc = (Pflat[idx_all] - c2w[:3, 3]) @ c2w[:3, :3]
        mx = (K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(np.float32)
        my = (K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(np.float32)
        sample = _remap_points(photo, mx, my)
        okp = (mx >= 0) & (mx <= iw - 1) & (my >= 0) & (my <= ih - 1)
        idx = idx_all[okp]
        img.reshape(-1, 3)[idx] = sample[okp]
        got.reshape(-1)[idx] = True

    holes = (inside & ~got).astype(np.uint8)
    if holes.any() and got.any():
        img = cv2.inpaint(img, holes, 3, cv2.INPAINT_TELEA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])   # img is already BGR
    return {"w": W, "h": H, "sub": T,
            "png": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}


def _remap_points(photo, mx, my, width=1024):
    """Sample a photo at a list of pixel positions (cv2.remap needs 2D maps under 32767 per side)."""
    n = len(mx)
    rows = -(-n // width)
    pad = rows * width - n
    mxp = np.concatenate([mx, np.full(pad, -1, np.float32)]).reshape(rows, width)
    myp = np.concatenate([my, np.full(pad, -1, np.float32)]).reshape(rows, width)
    out = cv2.remap(photo, mxp, myp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return out.reshape(-1, 3)[:n]


class PhotoCache:
    """Recently decoded keyframe photos, so rebuilding the texture doesn't re-decode everything."""

    def __init__(self, size=80):
        self.size, self.items = size, collections.OrderedDict()

    def get(self, k):
        if k in self.items:
            self.items.move_to_end(k)
            return self.items[k]
        return None

    def put(self, k, v):
        self.items[k] = v
        self.items.move_to_end(k)
        while len(self.items) > self.size:
            self.items.popitem(last=False)


def _surface_texture(u, v, colors, u0, v0, cell, nu, nv, grid):
    """
    Photo texture for the grid: average camera color on a raster TEX_SUB times finer than
    the grid, gaps filled by inpainting. Returns (message dict with a JPEG data URL,
    the RGB image, mask of texels the camera actually saw).
    """
    W, H = (nu - 1) * TEX_SUB + 1, (nv - 1) * TEX_SUB + 1
    x = np.clip(np.round((u - u0) / cell * TEX_SUB).astype(int), 0, W - 1)
    y = np.clip(np.round((v - v0) / cell * TEX_SUB).astype(int), 0, H - 1)
    flat = y * W + x
    counts = np.bincount(flat, minlength=W * H)
    img = np.zeros((W * H, 3), np.float64)
    for c in range(3):
        img[:, c] = np.bincount(flat, weights=colors[:, c], minlength=W * H)
    have = counts > 0
    img[have] /= counts[have, None]
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8).reshape(H, W, 3)
    have = have.reshape(H, W)
    holes = (~have).astype(np.uint8)
    if holes.any() and have.any():
        img = cv2.inpaint(img, holes, 3, cv2.INPAINT_TELEA)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return ({"w": W, "h": H, "sub": TEX_SUB,
             "png": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}, img, have)


def pack_points(pts, cols):
    """Compact point cloud for the dashboard: positions as 16-bit steps inside the bounding box, colors as bytes."""
    if len(pts) == 0:
        return {"n": 0}
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-9)
    q = np.round((pts - lo) / span * 65535).astype("<u2")
    c = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
    return {"n": len(pts), "min": lo.tolist(), "max": hi.tolist(),
            "q": base64.b64encode(q.tobytes()).decode(), "c8": base64.b64encode(c.tobytes()).decode()}


def _neighbors(grid):
    p = np.pad(grid, 1, constant_values=np.nan)
    H, W = grid.shape
    return np.stack([p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])


def _clean_grid(grid):
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        # 1. drop spikes that disagree with their neighbors. Only lone cells (1-2 in a row) and
        #    cells on the ragged edge of the scan: a group of odd cells is a real dent or bump,
        #    which the damage check needs to see, so it stays.
        nb = _neighbors(grid)
        med = np.nanmedian(nb, axis=0)
        diff = np.abs(grid - med)
        mad = np.nanmedian(diff)
        odd = (diff > 4 * max(mad, 1e-6)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(odd, connectivity=8)
        lone = (stats[labels, cv2.CC_STAT_AREA] <= 2) & (odd > 0)
        edge = np.isfinite(grid) & (np.isnan(nb).sum(0) > 0)
        grid = np.where(lone | (edge & (odd > 0)), np.nan, grid)
        # 2. fill holes from the edges inward (cells with >= 4 of 8 neighbors), up to 4 cells deep;
        #    only fills gaps surrounded by surface, never grows the outline much
        for _ in range(4):
            nb = _neighbors(grid)
            count = np.sum(~np.isnan(nb), axis=0)
            fill = np.isnan(grid) & (count >= 4)
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

    def __init__(self, runner, profile=None, conf_percentile=50, model_conf_percentile=30):
        self.runner = runner
        p = profile or gpu_profile(getattr(runner, "vram_gb", None))
        self.batch_size = p["batch"]
        self.min_overlap = p["min_overlap"]
        self.new_per_batch = p["new"]
        self.first_min = p["first"]
        if hasattr(runner, "depth_chunk"):
            runner.depth_chunk = p["depth_chunk"]
        self.conf_percentile = conf_percentile          # for lining batches up: only the surest points
        self.model_conf_percentile = model_conf_percentile  # for the model: keep more, fewer holes
        self.photos = PhotoCache()
        self.reset()

    def reset(self):
        self.pending = []      # [(kf_id, rgb)] waiting to be processed
        self.frames = {}       # kf_id -> {points, conf, color, cam_to_world, good_pts, good_cols} in world frame
        self.order = []        # kf_ids in the order they joined the model
        self.rgb = {}          # kf_id -> rgb, kept only for recent frames (possible overlap)
        self.batches = 0
        self.rejected = 0
        self.last_batch_seconds = None
        self.last_error = None
        self.photos = PhotoCache()
        self._anchor = None

    # -- input ------------------------------------------------------------- #
    def add_keyframe(self, kf_id, rgb):
        self.pending.append((kf_id, rgb))

    def ready(self, flush=False):
        if not self.order:
            return len(self.pending) >= (2 if flush else self.first_min)
        return len(self.pending) >= (1 if flush else self.new_per_batch)

    # -- processing -------------------------------------------------------- #
    def step(self, flush=False):
        """Process one batch if ready. Returns True if the model changed."""
        if not self.ready(flush):
            return False
        first = not self.order
        if first:
            overlap_ids = []
            n_new = min(len(self.pending), self.batch_size)
        else:
            # Fill the batch: when caught up, reuse lots of recent keyframes (better stitching,
            # the GPU has time to spare); when behind, take more new ones to catch up.
            n_overlap = min(len(self.order), max(self.min_overlap, self.batch_size - len(self.pending)))
            n_new = min(len(self.pending), self.batch_size - n_overlap)
            overlap_ids = self.order[-n_overlap:]
        new = self.pending[:n_new]
        batch_ids = overlap_ids + [k for k, _ in new]
        batch_rgb = [self.rgb[k] for k in overlap_ids] + [rgb for _, rgb in new]

        t = time.time()
        try:
            out = self.runner.predict(batch_rgb)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if getattr(self.runner, "depth_chunk", 1) > 1:
                self.runner.depth_chunk = max(1, self.runner.depth_chunk // 2)
                log(f"GPU out of memory -> depth step now {self.runner.depth_chunk} photo(s) at a time")
                return False
            if self.batch_size > self.min_overlap + 2:
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
            sim = self._refine(sim, out, overlap_ids)

        for kf_id, f, rgb in zip(batch_ids, out, batch_rgb):
            if kf_id in self.frames:
                continue  # overlap frame: keep the version already in the model
            s, R, tr = sim
            c2w = f["cam_to_world"].copy()
            c2w[:3, :3] = R @ c2w[:3, :3]
            c2w[:3, 3] = s * R @ c2w[:3, 3] + tr
            st = STORE_STRIDE
            pts = apply_sim(sim, f["points"][::st, ::st]).astype(np.float32)
            conf, color = f["conf"][::st, ::st], f["color"][::st, ::st]
            good = (conf >= np.percentile(conf, self.model_conf_percentile)) & np.isfinite(pts).all(-1)
            photo = None
            if f.get("K") is not None:
                ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                photo = enc.tobytes() if ok else None
            self.frames[kf_id] = {
                "points": pts, "conf": conf, "color": color, "cam_to_world": c2w,
                # confident points cached once, so rebuilding the grid stays quick as the model grows
                "good_pts": pts[good], "good_cols": color[good].astype(np.float32),
                # full-resolution photo + camera, for the sharp photo texture
                "K": f.get("K"), "photo": photo, "size": rgb.shape[:2],
            }
            self.order.append(kf_id)
            self.rgb[kf_id] = rgb

        # only recent keyframes can be reused as overlap in the next batch: drop the full-size
        # data of older ones (keeps long scans from eating RAM; their good points stay)
        keep = set(self.order[-(self.batch_size - 1):])
        for k in list(self.rgb):
            if k not in keep:
                del self.rgb[k]
                f = self.frames[k]
                for key in ("points", "conf", "color"):
                    f.pop(key, None)
        self.batches += 1
        log(f"batch {self.batches}: {len(batch_ids)} frames in {self.last_batch_seconds:.1f}s"
            + (f", peak GPU {self.runner.last_peak_gb:.2f} GB" if self.runner.last_peak_gb else "")
            + f" | model has {len(self.order)} keyframes, {len(self.pending)} waiting")
        return True

    def _refine(self, sim, out, overlap_ids):
        """If this batch revisits an older part of the model, pull it onto that surface (see icp_refine)."""
        if cKDTree is None:
            if not getattr(self, "_warned_scipy", False):
                log("scipy not installed: skipping revisit alignment (run start.bat to install it)")
                self._warned_scipy = True
            return sim
        # "older" = keyframes before the recent chain this batch was attached to
        recent = set(self.order[-self.batch_size:])
        old_ids = [k for k in self.order if k not in recent]
        if not old_ids or self._anchor is None:
            return sim
        rng = np.random.default_rng(self.batches)
        src = []
        for f in out[len(overlap_ids):]:   # only the NEW frames can reveal a double layer
            c = f["conf"]
            good = (c >= np.percentile(c, self.conf_percentile)) & np.isfinite(f["points"]).all(-1)
            src.append(f["points"][good])
        if not src:
            return sim
        src = apply_sim(sim, np.concatenate(src))
        if len(src) > 15000:
            src = src[rng.integers(0, len(src), 15000)]
        lo, hi = src.min(0), src.max(0)
        pad = 0.1 * (hi - lo)
        tgt = np.concatenate([self.frames[k]["good_pts"] for k in old_ids])
        tgt = tgt[np.all((tgt > lo - pad) & (tgt < hi + pad), axis=1)]
        if len(tgt) < 2000:
            return sim   # not a revisit
        if len(tgt) > 150000:
            tgt = tgt[rng.integers(0, len(tgt), 150000)]
        a_src, a_dst = self._anchor
        a_src = apply_sim(sim, a_src)
        t0 = time.time()
        res = icp_refine(src, tgt, a_src, a_dst)
        if res is None:
            return sim
        R, t, before, after = res
        extent = np.linalg.norm(hi - lo)
        angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        if not after < 0.9 * before or np.linalg.norm(t) > 0.05 * extent or angle > 5:
            return sim
        log(f"revisit: pulled onto earlier scan, gap {before / extent:.2%} -> {after / extent:.2%} ({time.time() - t0:.2f}s)")
        s, Rs, ts = sim
        return s, R @ Rs, R @ ts + t

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
        sel = np.random.default_rng(1).integers(0, len(src), min(len(src), 4000))
        self._anchor = (src[sel], dst[sel])
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
        cp = self.model_conf_percentile if conf_percentile is None else conf_percentile
        P, C = [], []
        for k in self.order:
            f = self.frames[k]
            if cp == self.model_conf_percentile or "conf" not in f:
                P.append(f["good_pts"]); C.append(f["good_cols"])
            else:
                good = (f["conf"] >= np.percentile(f["conf"], cp)) & np.isfinite(f["points"]).all(-1)
                P.append(f["points"][good]); C.append(f["color"][good])
        P, C = np.concatenate(P), np.concatenate(C)
        if max_points and len(P) > max_points:
            sel = np.random.default_rng(0).integers(0, len(P), max_points)  # fast sampling
            P, C = P[sel], C[sel]
        return P, C

    def views(self):
        return [(k, f["cam_to_world"], f["K"], f["photo"], f["size"])
                for k in self.order for f in (self.frames[k],) if f.get("K") is not None and f.get("photo")]

    def camera_positions(self):
        return np.array([self.frames[k]["cam_to_world"][:3, 3] for k in self.order])

    def model_message(self, scan_id, max_points=25000):
        pts, cols = self.all_points(max_points=max_points)
        grid = fit_grid(*self.all_points(max_points=300000), views=self.views(), cache=self.photos)
        return {
            "type": "model",
            "scan": scan_id,
            "keyframes": len(self.order),
            "batches": self.batches,
            "grid": grid,
            "points": pack_points(pts, cols),
            "cams": np.round(self.camera_positions(), 4).ravel().tolist(),
        }
