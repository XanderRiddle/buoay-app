"""
Surface checks for buoay-app: finds spots on the scanned surface that don't match their surroundings.

  bumps / dents (shown red):  the heightfield grid minus a smooth version of itself. VGGT reconstructs
                              flat and gently curved surfaces very evenly, so anything that sticks out of
                              (or sinks into) the local surface stands out clearly.
  color changes (shown yellow): the photo texture compared with the typical color around it
                              (rust, growth, paint loss, stains).

Both return scores, not yes/no answers: roughly "how many times bigger than the normal noise level".
The dashboard picks the cut-off (sensitivity slider), so the threshold can be tuned live.
Scores are sent as bytes: byte = score * SCORE_SCALE (so 255 means a score of 12.75 or more).
"""

import base64

import cv2
import numpy as np

SCORE_SCALE = 20.0

# Bumps
BUMP_SMOOTH = 0.12       # size of the "local surface" blur, as a fraction of the grid's long side
BUMP_MIN_CELLS = 2.0     # never blur less than this many cells
BUMP_FLOOR = 0.001       # smallest noise level assumed, as a fraction of the scan's size
                         # (keeps a near-perfect flat wall from flagging tiny ripples)
# Color changes
COLOR_SMOOTH = 0.12      # size of the "typical color around here" blur, fraction of the texture's long side
COLOR_FLOOR = 2.5        # smallest noise level assumed, in Lab color units
LIGHTNESS_WEIGHT = 0.5   # brightness counts half as much as hue: lighting and glare change brightness
                         # far more than they change color


def _blur_masked(values, weights, sigma):
    """Gaussian blur that only averages where weights > 0 (missing cells don't pull the average to 0)."""
    k = int(2 * round(3 * sigma) + 1)
    wts = weights.astype(np.float32)
    wv = wts[..., None] if values.ndim == 3 else wts
    v = np.where(wv > 0, values, 0).astype(np.float32) * wv
    num = cv2.GaussianBlur(v, (k, k), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(wts, (k, k), sigma, borderType=cv2.BORDER_REFLECT)
    if num.ndim == 3:
        den = den[..., None]
    with np.errstate(all="ignore"):
        return num / np.maximum(den, 1e-6)


def _detrend(values, mask, order):
    """
    Robustly fits and subtracts a smooth polynomial surface (order 1 = tilted plane, 2 = curved,
    like a hull's bend). Takes the big overall shape out first, so the local blur afterwards
    doesn't get pulled off at the edges of the scan where the surface is sloped.
    """
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xx, yy = xx / max(W - 1, 1) - 0.5, yy / max(H - 1, 1) - 0.5
    terms = [np.ones_like(xx), xx, yy] + ([xx * xx, xx * yy, yy * yy] if order >= 2 else [])
    A = np.stack(terms, -1)
    vals = values if values.ndim == 3 else values[..., None]
    keep = mask.copy()
    fit = np.zeros_like(vals)
    for _ in range(3):
        if keep.sum() < 3 * len(terms):
            break
        coef, *_ = np.linalg.lstsq(A[keep], vals[keep], rcond=None)
        fit = A @ coef
        r = np.linalg.norm(vals - fit, axis=-1)
        keep = mask & (r < 3.0 * np.median(r[mask]) + 1e-9)
    out = vals - fit
    return out if values.ndim == 3 else out[..., 0]


def _to_bytes(score):
    return np.clip(np.nan_to_num(score) * SCORE_SCALE, 0, 255).astype(np.uint8)


def bump_scores(grid, cell):
    """
    grid: (nv, nu) heights (NaN = no data). Returns (nv, nu) scores, 0 where there's no data.
    The local surface is re-fitted a few times, each time ignoring the cells that stood out,
    so a big dent doesn't drag its own reference surface down with it.
    """
    valid = np.isfinite(grid)
    if valid.sum() < 16:
        return np.zeros(grid.shape, np.float32)
    nv, nu = grid.shape
    sigma = max(BUMP_MIN_CELLS, BUMP_SMOOTH * max(nu, nv))
    floor = BUMP_FLOOR * max(nu, nv) * cell
    h = _detrend(np.where(valid, grid, 0).astype(np.float32), valid, order=2)
    w = valid.astype(np.float32)
    for _ in range(3):
        base = _blur_masked(h, w, sigma)
        res = h - base
        r = res[valid]
        noise = max(1.4826 * np.median(np.abs(r - np.median(r))), floor)
        w = (valid & (np.abs(res) < 2.5 * noise)).astype(np.float32)
    res = h - _blur_masked(h, w, sigma)
    r = res[w > 0]
    noise = max(1.4826 * np.median(np.abs(r - np.median(r))) if len(r) else 0, floor)
    score = np.abs(res) / noise
    # the outer ring of the scan is where VGGT is least sure: only trust cells with data all around
    inner = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return np.where(inner > 0, score, 0).astype(np.float32)


def color_scores(img, have, surface):
    """
    img: (H, W, 3) uint8 RGB texture, have: (H, W) bool texels actually seen by the camera,
    surface: (H, W) bool texels that lie on the grid. Returns (H, W) scores.
    """
    H, W = have.shape
    seen = have & surface
    if seen.sum() < 50:
        return np.zeros((H, W), np.float32)
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    lab[..., 0] *= LIGHTNESS_WEIGHT
    lab = _detrend(lab, seen, order=1)  # gradual lighting change across the scan
    sigma = max(3.0, COLOR_SMOOTH * max(H, W))
    w = seen.astype(np.float32)
    for _ in range(3):
        base = _blur_masked(lab, w, sigma)
        d = np.linalg.norm(lab - base, axis=-1)
        # median distance of 3-channel noise is ~1.54x its per-channel spread
        noise = max(np.median(d[seen]) / 1.54, COLOR_FLOOR)
        w = (seen & (d < 3.0 * noise)).astype(np.float32)
    d = np.linalg.norm(lab - _blur_masked(lab, w, sigma), axis=-1)
    d = cv2.GaussianBlur(d, (3, 3), 0.8)  # a single odd texel is noise, not a stain
    score = d / noise
    inner = cv2.erode(surface.astype(np.uint8), np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return np.where((inner > 0) & cv2.dilate(have.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool),
                    score, 0).astype(np.float32)


def surface_mask(grid, sub, H, W):
    """Which texels of the photo texture lie on grid cells that have data (texel x sits at cell x / sub)."""
    valid = np.isfinite(grid)
    iy = np.clip(np.round(np.arange(H) / sub).astype(int), 0, grid.shape[0] - 1)
    ix = np.clip(np.round(np.arange(W) / sub).astype(int), 0, grid.shape[1] - 1)
    return valid[iy[:, None], ix[None, :]]


def analyze(grid, cell, img=None, have=None, sub=4):
    """Everything the dashboard needs for the red / yellow highlights."""
    out = {"scale": SCORE_SCALE}
    b = bump_scores(grid, cell)
    out["bump"] = base64.b64encode(_to_bytes(b).tobytes()).decode()
    if img is not None and have is not None:
        H, W = have.shape
        c = color_scores(img, have, surface_mask(grid, sub, H, W))
        ok, png = cv2.imencode(".png", _to_bytes(c))
        out["color"] = {"w": W, "h": H, "png": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()}
    return out
