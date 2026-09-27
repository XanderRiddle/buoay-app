// 3D model viewer for buoay-app. Used by the dashboard and by the offline preview.
// Draws the reconstructed surface as a wireframe grid over an optional surface fill
// (translucent blue, or "photo": each cell colored from the camera images), plus optional
// points and camera path. Surface-check highlights glow on top: red where the shape has a bump or
// dent, yellow where the color changes (scores come from processing/anomaly.py).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const COLORS = {
  bg: 0x0b0e12,
  grid: 0x4da3ff,       // default "healthy" hull color
  fill: 0x1d4f8f,
  cams: 0xff5449,
};

// Highlight colors: [halo, core] as RGB 0-255
const GLOW = {
  damage: [[255, 30, 30], [255, 80, 64]],
  color: [[255, 190, 0], [255, 236, 90]],
};
const HL_SUB = 4;              // overlay texels per grid cell (same as the photo texture)
const MIN_TEXELS = { damage: 2 * HL_SUB * HL_SUB, color: 12 }; // smaller specks are ignored
const MAX_BEACONS = 12;        // glowing markers over the biggest areas of each kind

export function createModelViewer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);
  renderer.domElement.style.display = 'block';

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(COLORS.bg);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.001, 1000);
  camera.up.set(0, -1, 0); // VGGT uses camera-style axes where +y points down
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const groups = { lines: new THREE.Group(), surface: new THREE.Group(), points: new THREE.Group(), cams: new THREE.Group(),
                   highlights: new THREE.Group() };
  Object.values(groups).forEach((g) => scene.add(g));
  groups.points.visible = false;
  // surface-check highlights: which kinds are shown, and the score needed to count
  const hl = { damage: true, color: true, threshold: 3.5, summary: null, onChange: null };
  let hlState = null; // current grid's overlay pieces
  let surfaceMode = 'blue'; // 'off' | 'blue' | 'photo'
  const fills = { blue: null, photo: null };

  let userMoved = false; // keep auto-fitting the view as the model grows, until the user grabs it
  controls.addEventListener('start', () => { userMoved = true; });
  let lastModel = null;

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);
  resize();

  (function loop() {
    controls.update();
    pulseHighlights(performance.now() / 1000);
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  })();

  function clear(group) {
    for (const child of [...group.children]) {
      group.remove(child);
      if (child.children?.length) clear(child);
      child.geometry?.dispose();
      if (!child.userData.sharedMap) child.material?.map?.dispose();
      child.material?.dispose();
    }
  }

  function applySurfaceMode() {
    groups.surface.visible = surfaceMode !== 'off';
    if (fills.blue) fills.blue.visible = surfaceMode === 'blue';
    if (fills.photo) fills.photo.visible = surfaceMode === 'photo';
  }

  function buildGrid(g) {
    clear(groups.lines);
    clear(groups.surface);
    clear(groups.highlights);
    fills.blue = fills.photo = null;
    hlState = null;
    if (!g) { setSummary(null); return null; }
    const { nu, nv, cell } = g;
    const O = new THREE.Vector3(...g.origin), U = new THREE.Vector3(...g.u),
          V = new THREE.Vector3(...g.v), N = new THREE.Vector3(...g.n);
    const idx = (i, j) => j * nu + i;
    const pos = new Float32Array(nu * nv * 3);
    const valid = new Uint8Array(nu * nv);
    const p = new THREE.Vector3();
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
      const h = g.h[idx(i, j)];
      if (h === null) continue;
      p.copy(O).addScaledVector(U, i * cell).addScaledVector(V, j * cell).addScaledVector(N, h);
      pos.set([p.x, p.y, p.z], idx(i, j) * 3);
      valid[idx(i, j)] = 1;
    }
    const lines = [], tris = [];
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
      const a = idx(i, j);
      if (!valid[a]) continue;
      if (i + 1 < nu && valid[a + 1]) lines.push(a, a + 1);
      if (j + 1 < nv && valid[a + nu]) lines.push(a, a + nu);
      if (i + 1 < nu && j + 1 < nv && valid[a + 1] && valid[a + nu] && valid[a + nu + 1]) {
        tris.push(a, a + 1, a + nu + 1, a, a + nu + 1, a + nu);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const lineGeo = geo.clone(); lineGeo.setIndex(lines);
    groups.lines.add(new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ color: COLORS.grid })));

    const blueGeo = geo.clone(); blueGeo.setIndex(tris);
    fills.blue = new THREE.Mesh(blueGeo, new THREE.MeshBasicMaterial({
      color: COLORS.fill, transparent: true, opacity: 0.25, side: THREE.DoubleSide, depthWrite: false }));
    groups.surface.add(fills.blue);

    // "Photo" surface: an image of the average camera color over the surface,
    // 4x finer than the grid, stretched across the mesh
    if (g.texture) {
      const { w, h, sub } = g.texture;
      const uv = new Float32Array(nu * nv * 2);
      for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
        uv[idx(i, j) * 2] = (i * sub + 0.5) / w;
        uv[idx(i, j) * 2 + 1] = (j * sub + 0.5) / h;
      }
      const photoGeo = geo.clone(); photoGeo.setIndex(tris);
      photoGeo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
      const tex = new THREE.TextureLoader().load(g.texture.png);
      tex.flipY = false;
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.magFilter = THREE.LinearFilter;
      fills.photo = new THREE.Mesh(photoGeo, new THREE.MeshBasicMaterial({
        map: tex, side: THREE.DoubleSide,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 })); // keeps grid lines on top
      groups.surface.add(fills.photo);
    }
    buildHighlights(g, geo, tris, pos, valid);
    geo.dispose();
    applySurfaceMode();

    const center = O.clone().addScaledVector(U, (nu - 1) * cell / 2).addScaledVector(V, (nv - 1) * cell / 2);
    return { center, normal: N, size: Math.max(nu, nv) * cell };
  }

  const b64 = (s) => Uint8Array.from(atob(s), (ch) => ch.charCodeAt(0));

  // ---------------------------------------------------------------- highlights
  // A see-through layer draped over the surface: solid red/yellow where a spot scores above the
  // threshold, fading out into a soft glow around it, and pulsing gently. Plus a glowing marker
  // over each of the biggest spots so they're easy to find when zoomed out.

  let beaconTex = null;
  function beaconTexture() {
    if (beaconTex) return beaconTex;
    const c = document.createElement('canvas');
    c.width = c.height = 64;
    const ctx = c.getContext('2d');
    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.25, 'rgba(255,255,255,0.55)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);
    beaconTex = new THREE.CanvasTexture(c);
    return beaconTex;
  }

  function buildHighlights(g, geo, tris, pos, valid) {
    const a = g.anomalies;
    if (!a || !a.bump) { setSummary(null); return; }
    const { nu, nv } = g;
    const W = (nu - 1) * HL_SUB + 1, H = (nv - 1) * HL_SUB + 1;
    const uv = new Float32Array(nu * nv * 2);
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
      uv[(j * nu + i) * 2] = (i * HL_SUB + 0.5) / W;
      uv[(j * nu + i) * 2 + 1] = (j * HL_SUB + 0.5) / H;
    }
    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    const tex = new THREE.CanvasTexture(canvas);
    tex.flipY = false;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.LinearFilter;
    const hGeo = geo.clone(); hGeo.setIndex(tris);
    hGeo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    const mesh = new THREE.Mesh(hGeo, new THREE.MeshBasicMaterial({
      map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 })); // drawn over the surface and grid
    mesh.renderOrder = 2;
    groups.highlights.add(mesh);
    const beacons = new THREE.Group();
    groups.highlights.add(beacons);

    // per-texel scores: bumps come per grid cell (smoothly stretched), colors per texel (PNG, loads async)
    const scale = a.scale || 20;
    const cellScore = b64(a.bump);
    const bump = new Float32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const cx = x / HL_SUB, cy = y / HL_SUB;
      const i0 = Math.min(Math.floor(cx), nu - 2), j0 = Math.min(Math.floor(cy), nv - 2);
      const fx = cx - i0, fy = cy - j0;
      const s = (i, j) => cellScore[j * nu + i];
      bump[y * W + x] = ((s(i0, j0) * (1 - fx) + s(i0 + 1, j0) * fx) * (1 - fy)
                        + (s(i0, j0 + 1) * (1 - fx) + s(i0 + 1, j0 + 1) * fx) * fy) / scale;
    }
    // texels that lie on real surface (for the "% of surface" numbers)
    const onSurface = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = Math.round(x / HL_SUB), j = Math.round(y / HL_SUB);
      onSurface[y * W + x] = valid[j * nu + i];
    }
    hlState = { g, W, H, canvas, tex, mesh, beacons, bump, color: null, onSurface, pos, valid, nu, nv };

    if (a.color && a.color.w === W && a.color.h === H) {
      if (g._colorScores) hlState.color = g._colorScores;
      else {
        const state = hlState;
        const img = new Image();
        img.onload = () => {
          const c = document.createElement('canvas');
          c.width = W; c.height = H;
          const ctx = c.getContext('2d');
          ctx.drawImage(img, 0, 0);
          const d = ctx.getImageData(0, 0, W, H).data;
          const col = new Float32Array(W * H);
          for (let k = 0; k < W * H; k++) col[k] = d[k * 4] / scale;
          g._colorScores = col; // reused when the view is reset
          if (hlState === state) { state.color = col; paintHighlights(); }
        };
        img.src = a.color.png;
      }
    }
    paintHighlights();
  }

  // blur a 0..1 mask (three box blurs ~ a gaussian)
  function blur(src, W, H, r) {
    let a = Float32Array.from(src), b = new Float32Array(W * H);
    for (let pass = 0; pass < 3; pass++) {
      for (let y = 0; y < H; y++) {
        let acc = 0;
        for (let x = -r; x <= r; x++) acc += a[y * W + Math.min(W - 1, Math.max(0, x))];
        for (let x = 0; x < W; x++) {
          b[y * W + x] = acc / (2 * r + 1);
          acc += a[y * W + Math.min(W - 1, x + r + 1)] - a[y * W + Math.max(0, x - r)];
        }
      }
      for (let x = 0; x < W; x++) {
        let acc = 0;
        for (let y = -r; y <= r; y++) acc += b[Math.min(H - 1, Math.max(0, y)) * W + x];
        for (let y = 0; y < H; y++) {
          a[y * W + x] = acc / (2 * r + 1);
          acc += b[Math.min(H - 1, y + r + 1) * W + x] - b[Math.max(0, y - r) * W + x];
        }
      }
    }
    return a;
  }

  // connected areas above the threshold; drops specks. Returns { mask, regions: [{n, x, y, r}] }
  function findRegions(score, W, H, th, minTexels) {
    const mask = new Uint8Array(W * H), seen = new Uint8Array(W * H), regions = [];
    const stack = [];
    for (let k = 0; k < W * H; k++) {
      if (seen[k] || !(score[k] >= th)) continue;
      const members = [];
      stack.push(k); seen[k] = 1;
      while (stack.length) {
        const q = stack.pop(); members.push(q);
        const x = q % W, y = (q - x) / W;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
          const n = ny * W + nx;
          if (!seen[n] && score[n] >= th) { seen[n] = 1; stack.push(n); }
        }
      }
      if (members.length < minTexels) continue;
      let sx = 0, sy = 0;
      for (const q of members) { mask[q] = 1; sx += q % W; sy += Math.floor(q / W); }
      const x = sx / members.length, y = sy / members.length;
      let r2 = 0;
      for (const q of members) r2 = Math.max(r2, (q % W - x) ** 2 + (Math.floor(q / W) - y) ** 2);
      regions.push({ n: members.length, x, y, r: Math.sqrt(r2) });
    }
    regions.sort((p, q) => q.n - p.n);
    return { mask, regions };
  }

  function surfacePoint(st, x, y) {
    // nearest grid vertex with data to texel (x, y)
    const ci = x / HL_SUB, cj = y / HL_SUB;
    let best = null, bd = Infinity;
    for (let j = Math.max(0, Math.floor(cj) - 1); j <= Math.min(st.nv - 1, Math.ceil(cj) + 1); j++)
      for (let i = Math.max(0, Math.floor(ci) - 1); i <= Math.min(st.nu - 1, Math.ceil(ci) + 1); i++) {
        const k = j * st.nu + i;
        const d = (i - ci) ** 2 + (j - cj) ** 2;
        if (st.valid[k] && d < bd) { bd = d; best = k; }
      }
    return best === null ? null : new THREE.Vector3(st.pos[best * 3], st.pos[best * 3 + 1], st.pos[best * 3 + 2]);
  }

  function paintHighlights() {
    const st = hlState;
    if (!st) return;
    const { W, H, canvas, tex, beacons } = st;
    clear(beacons);
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(W, H);
    const d = img.data;
    const cellSize = st.g.cell;
    let surfaceTexels = 0;
    for (let k = 0; k < W * H; k++) surfaceTexels += st.onSurface[k];
    const summary = {};
    const layers = [];
    for (const kind of ['color', 'damage']) { // damage last: drawn on top of color
      const score = kind === 'damage' ? st.bump : st.color;
      if (!score) { summary[kind] = kind === 'color' && st.g.anomalies.color ? { loading: true } : null; continue; }
      const { mask, regions } = findRegions(score, W, H, hl.threshold, MIN_TEXELS[kind]);
      let n = 0;
      for (let k = 0; k < W * H; k++) n += mask[k] & st.onSurface[k];
      summary[kind] = { areas: regions.length, fraction: surfaceTexels ? n / surfaceTexels : 0 };
      if (!hl[kind] || !regions.length) continue;
      const halo = blur(mask, W, H, HL_SUB);
      layers.push({ kind, mask, halo });
      for (const r of regions.slice(0, MAX_BEACONS)) {
        const p = surfacePoint(st, r.x, r.y);
        if (!p) continue;
        const s = new THREE.Sprite(new THREE.SpriteMaterial({
          map: beaconTexture(), color: new THREE.Color(`rgb(${GLOW[kind][0].join(',')})`),
          transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0.8 }));
        s.userData.sharedMap = true;
        const size = Math.max(3 * cellSize, (r.r / HL_SUB) * cellSize * 2.6);
        s.userData.size = size;
        s.userData.phase = Math.random() * Math.PI * 2;
        s.scale.setScalar(size);
        s.position.copy(p);
        s.renderOrder = 3;
        beacons.add(s);
      }
    }
    for (let k = 0; k < W * H; k++) {
      let r = 0, gch = 0, b = 0, alpha = 0;
      for (const L of layers) { // later layers (damage) cover earlier ones (color)
        const core = L.mask[k];
        const glow = Math.min(1, L.halo[k] * 2.2);
        const a = Math.max(core * 0.8, glow * 0.6);
        if (a <= 0) continue;
        const [halo, hot] = GLOW[L.kind];
        const t = core ? 1 : 0;
        const cr = halo[0] + (hot[0] - halo[0]) * t, cg = halo[1] + (hot[1] - halo[1]) * t, cb = halo[2] + (hot[2] - halo[2]) * t;
        const na = a + alpha * (1 - a);
        r = (cr * a + r * alpha * (1 - a)) / na;
        gch = (cg * a + gch * alpha * (1 - a)) / na;
        b = (cb * a + b * alpha * (1 - a)) / na;
        alpha = na;
      }
      d[k * 4] = r; d[k * 4 + 1] = gch; d[k * 4 + 2] = b; d[k * 4 + 3] = Math.round(alpha * 255);
    }
    ctx.putImageData(img, 0, 0);
    tex.needsUpdate = true;
    setSummary(summary);
  }

  function pulseHighlights(t) {
    const st = hlState;
    if (!st || !groups.highlights.visible) return;
    st.mesh.material.opacity = 0.78 + 0.22 * Math.sin(t * 2.6);
    for (const s of st.beacons.children) {
      const k = 0.5 + 0.5 * Math.sin(t * 2.6 + s.userData.phase);
      s.scale.setScalar(s.userData.size * (0.85 + 0.3 * k));
      s.material.opacity = 0.45 + 0.45 * k;
    }
  }

  function setSummary(s) {
    hl.summary = s;
    hl.onChange?.(s);
  }

  // Points arrive packed: positions as 16-bit steps within [min, max], colors as bytes.
  function unpackPoints(pts) {
    if (pts.p) return { p: new Float32Array(pts.p), c: new Float32Array(pts.c) }; // older format
    const n = pts.n || 0;
    const q = new Uint16Array(b64(pts.q).buffer), c8 = b64(pts.c8);
    const p = new Float32Array(n * 3), c = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) {
      const a = i % 3;
      p[i] = pts.min[a] + (q[i] / 65535) * (pts.max[a] - pts.min[a]);
      c[i] = c8[i] / 255;
    }
    return { p, c };
  }

  function buildPoints(packed) {
    clear(groups.points);
    if (!packed || !(packed.n || packed.p?.length)) return;
    const pts = unpackPoints(packed);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pts.p, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(pts.c, 3));
    geo.computeBoundingSphere();
    const r = geo.boundingSphere.radius || 1;
    groups.points.add(new THREE.Points(geo, new THREE.PointsMaterial({ size: r / 250, vertexColors: true })));
  }

  function buildCams(c, size) {
    clear(groups.cams);
    if (!c || c.length < 3) return;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(c, 3));
    groups.cams.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: COLORS.cams, transparent: true, opacity: 0.6 })));
    groups.cams.add(new THREE.Points(geo.clone(), new THREE.PointsMaterial({ color: COLORS.cams, size: (size || 1) / 120 })));
  }

  function frame(info, cams) {
    if (!info) return;
    // look at the surface from the side the phone was on
    const n = info.normal.clone();
    if (cams && cams.length >= 3) {
      const cc = new THREE.Vector3();
      for (let i = 0; i < cams.length; i += 3) cc.add(new THREE.Vector3(cams[i], cams[i + 1], cams[i + 2]));
      cc.divideScalar(cams.length / 3);
      if (cc.sub(info.center).dot(n) < 0) n.negate();
    }
    controls.target.copy(info.center);
    // back off far enough that the whole grid fits, whatever the panel's shape
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const fit = Math.max(info.size / 2 / Math.tan(fov / 2), info.size / 2 / Math.tan(fov / 2) / camera.aspect);
    camera.position.copy(info.center).addScaledVector(n, fit * 1.15);
    camera.near = info.size / 1000; camera.far = info.size * 100;
    camera.updateProjectionMatrix();
    controls.update();
  }

  return {
    update(model) {
      lastModel = model;
      const info = buildGrid(model.grid);
      buildPoints(model.points);
      buildCams(model.cams, info?.size);
      if (!userMoved && info) frame(info, model.cams);
    },
    clear() {
      Object.values(groups).forEach(clear);
      hlState = null; setSummary(null);
      userMoved = false; lastModel = null;
    },
    resetView() {
      userMoved = false;
      if (lastModel?.grid) frame(buildGrid(lastModel.grid), lastModel.cams);
    },
    setVisible(name, on) { groups[name].visible = on; },
    isVisible(name) { return groups[name].visible; },
    setSurface(mode) { surfaceMode = mode; applySurfaceMode(); },
    getSurface() { return surfaceMode; },
    // surface-check highlights: kind is 'damage' (red) or 'color' (yellow)
    setHighlight(kind, on) { hl[kind] = on; paintHighlights(); },
    getHighlight(kind) { return hl[kind]; },
    // score an area needs to be highlighted (lower = more sensitive); 3.5 by default
    setThreshold(z) { hl.threshold = z; paintHighlights(); },
    onHighlights(fn) { hl.onChange = fn; fn(hl.summary); },
  };
}
