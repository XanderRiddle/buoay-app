// 3D model viewer for buoay-app. Used by the dashboard and by the offline preview.
// Draws the reconstructed surface as a wireframe grid over an optional surface fill
// (translucent blue, or "photo": each cell colored from the camera images), plus optional
// points and camera path.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const COLORS = {
  bg: 0x0b0e12,
  grid: 0x4da3ff,       // default "healthy" hull color
  fill: 0x1d4f8f,
  cams: 0xff5449,
};

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

  const groups = { lines: new THREE.Group(), surface: new THREE.Group(), points: new THREE.Group(), cams: new THREE.Group() };
  Object.values(groups).forEach((g) => scene.add(g));
  groups.points.visible = false;
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
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  })();

  function clear(group) {
    for (const child of [...group.children]) {
      group.remove(child);
      child.geometry?.dispose();
      child.material?.map?.dispose();
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
    fills.blue = fills.photo = null;
    if (!g) return null;
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
    geo.dispose();
    applySurfaceMode();

    const center = O.clone().addScaledVector(U, (nu - 1) * cell / 2).addScaledVector(V, (nv - 1) * cell / 2);
    return { center, normal: N, size: Math.max(nu, nv) * cell };
  }

  const b64 = (s) => Uint8Array.from(atob(s), (ch) => ch.charCodeAt(0));

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
  };
}
