// buoay-app server
// Serves /recording (phone camera) and /dashboard (live view) over HTTPS,
// and relays JPEG frames from the phone to every open dashboard via WebSocket.
//
//   node server.js            local network only
//   node server.js --tunnel   also reachable from anywhere through a free Cloudflare tunnel
//
// Every page and connection needs the access key (certs/access-key.txt). Opening a link
// with ?key=... once stores it in a cookie on that device.

const fs = require('fs');
const path = require('path');
const https = require('https');
const os = require('os');
const express = require('express');
const { WebSocketServer } = require('ws');
const selfsigned = require('selfsigned');
const crypto = require('crypto');
const { spawn } = require('child_process');
const QRCode = require('qrcode');

const PORT = process.env.PORT || 8443;
const CERT_DIR = path.join(__dirname, 'certs');
const LINKS_FILE = path.join(__dirname, 'LINKS.txt'); // in OneDrive, so it syncs to your other devices
const USE_TUNNEL = process.argv.includes('--tunnel');

// Phone browsers only allow camera access on HTTPS pages (or localhost),
// so we generate a self-signed certificate once and reuse it.
async function getCert() {
  const keyPath = path.join(CERT_DIR, 'key.pem');
  const certPath = path.join(CERT_DIR, 'cert.pem');
  if (fs.existsSync(keyPath) && fs.existsSync(certPath)) {
    return { key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) };
  }
  const pems = await selfsigned.generate([{ name: 'commonName', value: 'buoay-app.local' }]);
  fs.mkdirSync(CERT_DIR, { recursive: true });
  fs.writeFileSync(keyPath, pems.private);
  fs.writeFileSync(certPath, pems.cert);
  return { key: pems.private, cert: pems.cert };
}

// ---------- access key ----------
function getAccessKey() {
  if (process.env.BUOAY_KEY) return process.env.BUOAY_KEY;
  const p = path.join(CERT_DIR, 'access-key.txt');
  if (fs.existsSync(p)) return fs.readFileSync(p, 'utf8').trim();
  const key = crypto.randomBytes(9).toString('base64url');
  fs.mkdirSync(CERT_DIR, { recursive: true });
  fs.writeFileSync(p, key);
  return key;
}
const ACCESS_KEY = getAccessKey();
const COOKIE = 'buoay_key';

function sameKey(k) {
  if (typeof k !== 'string') return false;
  const a = Buffer.from(k), b = Buffer.from(ACCESS_KEY);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}
function cookieKey(req) {
  const m = (req.headers.cookie || '').match(new RegExp(`(?:^|;\\s*)${COOKIE}=([^;]+)`));
  return m ? decodeURIComponent(m[1]) : null;
}
function authorized(req) {
  const q = new URL(req.url, 'https://x').searchParams.get('key');
  return sameKey(q) || sameKey(cookieKey(req));
}

// ---------- links (local + public tunnel) ----------
const links = { public: null, local: [] };
function linkSet(base) {
  return { dashboard: `${base}/dashboard?key=${ACCESS_KEY}`, phone: `${base}/recording?key=${ACCESS_KEY}` };
}
function currentPhoneLink() {
  return links.public ? linkSet(links.public).phone : (links.local[0] ? linkSet(links.local[0]).phone : null);
}
function publishLinks() {
  const lines = [`buoay-app links (updated ${new Date().toLocaleString()})`, ''];
  if (links.public) {
    const l = linkSet(links.public);
    lines.push('FROM ANYWHERE (Cloudflare tunnel, changes every time the server restarts):');
    lines.push(`  Dashboard: ${l.dashboard}`, `  Phone:     ${l.phone}`, '');
  } else if (USE_TUNNEL) {
    lines.push('Tunnel starting... (this file updates when it is up)', '');
  }
  lines.push('SAME NETWORK AS THIS COMPUTER ONLY (self-signed certificate warning):');
  for (const base of links.local) {
    const l = linkSet(base);
    lines.push(`  Dashboard: ${l.dashboard}`, `  Phone:     ${l.phone}`);
  }
  if (ntfyTopic()) {
    lines.push('', `Get the new link on any device when it changes: https://ntfy.sh/${ntfyTopic()}`,
      '(open that page once, or subscribe to the topic in the free ntfy app)');
  }
  const text = lines.join('\n') + '\n';
  fs.writeFileSync(LINKS_FILE, text);
  // Also drop a copy in the OneDrive root (if this PC has OneDrive), so it syncs to your laptop
  // even when the project folder itself isn't in OneDrive.
  if (process.env.OneDrive && fs.existsSync(process.env.OneDrive)) {
    try { fs.writeFileSync(path.join(process.env.OneDrive, 'buoay-app-LINKS.txt'), text); } catch {}
  }
  console.log('\n' + text);
  notify();
}

// Optional notification with the new link each time it changes (free, no account): put a
// hard-to-guess topic name in certs/ntfy-topic.txt and open https://ntfy.sh/<topic> on your devices.
function ntfyTopic() {
  const p = path.join(CERT_DIR, 'ntfy-topic.txt');
  return fs.existsSync(p) ? fs.readFileSync(p, 'utf8').trim() || null : null;
}

let lastNotified = null;
function notify() {
  const topic = ntfyTopic();
  if (!links.public || !topic || lastNotified === links.public) return;
  lastNotified = links.public;
  const l = linkSet(links.public);
  fetch(`https://ntfy.sh/${encodeURIComponent(topic)}`, {
    method: 'POST', body: `Dashboard: ${l.dashboard}\nPhone: ${l.phone}`,
    headers: { Title: 'buoay-app is online', Click: l.dashboard },
  }).then(() => console.log('[ntfy] sent new link to', `https://ntfy.sh/${topic}`))
    .catch((e) => console.log('[ntfy] failed:', e.message));
}

function findCloudflared() {
  const local = path.join(__dirname, 'tools', 'cloudflared.exe');
  return fs.existsSync(local) ? local : 'cloudflared';
}

function startTunnel(port) {
  const exe = findCloudflared();
  console.log(`[tunnel] starting ${exe} ...`);
  const proc = spawn(exe, ['tunnel', '--no-autoupdate', '--url', `https://localhost:${port}`, '--no-tls-verify'],
    { windowsHide: true });
  const recent = [];
  const onData = (buf) => {
    recent.push(...buf.toString().split('\n').filter(Boolean));
    recent.splice(0, Math.max(0, recent.length - 4));
    const m = buf.toString().match(/https:\/\/(?!api\.)[-a-z0-9]+\.trycloudflare\.com/);
    if (m && m[0] !== links.public) {
      links.public = m[0];
      console.log(`[tunnel] public address: ${links.public}`);
      publishLinks();
    }
  };
  proc.stdout.on('data', onData);
  proc.stderr.on('data', onData);
  proc.on('error', (e) => {
    console.log(`[tunnel] could not start cloudflared (${e.message}). Run start.bat, which downloads it, or install it.`);
  });
  proc.on('exit', (code) => {
    console.log(`[tunnel] cloudflared exited (${code}); restarting in 5 s. The public link will change.`);
    if (recent.length) console.log('[tunnel] last output:\n  ' + recent.join('\n  '));
    links.public = null;
    setTimeout(() => startTunnel(port), 5000);
  });
}

function lanAddresses() {
  return Object.values(os.networkInterfaces())
    .flat()
    .filter((i) => i && i.family === 'IPv4' && !i.internal)
    .map((i) => i.address);
}

async function main() {
  const app = express();
  const pub = path.join(__dirname, 'public');
  app.get('/favicon.ico', (req, res) => res.status(204).end());

  // Access key check for everything below.
  app.use((req, res, next) => {
    const q = req.query.key;
    if (q !== undefined && sameKey(q)) {
      // remember the key on this device, then drop it from the address bar
      res.setHeader('Set-Cookie', `${COOKIE}=${encodeURIComponent(ACCESS_KEY)}; Max-Age=2592000; Path=/; Secure; HttpOnly; SameSite=Lax`);
      const clean = new URL(req.originalUrl, 'https://x');
      clean.searchParams.delete('key');
      return res.redirect(clean.pathname + clean.search);
    }
    if (sameKey(cookieKey(req))) return next();
    res.status(401).send(`<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">
      <body style="font-family:system-ui;background:#0f1216;color:#e6e9ee;padding:32px;line-height:1.5">
      <h2>buoay-app is locked</h2><p>Open the full link from <b>LINKS.txt</b> on the server computer
      (it ends in <code>?key=...</code>). You only need to do that once per device.</p></body>`);
  });

  app.get('/', (req, res) => res.redirect('/dashboard'));
  app.get('/api/links', (req, res) => {
    res.json({ phone: currentPhoneLink(), public: !!links.public });
  });
  app.get('/api/phone-qr.svg', async (req, res) => {
    const url = currentPhoneLink();
    if (!url) return res.status(404).end();
    res.type('image/svg+xml').send(await QRCode.toString(url, { type: 'svg', margin: 1, color: { dark: '#000', light: '#fff' } }));
  });
  app.get('/recording', (req, res) => res.sendFile(path.join(pub, 'recording.html')));
  app.get('/dashboard', (req, res) => res.sendFile(path.join(pub, 'dashboard.html')));
  app.get('/model-viewer.js', (req, res) => res.sendFile(path.join(pub, 'model-viewer.js')));
  // three.js served locally so the dashboard works without internet
  app.use('/vendor/three', express.static(path.join(__dirname, 'node_modules', 'three')));

  const server = https.createServer(await getCert(), app);
  const wss = new WebSocketServer({ server, path: '/ws', verifyClient: ({ req }) => authorized(req) });

  let camera = null; // the single active phone connection
  const viewers = new Set();
  let worker = null; // the Python reconstruction worker
  let lastModel = null; // latest 3D model (JSON string), replayed to new dashboards
  let lastRecon = null; // latest worker status (JSON string)
  let recording = false;
  let frameCount = 0;

  const broadcastStatus = () => {
    const msg = JSON.stringify({ type: 'status', cameraConnected: !!camera, recording });
    for (const v of viewers) if (v.readyState === 1) v.send(msg);
    if (worker && worker.readyState === 1) worker.send(msg);
  };
  const toViewers = (msg) => {
    for (const v of viewers) if (v.readyState === 1) v.send(msg);
  };

  wss.on('connection', (ws, req) => {
    const role = new URL(req.url, 'https://x').searchParams.get('role');

    if (role === 'camera') {
      if (camera) camera.close(4000, 'Replaced by a newer camera connection');
      camera = ws;
      recording = false;
      console.log(`[camera] connected from ${req.socket.remoteAddress}`);
      broadcastStatus();

      ws.on('message', (data, isBinary) => {
        if (isBinary) {
          // A JPEG frame: forward it to every dashboard that is keeping up.
          frameCount++;
          for (const v of viewers) {
            if (v.readyState === 1 && v.bufferedAmount < 2 * 1024 * 1024) v.send(data, { binary: true });
          }
          // The reconstruction worker gets every frame it can keep up with and picks keyframes.
          if (worker && worker.readyState === 1 && worker.bufferedAmount < 4 * 1024 * 1024) {
            worker.send(data, { binary: true });
          }
          return;
        }
        try {
          const msg = JSON.parse(data.toString());
          if (msg.type === 'recording') {
            recording = !!msg.recording;
            console.log(`[camera] recording ${recording ? 'STARTED' : 'STOPPED'}`);
            broadcastStatus();
          }
        } catch {}
      });

      ws.on('close', () => {
        if (camera === ws) {
          camera = null;
          recording = false;
          console.log('[camera] disconnected');
          broadcastStatus();
        }
      });
    } else if (role === 'worker') {
      if (worker) worker.close(4000, 'Replaced by a newer worker');
      worker = ws;
      console.log('[worker] connected');
      ws.send(JSON.stringify({ type: 'status', cameraConnected: !!camera, recording }));
      ws.on('message', (data, isBinary) => {
        if (isBinary) return;
        const text = data.toString();
        let type;
        try { type = JSON.parse(text).type; } catch { return; }
        if (type === 'model') lastModel = text;
        if (type === 'recon') lastRecon = text;
        toViewers(text);
      });
      ws.on('close', () => {
        if (worker === ws) {
          worker = null;
          lastRecon = JSON.stringify({ type: 'recon', state: 'offline' });
          toViewers(lastRecon);
          console.log('[worker] disconnected');
        }
      });
    } else {
      viewers.add(ws);
      console.log(`[dashboard] connected (${viewers.size} open)`);
      ws.send(JSON.stringify({ type: 'status', cameraConnected: !!camera, recording }));
      ws.send(lastRecon || JSON.stringify({ type: 'recon', state: 'offline' }));
      if (lastModel) ws.send(lastModel);
      ws.on('close', () => viewers.delete(ws));
    }
  });

  server.listen(PORT, '0.0.0.0', () => {
    console.log('\nbuoay-app is running.');
    links.local = [`https://localhost:${PORT}`, ...lanAddresses().map((ip) => `https://${ip}:${PORT}`)];
    publishLinks();
    if (USE_TUNNEL) startTunnel(PORT);
  });
}

main();
