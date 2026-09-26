// buoay-app server
// Serves /recording (phone camera) and /dashboard (live view) over HTTPS,
// and relays JPEG frames from the phone to every open dashboard via WebSocket.

const fs = require('fs');
const path = require('path');
const https = require('https');
const os = require('os');
const express = require('express');
const { WebSocketServer } = require('ws');
const selfsigned = require('selfsigned');

const PORT = process.env.PORT || 8443;
const CERT_DIR = path.join(__dirname, 'certs');

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

function lanAddresses() {
  return Object.values(os.networkInterfaces())
    .flat()
    .filter((i) => i && i.family === 'IPv4' && !i.internal)
    .map((i) => i.address);
}

async function main() {
  const app = express();
  const pub = path.join(__dirname, 'public');
  app.get('/', (req, res) => res.redirect('/dashboard'));
  app.get('/recording', (req, res) => res.sendFile(path.join(pub, 'recording.html')));
  app.get('/dashboard', (req, res) => res.sendFile(path.join(pub, 'dashboard.html')));

  const server = https.createServer(await getCert(), app);
  const wss = new WebSocketServer({ server, path: '/ws' });

  let camera = null; // the single active phone connection
  const viewers = new Set();
  let recording = false;
  let frameCount = 0;

  const broadcastStatus = () => {
    const msg = JSON.stringify({ type: 'status', cameraConnected: !!camera, recording });
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
          // TODO (later): hand `data` to the damage-detection model here.
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
    } else {
      viewers.add(ws);
      console.log(`[dashboard] connected (${viewers.size} open)`);
      ws.send(JSON.stringify({ type: 'status', cameraConnected: !!camera, recording }));
      ws.on('close', () => viewers.delete(ws));
    }
  });

  server.listen(PORT, '0.0.0.0', () => {
    console.log('\nbuoay-app is running.\n');
    console.log(`  Dashboard (this computer): https://localhost:${PORT}/dashboard`);
    for (const ip of lanAddresses()) {
      console.log(`  Phone camera page:         https://${ip}:${PORT}/recording`);
    }
    console.log('\nThe certificate is self-signed, so each browser will warn once. Choose "Advanced" -> "Proceed".\n');
  });
}

main();
