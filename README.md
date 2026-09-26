# buoay-app

Phone camera -> live web dashboard, over your local Wi-Fi.

## Run it

Requires Node.js 18+ (https://nodejs.org).

```
npm install
npm start
```

The terminal prints two links:

- **Dashboard** (open on this computer): `https://localhost:8443/dashboard`
- **Phone camera page** (open in Chrome on the Pixel): `https://<your-PC-IP>:8443/recording`

Notes:

- Phone and computer must be on the same Wi-Fi network.
- The certificate is self-signed, so each browser warns once. Tap **Advanced -> Proceed**. HTTPS is required because Chrome only allows camera access on secure pages.
- The first time you run it, Windows Firewall will ask about Node.js. Allow it on **Private networks**.
- On the phone, tap anywhere to start or stop. Red screen = recording.

## How it works

The phone grabs frames from the rear camera ~10 times per second, JPEG-encodes them, and sends them over a WebSocket to `server.js`. The server forwards each frame to every open dashboard. Because every frame passes through the server as a JPEG, that is where the damage-detection model will plug in later (see the `TODO` in `server.js`).

Tunables (FPS, resolution, JPEG quality) are at the top of the script in `public/recording.html`.
