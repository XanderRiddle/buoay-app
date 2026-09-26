# buoay-app

Phone camera -> live dashboard -> live 3D model of the surface you're scanning (a wall now, a hull later).

The **desktop at home** (RTX 3070 Ti) does all the work: web server, 3D reconstruction and a free Cloudflare
tunnel so the **phone** and **laptop** can reach it from any network.

```
phone /recording ---+                                    +--- laptop /dashboard
                    +--> Cloudflare tunnel --> desktop --+
                              (https)          server.js + processing/worker.py (VGGT on the GPU)
```

## One-time setup on the desktop

1. Install **Node.js** (nodejs.org, LTS) and **Python 3.12** (python.org, tick "Add python.exe to PATH").
2. Run **`processing\setup.bat`**. It should end with `CUDA GPU found: NVIDIA GeForce RTX 3070 Ti`.
   (The Python environment goes in `%LOCALAPPDATA%\buoay-app`, outside OneDrive, so it doesn't sync.)
3. Stop the desktop from sleeping while you're away: Settings > System > Power > Screen and sleep >
   "When plugged in, put my device to sleep after" > **Never**.

## Start it (on the desktop, before you leave)

Double-click **`start.bat`**. It downloads the tunnel program the first time, then opens two windows
(**buoay server** and **buoay 3D worker**) that restart themselves if they crash. The first start also
downloads the VGGT model (~5 GB).

About 10 seconds later, **`LINKS.txt`** in this folder has the links. The folder is in OneDrive, so the
file shows up on your laptop too:

```
Dashboard: https://something-random.trycloudflare.com/dashboard?key=...
Phone:     https://something-random.trycloudflare.com/recording?key=...
```

- **Laptop:** open the Dashboard link.
- **Phone:** on the dashboard, click **Phone link** and scan the QR code (or open the Phone link).

Each device only needs the `?key=...` link once; after that it remembers the key.
Anyone without the key gets a "locked" page.

**The tunnel address changes whenever the server restarts** (reboot, crash, Windows update). Check `LINKS.txt`
again if the links stop working. Optional: get a phone notification with the new link each time. Pick a
hard-to-guess topic name, put it in `certs\ntfy-topic.txt` on the desktop, and subscribe to that topic in the
free **ntfy** app.

`start.bat local` skips the tunnel (same Wi-Fi only, self-signed certificate warning).

## Scanning tips

- **Hold the phone sideways (landscape)**, point it straight at the surface, about 1 m away.
- **Move slowly and steadily sideways**, like spray-painting. If the dashboard's "Waiting" number keeps
  climbing, slow down.
- **Texture matters.** Brick, posters, a bookshelf: good. A blank painted wall: poor.
- Tap stop when done. The scan is saved on the desktop in `processing\scans\<date-time>\`.

## What the dashboard shows

- **Live feed** from the phone.
- **3D model**: the surface as a blue wireframe grid that grows as you scan. Drag to rotate, right-drag to pan,
  scroll to zoom. Toggle **Points** and **Camera path**. **Reset view** re-fits the view.
- **Reconstruction** stats: keyframes, backlog, batch time, GPU memory.

Scale is relative: the shape is right, but there are no real-world units yet.

## Rebuilding a model from photos (no phone needed)

```
cd processing
"%LOCALAPPDATA%\buoay-app\venv\Scripts\python.exe" worker.py --images test_images
```

Writes `processing\output\model_preview.html`.
