# buoay-app

Phone camera -> live dashboard -> live 3D model of the surface you're scanning (a wall now, a hull later).

Everything runs on one laptop (the one with the NVIDIA GPU). The phone just needs to be on the same Wi-Fi.

```
phone /recording  --JPEG frames-->  server.js  --frames-->  dashboard (live feed)
                                        |  ^
                               frames   v  |  3D model
                               processing/worker.py  (keyframes -> VGGT -> grid)
```

## First-time setup (on the laptop)

1. Install **Node.js** (nodejs.org, LTS).
2. Follow `processing/README.md` to set up Python + VGGT (`processing\setup.bat`).
3. Windows Firewall will ask about Node.js the first time: allow it on **Private networks**.

## Every time

Double-click **`start.bat`**. It opens two windows and the dashboard:

- **buoay server** prints the phone link: `https://<laptop-IP>:8443/recording`
- **buoay 3D worker** loads VGGT (about 20 s), then waits for the phone

On the phone (Chrome), open the link, accept the certificate warning (**Advanced -> Proceed**), and tap to start recording.

## Scanning tips

- **Hold the phone sideways (landscape)** and point it straight at the surface, about 1 m away.
- **Move slowly and steadily sideways**, like spray-painting. The dashboard's "Waiting" number should stay low; if it keeps climbing, slow down.
- **Texture matters.** Brick, posters, a bookshelf: good. A blank painted wall: poor.
- Tap stop when done. The last few keyframes get processed, then the scan is saved to `processing/scans/<date-time>/` (keyframes, `model.json`, `points.ply`).

## What the dashboard shows

- **Live feed** from the phone.
- **3D model**: the surface as a blue wireframe grid that grows every ~4 keyframes. Drag to rotate, right-drag to pan, scroll to zoom. Toggle **Points** (the raw colored 3D points) and **Camera path** (where the phone was). The view auto-fits until you grab it; **Reset view** brings that back.
- **Reconstruction** stats: keyframes, backlog, batch time, GPU memory.

Scale is relative: the model has the right shape, but no real-world units yet.

## Rebuilding a model from photos (no phone needed)

```
cd processing
.venv\Scripts\python.exe worker.py --images test_images
.venv\Scripts\python.exe worker.py --images scans\20260926-141500
```

Writes `processing/output/model_preview.html`.
