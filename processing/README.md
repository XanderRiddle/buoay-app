# processing: 3D reconstruction

- `worker.py`: the live reconstruction worker (started by `start.bat` in the main folder). Also `--images <folder>` to rebuild a model from photos. Batch size is picked from GPU memory (10 photos on an 8 GB card, 6 on 4 GB).
- `recon.py`: VGGT wrapper, batch stitching, grid fitting.
- `vggt_smoke_test.py`: one-off test of VGGT on a few photos (memory + timing).

`setup.bat` installs everything for all three into `%LOCALAPPDATA%\buoay-app\venv` (outside OneDrive, one per computer). Run it again on any machine that was set up before (it also removes the old `.venv` from this folder).

# VGGT smoke test

Checks whether VGGT (image -> 3D model) runs on this computer's GPU, and how much memory and time it needs.

## 1. One-time setup

1. Update the NVIDIA driver (GeForce Experience or nvidia.com/drivers). No NVIDIA GPU? Skip this; setup installs the CPU version instead.
2. Install **Python 3.11** from python.org. On the first installer screen, tick **"Add python.exe to PATH"**. (Python from MSYS2 / MinGW / Git Bash will not work: PyTorch doesn't support it. `setup.bat` checks for this.)
3. Double-click **`setup.bat`** in this folder. It downloads ~2.5 GB and takes a while. At the end it should print `CUDA GPU found: NVIDIA GeForce ...` with your card's name.

## 2. Take test photos

Put **6-10 photos** in `test_images`. How you take them matters more than anything else:

- Pick a wall with **texture**: brick, a bookshelf, posters, a whiteboard with writing. A blank painted wall will fail.
- Stand ~1-2 m away and **step sideways ~30 cm between shots**, keeping the camera pointed at the wall. Each photo should overlap the previous one by about two thirds.
- Hold the phone **sideways (landscape)**. Portrait photos use ~25% more GPU memory.
- Hold still so photos are sharp. Normal phone photos are fine; they get resized to 518 px wide.

## 3. Run it

Double-click **`run_test.bat`**. The first run downloads the model weights (~5 GB). When it finishes it prints the results and opens a 3D preview in the browser.

To pass options, run from a terminal in this folder:

```
run_test.bat --frames 4     # fewer photos if it runs out of GPU memory
run_test.bat --cpu          # no GPU, slow but always works (compare quality)
run_test.bat --conf 70      # keep only the 30% most confident points (less noise)
```

## What to send back

The **RESULTS** block it prints. The numbers that matter:

- **Peak GPU memory:** how close to the 4 GB limit we are. That decides how many photos per batch the real pipeline can use.
- **Inference time:** how long each batch takes.
- **Flatness check:** for a flat wall, a few % or less means the geometry is right.
- **Any warning about NaN points:** that would mean float16 isn't working well on this GPU.

A screenshot of `output/preview.html` helps too.
