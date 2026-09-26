@echo off
REM One-time setup for the VGGT processing test. Run from this folder.
cd /d "%~dp0"
if not exist test_images mkdir test_images
where python >nul 2>nul || (echo Python not found. Install Python 3.11 from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
python -m venv .venv || (pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo.
echo Installing PyTorch with CUDA (about 2.5 GB download)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || (pause & exit /b 1)
echo.
echo Installing VGGT...
pip install https://github.com/facebookresearch/vggt/archive/refs/heads/main.zip || (pause & exit /b 1)
echo.
python -c "import torch; ok=torch.cuda.is_available(); print('CUDA GPU found:', torch.cuda.get_device_name(0) if ok else 'NO - update the NVIDIA driver, then rerun setup.bat')"
echo.
echo Setup done. Put 6-10 photos in the test_images folder, then run run_test.bat
pause
