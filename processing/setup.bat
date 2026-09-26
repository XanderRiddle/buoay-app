@echo off
REM One-time setup for the VGGT processing test. Run from this folder.
REM Safe to rerun: it rebuilds .venv from scratch each time (downloads are cached).
cd /d "%~dp0"
if not exist test_images mkdir test_images

REM --- Pick a python.org Python (3.12 / 3.11 / 3.13 preferred via the py launcher) ---
set "PY="
where py >nul 2>nul && (
  py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
)
if not defined PY where py >nul 2>nul && py -3.11 -c "import sys" >nul 2>nul && set "PY=py -3.11"
if not defined PY where py >nul 2>nul && py -3.13 -c "import sys" >nul 2>nul && set "PY=py -3.13"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto :nopython
%PY% -c "import sys; sys.exit(0 if 'MSC' in sys.version else 1)" || goto :mingw
echo Using Python: %PY%
%PY% --version

REM --- Fresh virtual environment ---
if exist .venv (
  echo Removing old .venv...
  rmdir /s /q .venv
)
%PY% -m venv .venv || goto :fail
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

REM --- GPU build of PyTorch if an NVIDIA driver is present, otherwise the CPU build ---
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
where nvidia-smi >nul 2>nul && set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
echo.
echo Installing PyTorch from %TORCH_INDEX% ...
pip install torch torchvision --index-url %TORCH_INDEX% || goto :fail

REM --- VGGT. Its package pins numpy<2, which has no prebuilt Windows files for newer
REM     Python and ends up compiled with the wrong compiler. VGGT runs fine on numpy 2,
REM     so install it without its pins, and never compile anything from source. ---
echo.
echo Installing VGGT...
pip install --only-binary=:all: "numpy>=2" Pillow huggingface_hub einops safetensors opencv-python websockets || goto :fail
pip install --no-deps https://github.com/facebookresearch/vggt/archive/refs/heads/main.zip || goto :fail
echo.
python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__); ok=torch.cuda.is_available(); print('CUDA GPU found:', torch.cuda.get_device_name(0) if ok else 'none, will run on CPU')" || goto :fail
echo.
echo Setup done. Put 6-10 photos in the test_images folder, then run run_test.bat
pause
exit /b 0

:nopython
echo Python not found. Install Python 3.12 from python.org and tick "Add python.exe to PATH".
pause
exit /b 1

:mingw
echo The Python on your PATH is an MSYS2 / MinGW build, which PyTorch does not support.
echo Install Python 3.12 from python.org, then run setup.bat again.
pause
exit /b 1

:fail
echo.
echo Setup failed, see the error above.
pause
exit /b 1
