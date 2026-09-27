@echo off
REM One-time setup for the VGGT processing test. Run from this folder.
REM Safe to rerun: it rebuilds the environment from scratch each time (downloads are cached).
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
REM Lives outside OneDrive (per computer) so gigabytes of PyTorch don't sync between machines.
set "VENV=%LOCALAPPDATA%\buoay-app\venv"
if exist .venv (
  echo Removing the old .venv inside OneDrive. It now lives in %VENV%
  rmdir /s /q .venv
)
if exist "%VENV%" (
  echo Removing old environment...
  rmdir /s /q "%VENV%"
)
%PY% -m venv "%VENV%" || goto :fail
call "%VENV%\Scripts\activate.bat"
python -m pip install --upgrade pip

REM --- GPU build of PyTorch if an NVIDIA driver is present, otherwise the CPU build ---
REM Checks Windows' own list of graphics cards, so it works even when nvidia-smi isn't on PATH.
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
where nvidia-smi >nul 2>nul && set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
powershell -NoProfile -Command "if ((Get-CimInstance Win32_VideoController).Name -match 'NVIDIA') { exit 0 } else { exit 1 }" >nul 2>nul && set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
if /i "%~1"=="cpu" set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
echo.
echo Installing PyTorch from %TORCH_INDEX% ...
pip install torch torchvision --index-url %TORCH_INDEX% || goto :fail

REM --- VGGT. Its package pins numpy<2, which has no prebuilt Windows files for newer
REM     Python and ends up compiled with the wrong compiler. VGGT runs fine on numpy 2,
REM     so install it without its pins, and never compile anything from source. ---
echo.
echo Installing VGGT...
pip install --only-binary=:all: "numpy>=2" Pillow huggingface_hub einops safetensors opencv-python websockets scipy || goto :fail
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
