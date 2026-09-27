@echo off
REM Starts buoay-app on this computer: web server + 3D worker (+ Cloudflare tunnel).
REM   start.bat          reachable from anywhere through a free Cloudflare tunnel
REM   start.bat local    same Wi-Fi only (no tunnel)
REM Both windows restart themselves if they crash. Close them to stop.
cd /d "%~dp0"
where node >nul 2>nul || (echo Node.js not found. Install it from nodejs.org, then run this again. & pause & exit /b 1)
if not exist "%LOCALAPPDATA%\buoay-app\venv\Scripts\python.exe" (echo Run processing\setup.bat first. & pause & exit /b 1)
call npm install --no-audit --no-fund || (pause & exit /b 1)
REM newer versions need scipy (revisit alignment); install it if an older setup lacks it
"%LOCALAPPDATA%\buoay-app\venv\Scripts\python.exe" -c "import scipy" >nul 2>nul || "%LOCALAPPDATA%\buoay-app\venv\Scripts\python.exe" -m pip install --only-binary=:all: scipy

set "BUOAY_TUNNEL=--tunnel"
if /i "%~1"=="local" set "BUOAY_TUNNEL="
if defined BUOAY_TUNNEL if not exist tools\cloudflared.exe (
  echo Downloading cloudflared, the Cloudflare tunnel program...
  if not exist tools mkdir tools
  powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing -Uri https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe -OutFile tools\cloudflared.exe" || (echo Download failed. & pause & exit /b 1)
)

start "buoay server" cmd /c scripts\server_loop.bat
timeout /t 4 /nobreak >nul
start "buoay 3D worker" cmd /c scripts\worker_loop.bat
timeout /t 2 /nobreak >nul
set /p KEY=<certs\access-key.txt
start "" "https://localhost:8443/dashboard?key=%KEY%"
echo.
echo Running. Links for your phone and laptop are in LINKS.txt (it syncs through OneDrive).
echo The public link appears there about 10 seconds after start.
timeout /t 8 >nul
