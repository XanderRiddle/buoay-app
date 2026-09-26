@echo off
REM Starts everything on this laptop: the web server and the 3D reconstruction worker.
cd /d "%~dp0"
where node >nul 2>nul || (echo Node.js not found. Install it from nodejs.org, then run this again. & pause & exit /b 1)
if not exist processing\.venv\Scripts\python.exe (echo Run processing\setup.bat first. & pause & exit /b 1)
call npm install --no-audit --no-fund || (pause & exit /b 1)
start "buoay server" cmd /k "npm start"
timeout /t 3 /nobreak >nul
start "buoay 3D worker" cmd /k "cd processing && .venv\Scripts\python.exe worker.py"
echo.
echo Two windows opened: the server (shows the phone link) and the 3D worker.
echo Dashboard: https://localhost:8443/dashboard
timeout /t 5 >nul
start "" https://localhost:8443/dashboard
