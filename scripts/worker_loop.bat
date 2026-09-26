@echo off
REM Keeps the 3D reconstruction worker running; restarts it if it crashes.
cd /d "%~dp0\..\processing"
:loop
"%LOCALAPPDATA%\buoay-app\venv\Scripts\python.exe" worker.py
echo.
echo Worker stopped. Restarting in 5 seconds... (close this window to stop for good)
timeout /t 5 /nobreak >nul
goto loop
