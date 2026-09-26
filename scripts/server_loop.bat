@echo off
REM Keeps the web server (and tunnel) running; restarts it if it crashes.
cd /d "%~dp0\.."
:loop
node server.js %BUOAY_TUNNEL%
echo.
echo Server stopped. Restarting in 5 seconds... (close this window to stop for good)
timeout /t 5 /nobreak >nul
goto loop
