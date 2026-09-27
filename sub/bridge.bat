@echo off
REM Double-click on a Windows laptop joined to the sub's Wi-Fi (the Pixel hotspot). It asks for the dashboard link.
where python >nul 2>nul
if %errorlevel%==0 (python "%~dp0bridge.py" %*) else (py "%~dp0bridge.py" %*)
pause
