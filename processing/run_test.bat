@echo off
REM Runs the VGGT smoke test. Extra arguments pass through, e.g.  run_test.bat --frames 4
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo Run setup.bat first. & pause & exit /b 1)
python vggt_smoke_test.py %*
if exist output\preview.html start "" output\preview.html
if exist output\results.txt start "" notepad output\results.txt
pause
