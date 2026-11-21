@echo off
cd /d "%~dp0"
python "mechi\app.py"
if errorlevel 1 pause
