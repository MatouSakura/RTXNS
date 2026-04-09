@echo off
setlocal

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

call "%REPO_ROOT%\build-windows.cmd"
exit /b %errorlevel%
