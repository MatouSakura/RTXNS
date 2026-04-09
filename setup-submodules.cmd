@echo off
setlocal

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git not found in PATH.
    goto :fail
)

pushd "%REPO_ROOT%"
if errorlevel 1 goto :fail

echo [1/2] Syncing submodule URLs...
git submodule sync --recursive
if errorlevel 1 goto :sync_fail

echo [2/2] Initializing/updating submodules...
git submodule update --init --recursive
if errorlevel 1 goto :update_fail

echo.
echo Submodules are ready.
popd
exit /b 0

:sync_fail
echo.
echo [ERROR] Failed to sync submodule URLs.
popd
goto :fail

:update_fail
echo.
echo [ERROR] Failed to initialize or update submodules.
popd
goto :fail

:fail
echo.
pause
exit /b 1
