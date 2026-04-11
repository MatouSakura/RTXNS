@echo off
setlocal

if "%~1"=="" (
    echo Usage: run-sample.bat ^<SampleName^> [sample arguments]
    echo.
    echo Available samples:
    echo   SimpleInferencing
    echo   SimpleTraining
    echo   ShaderTraining
    echo   VolumetricCloudTraining
    echo   SlangpyInferencing
    echo   SlangpyTraining
    exit /b 1
)

set "SAMPLE_NAME=%~1"
shift

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

set "BIN_DIR=%REPO_ROOT%\bin\windows-x64"
set "SAMPLE_EXE=%BIN_DIR%\%SAMPLE_NAME%.exe"

if not exist "%SAMPLE_EXE%" (
    echo [ERROR] Sample executable not found:
    echo         %SAMPLE_EXE%
    echo.
    echo Build the project first with:
    echo   build.bat
    pause
    exit /b 1
)

echo Launching %SAMPLE_NAME%...
start "" /D "%BIN_DIR%" "%SAMPLE_EXE%" %*
exit /b 0
