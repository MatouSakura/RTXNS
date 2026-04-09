@echo off
setlocal

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

set "BUILD_DIR=%REPO_ROOT%\build"
set "GENERATOR=Visual Studio 17 2022"
set "PLATFORM=x64"
set "CONFIG=Release"

rem Update these two paths if your local tool locations change.
set "DXC_PATH=D:\Tools\DXC\dxc_2025_05_24\bin\x64\dxc.exe"
set "SLANG_PATH=D:\Tools\Slang\slang-2025.19.1-windows-x86_64\bin\slangc.exe"

if not exist "%DXC_PATH%" (
    echo [ERROR] DXC not found:
    echo         %DXC_PATH%
    goto :fail
)

if not exist "%SLANG_PATH%" (
    echo [ERROR] Slang not found:
    echo         %SLANG_PATH%
    goto :fail
)

pushd "%REPO_ROOT%"

echo [1/2] Configuring RTXNS...
cmake -S "%REPO_ROOT%" -B "%BUILD_DIR%" -G "%GENERATOR%" -A %PLATFORM% ^
  -DSHADERMAKE_FIND_DXC=OFF ^
  -DSHADERMAKE_DXC_PATH="%DXC_PATH%" ^
  -DSHADERMAKE_FIND_SLANG=OFF ^
  -DSHADERMAKE_SLANG_PATH="%SLANG_PATH%"
if errorlevel 1 goto :cmake_fail

echo [2/2] Building RTXNS...
cmake --build "%BUILD_DIR%" --config %CONFIG% --parallel
if errorlevel 1 goto :build_fail

echo.
echo Build completed successfully.
echo Binaries:
echo   %REPO_ROOT%\bin\windows-x64
popd
exit /b 0

:cmake_fail
echo.
echo [ERROR] CMake configure failed.
popd
goto :fail

:build_fail
echo.
echo [ERROR] Build failed.
popd
goto :fail

:fail
echo.
pause
exit /b 1
