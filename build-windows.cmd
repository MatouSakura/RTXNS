@echo off
setlocal

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

set "BUILD_DIR=%REPO_ROOT%\build"
set "GENERATOR=Visual Studio 17 2022"
set "PLATFORM=x64"
set "CONFIG=Release"
set "SUBMODULE_SETUP=%REPO_ROOT%\setup-submodules.cmd"

rem You can either edit these defaults or override them with environment variables.
if not defined DXC_PATH set "DXC_PATH=D:\Tools\DXC\dxc_2025_05_24\bin\x64\dxc.exe"
if not defined SLANG_PATH set "SLANG_PATH=D:\Tools\Slang\slang-2025.19.1-windows-x86_64\bin\slangc.exe"

where cmake >nul 2>nul
if errorlevel 1 (
    echo [ERROR] CMake not found in PATH.
    goto :fail
)

if not exist "%REPO_ROOT%\external\donut\CMakeLists.txt" goto :init_submodules
if not exist "%REPO_ROOT%\external\donut\nvrhi\CMakeLists.txt" goto :init_submodules
if not exist "%REPO_ROOT%\external\implot\implot.cpp" goto :init_submodules
goto :submodules_ready

:init_submodules
if not exist "%SUBMODULE_SETUP%" (
    echo [ERROR] Submodule bootstrap script not found:
    echo         %SUBMODULE_SETUP%
    goto :fail
)

echo [0/2] Submodules are missing. Running setup-submodules.cmd...
call "%SUBMODULE_SETUP%"
if errorlevel 1 goto :fail

:submodules_ready

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
cmake --build "%BUILD_DIR%" --config %CONFIG% --parallel --clean-first
if errorlevel 1 goto :build_fail

echo.
echo Build completed successfully.
echo Binaries:
echo   %REPO_ROOT%\bin\windows-x64
echo.
echo This build uses --clean-first so imported Slang shader changes are always picked up.
echo To rebuild after shader or source changes, run this script again.
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
