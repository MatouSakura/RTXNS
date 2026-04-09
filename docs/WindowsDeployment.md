# Windows Deployment Guide

This guide is for a clean Windows machine that needs to clone, initialize submodules, build, and run `RTXNS` with the helper scripts included in this repository.

## 1. Install prerequisites

Install the following first:

- `Git`
- `Visual Studio 2022` with the `Desktop development with C++` workload
- `CMake 3.24+`
- `DXC`
- `Slang`

Recommended local tool paths used by the scripts in this repository:

- `D:\Tools\DXC\dxc_2025_05_24\bin\x64\dxc.exe`
- `D:\Tools\Slang\slang-2025.19.1-windows-x86_64\bin\slangc.exe`

If you install them somewhere else, either:

- edit the `DXC_PATH` and `SLANG_PATH` lines at the top of [build-windows.cmd](../build-windows.cmd), or
- set `DXC_PATH` and `SLANG_PATH` as environment variables before running the script.

PowerShell example:

```powershell
$env:DXC_PATH="D:\Tools\DXC\dxc_2025_05_24\bin\x64\dxc.exe"
$env:SLANG_PATH="D:\Tools\Slang\slang-2025.19.1-windows-x86_64\bin\slangc.exe"
.\build-windows.cmd
```

## 2. Clone the repository

Preferred command:

```powershell
git clone --recursive https://github.com/MatouSakura/RTXNS.git
cd RTXNS
```

If the repository was already cloned without `--recursive`, run:

```powershell
.\setup-submodules.cmd
```

That script runs:

```powershell
git submodule sync --recursive
git submodule update --init --recursive
```

## 3. Build with one click

From the repository root:

```powershell
.\build.bat
```

Or, if you prefer the explicit script name:

```powershell
.\build-windows.cmd
```

What the script does:

- verifies `cmake` exists
- checks whether submodules are present and initializes them automatically if needed
- uses your local `DXC_PATH` and `SLANG_PATH`
- runs CMake configure
- runs a `Release` build in parallel

## 4. Build manually

If you prefer direct CMake commands:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DSHADERMAKE_FIND_DXC=OFF `
  -DSHADERMAKE_DXC_PATH="D:\Tools\DXC\dxc_2025_05_24\bin\x64\dxc.exe" `
  -DSHADERMAKE_FIND_SLANG=OFF `
  -DSHADERMAKE_SLANG_PATH="D:\Tools\Slang\slang-2025.19.1-windows-x86_64\bin\slangc.exe"

cmake --build build --config Release --parallel
```

## 5. Output location

After a successful build, binaries are placed in:

```text
bin\windows-x64
```

Common sample executables include:

- `SimpleInferencing.exe`
- `SimpleTraining.exe`
- `ShaderTraining.exe`
- `SlangpyInferencing.exe`
- `SlangpyTraining.exe`

Example:

```powershell
cd .\bin\windows-x64
.\SimpleInferencing.exe
```

## 6. Rebuild after editing code or shaders

The simplest way is to run the build script again:

```powershell
.\build-windows.cmd
```

Or rebuild directly:

```powershell
cmake --build build --config Release --parallel
```

## 7. Common issues

### Submodules are missing

Run:

```powershell
.\setup-submodules.cmd
```

### `DXC not found` or `Slang not found`

Update the paths at the top of [build-windows.cmd](../build-windows.cmd), or set:

- `DXC_PATH`
- `SLANG_PATH`

### CMake configure reused old cache data

Delete the `build` directory and build again:

```powershell
Remove-Item -Recurse -Force .\build
.\build-windows.cmd
```
