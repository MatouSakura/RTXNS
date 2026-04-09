# Windows 部署指南

这份指南面向一台全新的 Windows 机器，目标是把 `RTXNS` 克隆下来、初始化子模块、完成构建，并通过仓库内置脚本直接运行。

## 1. 安装前置依赖

请先安装以下工具：

- `Git`
- `Visual Studio 2022`，并勾选 `Desktop development with C++`
- `CMake 3.24+`
- `DXC`
- `Slang`

当前仓库脚本默认使用的本地工具路径为：

- `D:\Tools\DXC\dxc_2025_05_24\bin\x64\dxc.exe`
- `D:\Tools\Slang\slang-2025.19.1-windows-x86_64\bin\slangc.exe`

如果你把工具安装在别的位置，可以：

- 直接修改 [build-windows.cmd](../build-windows.cmd) 顶部的 `DXC_PATH` 和 `SLANG_PATH`
- 或者在运行脚本前，通过环境变量覆盖

PowerShell 示例：

```powershell
$env:DXC_PATH="D:\Tools\DXC\dxc_2025_05_24\bin\x64\dxc.exe"
$env:SLANG_PATH="D:\Tools\Slang\slang-2025.19.1-windows-x86_64\bin\slangc.exe"
.\build-windows.cmd
```

## 2. 克隆仓库

推荐命令：

```powershell
git clone --recursive https://github.com/MatouSakura/RTXNS.git
cd RTXNS
```

如果仓库之前已经 clone 过，但没有加 `--recursive`，请运行：

```powershell
.\setup-submodules.cmd
```

它内部执行的是：

```powershell
git submodule sync --recursive
git submodule update --init --recursive
```

## 3. 一键构建

在仓库根目录执行：

```powershell
.\build.bat
```

如果你更喜欢显式脚本名，也可以执行：

```powershell
.\build-windows.cmd
```

脚本会自动完成以下工作：

- 检查 `cmake` 是否在 `PATH` 中
- 检查子模块是否存在，如果缺失就自动初始化
- 使用你本地的 `DXC_PATH` 和 `SLANG_PATH`
- 执行 CMake 配置
- 并行构建 `Release` 版本

## 4. 手动使用 CMake 构建

如果你想自己执行命令行构建，可以使用：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DSHADERMAKE_FIND_DXC=OFF `
  -DSHADERMAKE_DXC_PATH="D:\Tools\DXC\dxc_2025_05_24\bin\x64\dxc.exe" `
  -DSHADERMAKE_FIND_SLANG=OFF `
  -DSHADERMAKE_SLANG_PATH="D:\Tools\Slang\slang-2025.19.1-windows-x86_64\bin\slangc.exe"

cmake --build build --config Release --parallel
```

## 5. 输出目录

构建成功后，可执行文件默认位于：

```text
bin\windows-x64
```

常见 sample 可执行文件包括：

- `SimpleInferencing.exe`
- `SimpleTraining.exe`
- `ShaderTraining.exe`
- `SlangpyInferencing.exe`
- `SlangpyTraining.exe`

例如：

```powershell
cd .\bin\windows-x64
.\SimpleInferencing.exe
```

你也可以直接使用仓库根目录里的启动脚本：

```powershell
.\run-simple-inferencing.bat
.\run-simple-training.bat
.\run-shader-training.bat
```

如果想带参数启动 sample，比如 `-dx12` 或 `-vk`：

```powershell
.\run-sample.bat SimpleInferencing -vk
```

## 6. 修改代码或 shader 后如何重编译

最简单的方式就是再次运行：

```powershell
.\build-windows.cmd
```

或者直接重建：

```powershell
cmake --build build --config Release --parallel
```

## 7. 常见问题

### 子模块缺失

运行：

```powershell
.\setup-submodules.cmd
```

### 提示 `DXC not found` 或 `Slang not found`

修改 [build-windows.cmd](../build-windows.cmd) 顶部路径，或者设置：

- `DXC_PATH`
- `SLANG_PATH`

### CMake 缓存脏了，重复使用了旧配置

删除 `build` 目录后重新构建：

```powershell
Remove-Item -Recurse -Force .\build
.\build-windows.cmd
```
