# RTX Neural Shading：快速开始指南

RTX Neural Shading 可以在 Windows 和 Linux 上构建并运行。

## 构建步骤

1. 递归克隆项目：

   ```sh
   git clone --recursive https://github.com/MatouSakura/RTXNS.git
   ```

2. 创建构建目录：

   ```sh
   cd RTXNS
   mkdir build
   ```

3. 使用你偏好的 CMake 生成器配置项目：

   ```sh
   cmake -S . -B build -G <generator>
   ```

   如果要启用 DX12 Cooperative Vector 预览功能，请打开 `ENABLE_DX12_COOP_VECTOR_PREVIEW` 选项（仅 Windows）：

   ```sh
   cmake -DENABLE_DX12_COOP_VECTOR_PREVIEW=ON
   ```

4. 打开 `build/RtxNeuralShading.sln`，在 Visual Studio 中构建所有项目；也可以直接使用 CMake 命令行：

   ```sh
   cmake --build build --config Release
   ```

5. 所有 sample 的可执行文件都会输出到 `/bin` 下，例如：

   ```sh
   bin/<platform>/SimpleInferencing
   ```

6. 支持的平台上，sample 可以通过命令行参数选择 DX12 或 Vulkan：

   - `-dx12`
   - `-vk`

## 说明

所有 sample 都使用 Slang 编写，并可分别编译到 DX12 或 Vulkan：

- [DirectX Preview Agility SDK](https://devblogs.microsoft.com/directx/directx12agility/)
- [Vulkan Cooperative Vector extension](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_NV_cooperative_vector.html)

## 驱动要求

- 使用 DirectX Preview Agility SDK 时，需要 Shader Model 6.9 Preview 驱动：
  - [GeForce](https://developer.nvidia.com/downloads/shadermodel6-9-preview-driver)
  - [Quadro](https://developer.nvidia.com/downloads/assets/secure/shadermodel6-9-preview-driver-quadro)
- 使用 Vulkan Cooperative Vector extension 时，需要 R570 及以上正式驱动：
  - [NVIDIA Driver](https://www.nvidia.com/en-gb/geforce/drivers)

### 示例

| 示例名称 | 结果图 | 说明 |
| -------- | ------ | ---- |
| [Simple Inferencing](SimpleInferencing.md) | [<img src="simple_inferencing.png" width="800">](simple_inferencing.png) | 展示如何使用 RTXNS 的底层构件实现一个推理 shader。该 sample 会从文件加载训练好的网络，并用它近似 Disney BRDF shader。运行时可以交互调节光源方向和部分材质参数。 |
| [Simple Training](SimpleTraining.md) | [<img src="simple_training.png" width="800">](simple_training.png) | 在 Simple Inferencing 基础上进一步展示如何训练一个可用于 shader 的神经网络。这个 sample 的目标是拟合一张经过变换的纹理。 |
| [Shader Training](ShaderTraining.md) | [<img src="shader_training.png" width="800">](shader_training.png) | 在 Simple Training 基础上引入 Slang AutoDiff 和完整的 MLP 抽象。这个 sample 使用之前介绍过的 `CoopVector` 训练代码来训练一个近似 Disney BRDF 的模型。 |
| [SlangPy Training](SlangpyTraining.md) | [<img src="slangpy_training.jpg" width="800">](slangpy_training.jpg) | 展示如何在 Python 中借助 SlangPy 训练不同的网络结构。你可以不改 C++ 代码就试验不同网络、编码方式和训练策略，并把结果导出给 C++ 侧加载。 |
| [SlangPy Inferencing](SlangpyInferencing.md) | [<img src="slangpy_inferencing_window.png" width="800">](slangpy_inferencing_window.png) | 展示如何先在 Python + SlangPy 中做神经网络推理，再把同一套 Slang 实现迁移到 C++。这样可以先快速原型验证，再落到生产环境。 |

### 教程

- [Tutorial](Tutorial.md)
  基于 [Shader Training](ShaderTraining.md) 示例，讲解如何开始编写你自己的 neural shader。

### 库文档

- [Library](LibraryGuide.md)
  讲解如何使用库里的辅助函数来创建、管理和运行神经网络。
