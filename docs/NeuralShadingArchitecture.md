# RTX Neural Shading: 当前架构与运行流程说明

这份文档面向当前 `RTXNS` 仓库，解释它现在是如何实现 neural shading 的，CPU 和 GPU 之间的数据怎么流动，各个 sample 是怎么接到同一套运行时上的，以及如果你想自己训练模型，应该从哪里开始改。

## 1. 一句话概览

当前的 `RTXNS` 不是一个通用深度学习框架，而是一个面向图形渲染的小型神经网络运行时。它的目标是把紧凑的 MLP 直接嵌进 shader 里，用 cooperative vector 硬件路径做推理或训练，然后再把输出接回正常的图形管线。

整体流程可以概括为 5 步：

1. CPU 侧创建或加载一个网络，使用可移植的 host layout
2. CPU 侧把它转换成 GPU 更适合执行的 device layout
3. shader 从 packed raw buffer 里按 offset 读取权重和偏置
4. GPU 在 compute 或 graphics pass 中执行推理或训练
5. 如果需要保存模型，再把 device layout 转回 host layout 写回磁盘

## 2. 项目里主要分成哪几层

当前仓库大致可以分成 4 层。

### 应用层

应用层基于 `donut` 和 `nvrhi`，负责：

- 创建设备和窗口
- 创建 pipeline、buffer、texture、binding set
- 组织每帧 command list
- 提供 ImGui / ImPlot UI

代表文件：

- [`samples/SimpleInferencing/SimpleInferencing.cpp`](../samples/SimpleInferencing/SimpleInferencing.cpp)
- [`samples/SimpleTraining/SimpleTraining.cpp`](../samples/SimpleTraining/SimpleTraining.cpp)
- [`samples/ShaderTraining/ShaderTraining.cpp`](../samples/ShaderTraining/ShaderTraining.cpp)

### Host 侧神经网络运行时

这一层是 C++ 代码，负责：

- 描述网络结构
- 计算每层的权重和偏置大小
- 计算每层在大 buffer 里的 offset
- 加载和保存网络文件
- 在 host layout 和 device layout 之间做转换

代表文件：

- [`src/NeuralShading/NeuralNetwork.h`](../src/NeuralShading/NeuralNetwork.h)
- [`src/NeuralShading/NeuralNetwork.cpp`](../src/NeuralShading/NeuralNetwork.cpp)
- [`src/NeuralShading/NeuralNetworkTypes.h`](../src/NeuralShading/NeuralNetworkTypes.h)

### Shader 侧神经网络运行时

这一层是 Slang shader 库，负责：

- 激活函数
- 线性层
- loss
- optimizer
- 通用 `InferenceMLP`
- 通用 `TrainingMLP`

代表文件：

- [`src/NeuralShading_Shaders/MLP.slang`](../src/NeuralShading_Shaders/MLP.slang)
- [`src/NeuralShading_Shaders/LinearOps.slang`](../src/NeuralShading_Shaders/LinearOps.slang)
- [`src/NeuralShading_Shaders/Activation.slang`](../src/NeuralShading_Shaders/Activation.slang)
- [`src/NeuralShading_Shaders/Loss.slang`](../src/NeuralShading_Shaders/Loss.slang)
- [`src/NeuralShading_Shaders/Optimizers.slang`](../src/NeuralShading_Shaders/Optimizers.slang)

### 结果统计和工具层

这一层负责：

- 把 batch loss 在 GPU 上做 reduction
- 把统计结果回读到 CPU
- 显示训练曲线和训练状态

代表文件：

- [`src/Utils/ProcessTrainingResults.slang`](../src/Utils/ProcessTrainingResults.slang)
- [`src/Utils/ResultsReadbackHandler.cpp`](../src/Utils/ResultsReadbackHandler.cpp)
- [`src/Utils/ResultsWidget.cpp`](../src/Utils/ResultsWidget.cpp)

## 3. 神经部分依赖什么 GPU 能力

`RTXNS` 当前依赖的是 `nvrhi` 暴露出来的 cooperative vector 路径。

启动时主要做 3 件事：

1. 为 Vulkan 或 DX12 打开 cooperative vector 相关扩展或实验特性
2. 用这些要求创建图形设备
3. 查询当前 GPU 是否真的支持当前 sample 需要的推理或训练格式

关键文件：

- [`src/Utils/DeviceUtils.cpp`](../src/Utils/DeviceUtils.cpp)
- [`src/NeuralShading/GraphicsResources.cpp`](../src/NeuralShading/GraphicsResources.cpp)

这意味着当前项目默认假设：

- GPU 支持 cooperative vector 推理和/或训练
- FP16 路径可用
- Slang 编译出来的 shader 能落到当前后端

如果这些能力缺失，sample 会直接退出，不会自动回退到 CPU 训练。

## 4. 网络在当前仓库里是怎么表示的

当前运行时有一个非常关键的区分：

- host layout
- device layout

### Host layout

Host layout 由下面几个结构描述：

- `NetworkArchitecture`
- `NetworkLayer`
- `NetworkLayout`

它通常是 `RowMajor` 的，适合：

- 初始化新网络
- 从磁盘读取网络
- 把训练结果写回磁盘

它是当前仓库里的“可移植格式”。

### Device layout

Device layout 主要有两种：

- `InferencingOptimal`
- `TrainingOptimal`

这两种 layout 是 GPU 友好的，但不保证跨设备、跨 API 可移植。也就是说：

- 文件里应该保存 host layout
- GPU 上执行时应该用 device layout

这就是当前代码里最重要的一个设计点：

1. host layout 负责可读写、可移植
2. device layout 负责执行效率
3. `NetworkUtilities::ConvertWeights` 负责桥接两者

## 5. 权重和偏置是怎么打包的

每一层都有两块数据：

- 权重矩阵
- 偏置向量

Host 侧会为每层计算：

- `weightSize`
- `biasSize`
- `weightOffset`
- `biasOffset`

并且按照 cooperative vector 对齐要求进行 padding。

这些 offset 会被写进 constant buffer，然后传给 shader。shader 自己不会“猜”布局，而是按 C++ 提前算好的偏移去 raw buffer 里读。

所以当前代码的典型模式都是：

1. 先构建 `NetworkLayout`
2. 再转换成 device-optimal layout
3. 上传一整块 packed 参数 buffer
4. 把每层 weight / bias offset 一起传给 shader

## 6. Shader 里真正执行了什么

当前 shader 侧的核心抽象是两个：

- `InferenceMLP`
- `TrainingMLP`

### `InferenceMLP`

只做前向传播：

1. 从 packed buffer 读取矩阵和偏置
2. 执行 `LinearOp`
3. 执行激活函数
4. 多层重复
5. 输出最终向量

### `TrainingMLP`

在前向传播基础上，再加反向传播。

当前实现里比较有代表性的一点是：

- 不是手写整套 backward graph
- 前向函数被标记为可微
- Slang autodiff 自动推出 backward 路径
- 梯度最终写进一个专门的梯度 buffer

这就是当前 `RTXNS` 把“神经网络训练”塞进 shader 的关键做法。

## 7. 底层数学路径是怎么铺起来的

从最底层看，一个神经层就是：

- 矩阵乘
- 加偏置
- 过激活函数

在当前仓库里，这几层是这么叠起来的：

1. cooperative vector 原语
2. `LinearOps.slang` 对这些原语做一层封装
3. `MLP.slang` 把多层前向和训练逻辑组织起来
4. 各个 sample 在上面定义自己的输入、target、loss 和可视化方式

也就是说，当前项目不是“每个 sample 都从零写一遍神经网络”，而是：

- 共享一套底层神经 runtime
- 每个 sample 只改具体任务

## 8. 当前设计里为什么大量使用频率编码

当前 sample 并不是直接把低维输入原封不动送进 MLP，而是经常先做 frequency encoding。

例子：

- `SimpleInferencing` 从 5 个 BRDF 输入扩展成 `5 * 6 = 30`
- `SimpleTraining` 从 2 个 UV 输入扩展成 `2 * 6 = 12`
- `ShaderTraining` 也是先编码 5 个 shading 标量再送进网络

原因很直接：当前网络都比较小，频率编码能显著提高对高频函数的表达能力和收敛速度。

## 9. 当前仓库里的 sample 梯度

当前 sample 的难度和抽象层次是逐级递进的。

### `SimpleInferencing`

用途：

- 展示最纯粹的神经推理路径

当前流程：

1. 从 `assets/data/disney.ns.bin` 加载网络
2. 构建 host-side network
3. 转换成 `InferencingOptimal`
4. 上传 packed 参数 buffer
5. 把 weight / bias offset 传给 pixel shader
6. 在 shader 里用 MLP 替换掉 Disney BRDF 的核心部分
7. 把 MLP 输出重新组合成最终材质响应

这里最重要的点是：

神经网络并没有替换整个 renderer，只是替换了一个局部 shading 子问题。

当前 sample 的网络大致是：

- 输入：`NdotL`、`NdotV`、`NdotH`、`LdotH`、`roughness`
- 编码后输入宽度：`30`
- hidden width：`32`
- 输出：`float4`
- 当前代码假定有 4 次 transition，也就是 3 层 hidden 加 1 层输出

关键文件：

- [`samples/SimpleInferencing/SimpleInferencing.cpp`](../samples/SimpleInferencing/SimpleInferencing.cpp)
- [`samples/SimpleInferencing/SimpleInferencing.slang`](../samples/SimpleInferencing/SimpleInferencing.slang)
- [`samples/SimpleInferencing/NetworkConfig.h`](../samples/SimpleInferencing/NetworkConfig.h)

### `SimpleTraining`

用途：

- 展示最容易理解的一整套 GPU 训练流程

当前 target：

- 输入纹理经过某种变换之后的 RGB

当前流程：

1. 根据 `NetworkConfig.h` 创建一个新网络
2. 转成 `TrainingOptimal`
3. 分配参数、梯度、Adam moments、随机状态、loss 等 GPU buffer
4. training shader 随机采样 UV
5. 网络根据编码后的 UV 预测 RGB
6. shader 从变换后的纹理里取 ground truth RGB
7. 把梯度累加到 gradient buffer
8. optimizer pass 执行 Adam 更新
9. reduction pass 统计 epoch loss
10. inference pass 输出当前网络预测图像，作为训练过程预览

当前网络配置：

- 编码后输入：`12`
- hidden layers：`4`
- hidden width：`64`
- 输出：`3`

它是当前仓库里最适合先上手理解训练闭环的 sample。

关键文件：

- [`samples/SimpleTraining/SimpleTraining.cpp`](../samples/SimpleTraining/SimpleTraining.cpp)
- [`samples/SimpleTraining/SimpleTraining_Training.slang`](../samples/SimpleTraining/SimpleTraining_Training.slang)
- [`samples/SimpleTraining/SimpleTraining_Optimizer.slang`](../samples/SimpleTraining/SimpleTraining_Optimizer.slang)
- [`samples/SimpleTraining/SimpleTraining_Inference.slang`](../samples/SimpleTraining/SimpleTraining_Inference.slang)
- [`samples/SimpleTraining/NetworkConfig.h`](../samples/SimpleTraining/NetworkConfig.h)

### `ShaderTraining`

用途：

- 展示最贴近“神经着色”本意的 sample

当前 target：

- Disney BRDF 核心着色函数

当前流程：

1. 在 GPU 上随机生成一批合法的光照和视角输入
2. 直接计算解析 Disney 函数，得到 ground truth
3. 用 `TrainingMLP` 计算神经近似结果
4. 计算相对 L2 loss
5. 用 Slang autodiff 驱动 backward
6. 执行 Adam 更新
7. 同时渲染 3 个 viewport：
   - 解析 Disney
   - neural 结果
   - difference

当前网络配置：

- 编码后输入：`30`
- hidden layers：`3`
- hidden width：`32`
- 输出：`4`

这个 sample 是当前仓库里最接近“我要做自己的 neural shader”时应该参考的实现。

还有一个当前实现细节要注意：

它为了方便每帧训练后立即可视化，推理阶段直接继续使用 `TrainingOptimal` layout，而不是每帧都转换成单独的 `InferencingOptimal` layout。

关键文件：

- [`samples/ShaderTraining/ShaderTraining.cpp`](../samples/ShaderTraining/ShaderTraining.cpp)
- [`samples/ShaderTraining/computeTraining.slang`](../samples/ShaderTraining/computeTraining.slang)
- [`samples/ShaderTraining/computeOptimizer.slang`](../samples/ShaderTraining/computeOptimizer.slang)
- [`samples/ShaderTraining/renderInference.slang`](../samples/ShaderTraining/renderInference.slang)
- [`samples/ShaderTraining/DisneyMLP.slang`](../samples/ShaderTraining/DisneyMLP.slang)
- [`samples/ShaderTraining/NetworkConfig.h`](../samples/ShaderTraining/NetworkConfig.h)

### `SlangpyInferencing` 和 `SlangpyTraining`

用途：

- 让你先在 Python / SlangPy 里快速试结构和 loss，再落回 C++ 运行时

这组 sample 说明当前项目并不只有 “C++ 应用 + Slang shader” 一条路，也支持：

- 先在 Python 里快速试模型
- 再把推理部署回 `RTXNS`

关键目录：

- [`samples/SlangpyInferencing`](../samples/SlangpyInferencing)
- [`samples/SlangpyTraining`](../samples/SlangpyTraining)

## 10. 一次训练 epoch 在当前项目里实际做了什么

当前训练类 sample 的骨架基本都是：

1. 生成一批输入
2. 跑 forward
3. 和 ground truth 比较
4. 写入每个 sample 的 loss
5. 累加参数梯度
6. 执行 optimizer pass
7. 对 batch loss 做 reduction，得到 epoch 统计
8. 可选地渲染当前预测结果
9. 可选地把统计结果回读到 CPU UI

变的只是 target 来源：

- `SimpleTraining` 的 target 来自纹理
- `ShaderTraining` 的 target 来自解析 Disney BRDF

训练框架本身并没有变化。

## 11. 当前 optimizer 是怎么工作的

当前训练 sample 使用的是 Adam。

当前实现的模式是：

1. training pass 先把梯度累加到 FP16 路径相关的 gradient buffer
2. optimizer pass 读取梯度
3. optimizer 内部维护 FP32 的 moment buffer
4. 参数的高精度影子副本保存在 FP32 buffer 中
5. 最终更新后的值再写回 FP16 参数 buffer，供网络继续执行

这样做的目的很明确：

- 网络执行路径继续保持 FP16 / cooperative-vector 友好
- optimizer 状态尽量维持更好的数值稳定性

关键文件：

- [`samples/SimpleTraining/SimpleTraining_Optimizer.slang`](../samples/SimpleTraining/SimpleTraining_Optimizer.slang)
- [`src/NeuralShading_Shaders/Optimizers.slang`](../src/NeuralShading_Shaders/Optimizers.slang)

## 12. Loss、统计和 UI 回读是怎么做的

当前仓库不会每个 sample 都把 loss 直接拉回 CPU 再算。

它的流程是：

1. training shader 先把每个 sample 的 loss 写进 GPU buffer
2. `ProcessTrainingResults.slang` 在 GPU 上做 reduction
3. 把汇总结果写进结构化结果 buffer
4. `ResultsReadbackHandler` 把结果拷回 CPU 可见内存
5. ImGui / ImPlot 负责显示训练曲线和状态

这让训练闭环尽量保持在 GPU 端，只把轻量级统计结果回读到 CPU。

## 13. 当前仓库里的“neural shading”到底是什么意思

在当前 `RTXNS` 代码里，neural shading 的含义不是：

- “整个 renderer 都交给 AI”

而是：

- 找出一个局部的 shading 子问题
- 用紧凑输入描述它
- 用小型 MLP 近似它
- 把结果重新接回正常图形管线

所以它的核心思想一直是：

- 只替换着色流程里的一小块高价值函数
- 不是替换全部渲染流程

## 14. 如果你要二次开发，最常改哪些地方

如果你想改当前项目，最常见的改动入口是下面几类。

### 改网络规模和超参数

改各个 sample 自己的 `NetworkConfig.h`。

### 改“网络要学什么”

改 sample 的 training shader：

- 输入怎么生成
- ground truth 怎么算
- loss 怎么定义

### 改推理结果怎么接回画面

改 sample 的 inference shader 或 pixel shader。

### 改共享神经模块

改：

- [`src/NeuralShading_Shaders`](../src/NeuralShading_Shaders)

### 改 host 侧 load/save/layout 转换

改：

- [`src/NeuralShading/NeuralNetwork.cpp`](../src/NeuralShading/NeuralNetwork.cpp)

## 15. 当前项目的心智模型

如果只记一句话，可以记成：

`RTXNS = 小型 MLP 运行时 + cooperative vector 执行路径 + sample 自定义训练目标`

再展开一点，就是：

1. 定义一个紧凑输入
2. 对输入做编码
3. 在 Slang 里跑一个小 MLP
4. 加载或训练权重
5. 把输出重新接回正常渲染流程

这就是当前仓库里 neural shading 的运作方式。

## 16. 实际要怎么开始训练

如果你真正想动手训练，最短答案是：

- 想先看最简单的完整训练流程，跑 `SimpleTraining`
- 想看最像“神经着色”的训练流程，跑 `ShaderTraining`
- 想先在 Python 里快速试模型结构和 loss，跑 `SlangpyTraining.py`

### Windows 侧运行入口

当前仓库根目录里最直接的入口是：

- `build.bat`
- `run-simple-training.bat`
- `run-shader-training.bat`
- `run-sample.bat SimpleTraining`
- `run-sample.bat ShaderTraining`

对应可执行文件是：

- `bin/windows-x64/SimpleTraining.exe`
- `bin/windows-x64/ShaderTraining.exe`
- `bin/windows-x64/SlangpyTraining.exe`

### C++ / Python 代码入口文件

如果你要从代码入口开始读，先看：

- [`samples/SimpleTraining/SimpleTraining.cpp`](../samples/SimpleTraining/SimpleTraining.cpp)
- [`samples/ShaderTraining/ShaderTraining.cpp`](../samples/ShaderTraining/ShaderTraining.cpp)
- [`samples/SlangpyTraining/SlangpyTraining.py`](../samples/SlangpyTraining/SlangpyTraining.py)

这几个文件就是当前训练路径的实际顶层入口。

## 17. 训练时到底要改哪些文件

这个问题要分情况看。

### A. 你想改网络层数、宽度、batch size、学习率这些参数

优先改 sample 自己的 `NetworkConfig.h`。

`SimpleTraining` 的参数入口：

- [`samples/SimpleTraining/NetworkConfig.h`](../samples/SimpleTraining/NetworkConfig.h)

当前最重要的宏有：

- `INPUT_FEATURES`
- `INPUT_NEURONS`
- `OUTPUT_NEURONS`
- `HIDDEN_NEURONS`
- `NUM_HIDDEN_LAYERS`
- `BASE_LEARNING_RATE`
- `MIN_LEARNING_RATE`
- `WARMUP_LEARNING_STEPS`
- `FLAT_LEARNING_STEPS`
- `DECAY_LEARNING_STEPS`
- `BATCH_COUNT`
- `BATCH_SIZE_X`
- `BATCH_SIZE_Y`
- `LOSS_SCALE`
- `RELU_LEAK`
- `MATRIX_LAYOUT`

`ShaderTraining` 的参数入口：

- [`samples/ShaderTraining/NetworkConfig.h`](../samples/ShaderTraining/NetworkConfig.h)

当前最重要的宏有：

- `INPUT_FEATURES`
- `INPUT_NEURONS`
- `OUTPUT_NEURONS`
- `HIDDEN_NEURONS`
- `NUM_HIDDEN_LAYERS`
- `BATCH_SIZE`
- `BATCH_COUNT`
- `LEARNING_RATE`
- `COMPONENT_WEIGHTS`
- `LOSS_SCALE`
- 各种 thread group 大小

`SlangpyTraining` 的参数入口：

- [`samples/SlangpyTraining/SlangpyTraining.py`](../samples/SlangpyTraining/SlangpyTraining.py)
- [`samples/SlangpyTraining/NetworkConfig.h`](../samples/SlangpyTraining/NetworkConfig.h)

当前 Python 侧最常改的是：

- `batch_shape`
- `learning_rate`
- `grad_scale`
- `num_batches_per_epoch`
- `num_epochs`
- `models = [...]` 里的模型定义
- `loss_name`

### B. 你想改训练数据和 target

这通常是最关键的改动点。

如果你从 `SimpleTraining` 改起，优先改：

- [`samples/SimpleTraining/SimpleTraining_Training.slang`](../samples/SimpleTraining/SimpleTraining_Training.slang)

这个文件当前决定了：

- 输入 UV 怎么生成
- 输入 UV 怎么变换
- ground truth RGB 怎么从纹理读取
- 预测值怎么和目标值比较

如果你从 `ShaderTraining` 改起，优先改：

- [`samples/ShaderTraining/computeTraining.slang`](../samples/ShaderTraining/computeTraining.slang)
- [`samples/ShaderTraining/Disney.slang`](../samples/ShaderTraining/Disney.slang)

这两个文件当前决定了：

- 随机 shading 输入怎么生成
- Disney 解析结果怎么计算
- loss 怎么针对 neural 结果来定义

如果你想训练的不是 Disney BRDF，而是别的 shading 子函数，这里就是第一修改点。

### C. 你想改推理路径和可视化效果

`SimpleTraining` 的推理显示入口：

- [`samples/SimpleTraining/SimpleTraining_Inference.slang`](../samples/SimpleTraining/SimpleTraining_Inference.slang)

`ShaderTraining` 的推理显示入口：

- [`samples/ShaderTraining/renderInference.slang`](../samples/ShaderTraining/renderInference.slang)
- [`samples/ShaderTraining/DisneyMLP.slang`](../samples/ShaderTraining/DisneyMLP.slang)

如果你训练完之后想换一种接入方式，或者想把 MLP 输出接回你自己的材质模型，这里就是入口。

### D. 你想改 optimizer

如果你只想改 sample 自己的 optimizer pass：

- [`samples/SimpleTraining/SimpleTraining_Optimizer.slang`](../samples/SimpleTraining/SimpleTraining_Optimizer.slang)
- [`samples/ShaderTraining/computeOptimizer.slang`](../samples/ShaderTraining/computeOptimizer.slang)

如果你想改共享 optimizer 实现：

- [`src/NeuralShading_Shaders/Optimizers.slang`](../src/NeuralShading_Shaders/Optimizers.slang)

### E. 你想改共享 MLP 运行时

这时应该改共享 shader runtime：

- [`src/NeuralShading_Shaders/MLP.slang`](../src/NeuralShading_Shaders/MLP.slang)
- [`src/NeuralShading_Shaders/LinearOps.slang`](../src/NeuralShading_Shaders/LinearOps.slang)
- [`src/NeuralShading_Shaders/Activation.slang`](../src/NeuralShading_Shaders/Activation.slang)
- [`src/NeuralShading_Shaders/Loss.slang`](../src/NeuralShading_Shaders/Loss.slang)

这里适合改：

- 激活函数
- 通用 loss
- 通用 MLP 前向 / 反向行为
- cooperative vector 封装方式

## 18. 当前训练循环的真实入口在哪里

如果你想追到“每帧到底是谁在发训练 dispatch”，可以按下面的入口看。

### `SimpleTraining`

Host 侧总入口：

- [`samples/SimpleTraining/SimpleTraining.cpp`](../samples/SimpleTraining/SimpleTraining.cpp)

关键函数是：

1. `Init()` 创建网络、buffer、texture、compute pipeline
2. `Render()` 每帧执行训练、优化、loss reduction 和 inference
3. `Animate()` 处理模型加载和保存

如果只看一个函数，优先看 `Render()`，因为当前真正的训练循环就在这里。

### `ShaderTraining`

Host 侧总入口：

- [`samples/ShaderTraining/ShaderTraining.cpp`](../samples/ShaderTraining/ShaderTraining.cpp)

关键函数是：

1. `Init()` 创建训练和渲染 pass
2. `CreateMLPBuffers()` 分配网络和 optimizer 资源
3. `Render()` 跑 batch 训练循环并绘制 3 个 viewport
4. `Animate()` 处理 reset 和模型加载保存

### `SlangpyTraining`

Python 侧总入口：

- [`samples/SlangpyTraining/SlangpyTraining.py`](../samples/SlangpyTraining/SlangpyTraining.py)

关键流程是：

1. `training_main()` 创建模型和 optimizer state
2. epoch 循环里把训练和优化命令 append 到 command buffer
3. 把训练好的权重写到 `weights.json`
4. 编译 inference shader，并调用 C++ 推理路径

## 19. 当前有哪些运行时参数可以不重编译直接调

当前 sample 已经有一部分 UI 控制可以不重编译直接改。

`SimpleTraining` 的 UI 入口：

- [`samples/SimpleTraining/UIWidget.cpp`](../samples/SimpleTraining/UIWidget.cpp)
- [`samples/SimpleTraining/UIData.h`](../samples/SimpleTraining/UIData.h)

当前可直接在 UI 里改的东西有：

- 启用或禁用训练
- 重置训练
- 选择纹理变换方式
- 从 `.bin` 加载模型
- 保存 `.bin` 模型

`ShaderTraining` 的 UI 入口：

- [`samples/ShaderTraining/UIWidget.cpp`](../samples/ShaderTraining/UIWidget.cpp)
- [`samples/ShaderTraining/UIData.h`](../samples/ShaderTraining/UIData.h)

当前可直接在 UI 里改的东西有：

- 光照强度
- specular
- roughness
- metallic
- 启用或禁用训练
- 重置训练
- 从 `.bin` 加载模型
- 保存 `.bin` 模型

如果某个参数 UI 里没有，那当前做法就是：

- 改 `NetworkConfig.h`
- 或改 training shader
- 或改 C++ 初始化代码

然后重新编译。

## 20. 如果你要自己训练一个新的 neural shader，推荐怎么走

结合当前仓库结构，最稳妥的顺序是：

1. 如果你的目标是 shading 函数，先从 `ShaderTraining` 改起
2. 如果你只是先验证一个 2D 函数逼近，先从 `SimpleTraining` 改起
3. 先改 `NetworkConfig.h`，把网络宽度、层数、batch size、学习率策略定下来
4. 再改 training shader，明确输入怎么生成、target 是什么、loss 怎么算
5. 再改 inference shader，明确训练好的结果怎么接回最终画面
6. 用 `build.bat` 重编译
7. 跑对应训练 sample，等 loss 稳定后再保存模型

如果只想记最短文件清单，可以记这 4 组：

- 网络结构和 batch 参数：`NetworkConfig.h`
- 训练目标和 loss：`*_Training.slang` 或 `computeTraining.slang`
- 推理接入方式：`*_Inference.slang` 或 `renderInference.slang`
- optimizer：`*_Optimizer.slang` 和 `Optimizers.slang`
