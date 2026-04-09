# RTX Neural Shading：如何编写你的第一个 Neural Shader

## 目的

这份教程以 [Shader Training](ShaderTraining.md) 为基础，简要说明如何开始编写你自己的 neural shader。

我们重点关注三个方面：

1. 如何从待训练的 shader 中提取关键特征
2. 如何调整网络配置
3. 如何修改激活函数和 loss 函数

这份文档不会展开讲 AI 训练和优化算法本身，而是聚焦于如何基于现有 sample 来配置并训练不同内容的网络。

## 提取训练输入中的关键特征

在把 Disney BRDF 接入 [Shader Training](ShaderTraining.md) 示例时，第一步其实是做特征提取。也就是说：

- 哪些特征应该交给网络去拟合
- 哪些特征仍然保留在普通 shader 逻辑里计算

这样做的目标，是避免网络过度专门化，也避免模型变得不必要地复杂。

Disney BRDF 示例里，网络输入包含：

- `view`
- `light`
- `normal`
- `material roughness`

而下面这些量则仍然保留在普通 shader 中：

- `light intensity`
- `material metallicness`
- 各类材质颜色分量

这本身是一个需要反复试验的权衡过程。

在确定哪些量适合作为训练输入后，下一步是尽量优化它们的表示形式：

- 尽量减少输入维度
- 尽量缩放到 `0-1` 或 `-1 - 1` 区间

因为神经网络通常更偏好这种数值范围。

在 Disney BRDF 中，这一步的做法是利用一个事实：

- 输入向量本身总是归一化的
- 实际最终使用的是它们之间的点积

因此输入就从 3 个 `float3` 向量，简化成了 4 个 `float` 标量点积。

接下来，网络输入通常还可以进一步做编码。很多研究表明，这一步能显著提升网络表现。当前库提供了两个编码器：

- `EncodeFrequency`
- `EncodeTriangle`

它们都会把输入编码成某种波形形式。

`ShaderTraining` 示例使用的是 frequency encoder。它会把输入维度扩大 6 倍，但通常能换来更好的拟合效果。你应该根据自己的数据集去试验不同编码方式。

当你已经知道：

- 编码后的输入维度
- 输出维度

就可以开始配置网络了。

## 修改网络配置

网络配置位于 [NetworkConfig.h](../samples/ShaderTraining/NetworkConfig.h) 中，通常需要按你的任务进行调整。

有些参数基本是由数据集直接决定的，比如输入和输出维度；另一些参数则需要实验来找到合适配置。

在 sample 里，这些配置为了易懂都被硬编码了，但在真正的生产环境里，它们理应成为训练流水线中的可配置项。

下面这些参数基本是由你要训练的 shader 任务直接决定的：

- `INPUT_NEURONS`
  应该等于编码后真正送进网络的输入维度
- `OUTPUT_NEURONS`
  应该等于网络输出维度，比如 RGB 三通道，或者像 Disney BRDF 那样的若干独立输出量

下面这些参数则更适合拿来实验：

- `NUM_HIDDEN_LAYERS`
  网络隐藏层数
- `HIDDEN_NEURONS`
  每层隐藏层神经元数量，这会显著影响模型精度和运行成本
- `LEARNING_RATE`
  需要调参以改善收敛速度和稳定性

未来库版本可能会支持更多精度配置，从而进一步影响质量和性能。当前版本固定为 `float16`。

只要这些参数通过宏在 C++ 和 shader 里共享，那么修改它们通常不需要额外改逻辑代码，只需要重新编译。唯一的例外是：

- 如果你修改了输入或输出 `CoopVec` 的大小
- 并且代码里有直接按下标访问元素的逻辑

比如：

```
float4 predictedDisney = { outputParams[0], outputParams[1], outputParams[2], outputParams[3] };
```

总之，真正找到适合你任务的网络配置，还是要依赖实验。

## 修改激活函数与 Loss 函数

Simple Shading 示例使用的是 `TrainingMLP`，它已经帮你封装掉大量训练 shader 中的重复逻辑：

```
var model = rtxns::mlp::TrainingMLP<half, 
    NUM_HIDDEN_LAYERS, INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, 
    CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
    gMLPParams, 
    gMLPParamsGradients, 
    rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
    rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets));

var hiddenActivation = rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>();
var finalActivation = rtxns::mlp::ExponentialAct<half, OUTPUT_NEURONS>();

var outputParams = model.forward(inputParams, hiddenActivation, finalActivation);
```

这里的激活函数 `ReLUAct` 和 `ExponentialAct` 会在 `TrainingMLP` 和 `InferenceMLP` 的前向、后向过程中被使用。相关实现位于：

- [CooperativeVectorFunctions.slang](../src/NeuralShading_Shaders/CooperativeVectorFunctions.slang)

如果你需要更多激活函数，可以在这里继续扩展。

Loss 函数的选择则依赖你的数据集：

- `Simple Training` 使用的是简单的 L2 loss
- `Shader Training` 使用的是更复杂的 L2 relative loss

在 Slang 中实现自定义 loss 并不难，因此你可以根据任务需要继续扩展。

## 超参数

下面是当前 sample 中一组可调的超参数示例：

| 参数 | 数值 |
| ---- | ---- |
| HIDDEN_NEURONS | 32 |
| NUM_HIDDEN_LAYERS | 3 |
| LEARNING_RATE | 1e-2f |
| BATCH_SIZE | (1 << 16) |
| BATCH_COUNT | 100 |
| Hidden Activation Functions | ReLUAct() |
| Final Activation Functions | ExponentialAct() |
| Loss Function | L2Relative() |

## 总结

如果你想训练自己的 neural shader，`Shader Training` 是当前仓库里最好的起点。

你需要先想清楚：

- shader 中哪些部分适合拆成网络输入
- 哪些部分仍然应该保留在常规 shader 逻辑里

然后再通过实验不断调整网络结构、激活函数、loss 和超参数，找到适合你任务的模型。
