# RTX Neural Shading：Shader Training 示例

## 目的

这个 sample 基于 [Simple Training](SimpleTraining.md) 进一步扩展，引入了 Slang 的 AutoDiff 功能，并通过完整的 MLP（Multi Layered Perceptron）抽象来训练神经网络。

这个 MLP 建立在之前介绍过的 `CoopVector` 训练代码之上，为使用 Slang 训练网络提供了更高层、更通用的接口。该 sample 创建了一个网络，并训练它去拟合 [Simple Inferencing](SimpleInferencing.md) 中使用的 Disney BRDF shader。

![Shader Training Output](shader_training.png)

运行后，窗口会展示 3 张图：

- 左侧：完整 Disney BRDF shader 的球体渲染结果
- 中间：训练中的神经网络渲染结果
- 右侧：两者的误差图

同时还提供一个 UI，用来：

- 调整部分材质参数
- 暂停或恢复训练
- 重置训练
- 保存或加载当前网络

## 训练流程

在 `RTXNS` 中创建并训练一个神经网络，一般要经历下面这些步骤。

这个 sample 和 [Simple Training](SimpleTraining.md) 最大的区别是：

- 训练和优化仍然通过 compute shader 完成
- 但推理阶段被直接集成进现有的 pixel shader 中

主要步骤如下：

1. 创建 host 侧神经网络存储并初始化
2. 创建 device-optimal layout 和 GPU buffer
3. 在 GPU 上把 host layout 转换成 device-optimal layout
4. 创建 loss、梯度和 optimizer 所需的辅助 buffer
5. 对随机输入反复执行训练 shader 和 optimizer shader，逐 epoch 调整参数
6. 通过推理 pixel shader 渲染球体，生成最终输出图像

## 网络配置

网络参数定义在 [NetworkConfig.h](../samples/ShaderTraining/NetworkConfig.h) 中，当前配置如下：

| 属性 | 数值 | 说明 |
| ---- | ---- | ---- |
| Input Features | 5 | 共 5 个输入参数 |
| Input Neurons | 30 | 5 个输入各自经过编码后扩展为 6 个值 |
| Output Neurons | 4 | 输出 4 个 BRDF 相关值 |
| Hidden Neurons | 32 | 每层隐藏层 32 个神经元 |
| Hidden Layers | 3 | 3 层隐藏层 |
| Precision | float16 | 当前使用半精度 |

## 应用层代码

Host 侧的神经网络创建逻辑和 [Simple Training](SimpleTraining.md) 大体类似，这里只强调它们的不同点。

### 训练循环

创建好 pipeline 并分配好 GPU buffer 后，这里的训练循环和 `Simple Training` 很接近。为了在保证可视化刷新的同时尽快收敛，训练与优化 pass 会在每一帧内多次执行（`g_trainingStepsPerFrame = 100`）。

```cpp
for (int i = 0; i < g_trainingStepsPerFrame; ++i)
{
    nvrhi::ComputeState state;

    // Training pass
    state.bindings = { m_trainingPass.bindingSet };
    state.pipeline = m_trainingPass.pipeline;
    m_commandList->setComputeState(state);
    m_commandList->dispatch(m_batchSize / 64, 1, 1);

    // Optimizer pass
    state.bindings = { m_optimizerPass.bindingSet };
    state.pipeline = m_optimizerPass.pipeline;
    m_commandList->setComputeState(state);
    m_commandList->dispatch(div_ceil(m_totalParameterCount, 32), 1, 1);
}
```

为了便于理解，上面的示例省略了一些 timer 相关代码。

训练 pass 完成后，会继续渲染两个球体，但使用的是不同的 pipeline：

- `m_directPass`：原始 Disney BRDF shader
- `m_inferencePass`：训练中的神经模型

## Shader 代码

当前这个 sample 试图用神经网络去近似下面这个函数：

```
Disney(NdotL, NdotV, NdotH, LdotH, roughness);
```

相比 [Simple Training](SimpleTraining.md)，这里的 shader 进一步使用了 Slang 的 [AutoDiff](https://shader-slang.org/slang/user-guide/autodiff.html) 来构建一个模板化训练类 `TrainingMLP`，实现位于 [MLP.slang](../src/NeuralShading_Shaders/MLP.slang)。

使用 AutoDiff 的好处是：

- 不需要手写完整的 backward pass
- 不需要逐个实现所有激活函数的导数反向传播
- Slang 可以从 forward 自动推导 backward

当前最核心的 3 个 shader 是：

- [training](../samples/ShaderTraining/computeTraining.slang)
- [optimizer](../samples/ShaderTraining/computeOptimizer.slang)
- [inference](../samples/ShaderTraining/renderInference.slang)

### Training

训练 shader 首先生成随机输入，并对其做编码，然后再送进神经网络：

```cpp
//----------- Training step
float params[INPUT_FEATURES] = {NdotL, NdotV, NdotH, LdotH, roughness};
var inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);
```

接着创建模型对象，并把输入送入网络做 forward：

```cpp
var model = rtxns::mlp::TrainingMLP<half, 
    NUM_HIDDEN_LAYERS, INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, 
    CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
    gMLPParams, 
    gMLPParamsGradients, 
    rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
    rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets));
```

`TrainingMLP` 的模板参数很多，但本质上主要包括：

- 隐藏层数量
- 输入神经元数量
- 每层隐藏层神经元数量
- 输出神经元数量
- 矩阵布局
- 精度

而运行时参数主要包括：

- 权重 / 偏置 buffer
- 梯度 buffer
- 每层权重 offset
- 每层偏置 offset

模型创建完成后，前向传播本身就非常直接了。只需要指定隐藏层激活函数和最终输出激活函数，然后调用 `forward` 即可。更完整的模板细节可参考 [Library Guide](LibraryGuide.md)。

```cpp
var hiddenActivation = rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>();
var finalActivation = rtxns::mlp::ExponentialAct<half, OUTPUT_NEURONS>();

var outputParams = model.forward(inputParams, hiddenActivation, finalActivation);
```

为了构造 loss gradient，这个例子使用了解析 Disney BRDF 输出和神经网络前向输出之间的 `L2Relative` 导数：

```cpp
float4 predictedDisney = { outputParams[0], outputParams[1], outputParams[2], outputParams[3] };

float4 lossGradient = rtxns::mlp::L2Relative<float, 4>.deriv(actualDisney, predictedDisney, float4(LOSS_SCALE / (gConst.batchSize * 4)) * COMPONENT_WEIGHTS);
```

最后，把 loss gradient 和输入向量一起送入 backward，更新梯度参数：

```cpp
model.backward(inputParams, hiddenActivation, finalActivation, rtxns::HCoopVec<OUTPUT_NEURONS>(lossGradient[0], lossGradient[1], lossGradient[2], lossGradient[3]));
```

<a id="l2-relative-loss-computation"></a>

#### L2 Relative Loss 的计算

下面这段代码在训练 shader 中计算每个 sample 的相对 L2 Loss。结果会写入 `gLossBuffer`，后续再在 batch 维度上做 reduction，以便在训练过程中可视化：

```cpp
// Store L2 relative loss 
float4 diff = predictedDisney - actualDisney;
float l2Error  = sqrt(dot(diff, diff));
float l2Target = sqrt(dot(actualDisney, actualDisney));
float epsilon = 1e-6f;

gLossBuffer[idx] = l2Error / (l2Target + epsilon);
```

### Optimizer

这里的 optimizer shader 和 [Simple Training](SimpleTraining.md) 中使用的是同样的思路。

```cpp
void adam_cs(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i >= maxParamSize)
        return;

    float gradient = (float)gMLPParamsGradients[i];
    gMLPParamsGradients[i] = half(0.0);

    float weightbias = gMLPParams32[i];

    optimizers::Adam optimizer = optimizers::Adam(gMoments1, gMoments2, learningRate, LOSS_SCALE);

    float adjustedWeightbias = optimizer.step(weightbias, i, gradient, currentStep);

    gMLPParams32[i] = adjustedWeightbias;
    gMLPParams[i] = (half)adjustedWeightbias;
}
```

### Inference

推理 pass 和训练 shader 里的 forward 基本一致。当前实现为了避免每个 batch 后再做一次 layout 转换，直接继续使用 `CoopVecMatrixLayout::TrainingOptimal`。但如果把它改成单独的纯推理 sample，默认更合理的 layout 其实应当是 `CoopVecMatrixLayout::InferencingOptimal`。

```cpp
float4 DisneyMLP<let HIDDEN_LAYERS : int, let HIDDEN_NEURONS : int>(
    float NdotL, float NdotV, float NdotH, float LdotH, float roughness, ByteAddressBuffer mlpBuffer,
    uint weightOffsets[HIDDEN_LAYERS+1], uint biasOffsets[HIDDEN_LAYERS+1])
{
    // Calculate approximated core shader part using MLP
    float params[INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, roughness };

    var inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);

    var model = rtxns::mlp::InferenceMLP<half, 
        HIDDEN_LAYERS, 
        INPUT_FEATURES * FREQUENCY_EXPANSION, 
        HIDDEN_NEURONS, 
        OUTPUT_NEURONS, 
        CoopVecMatrixLayout::TrainingOptimal, 
        CoopVecComponentType::Float16>
        (mlpBuffer, weightOffsets, biasOffsets);

    var outputParams = model.forward(inputParams, rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>(), rtxns::mlp::ExponentialAct<half, OUTPUT_NEURONS>());
    return float4(outputParams[0], outputParams[1], outputParams[2], outputParams[3]);
}
```
