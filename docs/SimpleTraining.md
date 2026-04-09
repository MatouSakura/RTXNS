# RTX Neural Shading：Simple Training 示例

## 目的

这个 sample 建立在 [Simple Inferencing](SimpleInferencing.md) 之上，用来介绍如何训练一个可用于 shader 的神经网络。

这个网络的目标是拟合一张经过变换的纹理。它的设计重点是“易于理解训练流程”，而不是做真实的纹理压缩。

![Simple Training Output](simple_training.png)

运行后，窗口里会显示三部分内容：

- 左侧：原始纹理
- 中间：当前训练中的神经网络输出
- 右侧：当前预测图像与参考图像之间经过放大的 loss delta

当模型训练充分后，右侧的误差图应该接近纯灰色。UI 同时还支持：

- 切换不同变换方式
- 保存网络
- 加载网络

## 训练流程

在 `RTXNS` 中创建并训练一个神经网络，一般需要下面这些步骤：

1. 创建 host 侧神经网络存储并初始化
2. 创建 device-optimal layout 和 GPU buffer
3. 在 GPU 上把 host layout 转换成 device-optimal layout
4. 创建 loss 梯度和 optimizer pass 所需的辅助 buffer
5. 反复执行训练 shader 和 optimizer shader
6. 执行 inference shader，生成输出图像
7. 如有需要，把网络存回文件

## 网络配置

网络配置位于 [NetworkConfig.h](../samples/SimpleTraining/NetworkConfig.h)，当前定义如下：

| 属性 | 数值 | 说明 |
| ---- | ---- | ---- |
| Input Features | 2 | U、V 坐标 |
| Input Neurons | 2 * 6 | U、V 经过 Frequency Encoding 后扩展 |
| Output Neurons | 3 | R、G、B |
| Hidden Neurons | 64 | 每层 64 个神经元 |
| Hidden Layers | 4 | 4 层隐藏层 |
| Precision | float16 | 当前使用半精度 |

在 RTX 4090 上，大约训练 10 到 15 秒后，这组配置就能给出对输入纹理比较合理的近似结果：

![Simple Training Output](simple_training_trained.png)

## 应用层代码

Host 侧的神经网络初始化过程比较直接。首先根据 `NetworkConfig.h` 填充一个网络结构描述，然后用它来初始化网络。

### 创建网络

这一步会为权重和偏置分配一块连续的 host 内存，尺寸和 host layout 相匹配，并用一个归一化分布进行初始化。同时，也会创建一个训练用的 device-optimal layout。

```
// Create Network
m_NeuralNetwork = std::make_unique<rtxns::Network>(GetDevice());
...
rtxns::NetworkArchitecture netArch = {};
netArch.inputNeurons = INPUT_NEURONS;
netArch.hiddenNeurons = HIDDEN_NEURONS;
netArch.outputNeurons = OUTPUT_NEURONS;
netArch.numHiddenLayers = NUM_HIDDEN_LAYERS;
netArch.biasPrecision = NETWORK_PRECISION;
netArch.weightPrecision = NETWORK_PRECISION;
...
m_NeuralNetwork->Initialise(netArch)
...
// Get a device optimized layout
m_deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(m_neuralNetwork->GetNetworkLayout(), rtxns::MatrixLayout::TrainingOptimal);

```

### GPU Buffer 分配

训练会用到多种 GPU buffer，它们的大小都依赖于当前网络规模。

训练本身至少需要：

- float16 参数 buffer
- 一个用于 optimizer pass 的梯度输出 buffer

而 Adam optimizer 还需要：

- float32 版本的参数 buffer
- 两个 moment buffer

#### Float16 参数 Buffer

前面的网络初始化代码会在 host 侧分配并填充权重和偏置。之后这些参数会写入 GPU，再转换成 `rtxns::MatrixLayout::TrainingOptimal`，供训练和推理 shader 直接使用。

虽然每层权重和偏置在 GPU 上对齐和尺寸有严格要求，但 sample 为了简化理解，把它们统一打包进一段连续 GPU 内存。

```
nvrhi::BufferDesc paramsBufferDesc;
paramsBufferDesc.byteSize = m_NeuralNetwork->GetNetworkParams().size();
...
m_mlpHostBuffer = GetDevice()->createBuffer(paramsBufferDesc);
...
m_CommandList->writeBuffer(m_mlpHostBuffer, m_NeuralNetwork->GetNetworkParams().data(), m_NeuralNetwork->GetNetworkParams().size());

paramsBufferDesc.byteSize = m_deviceNetworkLayout.networkSize;
...
m_mlpDeviceBuffer = GetDevice()->createBuffer(paramsBufferDesc);

// Convert to GPU optimized layout
m_networkUtils->ConvertWeights(m_neuralNetwork->GetNetworkLayout(), m_deviceNetworkLayout, m_mlpHostBuffer, 0, m_mlpDeviceBuffer, 0, GetDevice(), m_commandList);

```

#### Float32 参数 Buffer

当前 sample 的权重和偏置使用 `float16`，这样可以提高执行性能，但也会带来精度损失。

因此，这里需要一份 float32 版本的参数 buffer，用来在 optimizer 阶段做更稳定的参数更新，然后再把结果写回 float16 buffer 给训练 shader 使用。

```
paramsBufferDesc.byteSize = m_TotalParamCount * sizeof(float); // convert to float
paramsBufferDesc.structStride = sizeof(float);
m_MLPParametersfBuffer = GetDevice()->createBuffer(paramsBufferDesc);
```

#### 梯度 Buffer

训练 shader 完成反向传播后，每个神经元对应的梯度都会写到梯度 buffer 中，供 optimizer pass 消费。这个 buffer 的大小与参数 buffer 对应，当前使用 float16：

```
paramsBufferDesc.debugName = "MLPGradientsBuffer";
paramsBufferDesc.byteSize = m_TotalParamCount * sizeof(float16_t);
paramsBufferDesc.format = nvrhi::Format::R16_FLOAT;
m_MLPGradientsBuffer = GetDevice()->createBuffer(paramsBufferDesc);
```

#### Moments Buffer

最后还需要为 Adam optimizer 分配两个 float32 moment buffer，它们同样和参数 buffer 大小一致：

```
paramsBufferDesc.debugName = "MLPMoments1Buffer";
paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
paramsBufferDesc.byteSize = m_TotalParamCount * sizeof(float);
paramsBufferDesc.format = nvrhi::Format::R32_FLOAT;
m_MLPMoments1Buffer = GetDevice()->createBuffer(paramsBufferDesc);
...
paramsBufferDesc.debugName = "MLPMoments2Buffer";
m_MLPMoments2Buffer = GetDevice()->createBuffer(paramsBufferDesc);
```

### 权重和偏置的 Offset

在创建好 shader 侧要用到的 buffer 后，还需要把 device layout 里的权重和偏置 offset 提取出来，并写入 constant buffer。这个过程很简单，直接从每一层的 layout 信息中读取即可：

```
NeuralConstants neuralConstants = {};

for (int i = 0; i < NUM_TRANSITIONS; ++i)
{
    neuralConstants.weightOffsets[i / 4][i % 4] = m_deviceNetworkLayout.networkLayers[i].weightOffset;
    neuralConstants.biasOffsets[i / 4][i % 4] = m_deviceNetworkLayout.networkLayers[i].biasOffset;
}
```

### 训练循环

在 pipeline 创建完之后，训练循环本身并不复杂。训练按 batch 执行，当前 sample 使用的是：

| 属性 | 数值 |
| ---- | ---- |
| BATCH_COUNT | 128 |
| BATCH_SIZE_{X\|Y} | 32 |

这些 batch 大小需要根据你的模型自行调节。

```
for (uint32_t batch = 0; batch < BATCH_COUNT; batch++)
{
    // run the training pass
    state.bindings = { m_TrainingPass.m_BindingSet };
    state.pipeline = m_TrainingPass.m_Pipeline;
    m_CommandList->beginMarker("Training");
    m_CommandList->setComputeState(state);
    m_CommandList->dispatch(dm::div_ceil(BATCH_SIZE_X, 8), dm::div_ceil(BATCH_SIZE_Y, 8), 1);
    m_CommandList->endMarker();

    // optimizer pass
    state.bindings = { m_OptimizerPass.m_BindingSet };
    state.pipeline = m_OptimizerPass.m_Pipeline;
    m_CommandList->beginMarker("Update Weights");
    m_CommandList->setComputeState(state);
    m_CommandList->dispatch(dm::div_ceil(m_TotalParamCount, 32), 1, 1);
    m_CommandList->endMarker();

    neuralConstants.currentStep = ++m_AdamCurrentStep;
    m_CommandList->writeBuffer(m_NeuralConstantBuffer, &neuralConstants, sizeof(neuralConstants));
}
m_uiParams->epochs++;

...
// inference pass
state.bindings = { m_InferencePass.m_BindingSet };
state.pipeline = m_InferencePass.m_Pipeline;
m_CommandList->beginMarker("Inference");
m_CommandList->setComputeState(state);
m_CommandList->dispatch(dm::div_ceil(m_InferenceTexture->getDesc().width, 8), dm::div_ceil(m_InferenceTexture->getDesc().height, 8), 1);
m_CommandList->endMarker();
```

### 保存训练后的网络

当 UI 中点击 `save` 按钮时，sample 会调用 `rtxns::HostNetwork::UpdateFromBufferToFile()`，从 float16 GPU 参数 buffer 中读取训练结果，并写回文件：

```
 m_neuralNetwork->UpdateFromBufferToFile(
    m_mlpHostBuffer,
    m_mlpDeviceBuffer,
    m_neuralNetwork->GetNetworkLayout(),
    m_deviceNetworkLayout,
    m_uiParams->fileName,
    GetDevice(),
    m_commandList);
```

## Shader 代码

这个 sample 的神经网络要学习的是一个基于 UV 的简单 RGB 查询：

```
float4 colour = inputTexture[uv].rgb;
```

如果用 PyTorch 来表达，这个网络大致会长这样：

```
nn.Linear(2, hidden_layer_size),  # UV as input
nn.LeakyReLU(),
nn.Linear(hidden_layer_size, hidden_layer_size),
nn.LeakyReLU(),
nn.Linear(hidden_layer_size, hidden_layer_size),
nn.LeakyReLU(),
nn.Linear(hidden_layer_size, hidden_layer_size),
nn.LeakyReLU(),
nn.Linear(hidden_layer_size, 3),  # RGB as output
nn.Sigmoid()  # Ensure output is between 0 and 1
```

最关键的 3 个 shader 是：

- [training](../samples/SimpleTraining/SimpleTraining_Training.slang)
- [optimizer](../samples/SimpleTraining/SimpleTraining_Optimizer.slang)
- [inference](../samples/SimpleTraining/SimpleTraining_Inference.slang)

### Training

训练按 batch 进行。每个 batch 都会：

- 生成随机输入
- 做前向传播
- 和 ground truth 比较
- 计算 loss gradient
- 做反向传播
- 再交给 optimizer 更新权重和偏置

输入数据首先会做 frequency encoding，为网络提供更丰富的输入表示。虽然这一步不是绝对必须，但在这个 sample 中能带来大约 2 倍的训练效果提升和更高质量。当前使用的是 `EncodeFrequency()`：

```
// Get a random uv coordinate for the input and frequency encode it for improved convergance
float2 inputUV = clamp(float2(rng.next(), rng.next()), 0.0, 1.0);
CoopVec<VECTOR_FORMAT, INPUT_NEURONS> inputParams = rtxns::EncodeFrequency<half, 2>({inputUV.x, inputUV.y});
```

前向传播逻辑和 [Simple Inferencing](SimpleInferencing.md) 非常相似，只不过这里需要把每一层的中间结果缓存下来，以便反向传播重复使用：

```
// Create variables to cache the results from each stage
CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenParams[NUM_HIDDEN_LAYERS];
CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenActivated[NUM_HIDDEN_LAYERS];
CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputParams;
CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputActivated;

// Forward propagation through the neural network
// Input to hidden layer, then apply activation function
hiddenParams[0] = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(
    inputParams, gMLPParams, weightOffsets[0], biasOffsets[0], MATRIX_LAYOUT, TYPE_INTERPRETATION);
hiddenActivated[0] = rtxns::leakyReLU(hiddenParams[0], RELU_LEAK);

// Hidden layers to hidden layers, then apply activation function 
[ForceUnroll]
for (uint layer = 1; layer < NUM_HIDDEN_LAYERS; layer++)
{
    hiddenParams[layer] = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
        hiddenActivated[layer - 1], gMLPParams, weightOffsets[layer], biasOffsets[layer], 
        MATRIX_LAYOUT, TYPE_INTERPRETATION);
    hiddenActivated[layer] = rtxns::leakyReLU(hiddenParams[layer], RELU_LEAK);
}

// Hidden layer to output layer, then apply final activation function    
outputParams = rtxns::LinearOp<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(
    hiddenActivated[NUM_HIDDEN_LAYERS - 1], gMLPParams, weightOffsets[NUM_HIDDEN_LAYERS],
    biasOffsets[NUM_HIDDEN_LAYERS], MATRIX_LAYOUT, TYPE_INTERPRETATION);
outputActivated = rtxns::sigmoid(outputParams);
```

前向传播得到的是网络预测的 RGB，需要和 ground truth 进行比较。ground truth 的具体形式取决于 UI 中选择的网络变换模式：

```
// Take the output from the neural network as the output color
float3 predictedRGB = {outputActivated[0], outputActivated[1], outputActivated[2]};

// Now transform the input UVs according to the NetworkModel enum.
// This can easily be extended to try many different transforms.
uint2 actualUV;
if (gConst.networkTransform == NetworkTransform.Flip)
{
    float2 flipUV = inputUV.yx;
    actualUV = uint2(flipUV.xy * float2(gConst.imageHeight, gConst.imageWidth));
}
else if (gConst.networkTransform == NetworkTransform.Zoom)
{
    float2 zoomUV = inputUV * 0.5 + 0.25;
    actualUV = uint2(zoomUV.xy * float2(gConst.imageWidth, gConst.imageHeight));
}
else
{
    actualUV = uint2(inputUV.xy * float2(gConst.imageWidth, gConst.imageHeight));
}

// Load the texture according to the transformed input UVs. This will
// provide the RGB that the model is trying to train towards.
float3 actualRGB = inputTexture[actualUV].rgb;

// Output the loss, scaled to greyscale for output
uint2 lossUV = uint2(inputUV.xy * float2(gConst.imageWidth, gConst.imageHeight));
const float lossScaleFactor = 10.0f; // scale it up for better vis
lossTexture[lossUV] = float4((predictedRGB - actualRGB) * 0.5 * lossScaleFactor + 0.5, 1);

// Compute the L2 loss gradient
// L2Loss = (a-b)^2
// L2Loss Derivative = 2(a-b)
float3 lossGradient = 2.0 * (predictedRGB - actualRGB);
```

这个 loss gradient 对训练非常关键。但由于这里在半精度路径上工作，需要额外做缩放，尽量保住有效位数。这个缩放会在 optimizer 阶段被恢复：

```
// Scale by batch size 
lossGradient /= (batchSize.x * batchSize.y);

// Apply the LOSS_SCALE factor to retain precision. Remove it in the optimizer pass before use.
lossGradient *= LOSS_SCALE;

CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> lossGradientCV = CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS>(VECTOR_FORMAT(lossGradient[0]), VECTOR_FORMAT(lossGradient[1]), VECTOR_FORMAT(lossGradient[2]));
```

要计算反向传播，需要调用激活函数的导数版本以及线性层的 backward 版本。这些都已经在 `rtxns` 命名空间中实现好了，也可以自行扩展：

```
// Back-propogation pass, generate the gradients and accumulate the results into memory to be applied in the optimization pass.
CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputGradient;
CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenGradient;

// Output layer (loss gradient) to final hidden layer
outputGradient = rtxns::sigmoid_Backward(outputParams, lossGradientCV);
hiddenGradient = rtxns::LinearOp_Backward<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(
   hiddenActivated[NUM_HIDDEN_LAYERS - 1], outputGradient, gMLPParams, gMLPParamsGradients, 
   weightOffsets[NUM_HIDDEN_LAYERS], biasOffsets[NUM_HIDDEN_LAYERS], MATRIX_LAYOUT, TYPE_INTERPRETATION);

// Hidden layer to hidden layer 
for(int layer = NUM_HIDDEN_LAYERS - 1; layer >= 1; layer--)
{
    hiddenGradient = rtxns::leakyReLU_Backward(hiddenParams[layer], RELU_LEAK, hiddenGradient);
    hiddenGradient = rtxns::LinearOp_Backward<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>
        (hiddenActivated[layer - 1], hiddenGradient, gMLPParams, gMLPParamsGradients, 
        weightOffsets[layer], biasOffsets[layer], MATRIX_LAYOUT, TYPE_INTERPRETATION);
}

// First hidden layer to input layer
hiddenGradient = rtxns::leakyReLU_Backward(hiddenParams[0], RELU_LEAK, hiddenGradient);
rtxns::LinearOp_Backward<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(
    inputParams, hiddenGradient, gMLPParams, gMLPParamsGradients, weightOffsets[0], 
    biasOffsets[0], MATRIX_LAYOUT, TYPE_INTERPRETATION);
```

反向传播最终会把每个权重对应的梯度更新到 `gMLPParamsGradients` 中。

<a id="l2-loss-computation"></a>

#### L2 Loss 计算

下面这段代码用于在训练 shader 中计算每个 sample 的 L2 Loss，并把结果写入 `gLossBuffer`，以便后续做 batch reduction 和训练可视化：

```
float3 diff = predictedRGB - actualRGB;
gLossBuffer[dispatchThreadIdxy] = dot(diff, diff);
```


### Optimizer

正如训练循环中看到的，optimizer 会在每个训练 batch 之后执行一次。它的作用是做梯度下降，从而让模型逐步逼近最优解。当前 sample 使用的是 [Adam](https://arxiv.org/pdf/1412.6980)。

```
void adam_cs(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i >= maxParamSize)
        return;

    float gradient = (float)gMLPParamsGradients[i];
    gMLPParamsGradients[i] = half(0.0);

    // Get the floating point params, not float16
    float weightbias = gMLPParamsf[i];

    optimizers::Adam optimizer = optimizers::Adam(gMoments1, gMoments2, learningRate, LOSS_SCALE);

    float adjustedWeightbias = optimizer.step(weightbias, i, gradient, currentStep);

    gMLPParamsf[i] = adjustedWeightbias;
    gMLPParams[i] = (half)adjustedWeightbias;
}
```

这段 shader 逻辑相对直接：compute shader 遍历参数 buffer 中的每一个元素（可能是权重，也可能是偏置），然后调用 optimizer 去更新它。

这里使用的是 float32 版本的参数 buffer 来做梯度下降，以减小数值误差；更新完成后，再同步写回 float32 和 float16 两份 buffer。前面训练阶段施加的 `LOSS_SCALE` 会在这里恢复。梯度也会在使用后被清零，以备下一个 batch 继续训练。

如果需要，你也可以在 `Optimizers` 模块中增加新的优化算法来替换 Adam。

### Inference

推理 pass 基本就是训练阶段 forward 的精简版。当前因为它紧跟在训练 batch 后执行，所以仍然使用 `CoopVecMatrixLayout::TrainingOptimal`。如果单独抽成纯推理流程，更合理的 layout 应该是 `CoopVecMatrixLayout::InferencingOptimal`。

另外一点不同是：

- 推理阶段不再需要缓存每一层中间结果
- 输入来自 compute shader 的 thread ID，而不是随机数

```
// Set the input ID as the uv coordinate and frequency encode it for the network
float2 inputUV = float2(dispatchThreadID.x / float(gConst.imageWidth), dispatchThreadID.y / float(gConst.imageHeight));
CoopVec<VECTOR_FORMAT, INPUT_NEURONS> inputParams = rtxns::EncodeFrequency<half, 2>({inputUV.x, inputUV.y});

// Load offsets
uint weightOffsets[NUM_TRANSITIONS] = rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets);
uint biasOffsets[NUM_TRANSITIONS] = rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets);

CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenParams;
CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputParams;

// Forward propagation through the neural network
// Input to hidden layer, then apply activation function
hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(inputParams, gMLPParams, weightOffsets[0], biasOffsets[0], MATRIX_LAYOUT, TYPE_INTERPRETATION);
hiddenParams = rtxns::leakyReLU(hiddenParams, RELU_LEAK);

// Hidden layers to hidden layers, then apply activation function 
[ForceUnroll]
for (uint layer = 1; layer < NUM_HIDDEN_LAYERS; layer++)
{
    hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(hiddenParams, gMLPParams, weightOffsets[layer], biasOffsets[layer], MATRIX_LAYOUT, TYPE_INTERPRETATION);
    hiddenParams = rtxns::leakyReLU(hiddenParams, RELU_LEAK);
}

// Hidden layer to output layer, then apply final activation function
outputParams = rtxns::LinearOp<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(hiddenParams, gMLPParams, weightOffsets[NUM_HIDDEN_LAYERS], biasOffsets[NUM_HIDDEN_LAYERS], MATRIX_LAYOUT, TYPE_INTERPRETATION);
outputParams = rtxns::sigmoid(outputParams);

// Take the output from the neural network as the output color
float4 color = {outputParams[0], outputParams[1], outputParams[2], 1.f};
outputTexture[dispatchThreadID.xy] = color;
```
