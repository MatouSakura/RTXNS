# RTX Neural Shading：Simple Inferencing 示例

## 目的

这个 sample 用来展示如何基于 `RTXNS` 提供的一些底层构件实现一个推理 shader。

它会：

- 从文件中加载一个已经训练好的网络
- 使用这个网络去近似 Disney BRDF shader
- 在运行时允许你交互调整光源和部分材质参数

![Simple Inferencing Output](simple_inferencing.png)

可执行文件构建并运行后，窗口中会显示一个被照亮的球体，着色部分由神经网络来近似 Disney BRDF。

## 推理流程

在 `RTXNS` 中加载一个推理用神经网络，通常需要经历以下几个阶段：

1. 创建 host 侧神经网络存储并初始化
2. 创建一份 GPU 侧拷贝，并用 host 侧参数填充
3. 在正常渲染循环中，用推理代码替换 Disney shader 的核心部分来给球体着色

## 应用层代码

Host 侧对神经网络的创建和使用其实比较直接，主要依赖 `RTXNS` 封装在图形 API Cooperative Vector 之上的一些抽象。

### 创建网络

这里会创建一个 `rtxns::Network`，并从文件中初始化。

为了保持跨平台可移植性，网络文件应当保存为非 GPU 优化格式，例如 `rtxns::MatrixLayout::RowMajor`，之后再在当前设备上转换成 GPU 优化布局：

```
 m_networkUtils = std::make_shared<rtxns::NetworkUtilities>(GetDevice());
rtxns::HostNetwork hostNetwork(m_networkUtils);
if (!net.initializeFromFile(GetLocalPath("assets/data").string() + std::string("/disney.ns.bin")))
{
    log::debug("Loaded Neural Shading Network from file failed.");
    return false;
}

// Get a device optimized layout
rtxns::NetworkLayout deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(hostNetwork.GetNetworkLayout(), rtxns::MatrixLayout::InferencingOptimal);
```

这段代码会：

- 从文件里读取网络定义和参数
- 为每层权重和偏置分配一块连续的 host 内存
- 用文件中的数据填充这些参数
- 同时创建一个 GPU 侧优化布局 `rtxns::MatrixLayout::InferencingOptimal`

底层会借助 `CoopVector` 扩展去查询矩阵实际大小，并完成布局转换。

### GPU Buffer 分配

#### Float16 参数 Buffer

这里需要两份参数 buffer：

- 一份保存 host layout
- 一份保存 device-optimal layout

这两份 buffer 中存放的都是网络所有层的权重和偏置，通常使用 `float16` 精度。

当 host layout buffer 填充完成后，就可以把它转换成 device layout。后者会在推理 shader 中直接作为 Slang CoopVector 函数的输入。

```
// Create a buffer for the host side weight and bias parameters
nvrhi::BufferDesc bufferDesc;
bufferDesc.byteSize = hostNetwork.GetNetworkParams().size();
bufferDesc.debugName = "hostParamsBuffer";
bufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
bufferDesc.keepInitialState = true;
m_mlpHostBuffer = GetDevice()->createBuffer(bufferDesc);

// Create a buffer for a device optimized parameters layout
bufferDesc.byteSize = deviceNetworkLayout.networkSize;
bufferDesc.canHaveRawViews = true;
bufferDesc.canHaveUAVs = true;
bufferDesc.debugName = "deviceParamBuffer";
bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
m_mlpDeviceBuffer = GetDevice()->createBuffer(bufferDesc);

 // Upload the parameters
m_commandList->writeBuffer(m_mlpHostBuffer, params.data(), params.size());

// Convert to GPU optimized layout
m_networkUtils->ConvertWeights(hostNetwork.GetNetworkLayout(), deviceNetworkLayout, m_mlpHostBuffer, 0, m_mlpDeviceBuffer, 0, GetDevice(), m_commandList);

```

### 渲染循环

在这个 sample 中，推理 shader 逻辑直接从 pixel shader 内调用，因此渲染循环几乎不需要额外修改，只要确保参数 buffer 正确绑定即可。

## Shader 代码

正如前面提到的，这个 sample 的目标是用神经网络去近似 Disney BRDF shader。作为对照，调用 Disney shader 的代码大致会是这样：

```
void main_ps(float3 i_norm, float3 i_view, out float4 o_color : SV_Target0)
{
    //----------- Prepare input parameters
    float3 view = normalize(i_view);
    float3 norm = normalize(i_norm);
    float3 h = normalize(-lightDir.xyz + view);

    float NdotL = max(0.f, dot(norm, -lightDir.xyz));
    float NdotV = max(0.f, dot(norm, view));
    float NdotH = max(0.f, dot(norm, h));
    float LdotH = max(0.f, dot(h, -lightDir.xyz));

    //----------- Calculate core shader part DIRECTLY
    float4 outParams = DisneyBRDF(NdotL, NdotV, NdotH, LdotH, roughness);

    //----------- Calculate final color
    float3 Cdlin = float3(pow(baseColor[0], 2.2), pow(baseColor[1], 2.2), pow(baseColor[2], 2.2));
    float3 Cspec0 = lerp(specular * .08 * float3(1), Cdlin, metallic);
    float3 brdfn = outParams.x * Cdlin * (1-metallic) + outParams.y*lerp(Cspec0, float3(1), outParams.z) + outParams.w;
    float3 colorh = brdfn * float3(NdotL) * lightIntensity.rgb;

    o_color = float4(colorh, 1.f);
}
```

我们的目标是只替换掉 `DisneyBRDF()` 这一段，把它换成神经网络版本 `DisneyMLP`，其余着色代码尽量保持不变。

### 网络配置

神经网络的实际大小也可以从模型文件中读取出来，但为了简化示例，它也被硬编码在 [NetworkConfig.h](../samples/SimpleInferencing/NetworkConfig.h) 中，并在应用和 shader 之间共享：

```
#define VECTOR_FORMAT half
#define TYPE_INTERPRETATION CoopVecComponentType::Float16

#define INPUT_FEATURES 5
#define INPUT_NEURONS (INPUT_FEATURES * 6) // Frequency encoding increases the input by 6 for each input 
#define OUTPUT_NEURONS 4
#define HIDDEN_NEURONS 32
```

因此这个网络当前的结构是：

- 输入神经元：30 个
- 输出神经元：4 个
- 每层隐藏层神经元：32 个
- 精度：`float16`

### 输入参数

当前这个 Disney BRDF 模型会先把输入编码到 `0-1` 的频率域中，这通常更适合神经网络处理：

```
  float params[INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, roughness };
  inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);
```

### 推理 Shader

推理 shader 使用的是原生 Slang `CoopVec` 类型。它用 cooperative vector 的形式把神经网络的权重和偏置映射到底层硬件（例如 tensor core）。更完整的 `CoopVec` 说明可以参考 [Library Guide](LibraryGuide.md)，这里先做一个简要介绍。

```
  CoopVec<VECTOR_FORMAT, INPUT_NEURONS> inputParams;
```

这段代码声明了一个长度为 `INPUT_NEURONS`、精度为 `VECTOR_FORMAT` 的 `CoopVec`。

结合前面的宏定义，这里实际对应的是：

```
  CoopVec<half, 30> inputParams;
```

底层实现可能对向量大小有硬件限制，例如要求按 32 对齐，但 Slang 的 `CoopVec` 可以按逻辑大小创建，编译器会根据需要自动补齐。概念上，这些 `CoopVec` 很像 PyTorch 里的 tensor：每层网络以 cooperative vector 为输入，再输出另一个 cooperative vector。

为了执行推理模型，这里使用 `rtxns` 命名空间中的模板函数，把输入 cooperative vector 按层前向传播 through 网络。例如下面的 `LinearOp`，会把大小为 `INPUT_NEURONS` 的输入线性映射到 `HIDDEN_NEURONS`：

```
hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(...)
```

它可以理解成类似 `torch.nn.Linear` 的操作。

`LinearOp` 定义在 [LinearOps.slang](../src/NeuralShading_Shaders/LinearOps.slang) 中，本质上是对原生 `CoopVec` 接口的一层便利封装。它底层最终调用的是：

```
coopVecMatMulAdd<Type, Size>(...)
```

在线性层之后，通常还要接激活函数。这个例子里使用的是 `rtxns` 命名空间中的 `relu`，实现位于 [CooperativeVectorFunctions.slang](../src/NeuralShading_Shaders/CooperativeVectorFunctions.slang)：

```
hiddenParams = rtxns::relu(hiddenParams);
```

这套线性层 + 激活函数的流程会对网络的 4 层依次执行（1 层输入到隐藏层、3 层后续过渡）。最终输出是一个 `float4`，表示 Disney BRDF 近似结果，用于后续颜色计算：

```
CoopVec<VECTOR_FORMAT, INPUT_NEURONS> inputParams;
CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenParams;
CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputParams;

// Encode input parameters, 5 inputs to 30 parameters 
float params[INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, roughness };
inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);

// Forward propagation through the neural network
// Input to hidden layer, then apply activation function
hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(
    inputParams, gMLPParams, weightOffsets[0], biasOffsets[0], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
hiddenParams = rtxns::relu(hiddenParams);

// Hidden layer to hidden layer, then apply activation function 
hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
    hiddenParams, gMLPParams, weightOffsets[1], biasOffsets[1], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
hiddenParams = rtxns::relu(hiddenParams);

// Hidden layer to hidden layer, then apply activation function    
hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
    hiddenParams, gMLPParams, weightOffsets[2], biasOffsets[2], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
hiddenParams = rtxns::relu(hiddenParams);

// Hidden layer to output layer, then apply final activation function
outputParams = rtxns::LinearOp<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(
    hiddenParams, gMLPParams, weightOffsets[3], biasOffsets[3], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
outputParams = exp(outputParams);

// Take the output from the neural network as the output color
return float4(outputParams[0], outputParams[1], outputParams[2], outputParams[3]);
```

### 最终颜色计算

一旦网络输出了这 4 个值，它们就会直接接回后续普通着色代码中：

```
float3 Cdlin = float3(pow(baseColor[0], 2.2), pow(baseColor[1], 2.2), pow(baseColor[2], 2.2));
float3 Cspec0 = lerp(specular * .08 * float3(1), Cdlin, metallic);
float3 brdfn = outParams.x * Cdlin * (1-metallic) + outParams.y*lerp(Cspec0, float3(1), outParams.z) + outParams.w;
float3 colorh = brdfn * float3(NdotL) * lightIntensity.rgb;

o_color = float4(colorh, 1.f);

```

### 完整代码

```
float4 DisneyMLP(float NdotL, float NdotV, float NdotH, float LdotH, float roughness)
{   
    uint4 weightOffsets = gConst.weightOffsets; 
    uint4 biasOffsets = gConst.biasOffsets;  

   CoopVec<VECTOR_FORMAT, INPUT_NEURONS> inputParams;
   CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenParams;
   CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputParams;

   // Encode input parameters, 5 inputs to 30 parameters 
   float params[INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, roughness };
   inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);
   
   // Forward propagation through the neural network
   // Input to hidden layer, then apply activation function
   hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(
       inputParams, gMLPParams, weightOffsets[0], biasOffsets[0], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
   hiddenParams = rtxns::relu(hiddenParams);
   
   // Hidden layer to hidden layer, then apply activation function 
   hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
       hiddenParams, gMLPParams, weightOffsets[1], biasOffsets[1], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
   hiddenParams = rtxns::relu(hiddenParams);
   
   // Hidden layer to hidden layer, then apply activation function    
   hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
       hiddenParams, gMLPParams, weightOffsets[2], biasOffsets[2], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
   hiddenParams = rtxns::relu(hiddenParams);
   
   // Hidden layer to output layer, then apply final activation function
   outputParams = rtxns::LinearOp<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(
       hiddenParams, gMLPParams, weightOffsets[3], biasOffsets[3], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
   outputParams = exp(outputParams);

    // Take the output from the neural network as the output color
    return float4(outputParams[0], outputParams[1], outputParams[2], outputParams[3]);
}

[shader("fragment")]
void main_ps( 
    VertexOut vOut,
    out float4 o_color : SV_Target0)
{
    float4 lightIntensity = gConst.lightIntensity;
    float4 lightDir =  gConst.lightDir;
    float4 baseColor = gConst.baseColor;
    float specular = gConst.specular;
    float roughness = gConst.roughness;
    float metallic = gConst.metallic;

    // Prepare input parameters
    float3 view = normalize(vOut.view);
    float3 norm = normalize(vOut.norm);
    float3 h = normalize(-lightDir.xyz + view);

    float NdotL = max(0.f, dot(norm, -lightDir.xyz));
    float NdotV = max(0.f, dot(norm, view));
    float NdotH = max(0.f, dot(norm, h));
    float LdotH = max(0.f, dot(h, -lightDir.xyz));

    // Calculate approximated core shader part using MLP
    float4 outParams = DisneyMLP(NdotL, NdotV, NdotH, LdotH, roughness);

    // Calculate final color
    float3 Cdlin = float3(pow(baseColor.r, 2.2), pow(baseColor.g, 2.2), pow(baseColor.b, 2.2));
    float3 Cspec0 = lerp(specular * .08f * float3(1,1,1), Cdlin, metallic);
    float3 brdfn = outParams.x * Cdlin * (1 - metallic) + outParams.y * lerp(Cspec0, float3(1), outParams.z) + outParams.w;
    float3 colorh = brdfn * float3(NdotL) * lightIntensity.rgb;

    o_color = float4(colorh, 1.f);
 }
```

和本节开头的原始 shader 对比起来，唯一真正变化的地方，就是把原始的 `DisneyBRDF()` 替换成了神经网络执行函数 `DisneyMLP()`。
