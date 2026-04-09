# RTX Neural Shading：SlangPy Inferencing 示例

## 目的

这个 sample 展示了如何先在 Python 中借助 SlangPy 完成神经网络推理，再把同一套实现迁移到 C++。

它体现的是一种非常典型的工作流：

1. **原型阶段**：先在 Python + SlangPy 中快速试验
2. **部署阶段**：再把同样的 Slang 代码迁移到 C++ 应用中

因为 Python 版和 C++ 版复用了相同的 Slang 源代码，所以你既能利用 Python 的高迭代效率，也能在最终落地时获得 C++ 的性能。

这个 sample 同时提供了 Python 和 C++ 两套实现，它们执行的是同一个神经网络推理任务，因此非常适合用来理解两种环境之间如何平滑迁移。

<img src="slangpy_inferencing_chart.png" width="1000" alt="Transition chart">

## 动机

先用 SlangPy 和 Python 做实验，会更容易调整网络配置、快速验证效果。与此同时，Slang 的 autodiff 也能让训练代码的实现更简单。

当你对结果满意后，就可以把这套神经实现迁移到 C++ 图形引擎里。这个 sample 就是为了演示：如何在尽量少改动 Slang 神经网络实现的前提下，完成这次迁移。

这种工作流的主要优势包括：

1. **快速原型验证**：Python 的动态特性和生态更适合试验
2. **平滑迁移**：Python 和 C++ 共享同一套 Slang 逻辑，避免重复实现
3. **更高运行性能**：最终部署时由 C++ 提供实时性能
4. **统一代码来源**：神经网络逻辑只有一套“真实来源”
5. **更高调试效率**：先在 Python 里定位问题，再迁移到 C++

对于需要在“实验速度”和“运行效率”之间平衡的图形/机器学习工程师来说，这套流程非常有价值。

### Python 依赖

- Python 3.9 或更高版本
- `requirements.txt` 中列出的 Python 包：
  - SlangPy
  - NumPy

### 安装所需 Python 模块

在仓库根目录执行：

```sh
pip install -r samples/SlangpyInferencing/requirements.txt
```

### 运行 SlangPy 示例

在仓库根目录执行：

```sh
python samples\SlangpyInferencing\SlangpyInferencing.py
```

运行后会打开一个窗口，展示：

- 原始图像
- 推理结果图像
- 放大的误差图

<img src="slangpy_inferencing_window.png" width="800" alt="SlangPy Inferencing Steps">

## SlangPy 推理概览

Python 版 sample 使用 SlangPy 去调用用 Slang 编写的 GPU 代码。更详细内容可以参考 [官方文档](https://SlangPy.readthedocs.io/en/latest/)，这里仅做简要说明。

<img src="slangpy_inferencing_steps.png" width="600" alt="SlangPy Inferencing Steps">

### 架构概览

#### 设备初始化与 Slang 源码加载

首先通过 `app.py` 里的 `App` 类创建一个带 `Device` 的窗口：

```python
app = App(width=512*3+10*2, height=512, title="Mipmap Example", device_type=spy.DeviceType.vulkan)
```

然后加载一个 Slang 模块：

```python
module = spy.Module.load_from_file(app.device, "SlangpyInferencing_pyslang.slang")
```

#### 神经网络数据结构

在 Python 侧，神经网络由 `Network` 结构来表示：

```python
class Network(spy.InstanceList):
    def __init__(self, data: dict):
        super().__init__(module["Network"])
        assert len(data['layers']) == 3
        self.layer0 = NetworkParameters(data['layers'][0])
        self.layer1 = NetworkParameters(data['layers'][1])
        self.layer2 = NetworkParameters(data['layers'][2])
```

对应的 Slang 侧定义如下：

```c++
struct Network {
    NetworkParameters<16, 32>  layer0;
    NetworkParameters<32, 32>  layer1;
    NetworkParameters<32, 3>   layer2;

    float3 eval(no_diff float2 uv)
```

Python 里的 `NetworkParameters` 负责把权重和偏置转换成 cooperative vector 层。Slang 侧则为单层网络实现了 `forward`：

```c++
struct NetworkParameters<int Inputs, int Outputs>
{
    static const CoopVecComponentType ComponentType = CoopVecComponentType.Float16;
    ByteAddressBuffer weights, biases;

    CoopVec<half, Outputs> forward(CoopVec<half, Inputs> x)
    {
        return coopVecMatMulAdd<half, Outputs>(
            x, ComponentType,
            weights, 0, ComponentType,
            biases, 0, ComponentType,
            CoopVecMatrixLayout.InferencingOptimal, 
            false, 
            0
        );
    }
}
```

#### 推理过程

先从 JSON 文件加载权重和偏置，然后创建一个 `Network` 实例：

```python
trained_weights = json.load(open(AssetsPath / 'weights.json'))
network = Network(trained_weights)
```

在主循环里，程序会显示原图、调用 Slang 中的推理与 loss 函数，然后把输出结果 blit 到窗口上：

```python
while app.process_events():
    offset = 0
    app.blit(image, size=spy.int2(512), offset=spy.int2(offset,0), bilinear=True, tonemap=False)
    offset += 512 + 10
    res = spy.int2(256,256)

    lr_output = spy.Tensor.empty_like(image)
    module.inference(pixel = spy.call_id(), resolution = res, network = network, _result = lr_output)
    app.blit(lr_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), bilinear=True, tonemap=False)
    offset += 512 + 10

    loss_output = spy.Tensor.empty_like(image)
    module.loss(pixel = spy.call_id(),
                resolution = res,
                network = network,
                reference = image,
                _result = loss_output)
    app.blit(loss_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False)
    offset += 512 + 10

    app.present()
```

## C++ 推理概览

C++ 版本面向最终部署，目标是获得更高性能。SlangPy 非常适合原型开发，而 C++ 更适合实时应用。

C++ 版基于 `donut` 图形框架，并复用了与 Python 版相同的 Slang 代码。

<img src="slangpy_inferencing_steps_cpp.png" width="600" alt="C++ Inferencing Steps">

### 把 Slang 接进构建系统

在 Python 里，SlangPy 会自动加载并编译 Slang 代码；而在 C++ 中，则要通过 `slangc` 把 shader 编译成 DX12 / Vulkan 可执行二进制。

典型做法是把 Slang 编译过程集成到 CMake 里：

```cmake
include(../../external/donut/compileshaders.cmake)

set(SHADER_COMPILE_OPTIONS "--matrixRowMajor --hlsl2021" )
set(SHADER_COMPILE_OPTIONS_SPIRV " -X \"-Wno-41017 -capability spvCooperativeVectorNV -capability spvCooperativeVectorTrainingNV\" " )
set(SHADER_COMPILE_OPTIONS_DXIL " --shaderModel 6_9 --hlsl2021 -X \"-Wno-41012 -Wno-41016 -Wno-41017 -Xdxc -Vd\" " )

file(GLOB_RECURSE ${project}_shaders "*.hlsl" "*.hlsli" "*.slang")

donut_compile_shaders_all_platforms(
    TARGET ${project}_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    INCLUDES ${shader_includes}
    FOLDER ${folder}
    OUTPUT_BASE ${RTXNS_BINARY_DIR}/shaders/${project}
    SHADERMAKE_OPTIONS ${SHADER_COMPILE_OPTIONS}
    SHADERMAKE_OPTIONS_SPIRV ${SHADER_COMPILE_OPTIONS_SPIRV}
    SHADERMAKE_OPTIONS_DXIL ${SHADER_COMPILE_OPTIONS_DXIL}
    SOURCES ${${project}_shaders}
    SLANG
)
```

对应的 `shaders.cfg` 中，需要定义 Slang shader 的入口点：

```
SlangpyInferencing_cpp.slang -E inference_cs -T cs
```

### 初始化 Cooperative Vector 支持

如果要使用 cooperative vector，就必须在当前图形 API 上打开相关能力。

#### DirectX 12

要启用实验性 shader model 和 cooperative vector 支持：

```c++
UUID features[] = { D3D12ExperimentalShaderModels, D3D12CooperativeVectorExperiment };
HRESULT hr = D3D12EnableExperimentalFeatures(_countof(features), features, nullptr, nullptr);
```

#### Vulkan

在创建设备时，把 cooperative vector 扩展加入必需扩展列表：

```c++
deviceParams.requiredVulkanDeviceExtensions.push_back(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
```

### 加载神经网络参数

C++ 版和 Python 版一样，都会从同一个 JSON 文件读取网络参数：

```c++
m_networkUtils = std::make_shared<rtxns::NetworkUtilities>(GetDevice());
m_neuralNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);

if (!m_neuralNetwork->InitialiseFromJson(*nativeFS, (dataPath / "weights.json").string()))
{
    log::error("Failed to create a network.");
    return false;
}

assert(m_neuralNetwork->GetNetworkLayout().networkLayers.size() == 3);

m_deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(m_neuralNetwork->GetNetworkLayout(), rtxns::MatrixLayout::InferencingOptimal);
```

### GPU Buffer 分配与参数转换

首先要创建两套 buffer：

- 一套保存原始 host layout
- 一套保存 cooperative vector 可直接使用的 device-optimal layout

```c++
const auto& params = m_neuralNetwork->GetNetworkParams();

nvrhi::BufferDesc paramsBufferDesc;
paramsBufferDesc.byteSize = params.size();
paramsBufferDesc.debugName = "MLPParamsHostBuffer";
paramsBufferDesc.canHaveUAVs = true;
paramsBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
paramsBufferDesc.keepInitialState = true;
m_mlpHostBuffer = GetDevice()->createBuffer(paramsBufferDesc);

paramsBufferDesc.structStride = sizeof(uint16_t);
paramsBufferDesc.byteSize = m_deviceNetworkLayout.networkSize;
paramsBufferDesc.canHaveRawViews = true;
paramsBufferDesc.canHaveUAVs = true;
paramsBufferDesc.canHaveTypedViews = true;
paramsBufferDesc.format = nvrhi::Format::R16_FLOAT;
paramsBufferDesc.debugName = "MLPParamsByteAddressBuffer";
paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
m_mlpDeviceBuffer = GetDevice()->createBuffer(paramsBufferDesc);
```

然后上传 host 参数，并转换成 GPU 优化布局：

```c++
// Upload the host side parameters
m_commandList->setBufferState(m_mlpHostBuffer, nvrhi::ResourceStates::CopyDest);
m_commandList->commitBarriers();
m_commandList->writeBuffer(m_mlpHostBuffer, m_neuralNetwork->GetNetworkParams().data(), m_neuralNetwork->GetNetworkParams().size());

// Convert to GPU optimized layout
m_networkUtils->ConvertWeights(m_neuralNetwork->GetNetworkLayout(), 
        m_deviceNetworkLayout, m_mlpHostBuffer, 0, m_mlpDeviceBuffer, 0, GetDevice(), m_commandList);

// Update barriers for use
m_commandList->setBufferState(m_mlpDeviceBuffer, nvrhi::ResourceStates::ShaderResource);
m_commandList->commitBarriers();
```

### Compute Shader

在 Python 版中，SlangPy 会自动处理 compute shader 和 binding；而在 C++ 版中，你需要自己写 compute shader，明确调用 `inference` 和 `loss`：

```c++
DECLARE_CBUFFER(NeuralConstants, gConst, 0, 0);
RWTexture2D<float4> inferenceTexture : REGISTER_UAV(0, 0);
RWTexture2D<float4> lossTexture : REGISTER_UAV(1, 0);

Texture2D<float4> inputTexture : REGISTER_SRV(0, 0);

Network network;

[shader("compute")]
[numthreads(8, 8, 1)]
void inference_cs(uint3 pixel: SV_DispatchThreadID)
{
    inferenceTexture[pixel.xy].rgb = inference(pixel.xy, gConst.resolution, network);
    lossTexture[pixel.xy].rgb = loss(pixel.xy, gConst.resolution, inputTexture[pixel.xy].rgb, network);
}
```

为了让 `Network` 结构能拿到参数，需要给它加入合适的 binding：

```c++
struct NetworkParameters<int Inputs, int Outputs, int WeightReg, int BiasReg>
{
    static const CoopVecComponentType ComponentType = CoopVecComponentType.Float16;

    ByteAddressBuffer weights : REGISTER_SRV(WeightReg, 0);
    ByteAddressBuffer biases : REGISTER_SRV(BiasReg, 0);

    CoopVec<half, Outputs> forward(CoopVec<half, Inputs> x)
    {
        return coopVecMatMulAdd<half, Outputs>(
            x, ComponentType,
            weights, 0, ComponentType,
            biases, 0, ComponentType,
            CoopVecMatrixLayout.InferencingOptimal, false, 0
        );
    }
}

struct Network {
    NetworkParameters<16, 32, 0, 1> layer0;
    NetworkParameters<32, 32, 2, 3> layer1;
    NetworkParameters<32, 3, 4, 5>  layer2;
```

C++ 侧通常把所有神经网络参数都打包在一个总 buffer 中，因此需要把这块大 buffer 按不同 range 绑定到 Slang 侧：

```c++
nvrhi::BindingSetDesc bindingSetDesc;
bindingSetDesc.bindings = {
    nvrhi::BindingSetItem::ConstantBuffer(0, m_NeuralConstantBuffer),
    nvrhi::BindingSetItem::Texture_UAV(0, m_InferenceTexture),
    nvrhi::BindingSetItem::Texture_UAV(1, m_LossTexture),
    nvrhi::BindingSetItem::Texture_SRV(0, m_InputTexture),
};
{
    int i = 1;
    for (const auto& l : m_deviceNetworkLayout.networkLayers)
    {
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::RawBuffer_SRV(i++, m_mlpDeviceBuffer, nvrhi::BufferRange(l.weightOffset, l.weightSize)));
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::RawBuffer_SRV(i++, m_mlpDeviceBuffer, nvrhi::BufferRange(l.biasOffset, l.biasSize)));
    }
}

nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_BindingLayout, m_BindingSet);
```

最后，对输出纹理大小范围执行 compute dispatch：

```c++
nvrhi::ComputeState state;

// inference pass
state.bindings = { m_BindingSet };
state.pipeline = m_Pipeline;
m_commandList->beginMarker("Inference");
m_commandList->setComputeState(state);
m_commandList->dispatch(dm::div_ceil(m_InferenceTexture->getDesc().width, 8), dm::div_ceil(m_InferenceTexture->getDesc().height, 8), 1);
m_commandList->endMarker();
```

### 运行 C++ 示例

要运行 C++ 版 sample，只需要构建工程并执行 `SlangpyInferencing`。

运行后会看到一个与 Python 版类似的窗口，展示：

- 原始图像
- 神经推理结果
- 误差可视化

C++ 版使用与 Python 版完全相同的网络权重，并会得到视觉上一致的结果，只是运行在原生代码路径上，性能更高。

<img src="slangpy_inferencing_window_cpp.png" width="800" alt="C++ Inferencing Window">
