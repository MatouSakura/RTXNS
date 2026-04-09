# RTX Neural Shading：库使用指南

这个库大致可以分成两个部分：

- 应用侧
- shader 侧

应用侧提供了一组辅助函数，用来：

- 创建神经网络
- 把网络序列化到磁盘 / 从磁盘反序列化
- 修改网络精度和矩阵布局
- 分配和销毁底层存储

它们建立在 `nvrhi` SDK 之上，因此对图形 API 本身相对无关，也可以较容易迁移到别的引擎体系中。

shader 侧则提供了训练和推理小型神经网络所需的 Slang 辅助函数。

## 应用层代码

应用层中最核心的神经网络工具类定义在 `NeuralNetwork.h` 中，主要包括：

- `rtxns::HostNetwork`
- `rtxns::NetworkUtilities`

其中：

- `rtxns::HostNetwork` 负责 host 侧权重和偏置的分配，以及网络文件的加载 / 保存
- `rtxns::NetworkUtilities` 负责在 host matrix layout 和 device-optimal matrix layout 之间做转换

`rtxns::HostNetwork` 在使用前必须先创建并初始化。它可以通过以下几种方式初始化：

- 用输入参数描述的网络结构初始化
- 从文件初始化
- 从另一个 `rtxns::Network` 初始化

无论哪种方式，初始化后的网络一开始都处于 host layout，也就是：

- `rtxns::MatrixLayout::RowMajor`
- 或 `rtxns::MatrixLayout::ColumnMajor`

### 从参数初始化网络

```cpp
// Initialise an empty network from parameters
nvrhi::IDevice* device = ...
rtxns::HostNetwork hostNetwork = rtxns::HostNetwork(device);

rtxns::NetworkArchitecture netArch = {};
netArch.inputNeurons = 2;
netArch.hiddenNeurons = 32;
netArch.outputNeurons = 3;
netArch.numHiddenLayers = 3;
netArch.biasPrecision = rtxns::Precision::f16;
netArch.weightPrecision = rtxns::Precision::f16;

if (!hostNetwork.Initialise(netArch))
    log::error("Failed to create a network from an arch!");
```

### 从文件初始化网络

```cpp
// Initialise a network from a file
nvrhi::IDevice* device = ...
rtxns::HostNetwork hostNetwork = rtxns::Network(device);
if (!hostNetwork.InitialiseFromFile("myNN.bin"))
    log::error("Failed to create a network from myNN.bin!");
```

创建网络时，只会分配 host 侧用于存放每层权重和偏置的内存，并不会自动分配 GPU 内存。相反，它会把大小和 offset 等信息整理好，供你自己分配 GPU buffer 并拷贝数据。

Host 侧的权重和偏置尺寸已经可以直接拷到 GPU。参数可以通过参数访问器获得，offset 则从 layer 信息里读取：

```cpp
const std::vector<uint8_t>& params = neuralNetwork.GetNetworkParams();

// Copy to GPU buffer
copy(hostBuffer, params.data(), params.size());
```

### 矩阵布局

网络内部有“矩阵布局”的概念，它可以分成 host layout 和 device-optimal layout 两类。host layout 当然也可以在 GPU 上使用，但性能通常不如 device-optimal layout。

```cpp
enum class MatrixLayout
{
    RowMajor,
    ColumnMajor,
    InferencingOptimal,
    TrainingOptimal,
};
```

其中：

- `rtxns::MatrixLayout::RowMajor`
- `rtxns::MatrixLayout::ColumnMajor`

属于 host layout。因为它们与具体硬件和 API 无关，所以适合写入文件。

而：

- `rtxns::MatrixLayout::InferencingOptimal`
- `rtxns::MatrixLayout::TrainingOptimal`

属于 device-optimal layout。它们是与硬件强相关的“黑盒格式”，不能保证跨 GPU、跨 API 直接通用，且通常会带有额外的对齐和 padding 要求。

### 网络的一般生命周期

一个网络的典型生命周期通常是：

1. 先在 host layout 中创建或从文件加载
2. 上传到 GPU
3. 调用 `rtxns::NetworkUtilities::ConvertWeights` 转成 device-optimal layout
4. 训练时使用 `rtxns::MatrixLayout::TrainingOptimal`
5. 训练完成后再转成 `rtxns::MatrixLayout::InferencingOptimal`
6. 如果要保存模型，再转回 host layout 并写入文件

### 创建新的 device layout

如果要修改网络布局，通常先调用 `rtxns::NetworkUtilities::GetNewMatrixLayout()`，为相同网络结构创建一份新的 device-optimal layout。随后把权重和偏置的 offset 取出来，通过 constant buffer 传给 GPU。

```cpp
// Get a device optimized layout
rtxns::NetworkLayout deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(neuralNetwork.GetNetworkLayout(), rtxns::MatrixLayout::TrainingOptimal);

// Store the device layout offsets 
weightOffsets = dm::uint4(
    deviceNetworkLayout.networkLayers[0].weightOffset,
    deviceNetworkLayout.networkLayers[1].weightOffset,
    deviceNetworkLayout.networkLayers[2].weightOffset,
    deviceNetworkLayout.networkLayers[3].weightOffset);
biasOffsets = dm::uint4(
    deviceNetworkLayout.networkLayers[0].biasOffset,
    deviceNetworkLayout.networkLayers[1].biasOffset,
    deviceNetworkLayout.networkLayers[2].biasOffset,
    deviceNetworkLayout.networkLayers[3].biasOffset);

```

在为 `deviceNetworkLayout.networkSize` 分配好 GPU buffer 之后，再调用 `rtxns::NetworkUtilities::ConvertWeights()`，把 host layout 转成 device-optimal layout：

```cpp
ConvertWeights(hostNetwork.GetNetworkLayout(),
    deviceNetworkLayout,
    hostBuffer,
    hostOffset,
    deviceBuffer,
    deviceOffset,
    device,
    commandList);
```

此时这份 device buffer 就可以直接用于训练了。

从 `TrainingOptimal` 转到 `InferencingOptimal` 时，也要再次执行：

- `rtxns::NetworkUtilities::GetNewMatrixLayout()`
- `rtxns::NetworkUtilities::ConvertWeights()`

而当你想把训练结果保存回 host layout 时，同样也是通过 `ConvertWeights` 完成转换。这个过程在 `rtxns::HostNetwork::UpdateFromBufferToFile()` 中已经做了封装。

### Cooperative Vectors

如果你想自己实现神经网络类，而不是只使用现成封装，那么值得重点研究 `CoopVector.h` 中的 `ICoopVectorUtils`，以及它在 `NeuralNetwork.h` 中的用法。

它提供了一个对 Vulkan 和 DX12 Cooperative Vector 扩展相对无关的接口，主要用来：

- 查询矩阵大小
- 在 GPU 上进行不同布局之间的数据转换
- 处理不同精度下的矩阵表示

### Loss 可视化

训练过程中，如果想把每个 sample 的 loss 汇总、回读并最终画出来，通常要经历下面几个步骤：

1. **把每个 sample 的 loss 写入 Loss Buffer**
   在每个训练 batch 中，为每个 sample 计算一个标量 loss，并写入 GPU loss buffer

2. **把 batch loss 累加到 Accumulation Buffer**
   每个 batch 完成后，运行 loss reduction shader，把当前 batch 的总 loss 原子地累加到全局 accumulation buffer

3. **在 epoch 结束时计算平均 loss**
   当这一轮 epoch 的所有 batch 都处理完成后，计算平均 epoch loss，并写入输出 buffer

4. **把 epoch 结果回读到 CPU**
   将输出 buffer 拷回 CPU，从而获取最终 loss 指标

5. **使用 ImPlot 绘图**
   把 epoch loss 追加到历史数组里，再用 ImPlot 绘制曲线

SDK 已经提供了若干帮助类来实现这一流程。后文中的 [Loss Reduction Shader Workflow](#loss-reduction-shader-workflow) 章节展示了 reduction shader 的典型写法。

#### ResultsReadbackHandler

`ResultsReadbackHandler` 负责在每个 epoch 结束时，把训练结果从 GPU 拷回 CPU。

它提供了一个比较简单的接口，用来：

- 同步 GPU 输出 buffer
- 读取最终 loss 值
- 为下一次训练清空内部状态

```cpp
class ResultsReadbackHandler
{
public:
    ResultsReadbackHandler(nvrhi::DeviceHandle device);
    void SyncResults(nvrhi::CommandListHandle commandList);
    nvrhi::BufferHandle GetResultsBuffers() const;
    bool GetResults(TrainingResults& results) const;
    void Reset();
};
```

几个关键接口分别是：

`void SyncResults(nvrhi::CommandListHandle commandList)`

会提交 GPU copy 命令，把训练结果 buffer 拷到 CPU 可见的 staging buffer 中。通常应当在 GPU 已经计算出 epoch 结果之后调用，也就是训练循环每个 epoch 结束时调用。

`bool GetResults(TrainingResults& results) const`

尝试从 staging buffer 中读取训练结果。如果当前还没有通过 `SyncResults` 完成同步，则返回 `false`。

#### ResultsWidget

`ResultsWidget` 是一个简单的 UI 组件，继承自 `IWidget`，负责按 epoch 展示训练结果。

它会：

- 收集新的 loss 指标
- 维护历史数据
- 使用 **ImPlot** 画出训练曲线

```cpp
class ResultsWidget : public IWidget
{
public:
    ResultsWidget();
    void Draw() override;
    void Reset();
    void Update(const TrainingResults& trainingResults);

private:
    std::vector<float> m_epochHistory;
    std::vector<float> m_averageL2LossHistory;
};
```

## Shader 侧代码

shader 库主要分为几个模块。

### 线性操作模块

线性操作模块是训练和推理的核心，主要提供：

- `LinearOp`
- `LinearOp_Backward`

同时它还提供了一个可与 Slang autodiff 协作的 `LinearOp` backward derivative 实现。

`LinearOp` 用于在神经网络中执行一次线性前向步骤：把大小为 `K` 的输入层映射到大小为 `M` 的下一层。权重和偏置都存放在同一个 buffer 中。`CoopVecMatrixLayout` 指定权重矩阵的布局，它应当与 C++ 侧的 `MatrixLayout` 一致；`CoopVecComponentType` 则指定矩阵按什么精度解释，通常应与类型 `T` 相匹配。

```cpp
CoopVec<T, M> LinearOp<T : __BuiltinFloatingPointType, let M : int, let K : int>( 
    CoopVec<T, K> ip, 
    ByteAddressBuffer matrixBiasBuffer, 
    uint matrixOffset, 
    int biasOffset, 
    constexpr CoopVecMatrixLayout matrixLayout, 
    constexpr CoopVecComponentType componentType)
```

`LinearOp_Backward` 用于反向传播时的线性步骤：把大小为 `M` 的梯度应用到前一层大小为 `K` 的输入上。权重、偏置以及它们的导数分别存放在对应 buffer 中。

```cpp
 CoopVec<T, K> LinearOp_Backward<T : __BuiltinFloatingPointType, let M : int, let K : int>(
    CoopVec<T, K> ip, 
    CoopVec<T, M> grad, 
    ByteAddressBuffer matrixBiasBuffer, 
    RWByteAddressBuffer matrixBiasBufferDerivative, 
    uint matrixOffset, 
    int biasOffset, 
    constexpr CoopVecMatrixLayout matrixLayout, 
    constexpr CoopVecComponentType componentType)
```

#### 可微的 LinearOps

模块的后半部分扩展了 cooperative vector，使其能与 Slang 自动微分配合使用，因为这部分能力原生并不直接可用。

`MatrixBiasBuffer` 和 `MatrixBiasBufferDifferential` 都继承自 Slang 的 `IDifferentiablePtrType`，从而让矩阵 buffer 和它的导数支持 autodiff：

```cpp
struct MatrixBiasBufferDifferential : IDifferentiablePtrType
    {
        typealias Differential = MatrixBiasBufferDifferential;

        __init(RWByteAddressBuffer buf) 
        { 
            buffer = buf;
        }

        RWByteAddressBuffer buffer;
};

struct MatrixBiasBuffer : IDifferentiablePtrType
    {
        typealias Differential = MatrixBiasBufferDifferential;

        __init(ByteAddressBuffer buf) 
        { 
            buffer = buf;
        }

        ByteAddressBuffer buffer;
};
```

随后，这个模块又提供了一个可微版本的 `LinearOp`，它把 `matrixBiasBuffer` 替换成了 `MatrixBiasBuffer`，并且通过 `offsets` 传递权重和偏置位置：

```cpp
CoopVec<T, M> LinearOp<T : __BuiltinFloatingPointType, let M : int, let K : int>( 
    CoopVec<T, K> ip, 
    MatrixBiasBuffer matrixBiasBuffer, 
    uint2 offsets,
    constexpr CoopVecMatrixLayout matrixLayout, 
    constexpr CoopVecComponentType componentType)
```

`LinearOp_BackwardAutoDiff` 则是这个 `LinearOp` 的 backward derivative，其中输入 `CoopVec` 和 `MatrixBiasBuffer` 都通过 `DifferentialPair` 传递。更具体的语义可以参考 Slang 的 autodiff 文档。

```cpp
[BackwardDerivativeOf(LinearOp)]
void LinearOp_BackwardAutoDiff<T : __BuiltinFloatingPointType, let M : int, let K : int>( 
    inout DifferentialPair<CoopVec<T, K>> ip, 
    DifferentialPtrPair<MatrixBiasBuffer> MatrixBiasBuffer, 
    uint2 offsets,
    constexpr CoopVecMatrixLayout matrixLayout, 
    constexpr CoopVecComponentType componentType, 
    CoopVec<T, M>.Differential grad)
```

### 激活函数模块

这个模块提供了一系列常见激活函数实现：

```cpp
struct NoneAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct LinearAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct ExponentialAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct ShiftedExponentialAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct ReLUAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct LeakyReLUAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct SigmoidAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct SwishAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct TanhAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

```

### MLP 模块

MLP 模块里有两个核心结构：

- `InferenceMLP`
- `TrainingMLP`

二者都能执行完整的 forward pass，但使用场景不同：

- `InferenceMLP` 只负责推理
- `TrainingMLP` 负责训练，除了 forward 之外，还包含额外的导数 buffer 和 backward 接口

`TrainingMLP` 的 backward 不是手写出来的，而是利用 Slang 的 autodiff 自动生成。

#### InferenceMLP

```cpp
 struct InferenceMLP<
        T : __BuiltinFloatingPointType, 
        let HIDDEN_LAYERS : int, 
        let INPUTS : int, 
        let HIDDEN : int, 
        let OUTPUTS : int, 
        let matrixLayout : CoopVecMatrixLayout, 
        let componentType : CoopVecComponentType
    >
    {
        ...
        CoopVec<T, OUTPUTS> forward<Act : IActivation<T, HIDDEN>, FinalAct : IActivation<T, OUTPUTS>>(
            CoopVec<T, INPUTS> inputParams, 
            Act act, 
            FinalAct finalAct);
        ...
    }
```

#### TrainingMLP

```cpp
 struct TrainingMLP<
        T : __BuiltinFloatingPointType, 
        let HIDDEN_LAYERS : int, 
        let INPUTS : int, 
        let HIDDEN : int, 
        let OUTPUTS : int, 
        let matrixLayout : CoopVecMatrixLayout, 
        let componentType : CoopVecComponentType
    >
    {
        ...
        CoopVec<T, OUTPUTS> forward<Act : IActivation<T, HIDDEN>, FinalAct : IActivation<T, OUTPUTS>>(CoopVec<T, INPUTS> inputParams, Act act, FinalAct finalAct);

        void backward<Act : IActivation<T, HIDDEN>, FAct : IActivation<T, OUTPUTS>>(CoopVec<T, INPUTS> ip, Act act, FAct fact, CoopVec<T, OUTPUTS> loss);
        ...
    }
```

### Optimizer 模块

这个模块定义了若干优化器实现的统一接口。

统一接口要求每个优化器都实现一步更新：

```cpp
interface IOptimizer
{
    float step(float weightBias, uint parameterID, float gradient, const float currentStep);
};
```

当前模块中提供了 Adam 优化器实现。它额外引入了两个 moment buffer 和一些超参数：

```cpp
struct Adam : IOptimizer
{
    RWBuffer<float> m_moments1;
    RWBuffer<float> m_moments2;
    float m_learningRate;
    float m_lossScale;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
}
```

### Utility 模块

这个模块主要提供：

- 输入编码
- 权重 / 偏置 offset 解包

编码器的目的，是扩展神经网络输入维度，为学习过程提供更丰富的信息。是否真的提升质量或性能，仍然应该结合具体任务验证。

`CoopVecFromArray` 用于从一个 float 数组创建同尺寸的 `CoopVec`：

```cpp
CoopVec<T, PARAMS_COUNT> CoopVecFromArray<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(float parameters[PARAMS_COUNT])
```

`EncodeFrequency` 会把每个输入参数扩展为 6 个分量，并通过正弦 / 余弦波形式编码：

```cpp
CoopVec<T, PARAMS_COUNT * FREQUENCY_ENCODING_COUNT> EncodeFrequency<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(float parameters[PARAMS_COUNT])
```

`EncodeTriangle` 也会把每个输入扩展为 6 个分量，但编码形式是三角波：

```cpp
CoopVec<T, PARAMS_COUNT * TRIANGLE_ENCODING_COUNT> EncodeTriangle<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(float parameters[PARAMS_COUNT])
```

`UnpackArray` 用于把 constant buffer 中按 `uint4` 打包对齐的权重 / 偏置 offset 解包出来：

```cpp
uint[NUM_UNPACKED] UnpackArray<let NUM_PACKED4 : int, let NUM_UNPACKED : int>(uint4 ps[NUM_PACKED4])
```

### LossAccumulation 模块

`LossAccumulation` 模块提供了一套轻量、适合 GPU 的浮点 loss 累加机制，用来把多分量 loss 累加进 `RWByteAddressBuffer`。

它支持：

- 动态长度的 loss 向量
- 对组件数量做安全裁剪
- 可移植的原子浮点加法（DXIL + 厂商相关路径）
- 初始化、相加和累加 loss 数组的辅助函数

这个模块适合：

- 神经网络训练循环
- 指标收集
- 以及所有需要在 GPU 上做多分量浮点归约的场景

#### 常量

```cpp
public static const uint LOSS_ACCUM_MAX_COMPONENTS = 128;
```

它表示 loss 向量允许累加的最大组件数。任何请求的分量数都会被裁剪到这个上限。

#### LossConfig

```cpp
public struct LossConfig
{
    uint componentCount;
    uint baseByteOffset;

    public uint GetComponentCount()
    { 
        return min(componentCount, LOSS_ACCUM_MAX_COMPONENTS);
    }
}
```

`LossConfig` 描述了一段 loss 分量在目标 buffer 中的存放方式：

- `componentCount`：需要累加的分量个数
- `baseByteOffset`：在 `RWByteAddressBuffer` 中的起始字节偏移

`GetComponentCount()` 会保证最终值不超过 `LOSS_ACCUM_MAX_COMPONENTS`。

每个分量占用 4 字节，也就是一个 32-bit float。

#### 原子浮点加法

##### 基于 CAS 的 Atomic Add（DXIL 回退实现）

```cpp
float AtomicAddFloat(RWByteAddressBuffer buffer, uint byteOffset, float valueToAdd)
```

这个函数通过 compare-and-swap 实现原子浮点加法。当原生 `InterlockedAddF32` 不可用时就使用它。返回值是更新前的旧值。

##### 可移植选择封装

```cpp
public float AtomicAddF32Portable(
    RWByteAddressBuffer buffer,
    uint byteOffset,
    float value)
```

它会自动选择：

- 原生 `InterlockedAddF32`
- 或在 DXIL 上使用 `AtomicAddFloat` 作为回退路径

#### 组件数组辅助函数

##### 清零数组

```cpp
public void Zero<let N : int>(out float components[N])
```

把长度为 `N` 的数组全部初始化为 `0.0f`。适合初始化每线程或每 sample 的临时 loss 累加器。

##### 数组逐项相加

```cpp
public void AddInPlace<let N : int>(
    inout float dst[N],
    float src[N],
    uint componentCount)
```

把 `src` 中前 `componentCount` 个分量累加到 `dst` 中。如果 `componentCount < N`，就只加前面那一部分。

#### 累加到内存

```cpp
public void AccumulateComponents<let N : int>(
    RWByteAddressBuffer buffer,
    float components[N],
    LossConfig config)
```

这个函数会从 `config.baseByteOffset` 开始，把每个分量累加到目标 buffer 中。

##### 行为说明

- 字节偏移计算方式为：`byteOffset = config.baseByteOffset + (i * 4)`
- 每个分量通过 `AtomicAddF32Portable` 执行原子浮点加
- 实际更新分量数不会超过 `config.GetComponentCount()`

##### 典型用法

1. 每线程或每 sample 先维护一个局部的分量数组
2. 先做局部 reduction
3. 再调用 `AccumulateComponents` 把它合并到全局 GPU buffer 中

<a id="loss-reduction-shader-workflow"></a>

### Loss Reduction Shader 工作流

在 reduction 之前，每个 sample 都会先算出自己的标量 loss。具体可以参考：

- [SimpleTraining](SimpleTraining.md#l2-loss-computation)
- [ShaderTraining](ShaderTraining.md#l2-relative-loss-computation)

`lossReduction_cs` compute shader 的工作，是把这些 per-sample loss 归约成一个 batch 级别的 loss，并进一步累加到全局 buffer，供 epoch 级统计使用。

每个线程会从 `lossBuffer` 中读取一个标量 loss（索引不超过 `gConst.batchSize`）。这些值会先写进 group shared 内存 `gLossShared`，然后在 thread group 内做树形 reduction，借助 `AddInPlace` 把数据就地累加。最终 reduction 完成后，线程 `0` 持有整个 batch 的总 loss。

接下来，这个 batch loss 会通过 `AccumulateComponents` 和 `MakeLossConfig` 原子地写入全局 `accumulationBuffer`。因为这里的 `MAX_COMPONENTS` 设为 `1`，所以它本质上就是按 batch 累加一个标量 loss，非常适合：

- 计算 epoch 平均值
- 或者绘制训练曲线

#### 可适配任意 per-sample 标量 loss

这种 reduction 模式本身并不依赖具体 loss 的定义。任何阶段只要能往 `lossBuffer` 中写“每个 sample 一个 float”，这个 reduction shader 都能正常工作。例如：

- 每像素 / 每 sample 的 MSE
- 每 sample 的 L2、L1 或 Huber loss
- 分类问题里的负对数似然
- 任何自定义标量 loss

只要每个 sample 最终贡献的是一个标量，这个 shader 就能把 batch 中的所有值求和，并进一步为 epoch 级统计和可视化服务。

```cpp
static const uint THREADS_PER_GROUP = RESULTS_THREADS_PER_GROUP;
static const uint MAX_COMPONENTS = 1; 

groupshared float gLossShared[THREADS_PER_GROUP][MAX_COMPONENTS];

[shader("compute")]
[numthreads(THREADS_PER_GROUP, 1, 1)]
void lossReduction_cs(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    const uint index = dispatchThreadID.x;
    
    const LossConfig lossConfig = MakeLossConfig(MAX_COMPONENTS, gConst.bufferOffset);

    float loss = 0.f;

    if (index < gConst.batchSize)
    {
        loss = lossBuffer[index];
    }
    gLossShared[groupThreadID.x][0] = loss;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1)
    {
        if (groupThreadID.x < stride)
        {
            AddInPlace<MAX_COMPONENTS>(gLossShared[groupThreadID.x], gLossShared[groupThreadID.x + stride], lossConfig.GetComponentCount());
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (groupThreadID.x == 0)
    {
       AccumulateComponents<MAX_COMPONENTS>(accumulationBuffer, gLossShared[0], lossConfig);
    }
}
```

#### 计算 Epoch 平均 Loss

当一个 epoch 中所有 batch 都已经处理完，并且每个 batch 的 loss 都已经累加到全局 `accumulationBuffer` 后，就可以读出整轮 epoch 的总 loss，并把它换算成平均 loss。

`accumulationBuffer` 中保存的是整个 epoch 的 per-sample loss 总和。要得到平均值，只需要读出它，并除以这一轮 epoch 处理过的 sample 数量：

```cpp
    float lossSum =  asfloat(accumulationBuffer.Load(0u));
    float epochLoss = lossSum / gConst.epochSampleCount;
    
    TrainingResults result = {}; 
    result.l2Loss = epochLoss; 
    
    outputBuffer[0] = result; 
```
