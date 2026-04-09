# RTX Neural Shading：SlangPy Training 示例

## 目的

这个 sample 展示了如何使用 SlangPy 在 Python 中创建并训练网络结构。这样你就可以在不改 C++ 代码、也不重新编译 C++ 的情况下，快速试验不同网络、不同编码方式和不同训练策略。

作为演示，它会同时实例化多个不同的网络结构，并在同一份数据上并排训练。它还展示了一种把最终模型参数和网络结构导出到磁盘的方法，这样后续就可以在 C++ 侧加载，而不必手改头文件。

这里重新使用了 [Simple Training](SimpleTraining.md) 中的纹理训练例子，用它说明：只要对网络结构做一些小调整，就可能明显改变训练质量。如何为某个实际应用设计“好模型”远超这份文档的范围，因此这个 sample 更偏向示意性质，不应该被当成真正的纹理压缩方案。

![SlangPy Training Output](slangpy_training.jpg)

## 动机

在实际项目里，几乎不可能事先准确知道哪一种网络结构最适合某个应用。神经模型开发通常都包含一个探索阶段，在这个阶段里你需要频繁尝试：

- 不同的网络结构
- 不同的训练设置
- 不同的激活函数
- 不同的 loss

前面的 sample 为了易懂，都使用了预先固定的网络结构。如果你已经非常确定这个结构和你的问题相匹配，那当然没问题；但如果你还在探索阶段，那种做法会比较别扭，因为每次改动都需要修改多份文件并重新编译。

像 `pytorch` 或 `jax` 这类框架非常适合做这种探索，但它们更偏向大网络场景。如果直接拿来训练这里这种小网络，往往会比 `RTXNS` 自己的路径慢很多。

这个 sample 展示了一种替代思路：

- 底层仍然使用 `RTXNS` 的 building block
- 但在 Python 侧用一种更模块化、接近 `pytorch` 的方式来组织模型

## 环境准备

运行这个 sample 之前，需要安装额外依赖。

### 前置要求

需要较新的 Python 版本：

- `Python >= 3.9`

具体安装方式随平台而异。

### 安装依赖

在仓库根目录执行：

```
pip install -r samples/SlangpyTraining/requirements.txt
```

### 运行示例

在仓库根目录执行：

```
python samples\SlangpyTraining\SlangpyTraining.py
```

程序会打开一个窗口，并依次训练 4 种不同的网络结构。每个模型都会显示：

- 上方：当前拟合结果
- 下方：参考图像与拟合结果之间放大的误差图

当训练完成后，sample 会：

1. 把最佳模型写到磁盘
2. 为这个模型编译一个推理 shader
3. 启动一个 C++ sample 来执行推理，效果与 [Simple Training](SimpleTraining.md) 类似

## SlangPy 概览

这个 sample 使用 SlangPy，它提供了一种非常方便的方式，让 Python 直接调用用 Slang 编写的 GPU 代码。

更完整的说明请参考 [官方文档](https://SlangPy.readthedocs.io/en/latest/)，这里先快速梳理一下最核心的概念。

### 设备初始化

第一步是创建设备对象。为了帮 sample 自动设置好 include 路径，这里通过 `Helpers.py` 中的 `SDKSample` 来完成：

```
sample = SDKSample(sys.argv[1:])
device = sample.device
```

接下来加载一个 Slang 模块：

```
module = Module.load_from_file(device, "SlangpyTraining.slang")
```

返回值是一个 `Module`，它包含了这个 Slang 模块中的所有类型和函数。

### 简单函数调用

假设 Slang 模块里有这样一个函数：

```
float add(float a, float b)
{
    return a + b;
}
```

那么你在 Python 中可以这样调用它：

```
result = module.add(1.0, 2.0)
print(result) # 3.0

# Named parameters are also supported
result = module.add(a=1.0, b=2.0)
```

其内部会自动生成一个 compute shader，并在 GPU 上执行它。

如果只是拿它来算两个数字，当然不算高效。但 SlangPy 真正强大的地方在于，它可以很轻松地把同一个函数应用到大批量数据上。

例如：

```
a = np.random.rand(1000000)
b = np.random.rand(1000000)

result = module.add(a, b, _result=np.ndarray)

print(result[:10])
```

这里执行的还是同一个函数，只不过它被向量化后，分别作用于 100 万个元素。

这种“对一大批数据逐元素执行同一种函数”的模式，本来就是图形编程中的常见模式。SlangPy 的目标就是把这种调用方式变得简单自然。

SlangPy 支持的参数类型很多，包括：

- NumPy 数组
- 纹理
- torch.Tensors

默认情况下，SlangPy 的返回类型是 `NDBuffer`，它表示一个具有指定形状和 Slang 元素类型的 GPU buffer。你也可以像前面的例子一样，通过 `_result` 参数指定输出格式，例如直接拿到一个 NumPy 数组。

### 传递 struct

Slang 中的 struct 可以用 Python 字典传入。例如：

```
struct Color { float red; float green; float blue; }
void processColor(Color c) { /* ... */ }
```

可以这样调用：

```
module.processColor({"red": 1.0, "green": 0.5, "blue": 0.0})
```

也可以在 Python 中创建一个同构类，并提供 `get_this()` 方法：

```
class Color:
    # ...

    def get_this(self):
        return {"red": self.red, "green": self.green, "blue": self.blue}

c = Color( .... )
module.processColor(c)
```

当 SlangPy 发现一个 Python 对象提供了 `get_this()`，就会调用它，并把返回字典翻译成对应的 Slang struct。重点不在 Python 类名，而在字典字段必须和 Slang struct 字段一致。

## 架构概览

如果我们想频繁尝试不同网络，就需要一种办法，在不手写每个训练 / 推理 shader 的前提下，直接运行 `RTXNS` 库里的神经网络 building block。

这里的核心思路和 `pytorch` 很接近：

1. 先把 `RTXNS` 的模块都适配到统一接口上
2. 再像 `torch.nn.Sequential` 那样，把这些模块自由组合起来构建网络

为此，这个 sample 在 `NeuralModules.slang` 里包装了一层新的抽象，首先定义了一个接口：

```
interface IModule<T : __BuiltinFloatingPointType, let NumInputs : int, let NumOutputs : int>
{
    [BackwardDifferentiable]
    CoopVec<T, NumOutputs> forward(CoopVec<T, NumInputs> inputParams);
}
```

任何满足 `IModule<T, NumIn, NumOut>` 的类型，都必须提供一个 `forward`：输入是一个 `NumIn` 大小的 `CoopVec`，输出是一个 `NumOut` 大小的 `CoopVec`。

这意味着我们可以写出与具体模型无关的泛型代码。例如 `SlangpyTraining.slang` 中的这个函数：

```
[Differentiable]
float3 EvalModel<Model: rtxns::IModule<half, 2, 3>>(Model model, no_diff float2 inputUV)
{
    var inputVec = rtxns::CoopVecFromVector<half>(inputUV);

    var result = model.forward(inputVec);

    return rtxns::VectorFromCoopVec(result);
}
```

由于约束里写明了 `Model: rtxns::IModule<half, 2, 3>`，所以我们知道：

- 模型接收 2 维输入
- 输出 3 维结果

也就是说，只要某个模型符合这个接口，它就能被这个函数正确执行。

而且：

- `IModule::forward`
- `EvalModel`

都被标记为可微，因此只需调用 `bwd_diff(EvalModel)` 就能自动完成整条模型路径的反向传播。

于是就可以得到一个完全泛化的训练函数，用于拟合参考纹理：

```
void TrainTexture<
    Model : rtxns::IModule<half, 2, 3>,
    Loss : rtxns::mlp::ILoss<float, 3>
>(Model model, inout RNG rng, Texture2D<float4> targetTex, float lossScale)
{
    // Get a random uv coordinate for the input
    float2 inputUV = clamp(float2(rng.next(), rng.next()), 0.0, 1.0);

    // Sample the target texture at the generated UV
    float3 targetRGB = SampleTexture(targetTex, inputUV).rgb;

    // Evaluate the current output of the model
    float3 predictedRGB = EvalModel(model, inputUV);

    // Evaluate the loss gradient
    float3 lossGradient = Loss.deriv(targetRGB, predictedRGB, lossScale);

    // Backpropragate gradient through network parameters
    bwd_diff(EvalModel)(model, inputUV, lossGradient);
}
```

和 [Simple Training](SimpleTraining.md) 相比，这里最大的变化是：模型本身不再写死，而是被抽象成一个泛型参数。

整个流程仍然一样：

- 随机生成 UV
- 从目标纹理采样 `targetRGB`
- 用 `EvalModel` 得到预测 `predictedRGB`
- 用 `Loss.deriv(...)` 计算 loss gradient
- 用 `bwd_diff(EvalModel)` 做反向传播

但好处是，这个函数现在是可复用的：

- 任何“2 输入、3 输出”的模型都可以拿来训练
- Slang 会自动为当前模型生成正确的前向和梯度代码

### 神经模块实现

这些 `IModule` 实现本质上只是对 `RTXNS` 现有函数的一层薄包装。

例如 `NeuralModules.slang` 里的 `FrequencyEncoding`：

```
struct FrequencyEncoding<
    T : __BuiltinFloatingPointType,
    let NumInputs : int,
    let NumScales : int
> : IModule<T, NumInputs, NumScales * NumInputs * 2>
{
    [BackwardDifferentiable]
    CoopVec<T, NumScales * NumInputs * 2> forward(CoopVec<T, NumInputs> inputParams)
    {
        return rtxns::EncodeFrequencyN<T, NumInputs, NumScales>(inputParams);
    }
}
```

它的 `forward` 只是直接调用 `rtxns::EncodeFrequencyN`。关键点在于：

- 要为 `IModule` 正确填写泛型参数
- 让调用方能知道这个模块的输入输出维度

Slang AutoDiff 会自动帮你推导梯度。

除此之外，这里还提供了两个重要构件：

- `TrainableMLPModule`
- `ModuleChain`

前者本质上是带激活函数的训练型 MLP 封装；后者则负责把多个模块按顺序串起来，让前一个模块的输出作为后一个模块的输入。

### Python 侧的神经模块

虽然 `NeuralModules.slang` 已经允许我们在 Slang 里定义各种架构，但直接这么写类型会很冗长。例如，如果要复现 [Simple Training](SimpleTraining.md) 里的网络，大概要写成这样：

```
rtxns::ModuleChain<half, 2, 12, 3,
    rtxns::FrequencyEncoding<half, 2, 3>,
    rtxns::InferenceMLPModule<half, 4, 12, 32, 3,
        rtxns::mlp::LeakyReLUAct<half, 32>,
        rtxns::mlp::SigmoidAct<half, 3>
    >
>
```

这显然不够友好。好在 Python 可以替你生成它。

`NeuralModules.py` 中定义了一组和 `NeuralModules.slang` 对应的 Python 类型，它们都以 `CoopVecModule` 为根基。之所以不直接叫 `Module`，是为了避免和 SlangPy 自己的 `Module` 或 PyTorch 的 `Module` 混淆。

`CoopVecModule` 提供了几个重要接口：

- `type_name`
  返回这个模块对应的 Slang 类型名
- `get_this()`
  返回模块字段字典，使得 Python 对象可直接传给 Slang
- `parameters()` / `gradients()`
  返回训练时会被读取或写入的参数 buffer 和梯度 buffer 列表

有了这些 Python 模块后，就能像下面这样搭建网络：

```
encoding = FrequencyEncoding(DataType.float16, 2, 3)
mlp_with_encoding = ModuleChain(
    encoding,
    TrainableMLP(device, DataType.float16,
                    num_hidden_layers=3,
                    input_width=encoding.fan_out,
                    hidden_width=32,
                    output_width=3,
                    hidden_act=LeakyReLUAct(),
                    output_act=SigmoidAct())
)
```

这段代码做了几件关键事情：

1. `TrainableMLP` 会创建用于存储 CoopVector 权重和梯度的 buffer
2. 这些 buffer 可以通过 `mlp_with_encoding.parameters()` 和 `.gradients()` 取到
3. 模型类型名已经自动生成
4. 借助 `get_this()`，我们可以把整个 Python 对象直接传给 Slang

比如，调用 `EvalModel` 时可以这样：

```
module[f"EvalModel<{mlp_with_encoding.type_name}>"](mlp_with_encoding, ....)
```

由于 `EvalModel` 是泛型函数，因此需要先通过字符串拼出特化后的函数签名，再用 `[]` 去模块里取它。

这样一来，模型架构和参数都集中定义在同一个地方。你只要改 `mlp_with_encoding` 的构造方式，后面所有训练逻辑就会自动适配，而且：

- 不需要重编译 C++
- 甚至不需要改 Slang 里的训练函数

## Python 运行流程概览

理解了整体设计之后，再来看 sample 实际是怎么跑的。

### 超参数

设备初始化后，首先定义一批训练超参数，比如学习率、batch 大小等。这些值只是一个起点，实际中都可以继续调：

```
    batch_shape = (256, 256)
    learning_rate = 0.005
    grad_scale = 128.0
    loss_scale = grad_scale / math.prod(batch_shape)

    sample_target = 1000000000
    num_batches_per_epoch = 1000 if INTERACTIVE else 5000
    num_epochs = sample_target // (num_batches_per_epoch * math.prod(batch_shape))
```

### 模型初始化

接着，sample 会创建 4 种不同的网络结构。最简单的是一个基础 MLP：

```
basic_mlp = TrainableMLP(device, DataType.float16,
                         num_hidden_layers=3,
                         input_width=2,
                         hidden_width=32,
                         output_width=3,
                         hidden_act=ReLUAct(),
                         output_act=NoneAct())
```

它的参数和 `MLP.slang` 中的 `TrainingMLP` 基本一致。内部会自动创建 CoopVector 参数 buffer，并执行随机初始化。

如果要构造更复杂的网络，例如“输入编码 + MLP”的组合，就可以用 `ModuleChain`：

```
encoding = FrequencyEncoding(DataType.float16, 2, 3)
mlp_with_encoding = ModuleChain(
    encoding,
    TrainableMLP(device, DataType.float16,
                 num_hidden_layers=3,
                 input_width=encoding.fan_out,
                 hidden_width=32,
                 output_width=3,
                 hidden_act=LeakyReLUAct(),
                 output_act=SigmoidAct())
)
```

`ModuleChain` 接受一个模块列表，并按顺序把它们串起来。

### 训练数据生成

这个 sample 通过对参考纹理做随机采样来生成训练数据。第一步先把目标纹理加载进来：

```
target_tex = sample.load_texture("nvidia-logo.png")
```

然后需要一个随机数生成器。`SlangpyTraining.slang` 中提供了一个简单的 `RNG`：

```
struct RNG
{
    uint state;

    __init(uint state) { this.state = state; }
    /* ... */
}
```

训练时我们希望 batch 中每个线程都有自己的 `RNG` 实例。做法是：

先加载包含 `RNG` 的 Slang 模块：

```
module = Module.load_from_file(device, "SlangpyTraining.slang")
```

再用 NumPy 为 batch 中每个位置生成一个随机初始种子：

```
pcg = np.random.PCG64(seed=12345)
seeds = pcg.random_raw(batch_shape).astype(np.uint32)
```

这样就得到一个形状为 `batch_shape` 的 `ndarray`。然后调用 Slang 构造函数：

```
rng = module.RNG(seeds)
```

虽然 Slang 构造函数只接收一个 `uint`，但因为传进去的是一个数组，SlangPy 会自动做向量化调用，最终得到一个形状同样为 `batch_shape` 的 `NDBuffer<RNG>`。

### 训练准备

真正开始训练之前，还需要先准备 optimizer。

`SlangpyTraining.slang` 中 optimizer 的函数签名如下：

```
void OptimizerStep(
    RWBuffer<float> moments1,
    RWBuffer<float> moments2,
    RWBuffer<float> paramF,
    RWBuffer<half> paramH,
    RWBuffer<half> grad,
    uint idx,
    float learningRate,
    float gradScale,
    int iteration)
```

其中 `paramH` 和 `grad` 分别就是模型的半精度参数和对应梯度，可以直接从模型对象拿到：

```
grads = model.gradients()[0]
parameters = model.parameters()[0]
```

这个 sample 为了简单，假设模型里只有一块参数 buffer。支持多个参数 buffer 的情况可以自行扩展。

接下来，还需要准备一份 float32 参数副本，避免训练过程中不断累积量化误差。

前面的 sample 里，这通常需要额外写一段 compute shader；而在 SlangPy 里可以更简单。`SlangpyTraining.slang` 中定义了：

```
float ConvertToFloat(half paramH)
{
    return (float)paramH;
}
```

直接这样调用即可：

```
parametersF = module.ConvertToFloat(parameters)
```

它会对 `parameters` 中每一个 `half` 元素逐个执行转换，并返回一个同形状的 float buffer。

然后还要创建两个零初始化的 moment buffer。`NDBuffer` 提供了 `zeros_like`：

```
optimizer_state = {
    "moments1": NDBuffer.zeros_like(parametersF),
    "moments2": NDBuffer.zeros_like(parametersF),
    "paramF": parametersF,
    "paramH": parameters,
    "grad": grads,
    "learningRate": learning_rate,
    "gradScale": grad_scale
}
```

这个字典的键名和 `OptimizerStep` 的参数名是对齐的，因此后续就能直接：

```
OptimizerStep(**optimizer_state)
```

当然，`idx` 和 `iteration` 这两个参数仍然需要单独传。

接下来，还要从 `SlangpyTraining.slang` 中取出训练循环里真正要调用的几个函数：

```
OptimizerStep /* ... */
TrainTexture<Model : rtxns::IModule<half, 2, 3>, Loss : rtxns::mlp::ILoss<float, 3>> /* ... */
EvalModel<Model: rtxns::IModule<half, 2, 3>> /* ... */
EvalLoss<Loss : rtxns::mlp::ILoss<float, 3>> /* ... */
```

其中 `OptimizerStep` 不是泛型，可以直接取：

```
optimizer_step = module.OptimizerStep
```

其余几个函数带了模型类型或 loss 类型这样的泛型参数，因此需要先特化：

```
train_texture = module[f"TrainTexture<{model.type_name}, {loss_name} >"]
eval_model = module[f"EvalModel<{model.type_name} >"]
eval_loss = module[f"EvalLoss<{loss_name} >"]
```

### 主训练循环

训练循环的目标就是不断交替调用 `train_texture` 和 `optimizer_step`，直到模型收敛。最简单的形式如下：

```
iteration = 0
for batch in range(num_batches):
    train_texture(model, rng, target_tex, loss_scale)
    optimizer_step(idx=call_id((num_params, )), iteration=iteration, **optimizer_state)
    iteration += 1
```

这里唯一新的东西是 `call_id(call_shape)`，它提供当前向量化调用中的线程索引，用来访问参数 buffer。

如果要在一个紧凑循环里执行很多次调用，先把它们 append 到命令缓冲区里再一次性提交，通常会更快：

```
cmd = device.create_command_buffer()
cmd.open()
for batch in range(num_batches_per_epoch):
    train_texture.append_to(cmd, model, rng, target_tex, loss_scale)
    optimizer_step.append_to(cmd, idx=call_id((num_params, )), iteration=iteration, **optimizer_state)
    iteration += 1
cmd.close()
device.submit_command_buffer(cmd)
```

`SlangpyTraining.py` 在此基础上又进一步按 **epoch** 来组织训练。每个 epoch 之间，它会刷新界面，并打印训练信息。

### 训练完成后

训练结束后，模型参数会以 JSON 格式写到磁盘。`CoopVecModule` 基类已经提供了 `serialize()`：

```
param_dict = best_model.serialize()
open(weight_path, "w").write(json.dumps(param_dict, indent=4))
```

但光有权重还不够，还必须保存架构信息，后续推理时才能知道如何还原模型。

理论上，模型架构完全编码在类型里，因此可以直接存 `type_name`。不过这里有两个实际问题：

1. 推理时我们不想继续使用 `TrainingMLP`
2. 除了权重，还需要初始化模型里的常量，例如：
   - `LeakyReLUAct` 的负斜率
   - MLP 的参数 buffer
   - 权重和偏置 offset

这些都属于运行时信息，不能直接原样写进静态文件。

当前 sample 采取的是一种很直接、但不算特别健壮的做法：

- `CoopVecModule` 提供 `inference_type_name`
- 它返回一个用于推理的类型名
- 对于 MLP，这通常会返回 `InferenceMLP` 而不是 `TrainingMLP`
- 同时还提供 `get_initializer()`，返回一段 Slang 可用的 braced initializer 字符串

这部分最终是这样用的：

```
sample.compile_inference_shader(best_model)
sample.run_sdk_inference(weight_path)
```

其中被编译的 shader 是 `SlangpyInference.slang`，里面通过如下方式创建模型：

```
float3 evalModel(StructuredBuffer<half> weights, uint wo[MAX_LAYER_COUNT], uint bo[MAX_LAYER_COUNT], float2 uv)
{
    // Auto-generated defines from SlangpyTraining.py
    MODEL_TYPE model = MODEL_INITIALIZER;

    /* ... */
}
```

`SlangpyTraining.py` 会把训练好的模型类型和初始化器写进两个宏：

- `MODEL_TYPE`
- `MODEL_INITIALIZER`

如果把其中一个 sample 训练出来的模型展开，效果大概是：

```
float3 evalModel(StructuredBuffer<half> weights, uint wo[MAX_LAYER_COUNT], uint bo[MAX_LAYER_COUNT], float2 uv)
{
    // Auto-generated defines from SlangpyTraining.py
    ModuleChain<half, 2, 12, 3,
        FrequencyEncoding<half, 2, 3>,
        InferenceMLPModule<half, 4, 12, 32, 3,
            rtxns::mlp::LeakyReLUAct<half, 32>,
            rtxns::mlp::SigmoidAct<half, 3>
        >
    > model = {
        {},
        {
            weights,
            { wo[0], wo[1], wo[2], wo[3], wo[4] },
            { bo[0], bo[1], bo[2], bo[3], bo[4] },
            { 0.01h },
            {}
        }
    }

    /* ... */
}
```

这里会同时初始化：

- 激活函数常量（例如负斜率）
- 参数 buffer
- weight / bias offset

当前 sample 之所以采用这种方案，主要是为了简单。但它并不算很健壮，例如：

- 默认假设周围作用域里一定有 `weights`、`wo` 这类变量
- 如果一个模型里有多个 MLP，就不够灵活

更通用的方案就留给实际项目自行扩展了。

## 附录：NeuralModule 细节

更复杂一点的 `IModule` 实现，是 `TrainableMLPModule`。

它的开头大致如下：

```
struct TrainableMLPModule<
    T : __BuiltinFloatingPointType,
    let NumHiddenLayers : int,
    let InputNeurons : int,
    let HiddenNeurons : int,
    let OutputNeurons : int,
    let ComponentType : CoopVecComponentType,
    HiddenAct : mlp::IActivation<T, HiddenNeurons>,
    OutputAct : mlp::IActivation<T, OutputNeurons>
> : IModule<T, InputNeurons, OutputNeurons>
/* ... */
```

它接受的泛型参数和 [Shader Training](ShaderTraining.md) 中使用的 `TrainingMLP` 基本一致，只是额外把 hidden / output activation 也纳入了模板参数。

在内部，它保存了创建和执行一个 `TrainingMLP` 所需的全部信息：

```
/* ... */
    ByteAddressBuffer parameters;
    RWByteAddressBuffer derivatives; 
    uint matrixOffsets[NumHiddenLayers + 1];
    uint biasOffsets[NumHiddenLayers + 1];

    HiddenAct hiddenAct;
    OutputAct outputAct;
/* ... */
```

它的 `forward` 做的事情也非常直接：

```
/* ... */
[BackwardDerivative(backward)]
CoopVec<T, OutputNeurons> forward(CoopVec<T, InputNeurons> inputParams)
{
    var mlp = mlp::TrainingMLP<
        T, 
        NumHiddenLayers, 
        InputNeurons, 
        HiddenNeurons, 
        OutputNeurons, 
        CoopVecMatrixLayout::TrainingOptimal, 
        ComponentType
    >(parameters, derivatives, matrixOffsets, biasOffsets);
    return mlp.forward(inputParams, hiddenAct, outputAct);
}
/* ... */
```

也就是说，它只是把 `TrainingMLP` 包装成了一个满足 `IModule` 接口的模块，从而可以被更高层的泛型训练代码复用。

#### 组合多个模块

为了让 `IModule` 真正实用，还必须有一种方式能把多个模块串起来，让前一层的输出成为后一层的输入。

这就是 `ModuleChain` 的作用：

```
struct ModuleChain<
    T : __BuiltinFloatingPointType,
    let NumInputs : int,
    let NumHidden : int,
    let NumOutputs : int,
    First : IModule<T, NumInputs, NumHidden>,
    Second : IModule<T, NumHidden, NumOutputs>
> : IModule<T, NumInputs, NumOutputs>
```

它接收两个模块：

- `First`
- `Second`

模板约束保证了 `Second` 的输入维度必须和 `First` 的输出维度匹配。

剩下的实现就很自然了：

```
{
    First first;
    Second second;

    [BackwardDifferentiable]
    CoopVec<T, NumOutputs> forward(CoopVec<T, NumInputs> inputParams)
    {
        CoopVec<T, NumHidden> middle = first.forward(inputParams);
        return second.forward(middle);
    }
}
```

它本身只能串两个模块，但这已经足够了。因为如果你想串 3 个甚至更多模块，只需要继续嵌套 `ModuleChain` 就行。例如：

- 先把 `A` 和 `B` 组成一个 `ModuleChain`
- 再把这个结果和 `C` 继续组合

这样就能构建任意复杂的模型。
