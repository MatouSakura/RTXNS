# RTX Neural Shading：体积云训练样例

## 目的

`VolumetricCloudTraining` 是一个基于 `ShaderTraining` 改出来的体积云演示样例。
它不是直接让网络输出整张最终画面，而是让网络学习局部云密度函数：

```text
MLP(x, y, z, time, coverage) -> density
```

这样做的好处是：

- 训练目标更稳定
- 更适合程序体积云 / 云海这类问题
- 仍然保留常规 raymarch 渲染流程，便于继续向 Shadertoy 风格靠近

当前窗口分为 3 个视图：

- 左侧：解析体积云真值
- 中间：神经网络近似结果
- 右侧：两者误差图

## 如何运行

先编译：

```powershell
cd D:\Project\Cpp\RTXNS
cmake --build build --config Release --target VolumetricCloudTraining --parallel
```

然后运行：

```powershell
.\run-volumetric-cloud-training.bat
```

或直接启动：

```powershell
D:\Project\Cpp\RTXNS\bin\windows-x64\VolumetricCloudTraining.exe
```

## 交互方式

- 鼠标左键拖动：旋转太阳方向
- `Sun Intensity`：太阳亮度
- `Coverage`：云层覆盖度
- `Density Scale`：密度强度
- `Absorption`：吸收强度
- `Time`：云层时间相位
- `Animate Time`：自动播放风场动画
- `Disable Training / Enable Training`：暂停或恢复训练
- `Reset Training`：重置网络权重
- `Load Model / Save Model`：加载或保存模型

## 主要文件

- `samples/VolumetricCloudTraining/VolumetricCloudTraining.cpp`
- `samples/VolumetricCloudTraining/CloudScene.slang`
- `samples/VolumetricCloudTraining/CloudMLP.slang`
- `samples/VolumetricCloudTraining/computeTraining.slang`
- `samples/VolumetricCloudTraining/renderCloudDirect.slang`
- `samples/VolumetricCloudTraining/renderCloudInference.slang`
- `samples/VolumetricCloudTraining/renderCloudDifference.slang`
- `samples/VolumetricCloudTraining/NetworkConfig.h`

## 当前实现思路

当前版本采用的是：

1. 用程序噪声生成一个分层云海密度场
2. 随机采样 `(x, y, z, time, coverage)` 训练 MLP
3. 渲染时继续做体积步进
4. 每个步进位置调用 MLP 预测局部密度

它更接近“神经化局部体积函数”，而不是“神经化整张屏幕颜色”。

## 和你给的 Shadertoy 的关系

你给的那个 Shadertoy 更偏向：

- 程序噪声体积云
- 分层云海
- raymarch 累积
- 太阳方向近似照明

所以在 RTXNS 里，最合理的改法不是直接学最终 `RGB`，而是优先学 `density`。

当前样例已经按这个方向搭好了骨架，后续可以继续加强：

- 更复杂的多级 `map` 结构
- 更接近原版的光照项
- 更强的远景大气透视
- 更贴近原版的时间流动和卷积感
