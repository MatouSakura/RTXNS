/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <donut/app/ApplicationBase.h>
#include <donut/app/imgui_renderer.h>
#include <donut/app/DeviceManager.h>
#include <donut/app/UserInterfaceUtils.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/json.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/BindingCache.h>
#include <nvrhi/utils.h>

#include "DeviceUtils.h"
#include "GraphicsResources.h"
#include "GeometryUtils.h"
#include "NeuralNetwork.h"
#include "Float16.h"

#include "UIData.h"
#include "UserInterface.h"
#include "UIWidget.h"
#include "ResultsWidget.h"
#include "TrainingResults.h"
#include "ResultsReadbackHandler.h"

#include <iostream>
#include <fstream>
#include <random>
#include <numeric>
#include <format>

using namespace donut;
using namespace donut::math;

#include "NetworkConfig.h"

static const char* g_windowTitle = "RTX Neural Shading Example: SDF Blob Training (Ground Truth | Neural | Error)";
constexpr int g_maxViewsNum = 3;
constexpr int g_statisticsPerFrames = 100;

static std::random_device rd;

class SimpleShading : public app::IRenderPass
{

public:
    SimpleShading(app::DeviceManager* deviceManager, UserInterface& ui, rtxns::GraphicsResources& graphicsResources)
        : IRenderPass(deviceManager), m_ui(ui), m_graphicsResources(graphicsResources)
    {
        ResetCamera();
    }

    bool Init()
    {
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();

        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/SDFBlobTraining" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path utilShaderPath = app::GetDirectoryWithExecutable() / "shaders/Utils" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

        std::shared_ptr<vfs::RootFileSystem> rootFS = std::make_shared<vfs::RootFileSystem>();
        rootFS->mount("/shaders/donut", frameworkShaderPath);
        rootFS->mount("/shaders/app", appShaderPath);
        rootFS->mount("/shaders/utils", utilShaderPath);

        m_shaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFS, "/shaders");
        m_commonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_shaderFactory);
        m_bindingCache = std::make_unique<engine::BindingCache>(GetDevice());

        ////////////////////
        //
        // Create the Neural network class and initialize it the hyper parameters from NetworkConfig.h.
        //
        ////////////////////
        m_networkUtils = std::make_shared<rtxns::NetworkUtilities>(GetDevice());
        m_neuralNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
        if (!m_neuralNetwork->Initialise(m_netArch))
        {
            log::error("Failed to create a network.");
            return false;
        }

        ////////////////////
        //
        // Create UI components
        //
        ////////////////////
        m_resultsWidget = std::make_unique<ResultsWidget>(3000.f, 0.f, 0.05f);
        m_uiWidget = std::make_unique<UIWidget>(m_uiData);
        m_ui.AddWidget(m_uiWidget.get());
        m_ui.AddWidget(m_resultsWidget.get());

        ////////////////////
        //
        // Create the shaders/buffers for the Neural Training
        //
        ////////////////////
        m_trainingPass.computeShader = m_shaderFactory->CreateShader("app/computeTraining", "main_cs", nullptr, nvrhi::ShaderType::Compute);
        m_optimizerPass.computeShader = m_shaderFactory->CreateShader("app/computeOptimizer", "adam_cs", nullptr, nvrhi::ShaderType::Compute);

        m_trainingConstantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(TrainingConstantBufferEntry), "TrainingConstantBuffer")
                                                                 .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                                                                 .setKeepInitialState(true));
        // Initialize direct pass
        {
            m_directPass.constantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(DirectConstantBufferEntry), "DirectConstantBuffer")
                                                                        .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                                                                        .setKeepInitialState(true));
            m_directPass.vertexShader = m_shaderFactory->CreateShader("app/renderCloudDirect", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
            m_directPass.pixelShader = m_shaderFactory->CreateShader("app/renderCloudDirect", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
            assert(m_directPass.vertexShader && m_directPass.pixelShader);
        }

        // Initialize neural pass
        {
            m_inferencePass.constantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(InferenceConstantBufferEntry), "NeuralConstantBuffer")
                                                                           .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                                                                           .setKeepInitialState(true));
            m_inferencePass.vertexShader = m_shaderFactory->CreateShader("app/renderCloudInference", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
            m_inferencePass.pixelShader = m_shaderFactory->CreateShader("app/renderCloudInference", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
            assert(m_inferencePass.vertexShader && m_inferencePass.pixelShader);
        }

        // Initialize difference pass
        {
            m_differencePass.constantBuffer = m_inferencePass.constantBuffer;
            m_differencePass.vertexShader = m_shaderFactory->CreateShader("app/renderCloudDifference", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
            m_differencePass.pixelShader = m_shaderFactory->CreateShader("app/renderCloudDifference", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
            assert(m_differencePass.vertexShader && m_differencePass.pixelShader);
        }

        ////////////////////
        //
        // Create the shaders/buffers for the data processing
        //
        ////////////////////
        std::vector<donut::engine::ShaderMacro> nvapiMacro;
        if (m_graphicsResources.NvAPIInitialised())
        {
            nvapiMacro.push_back({ donut::engine::ShaderMacro("NVAPI_INIT", "1") });
        }
        else
        {
            nvapiMacro.push_back({ donut::engine::ShaderMacro("NVAPI_INIT", "0") });
        }
        m_lossReductionPass.computeShader = m_shaderFactory->CreateShader("utils/ProcessTrainingResults", "lossReduction_cs", &nvapiMacro, nvrhi::ShaderType::Compute);
        m_averageLossPass.computeShader = m_shaderFactory->CreateShader("utils/ProcessTrainingResults", "average_cs", nullptr, nvrhi::ShaderType::Compute);

        // Direct binding
        {
            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = { nvrhi::BindingSetItem::ConstantBuffer(0, m_directPass.constantBuffer) };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_directPass.bindingLayout, m_directPass.bindingSet);
        }

        CreateMLPBuffers();

        // Create timers
        m_disneyTimer = GetDevice()->createTimerQuery();
        m_neuralTimer = GetDevice()->createTimerQuery();
        m_trainingTimer = GetDevice()->createTimerQuery();
        m_optimizerTimer = GetDevice()->createTimerQuery();

        return true;
    }

    void CreateMLPBuffers()
    {
        // Get a device optimized layout
        m_deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(m_neuralNetwork->GetNetworkLayout(), rtxns::MatrixLayout::TrainingOptimal);

        for (int i = 0; i < NUM_TRANSITIONS; ++i)
        {
            m_weightOffsets[i / 4][i % 4] = m_deviceNetworkLayout.networkLayers[i].weightOffset;
            m_biasOffsets[i / 4][i % 4] = m_deviceNetworkLayout.networkLayers[i].biasOffset;
        }

        const auto hostBufferSize = m_neuralNetwork->GetNetworkLayout().networkByteSize;
        const auto deviceBufferSize = m_deviceNetworkLayout.networkByteSize;

        assert((deviceBufferSize % sizeof(uint16_t)) == 0 && "fp16 parameter buffer must be 2-byte aligned");
        m_totalParameterCount = uint(deviceBufferSize / sizeof(uint16_t)); // Includes possible padding
        m_batchSize = BATCH_SIZE;

        // Create and fill buffers
        {
            m_commandList = GetDevice()->createCommandList();
            m_commandList->open();

            nvrhi::BufferDesc paramsBufferDesc;

            paramsBufferDesc.debugName = "MLPParamsHostBuffer";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
            paramsBufferDesc.byteSize = hostBufferSize;
            paramsBufferDesc.canHaveUAVs = true;
            paramsBufferDesc.keepInitialState = true;
            m_mlpHostBuffer = GetDevice()->createBuffer(paramsBufferDesc);

            paramsBufferDesc.debugName = "MLPParamsDeviceBuffer";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            paramsBufferDesc.byteSize = deviceBufferSize;
            paramsBufferDesc.canHaveRawViews = true;
            paramsBufferDesc.canHaveTypedViews = true;
            paramsBufferDesc.format = nvrhi::Format::R16_FLOAT;
            m_mlpDeviceBuffer = GetDevice()->createBuffer(paramsBufferDesc);

            paramsBufferDesc.debugName = "MLPParamsDeviceBuffer32";
            paramsBufferDesc.canHaveRawViews = false;
            paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            paramsBufferDesc.byteSize = m_totalParameterCount * sizeof(float);
            paramsBufferDesc.format = nvrhi::Format::R32_FLOAT;
            m_mlpDeviceFP32Buffer = GetDevice()->createBuffer(paramsBufferDesc);

            paramsBufferDesc.debugName = "MLPGradientsBuffer";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            paramsBufferDesc.canHaveRawViews = true;
            paramsBufferDesc.byteSize = (m_totalParameterCount * sizeof(uint16_t) + 3) & ~3; // Round up to nearest multiple of 4
            paramsBufferDesc.structStride = sizeof(uint16_t);
            paramsBufferDesc.format = nvrhi::Format::R16_FLOAT;
            m_mlpGradientsBuffer = GetDevice()->createBuffer(paramsBufferDesc);

            m_commandList->beginTrackingBufferState(m_mlpGradientsBuffer, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->clearBufferUInt(m_mlpGradientsBuffer, 0);

            paramsBufferDesc.debugName = "MLPMoments1Buffer";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            paramsBufferDesc.byteSize = m_totalParameterCount * sizeof(float);
            paramsBufferDesc.format = nvrhi::Format::R32_FLOAT;
            paramsBufferDesc.canHaveRawViews = false;
            m_mlpMoments1Buffer = GetDevice()->createBuffer(paramsBufferDesc);

            m_commandList->beginTrackingBufferState(m_mlpMoments1Buffer, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->clearBufferUInt(m_mlpMoments1Buffer, 0);

            paramsBufferDesc.debugName = "MLPMoments2Buffer";
            m_mlpMoments2Buffer = GetDevice()->createBuffer(paramsBufferDesc);

            m_commandList->beginTrackingBufferState(m_mlpMoments2Buffer, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->clearBufferUInt(m_mlpMoments2Buffer, 0);

            // Upload parameters and convert to GPU optimal layout
            const auto& params = m_neuralNetwork->GetNetworkParams();

            // Upload the host side parameters
            m_commandList->writeBuffer(m_mlpHostBuffer, params.data(), params.size());

            // Convert to GPU optimized layout
            m_networkUtils->ConvertWeights(m_neuralNetwork->GetNetworkLayout(), m_deviceNetworkLayout, m_mlpHostBuffer, 0, m_mlpDeviceBuffer, 0, GetDevice(), m_commandList);

            m_commandList->close();
            GetDevice()->executeCommandList(m_commandList);
        }

        {
            // Create results resources
            if (m_resultsReadback == nullptr)
            {
                m_resultsReadback = std::make_unique<ResultsReadbackHandler>(GetDevice());
            }
            else
            {
                m_resultsReadback->Reset();
            }

            nvrhi::BufferDesc bufferDesc = {};
            bufferDesc.byteSize = BATCH_SIZE * sizeof(float);
            bufferDesc.format = nvrhi::Format::R32_FLOAT;
            bufferDesc.canHaveTypedViews = true;
            bufferDesc.canHaveUAVs = true;
            bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            bufferDesc.keepInitialState = true;
            bufferDesc.debugName = "lossBuffer";
            m_lossBuffer = GetDevice()->createBuffer(bufferDesc);

            nvrhi::BufferDesc accumDesc;
            accumDesc.byteSize = sizeof(float);
            accumDesc.canHaveRawViews = true;
            accumDesc.canHaveUAVs = true;
            accumDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            accumDesc.keepInitialState = true;
            accumDesc.debugName = "accumulationBuffer";
            m_accumulationBuffer = GetDevice()->createBuffer(accumDesc);

            m_lossConstantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(LossConstants), "NeuralConstantBuffer")
                                                                 .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                                                                 .setKeepInitialState(true));
        }

        nvrhi::BindingSetDesc bindingSetDesc = {};
        // Training binding
        {
            m_trainingPass.bindingSet = nullptr;
            m_trainingPass.bindingLayout = nullptr;

            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_trainingConstantBuffer),
                nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer),
                nvrhi::BindingSetItem::RawBuffer_UAV(0, m_mlpGradientsBuffer),
                nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_lossBuffer),
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_trainingPass.bindingLayout, m_trainingPass.bindingSet);

            nvrhi::ComputePipelineDesc pipelineDesc;
            pipelineDesc.bindingLayouts = { m_trainingPass.bindingLayout };
            pipelineDesc.CS = m_trainingPass.computeShader;
            m_trainingPass.pipeline = GetDevice()->createComputePipeline(pipelineDesc);
        }

        // Optimization binding
        {
            m_optimizerPass.bindingSet = nullptr;
            m_optimizerPass.bindingLayout = nullptr;

            bindingSetDesc = {};
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_trainingConstantBuffer), nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_mlpDeviceBuffer),
                nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_mlpDeviceFP32Buffer),   nvrhi::BindingSetItem::TypedBuffer_UAV(2, m_mlpGradientsBuffer),
                nvrhi::BindingSetItem::TypedBuffer_UAV(3, m_mlpMoments1Buffer),     nvrhi::BindingSetItem::TypedBuffer_UAV(4, m_mlpMoments2Buffer),
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_optimizerPass.bindingLayout, m_optimizerPass.bindingSet);

            nvrhi::ComputePipelineDesc pipelineDesc;
            pipelineDesc.bindingLayouts = { m_optimizerPass.bindingLayout };
            pipelineDesc.CS = m_optimizerPass.computeShader;
            m_optimizerPass.pipeline = GetDevice()->createComputePipeline(pipelineDesc);
        }

        // Inference binding
        {
            m_inferencePass.pipeline = nullptr;
            m_inferencePass.bindingSet = nullptr;
            m_inferencePass.bindingLayout = nullptr;

            bindingSetDesc = {};
            bindingSetDesc.bindings = { nvrhi::BindingSetItem::ConstantBuffer(0, m_inferencePass.constantBuffer), nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer) };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_inferencePass.bindingLayout, m_inferencePass.bindingSet);
        }

        // Difference binding
        {
            m_differencePass.pipeline = nullptr;
            m_differencePass.bindingSet = nullptr;
            m_differencePass.bindingLayout = nullptr;

            bindingSetDesc = {};
            bindingSetDesc.bindings = { nvrhi::BindingSetItem::ConstantBuffer(0, m_differencePass.constantBuffer), nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer) };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_differencePass.bindingLayout, m_differencePass.bindingSet);
        }

        // Loss reduction binding
        {
            m_lossReductionPass.pipeline = nullptr;
            m_lossReductionPass.bindingSet = nullptr;
            m_lossReductionPass.bindingLayout = nullptr;

            bindingSetDesc = {};
            bindingSetDesc.bindings = { nvrhi::BindingSetItem::ConstantBuffer(0, m_lossConstantBuffer), nvrhi::BindingSetItem::TypedBuffer_SRV(0, m_lossBuffer),
                                        nvrhi::BindingSetItem::RawBuffer_UAV(0, m_accumulationBuffer),
                                        nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_resultsReadback->GetResultsBuffers()), nvrhi::BindingSetItem::TypedBuffer_UAV(99, nullptr) };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_lossReductionPass.bindingLayout, m_lossReductionPass.bindingSet);

            nvrhi::ComputePipelineDesc pipelineDesc;
            pipelineDesc.bindingLayouts = { m_lossReductionPass.bindingLayout };
            pipelineDesc.CS = m_lossReductionPass.computeShader;
            m_lossReductionPass.pipeline = GetDevice()->createComputePipeline(pipelineDesc);
        }

        // Average loss binding
        {
            m_averageLossPass.pipeline = nullptr;
            m_averageLossPass.bindingSet = nullptr;
            m_averageLossPass.bindingLayout = nullptr;

            bindingSetDesc = {};
            bindingSetDesc.bindings = { nvrhi::BindingSetItem::ConstantBuffer(0, m_lossConstantBuffer), nvrhi::BindingSetItem::TypedBuffer_SRV(0, m_lossBuffer),
                                        nvrhi::BindingSetItem::RawBuffer_UAV(0, m_accumulationBuffer),
                                        nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_resultsReadback->GetResultsBuffers()) };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_averageLossPass.bindingLayout, m_averageLossPass.bindingSet);

            nvrhi::ComputePipelineDesc pipelineDesc;
            pipelineDesc.bindingLayouts = { m_averageLossPass.bindingLayout };
            pipelineDesc.CS = m_averageLossPass.computeShader;
            m_averageLossPass.pipeline = GetDevice()->createComputePipeline(pipelineDesc);
        }

        // Reset training parameters
        m_currentOptimizationStep = 0;
        m_uiData.epochs = 0;
        m_uiData.trainingTime = 0.0f;
    }

    std::shared_ptr<engine::ShaderFactory> GetShaderFactory() const
    {
        return m_shaderFactory;
    }

    void ResetCamera()
    {
        m_cameraPosition = float3(0.f, 0.2f, -6.5f);
        m_cameraTarget = float3(0.f, 0.1f, 0.f);
    }

    void ConsumeCameraCommands()
    {
        if (m_uiData.resetCamera)
        {
            ResetCamera();
            m_uiData.resetCamera = false;
        }

        const int moveForward = m_uiData.cameraMoveForward;
        const int moveRight = m_uiData.cameraMoveRight;
        const int moveUp = m_uiData.cameraMoveUp;
        m_uiData.cameraMoveForward = 0;
        m_uiData.cameraMoveRight = 0;
        m_uiData.cameraMoveUp = 0;

        if (moveForward == 0 && moveRight == 0 && moveUp == 0)
        {
            return;
        }

        const float3 worldUp(0.f, 1.f, 0.f);
        float3 forward = m_cameraTarget - m_cameraPosition;
        forward.y = 0.f;

        const float forwardLength = length(forward);
        if (forwardLength < 1e-4f)
        {
            forward = float3(0.f, 0.f, 1.f);
        }
        else
        {
            forward /= forwardLength;
        }

        float3 right = cross(worldUp, forward);
        const float rightLength = length(right);
        if (rightLength < 1e-4f)
        {
            right = float3(1.f, 0.f, 0.f);
        }
        else
        {
            right /= rightLength;
        }

        float3 delta = forward * (float(moveForward) * m_uiData.cameraMoveStep);
        delta += right * (float(moveRight) * m_uiData.cameraMoveStep);
        delta += worldUp * (float(moveUp) * m_uiData.cameraHeightStep);

        m_cameraPosition += delta;
        m_cameraTarget += delta;
    }

    float GetCameraDistance() const
    {
        return length(m_cameraPosition - m_cameraTarget);
    }

    void OrbitCamera(float2 delta)
    {
        float azimuth, elevation, distance;
        cartesianToSpherical(m_cameraPosition - m_cameraTarget, azimuth, elevation, distance);
        azimuth -= delta.x * 0.008f;
        elevation = clamp(elevation + delta.y * 0.006f, -PI_f * 0.45f, PI_f * 0.45f);
        m_cameraPosition = m_cameraTarget + sphericalToCartesian(azimuth, elevation, distance);
    }

    void PanCamera(float2 delta)
    {
        const float distance = max(1.0f, GetCameraDistance());
        const float panScale = distance * 0.0025f;
        float3 forward = normalize(m_cameraTarget - m_cameraPosition);
        float3 right = normalize(cross(forward, float3(0.f, 1.f, 0.f)));
        float3 up = normalize(cross(right, forward));
        float3 translation = (-right * delta.x + up * delta.y) * panScale;
        m_cameraPosition += translation;
        m_cameraTarget += translation;
    }

    void ZoomCamera(float wheelDelta)
    {
        float3 offset = m_cameraPosition - m_cameraTarget;
        float distance = length(offset);
        if (distance < 1e-4f)
        {
            offset = float3(0.f, 10.f, -10.f);
            distance = length(offset);
        }

        const float zoomFactor = pow(0.88f, wheelDelta);
        distance = clamp(distance * zoomFactor, 18.0f, 480.0f);
        m_cameraPosition = m_cameraTarget + normalize(offset) * distance;
    }

    bool MousePosUpdate(double xpos, double ypos) override
    {
        float2 newPos = float2(float(xpos), float(ypos));
        float2 delta = newPos - m_currentXY;

        if (m_mouseOrbiting)
        {
            OrbitCamera(delta);
        }
        else if (m_mousePanning)
        {
            PanCamera(delta);
        }

        m_currentXY = newPos;
        return m_mouseOrbiting || m_mousePanning;
    }

    bool MouseScrollUpdate(double xoffset, double yoffset) override
    {
        if (yoffset == 0.0)
        {
            return false;
        }

        ZoomCamera(float(yoffset));
        return true;
    }

    bool MouseButtonUpdate(int button, int action, int mods) override
    {
        const bool pressed = (action == GLFW_PRESS);
        if (pressed)
        {
            double xpos = 0.0;
            double ypos = 0.0;
            glfwGetCursorPos(GetDeviceManager()->GetWindow(), &xpos, &ypos);
            m_currentXY = float2(float(xpos), float(ypos));
        }

        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            m_mouseOrbiting = pressed;
            if (pressed)
            {
                m_mousePanning = false;
            }
            return true;
        }

        if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            m_mousePanning = pressed;
            if (pressed)
            {
                m_mouseOrbiting = false;
            }
            return true;
        }

        return false;
    }

    void Animate(float seconds) override
    {
        if (m_uiData.animateTime)
        {
            m_uiData.time += seconds * 0.05f;
            while (m_uiData.time > 1.0f)
            {
                m_uiData.time -= 1.0f;
            }
        }

        if (m_uiData.training)
        {
            m_uiData.trainingTime += seconds;
        }

        auto toMicroSeconds = [&](const auto& timer) { return int(GetDevice()->getTimerQueryTime(timer) * 1000000); };

        auto t = toMicroSeconds(m_disneyTimer);
        if (t != 0)
        {
            m_extraStatus = std::format(" - Ground Truth - {:3d}us, Neural - {:3d}us, Training - {:3d}us, Optimization - {:3d}us", t, toMicroSeconds(m_neuralTimer),
                                        toMicroSeconds(m_trainingTimer), toMicroSeconds(m_optimizerTimer));
        }
        GetDeviceManager()->SetInformativeWindowTitle(g_windowTitle, true, m_extraStatus.c_str());

        ////////////////////
        //
        // Reset/Load/Save the Neural network if required
        //
        ////////////////////
        if (m_uiData.reset)
        {
            m_neuralNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
            if (m_neuralNetwork->Initialise(m_netArch))
            {
                CreateMLPBuffers();
            }
            else
            {
                log::error("Failed to create a network.");
            }
            m_resultsWidget->Reset();
            m_uiData.reset = false;
        }

        if (!m_uiData.fileName.empty())
        {
            if (m_uiData.load)
            {
                auto newNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
                if (newNetwork->InitialiseFromFile(m_uiData.fileName))
                {
                    // Validate the loaded file against what the shaders expect
                    if (!m_networkUtils->CompareNetworkArchitecture(newNetwork->GetNetworkArchitecture(), m_netArch))
                    {
                        log::error("The loaded network does not match the network architecture in the compiled shaders. Load aborted");
                    }
                    else
                    {
                        m_neuralNetwork = std::move(newNetwork);
                        CreateMLPBuffers();
                        m_resultsWidget->Reset();
                    }
                }
            }
            else
            {
                m_neuralNetwork->UpdateFromBufferToFile(
                    m_mlpHostBuffer, m_mlpDeviceBuffer, m_neuralNetwork->GetNetworkLayout(), m_deviceNetworkLayout, m_uiData.fileName, GetDevice(), m_commandList);
            }
            m_uiData.fileName = "";
        }
    }

    void BackBufferResizing() override
    {
        m_directPass.pipeline = nullptr;
        m_inferencePass.pipeline = nullptr;
        m_differencePass.pipeline = nullptr;
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        ConsumeCameraCommands();

        std::uniform_int_distribution<uint64_t> ldist;
        uint64_t seed = ldist(rd);

        const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();
        const float height = float(fbinfo.height);
        const int activeViews = m_uiData.showNeuralView ? (m_uiData.showErrorView ? 3 : 2) : 1;
        const float width = float(fbinfo.width) / float(activeViews);

        // Update statistics every g_statisticsPerFrames frames
        bool updateStat = GetDeviceManager()->GetCurrentBackBufferIndex() % g_statisticsPerFrames == 0;

        float3 cameraUp(0, 1, 0);
        float4 viewDir(normalize(m_cameraTarget - m_cameraPosition), 0.f);

        DirectConstantBufferEntry directModelConstant = {};
        directModelConstant.cameraPos = float4(m_cameraPosition, 0.f);
        directModelConstant.lightDir = float4(normalize(m_lightDir), 0.f);
        directModelConstant.sunColor = float4(float3(1.00f, 0.95f, 0.85f) * m_uiData.sunIntensity, 0.f);
        directModelConstant.skyColor = float4(0.49f, 0.61f, 0.82f, 0.f);
        directModelConstant.horizonColor = float4(0.95f, 0.74f, 0.56f, 0.f);
        directModelConstant.volumeMin = float4(0.f, 10.0f, 0.f, 0.f);
        directModelConstant.volumeMax = float4(0.f, 36.0f, 0.f, 0.f);
        directModelConstant.time = m_uiData.time;
        directModelConstant.coverage = m_uiData.coverage;
        directModelConstant.densityScale = m_uiData.densityScale;
        directModelConstant.absorption = m_uiData.absorption;
        directModelConstant.unlitAndPadding = float4(m_uiData.unlit ? 1.0f : 0.0f, m_uiData.opacity, 0.f, 0.f);
        directModelConstant.viewProject = affineToHomogeneous(translation(-m_cameraPosition) * lookatZ(-viewDir.xyz(), cameraUp)) * perspProjD3DStyle(radians(52.0f), float(width) / float(height), 0.1f, 120.f);
        directModelConstant.viewProjectInverse = inverse(directModelConstant.viewProject);

        ////////////////////
        //
        // Fill out the inference constant buffer including the neural weight/bias offsets.
        //
        ////////////////////
        InferenceConstantBufferEntry inferenceModelConstant;
        inferenceModelConstant.directConstants = directModelConstant;
        std::ranges::copy(m_weightOffsets, inferenceModelConstant.weightOffsets);
        std::ranges::copy(m_biasOffsets, inferenceModelConstant.biasOffsets);

        m_commandList->open();

        ////////////////////
        //
        // Start the training loop
        //
        ////////////////////
        if (m_uiData.training)
        {
            LossConstants lossConstants = {};
            lossConstants.epochSampleCount = BATCH_SIZE * BATCH_COUNT;
            lossConstants.batchSize = BATCH_SIZE;
            lossConstants.epoch = m_uiData.epochs;
            m_commandList->writeBuffer(m_lossConstantBuffer, &lossConstants, sizeof(LossConstants));

            m_commandList->clearBufferUInt(m_accumulationBuffer, 0);

            for (int i = 0; i < BATCH_COUNT; ++i)
            {
                TrainingConstantBufferEntry trainingModelConstant = {
                    .maxParamSize = m_totalParameterCount, .learningRate = m_learningRate, .currentStep = float(++m_currentOptimizationStep), .batchSize = m_batchSize, .seed = seed
                };
                std::ranges::copy(m_weightOffsets, trainingModelConstant.weightOffsets);
                std::ranges::copy(m_biasOffsets, trainingModelConstant.biasOffsets);

                m_commandList->writeBuffer(m_trainingConstantBuffer, &trainingModelConstant, sizeof(trainingModelConstant));

                nvrhi::ComputeState state;

                // Training pass
                state.bindings = { m_trainingPass.bindingSet };
                state.pipeline = m_trainingPass.pipeline;
                m_commandList->beginMarker("Training");

                if (updateStat && i == 0)
                {
                    GetDevice()->resetTimerQuery(m_trainingTimer);
                    m_commandList->beginTimerQuery(m_trainingTimer);
                }

                m_commandList->setComputeState(state);
                m_commandList->dispatch(m_batchSize / THREADS_PER_GROUP_TRAIN, 1, 1);

                if (updateStat && i == 0)
                {
                    m_commandList->endTimerQuery(m_trainingTimer);
                }
                m_commandList->endMarker();

                // Optimizer pass
                state.bindings = { m_optimizerPass.bindingSet };
                state.pipeline = m_optimizerPass.pipeline;
                m_commandList->beginMarker("Update Weights");

                if (updateStat && i == 0)
                {
                    GetDevice()->resetTimerQuery(m_optimizerTimer);
                    m_commandList->beginTimerQuery(m_optimizerTimer);
                }

                m_commandList->setComputeState(state);
                m_commandList->dispatch(div_ceil(m_totalParameterCount, THREADS_PER_GROUP_OPTIMIZE), 1, 1);

                if (updateStat && i == 0)
                {
                    m_commandList->endTimerQuery(m_optimizerTimer);
                }
                m_commandList->endMarker();

                // Sum L2 Loss for the batch
                state.bindings = { m_lossReductionPass.bindingSet };
                state.pipeline = m_lossReductionPass.pipeline;
                m_commandList->beginMarker("Loss reduction");

                m_commandList->setComputeState(state);
                m_commandList->dispatch(dm::div_ceil(lossConstants.batchSize, RESULTS_THREADS_PER_GROUP), 1, 1);
                m_commandList->endMarker();
            }
            ++m_uiData.epochs;

            {
                // Calculate average loss for the epoch
                nvrhi::ComputeState state;
                state.bindings = { m_averageLossPass.bindingSet };
                state.pipeline = m_averageLossPass.pipeline;
                m_commandList->beginMarker("Average loss");

                m_commandList->setComputeState(state);
                m_commandList->dispatch(1, 1, 1);
                m_commandList->endMarker();

                // Copy the buffers to the CPU
                m_resultsReadback->SyncResults(m_commandList);
            }
        }

        nvrhi::utils::ClearColorAttachment(m_commandList, framebuffer, 0, nvrhi::Color(0.f));

        RenderPass* passes[] = { &m_directPass, &m_inferencePass, &m_differencePass };
        for (int viewIndex = 0; viewIndex < activeViews; ++viewIndex)
        {
            nvrhi::TimerQueryHandle timer;
            if (viewIndex < 2 && updateStat)
            {
                timer = viewIndex == 0 ? m_disneyTimer.Get() : m_neuralTimer.Get();
                GetDevice()->resetTimerQuery(timer);
                m_commandList->beginTimerQuery(timer);
            }

            auto& pass = *passes[viewIndex];

            if (!pass.pipeline)
            {
                nvrhi::GraphicsPipelineDesc psoDesc;
                psoDesc.VS = pass.vertexShader;
                psoDesc.PS = pass.pixelShader;
                psoDesc.inputLayout = pass.inputLayout;
                psoDesc.bindingLayouts = { pass.bindingLayout };
                psoDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
                psoDesc.renderState.depthStencilState.depthTestEnable = false;

                pass.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer->getFramebufferInfo());
            }

            if (viewIndex == 0)
            {
                m_commandList->writeBuffer(pass.constantBuffer, &directModelConstant, sizeof(directModelConstant));
            }
            else
            {
                m_commandList->writeBuffer(pass.constantBuffer, &inferenceModelConstant, sizeof(inferenceModelConstant));
            }

            nvrhi::GraphicsState state;
            state.bindings = { pass.bindingSet };
            state.pipeline = pass.pipeline;
            state.framebuffer = framebuffer;

            // Construct the viewport so that all viewports form a grid.
            const float left = width * viewIndex;
            const float top = 0;

            const nvrhi::Viewport viewport = nvrhi::Viewport(left, left + width, 0, height, 0.f, 1.f);
            state.viewport.addViewportAndScissorRect(viewport);

            // Update the pipeline, bindings, and other state.
            m_commandList->setGraphicsState(state);

            // Draw the model.
            nvrhi::DrawArguments args;
            args.vertexCount = 4;
            m_commandList->draw(args);

            if (viewIndex < 2 && updateStat)
            {
                m_commandList->endTimerQuery(timer);
            }
        }
        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);

        if (m_uiData.training)
        {
            TrainingResults results;
            if (m_resultsReadback->GetResults(results))
            {
                m_resultsWidget->Update(results);
            }
        }
    }

private:
    std::string m_extraStatus;
    nvrhi::TimerQueryHandle m_disneyTimer;
    nvrhi::TimerQueryHandle m_neuralTimer;
    nvrhi::TimerQueryHandle m_trainingTimer;
    nvrhi::TimerQueryHandle m_optimizerTimer;

    std::shared_ptr<engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<engine::CommonRenderPasses> m_commonPasses;
    std::unique_ptr<engine::BindingCache> m_bindingCache;

    struct RenderPass
    {
        nvrhi::ShaderHandle vertexShader;
        nvrhi::ShaderHandle pixelShader;
        nvrhi::BufferHandle constantBuffer;
        nvrhi::InputLayoutHandle inputLayout;
        nvrhi::BindingLayoutHandle bindingLayout;
        nvrhi::BindingSetHandle bindingSet;
        nvrhi::GraphicsPipelineHandle pipeline;
    };

    RenderPass m_directPass;
    RenderPass m_inferencePass;
    RenderPass m_differencePass;

    float3 m_lightDir{ -0.62f, -0.22f, -0.75f };
    float3 m_cameraPosition;
    float3 m_cameraTarget;
    float2 m_currentXY;
    bool m_mouseOrbiting = false;
    bool m_mousePanning = false;

    nvrhi::BufferHandle m_vertexBuffer;
    nvrhi::BufferHandle m_indexBuffer;

    nvrhi::BufferHandle m_trainingConstantBuffer;
    nvrhi::BufferHandle m_mlpHostBuffer;
    nvrhi::BufferHandle m_mlpDeviceBuffer;
    nvrhi::BufferHandle m_mlpDeviceFP32Buffer;
    nvrhi::BufferHandle m_mlpGradientsBuffer;
    nvrhi::BufferHandle m_mlpMoments1Buffer;
    nvrhi::BufferHandle m_mlpMoments2Buffer;

    // Feedback buffers
    nvrhi::BufferHandle m_lossBuffer;
    nvrhi::BufferHandle m_accumulationBuffer;
    nvrhi::BufferHandle m_lossConstantBuffer;

    std::unique_ptr<ResultsReadbackHandler> m_resultsReadback;

    uint m_totalParameterCount = 0;
    uint m_batchSize = BATCH_SIZE;
    uint m_currentOptimizationStep = 0;
    float m_learningRate = LEARNING_RATE;

    nvrhi::CommandListHandle m_commandList;

    int m_indicesNum = 0;

    struct NeuralPass
    {
        nvrhi::ShaderHandle computeShader;
        nvrhi::BindingLayoutHandle bindingLayout;
        nvrhi::BindingSetHandle bindingSet;
        nvrhi::ComputePipelineHandle pipeline;
    };

    NeuralPass m_trainingPass;
    NeuralPass m_optimizerPass;
    NeuralPass m_lossReductionPass;
    NeuralPass m_averageLossPass;

    uint4 m_weightOffsets[NUM_TRANSITIONS_ALIGN4];
    uint4 m_biasOffsets[NUM_TRANSITIONS_ALIGN4];

    rtxns::GraphicsResources& m_graphicsResources;
    UserInterface& m_ui;
    UIData m_uiData;
    std::unique_ptr<UIWidget> m_uiWidget;
    std::unique_ptr<ResultsWidget> m_resultsWidget;

    std::shared_ptr<rtxns::NetworkUtilities> m_networkUtils;
    std::unique_ptr<rtxns::HostNetwork> m_neuralNetwork;
    rtxns::NetworkLayout m_deviceNetworkLayout;

    const rtxns::NetworkArchitecture m_netArch = {
        .numHiddenLayers = NUM_HIDDEN_LAYERS,
        .inputNeurons = INPUT_NEURONS,
        .hiddenNeurons = HIDDEN_NEURONS,
        .outputNeurons = OUTPUT_NEURONS,
        .weightPrecision = rtxns::Precision::F16,
        .biasPrecision = rtxns::Precision::F16,
    };
};

#ifdef WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI graphicsApi = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    if (graphicsApi == nvrhi::GraphicsAPI::D3D11)
    {
        log::error("This sample does not support D3D11.");
        return 1;
    }
    std::unique_ptr<app::DeviceManager> deviceManager(app::DeviceManager::Create(graphicsApi));

    app::DeviceCreationParameters deviceParams;
    deviceParams.backBufferWidth = deviceParams.backBufferHeight * g_maxViewsNum;

#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableNvrhiValidationLayer = true;
#endif

    ////////////////////
    //
    // Setup the CoopVector extensions.
    //
    ////////////////////
    SetCoopVectorExtensionParameters(deviceParams, graphicsApi, true, g_windowTitle);

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_windowTitle))
    {
        if (graphicsApi == nvrhi::GraphicsAPI::VULKAN)
        {
            log::fatal("Cannot initialize a graphics device with the requested parameters. Please try a NVIDIA driver version greater than 570");
        }
        if (graphicsApi == nvrhi::GraphicsAPI::D3D12)
        {
            log::fatal("Cannot initialize a graphics device with the requested parameters. Please use the Shader Model 6-9-Preview Driver, link in the README");
        }
        return 1;
    }

    rtxns::GraphicsResources graphicsResources(deviceManager->GetDevice());
    if (!graphicsResources.GetCoopVectorFeatures().inferenceSupported && !graphicsResources.GetCoopVectorFeatures().trainingSupported &&
        !graphicsResources.GetCoopVectorFeatures().fp16InferencingSupported && !graphicsResources.GetCoopVectorFeatures().fp16TrainingSupported)
    {
        log::fatal("Not all required Coop Vector features are available");
        return 1;
    }

    {
        UserInterface gui(deviceManager.get());
        SimpleShading example(deviceManager.get(), gui, graphicsResources);

        if (example.Init() && gui.Init(example.GetShaderFactory()))
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->AddRenderPassToBack(&gui);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&gui);
            deviceManager->RemoveRenderPass(&example);
        }
    }

    deviceManager->Shutdown();

    return 0;
}
