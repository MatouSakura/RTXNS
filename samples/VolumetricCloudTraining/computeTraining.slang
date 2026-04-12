/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

import CooperativeVectorAutoDiff;
import CooperativeVectorFunctions;
import Utils;
import Activation;
import MLP;
import Loss;
import PCG32;
import CloudScene;

DECLARE_CBUFFER(TrainingConstantBufferEntry, gConst, 0, 0);
ByteAddressBuffer gMLPParams               : REGISTER_SRV(0, 0);
RWByteAddressBuffer gMLPParamsGradients    : REGISTER_UAV(0, 0);
RWBuffer<float>     gLossBuffer            : REGISTER_UAV(1, 0);

[shader("compute")]
[numthreads(THREADS_PER_GROUP_TRAIN, 1, 1)] 
void main_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    PCG32 rng = PCG32(gConst.seed, idx);

    float3 samplePos = float3(
        lerp(-6.0f, 6.0f, rng.nextFloat()),
        lerp(-1.15f, 1.15f, rng.nextFloat()),
        lerp(-6.0f, 6.0f, rng.nextFloat()));

    float time = rng.nextFloat();
    float coverage = lerp(0.3f, 0.82f, rng.nextFloat());
    float actualDensity = CloudDensity(samplePos, time, coverage);

    float params[INPUT_FEATURES] = { samplePos.x, samplePos.y, samplePos.z, time, coverage };
    var inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);

    var model = rtxns::mlp::TrainingMLP<half, 
        NUM_HIDDEN_LAYERS, INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, 
        CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
        gMLPParams, 
        gMLPParamsGradients, 
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets), 
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets));

    var hiddenActivation = rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>();
    var finalActivation = rtxns::mlp::SigmoidAct<half, OUTPUT_NEURONS>();

    var outputParams = model.forward(inputParams, hiddenActivation, finalActivation);
    float predictedDensity = outputParams[0];

    float1 lossGradient = rtxns::mlp::LossDerivFromVector<1, rtxns::mlp::L2<float, 1>>(
        float1(actualDensity),
        float1(predictedDensity),
        float1(LOSS_SCALE / gConst.batchSize));

    model.backward(inputParams, hiddenActivation, finalActivation, rtxns::HCoopVec<OUTPUT_NEURONS>(lossGradient[0]));

    float diff = predictedDensity - actualDensity;
    gLossBuffer[idx] = diff * diff;
}
