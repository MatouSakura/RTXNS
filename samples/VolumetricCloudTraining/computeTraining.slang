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

    float time = rng.nextFloat();
    float coverage = lerp(0.45f, 0.86f, rng.nextFloat());

    // Candidate A: full-domain uniform sample.
    float3 samplePosA = float3(
        lerp(CLOUD_TRAIN_DOMAIN_MIN.x, CLOUD_TRAIN_DOMAIN_MAX.x, rng.nextFloat()),
        lerp(CLOUD_TRAIN_DOMAIN_MIN.y, CLOUD_TRAIN_DOMAIN_MAX.y, rng.nextFloat()),
        lerp(CLOUD_TRAIN_DOMAIN_MIN.z, CLOUD_TRAIN_DOMAIN_MAX.z, rng.nextFloat()));
    float densityA = CloudDensity(samplePosA, time, coverage);

    // Candidate B: height-focused sample (bias to cloud layer center).
    float centerBiased = (rng.nextFloat() + rng.nextFloat()) * 0.5f;
    float focusedY = lerp(
        CLOUD_TRAIN_DOMAIN_MIN.y + (CLOUD_TRAIN_DOMAIN_MAX.y - CLOUD_TRAIN_DOMAIN_MIN.y) * 0.20f,
        CLOUD_TRAIN_DOMAIN_MIN.y + (CLOUD_TRAIN_DOMAIN_MAX.y - CLOUD_TRAIN_DOMAIN_MIN.y) * 0.80f,
        centerBiased);
    float3 samplePosB = float3(
        lerp(CLOUD_TRAIN_DOMAIN_MIN.x, CLOUD_TRAIN_DOMAIN_MAX.x, rng.nextFloat()),
        focusedY,
        lerp(CLOUD_TRAIN_DOMAIN_MIN.z, CLOUD_TRAIN_DOMAIN_MAX.z, rng.nextFloat()));
    float densityB = CloudDensity(samplePosB, time, coverage);

    // Keep 25% uniform coverage, 75% prefer denser candidate to avoid collapsing to near-zero output.
    float3 samplePos = samplePosA;
    float actualDensity = densityA;
    if (rng.nextFloat() > 0.25f && densityB > densityA)
    {
        samplePos = samplePosB;
        actualDensity = densityB;
    }

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

    float densityWeight = lerp(1.0f, 4.0f, saturate(actualDensity * 2.0f));
    float1 lossGradient = rtxns::mlp::LossDerivFromVector<1, rtxns::mlp::L2<float, 1>>(
        float1(actualDensity),
        float1(predictedDensity),
        float1((LOSS_SCALE / gConst.batchSize) * densityWeight));

    model.backward(inputParams, hiddenActivation, finalActivation, rtxns::HCoopVec<OUTPUT_NEURONS>(lossGradient[0]));

    float diff = predictedDensity - actualDensity;
    gLossBuffer[idx] = diff * diff * densityWeight;
}
