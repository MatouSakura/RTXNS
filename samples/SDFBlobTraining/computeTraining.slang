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
const static float PI = 3.14159265358979323846f;

[shader("compute")]
[numthreads(THREADS_PER_GROUP_TRAIN, 1, 1)]
void main_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    PCG32 rng = PCG32(gConst.seed, idx);

    float time = rng.nextFloat();
    float coverage = lerp(0.2f, 0.85f, rng.nextFloat());

    float3 samplePosA = float3(
        lerp(SDF_DOMAIN_MIN.x, SDF_DOMAIN_MAX.x, rng.nextFloat()),
        lerp(SDF_DOMAIN_MIN.y, SDF_DOMAIN_MAX.y, rng.nextFloat()),
        lerp(SDF_DOMAIN_MIN.z, SDF_DOMAIN_MAX.z, rng.nextFloat()));
    float sdfA = BlobSceneSdf(samplePosA, time, coverage);

    float3 c0, c1, c2;
    GetBlobCenters(time, c0, c1, c2);
    float cPick = rng.nextFloat();
    float3 center = cPick < (1.0f / 3.0f) ? c0 : (cPick < (2.0f / 3.0f) ? c1 : c2);

    float az = rng.nextFloat() * PI * 2.0f;
    float z = rng.nextFloat() * 2.0f - 1.0f;
    float rxy = sqrt(max(0.0f, 1.0f - z * z));
    float saz, caz;
    sincos(az, saz, caz);
    float3 dir = float3(rxy * caz, z, rxy * saz);
    float shellRadius = lerp(0.45f, 1.10f, rng.nextFloat());
    float3 samplePosB = center + dir * shellRadius;
    float sdfB = BlobSceneSdf(samplePosB, time, coverage);

    float3 samplePos = samplePosA;
    float actualSdf = sdfA;
    if (rng.nextFloat() > 0.25f && abs(sdfB) < abs(sdfA))
    {
        samplePos = samplePosB;
        actualSdf = sdfB;
    }

    float3 pNorm = NormalizeSdfPos(samplePos);
    float coverageNorm = NormalizeSdfCoverage(coverage);
    float params[INPUT_FEATURES] = { pNorm.x, pNorm.y, pNorm.z, saturate(time), coverageNorm };
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
    float predictedEncoded = outputParams[0];
    float actualEncoded = EncodeSdfForMLP(actualSdf);

    float surfaceWeight = 1.0f + 4.0f * exp(-abs(actualSdf) * 6.0f);
    float1 lossGradient = rtxns::mlp::LossDerivFromVector<1, rtxns::mlp::L2<float, 1>>(
        float1(actualEncoded),
        float1(predictedEncoded),
        float1((LOSS_SCALE / gConst.batchSize) * surfaceWeight));

    model.backward(inputParams, hiddenActivation, finalActivation, rtxns::HCoopVec<OUTPUT_NEURONS>(lossGradient[0]));

    float diff = predictedEncoded - actualEncoded;
    gLossBuffer[idx] = diff * diff * surfaceWeight;
}
