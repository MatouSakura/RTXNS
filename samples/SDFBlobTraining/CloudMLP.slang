/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

import MLP;
import CooperativeVectorFunctions;
import Activation;
import Utils;
import CloudScene;

static const int SDF_INPUT_FEATURES = 5;
static const int SDF_OUTPUT_NEURONS = 1;
static const int SDF_FREQUENCY_EXPANSION = 6;

float PredictSdfMLP<let HiddenLayerCount : int, let HiddenWidth : int>(
    float3 p,
    float time,
    float coverage,
    ByteAddressBuffer mlpBuffer,
    uint weightOffsets[HiddenLayerCount + 1],
    uint biasOffsets[HiddenLayerCount + 1])
{
    float3 pNorm = NormalizeSdfPos(p);
    float coverageNorm = NormalizeSdfCoverage(coverage);

    float params[SDF_INPUT_FEATURES] = { pNorm.x, pNorm.y, pNorm.z, saturate(time), coverageNorm };
    var inputParams = rtxns::EncodeFrequency<half, SDF_INPUT_FEATURES>(params);

    var model = rtxns::mlp::InferenceMLP<
        half,
        HiddenLayerCount,
        SDF_INPUT_FEATURES * SDF_FREQUENCY_EXPANSION,
        HiddenWidth,
        SDF_OUTPUT_NEURONS,
        CoopVecMatrixLayout::TrainingOptimal,
        CoopVecComponentType::Float16>(
            mlpBuffer,
            weightOffsets,
            biasOffsets);

    var outputParams = model.forward(
        inputParams,
        rtxns::mlp::ReLUAct<half, HiddenWidth>(),
        rtxns::mlp::SigmoidAct<half, SDF_OUTPUT_NEURONS>());

    return DecodeSdfFromMLP(outputParams[0]);
}

float PredictSdfMLPSafe<let HiddenLayerCount : int, let HiddenWidth : int>(
    float3 p,
    float time,
    float coverage,
    ByteAddressBuffer mlpBuffer,
    uint weightOffsets[HiddenLayerCount + 1],
    uint biasOffsets[HiddenLayerCount + 1])
{
    if (!IsInsideSdfDomain(p))
    {
        return BlobSceneSdf(p, time, coverage);
    }

    return PredictSdfMLP<HiddenLayerCount, HiddenWidth>(
        p,
        time,
        coverage,
        mlpBuffer,
        weightOffsets,
        biasOffsets);
}
