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

static const int CLOUD_INPUT_FEATURES = 5;
static const int CLOUD_OUTPUT_NEURONS = 1;
static const int CLOUD_FREQUENCY_EXPANSION = 6;

float CloudDensityMLP<let HiddenLayerCount : int, let HiddenWidth : int>(
    float3 p,
    float time,
    float coverage,
    ByteAddressBuffer mlpBuffer,
    uint weightOffsets[HiddenLayerCount + 1],
    uint biasOffsets[HiddenLayerCount + 1])
{
    float params[CLOUD_INPUT_FEATURES] = { p.x, p.y, p.z, time, coverage };
    var inputParams = rtxns::EncodeFrequency<half, CLOUD_INPUT_FEATURES>(params);

    var model = rtxns::mlp::InferenceMLP<
        half,
        HiddenLayerCount,
        CLOUD_INPUT_FEATURES * CLOUD_FREQUENCY_EXPANSION,
        HiddenWidth,
        CLOUD_OUTPUT_NEURONS,
        CoopVecMatrixLayout::TrainingOptimal,
        CoopVecComponentType::Float16>(
            mlpBuffer,
            weightOffsets,
            biasOffsets);

    var outputParams = model.forward(
        inputParams,
        rtxns::mlp::ReLUAct<half, HiddenWidth>(),
        rtxns::mlp::SigmoidAct<half, CLOUD_OUTPUT_NEURONS>());

    return outputParams[0];
}
