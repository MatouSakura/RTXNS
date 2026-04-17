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
import CloudScene;
import CloudMLP;

DECLARE_CBUFFER(InferenceConstantBufferEntry, gConst, 0, 0);
ByteAddressBuffer gMLPParams : REGISTER_SRV(0, 0);

struct PS_INPUT
{
    float4 pos : SV_Position;
    float2 uv : UV;
}

[shader("vertex")]
void main_vs(
    uint vertexId : SV_VertexID,
    out PS_INPUT output)
{
    uint u = vertexId & 1;
    uint v = (vertexId >> 1) & 1;

    output.uv = float2(u, v);
    output.pos = float4(float(u) * 2.0f - 1.0f, 1.0f - float(v) * 2.0f, 0.0f, 1.0f);
}

float3 RaymarchDirect(float3 rayOrigin, float3 rayDir, float jitter)
{
    DirectConstantBufferEntry directConstants = gConst.directConstants;
    CloudRaymarchContext context = InitCloudRaymarch(
        rayOrigin,
        rayDir,
        directConstants.lightDir.xyz,
        directConstants.sunColor.rgb,
        directConstants.skyColor.rgb,
        directConstants.horizonColor.rgb,
        directConstants.volumeMin.xyz,
        directConstants.volumeMax.xyz);

    if (!context.valid)
    {
        return context.background;
    }

    float transmittance = 1.0f;
    float3 accumColor = 0.0f;
    float t = context.tStart + CloudMarchStep(context.tStart) * jitter;

    [loop]
    for (uint stepIndex = 0; stepIndex < CLOUD_MARCH_STEPS; ++stepIndex)
    {
        if (t >= context.tEnd)
        {
            break;
        }

        float currentT = t;
        float stepSize = CloudMarchStep(currentT);
        float3 samplePos = rayOrigin + rayDir * currentT;
        float density = CloudDensity(WorldToCloudFieldPos(samplePos), directConstants.time, directConstants.coverage);

        if (CloudHasVisibleDensity(density))
        {
            float densityAhead = CloudDensity(WorldToCloudFieldPos(samplePos - directConstants.lightDir.xyz * CLOUD_LIGHT_PROBE_DISTANCE), directConstants.time, directConstants.coverage);
            CloudAccumulateStep(
                density,
                densityAhead,
                currentT,
                stepSize,
                rayDir,
                directConstants.lightDir.xyz,
                context.background,
                directConstants.sunColor.rgb,
                directConstants.densityScale,
                directConstants.absorption,
                transmittance,
                accumColor);

            if (CloudShouldTerminate(transmittance))
            {
                break;
            }
        }

        t += stepSize;
    }

    return ComposeCloudColor(accumColor, transmittance, context.background);
}

float3 RaymarchInference(float3 rayOrigin, float3 rayDir, float jitter)
{
    DirectConstantBufferEntry directConstants = gConst.directConstants;
    CloudRaymarchContext context = InitCloudRaymarch(
        rayOrigin,
        rayDir,
        directConstants.lightDir.xyz,
        directConstants.sunColor.rgb,
        directConstants.skyColor.rgb,
        directConstants.horizonColor.rgb,
        directConstants.volumeMin.xyz,
        directConstants.volumeMax.xyz);

    if (!context.valid)
    {
        return context.background;
    }

    float transmittance = 1.0f;
    float3 accumColor = 0.0f;
    float t = context.tStart + CloudMarchStep(context.tStart) * jitter;

    [loop]
    for (uint stepIndex = 0; stepIndex < CLOUD_MARCH_STEPS; ++stepIndex)
    {
        if (t >= context.tEnd)
        {
            break;
        }

        float currentT = t;
        float stepSize = CloudMarchStep(currentT);
        float3 samplePos = rayOrigin + rayDir * currentT;
        float3 fieldPos = WorldToCloudFieldPos(samplePos);
        float density = CloudDensityMLPSafe<NUM_HIDDEN_LAYERS, HIDDEN_NEURONS>(
            fieldPos,
            directConstants.time,
            directConstants.coverage,
            gMLPParams,
            rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
            rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets));

        if (CloudHasVisibleDensity(density))
        {
            float densityAhead = CloudDensityMLPSafe<NUM_HIDDEN_LAYERS, HIDDEN_NEURONS>(
                WorldToCloudFieldPos(samplePos - directConstants.lightDir.xyz * CLOUD_LIGHT_PROBE_DISTANCE),
                directConstants.time,
                directConstants.coverage,
                gMLPParams,
                rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
                rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets));
            CloudAccumulateStep(
                density,
                densityAhead,
                currentT,
                stepSize,
                rayDir,
                directConstants.lightDir.xyz,
                context.background,
                directConstants.sunColor.rgb,
                directConstants.densityScale,
                directConstants.absorption,
                transmittance,
                accumColor);

            if (CloudShouldTerminate(transmittance))
            {
                break;
            }
        }

        t += stepSize;
    }

    return ComposeCloudColor(accumColor, transmittance, context.background);
}

[shader("fragment")]
void main_ps(
    PS_INPUT input,
    out float4 o_color : SV_Target0)
{
    float2 ndc = float2(input.uv.x * 2.0f - 1.0f, 1.0f - input.uv.y * 2.0f);
    float4 farPointH = mul(float4(ndc, 1.0f, 1.0f), gConst.directConstants.viewProjectInverse);
    float3 farPoint = farPointH.xyz / farPointH.w;

    float3 rayOrigin = gConst.directConstants.cameraPos.xyz;
    float3 rayDir = normalize(farPoint - rayOrigin);
    float jitterSeed = frac(gConst.directConstants.time * 131.0f) * 71.0f;
    float jitter = lerp(0.12f, 0.98f, Hash12(input.pos.xy + jitterSeed));

    float3 directColor = ApplyCloudPost(RaymarchDirect(rayOrigin, rayDir, jitter));
    float3 inferenceColor = ApplyCloudPost(RaymarchInference(rayOrigin, rayDir, jitter));
    float3 difference = abs(directColor - inferenceColor) * 5.0f;
    o_color = float4(saturate(difference + 0.03f), 1.0f);
}
