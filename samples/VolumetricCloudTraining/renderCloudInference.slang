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

float3 RaymarchInference(float3 rayOrigin, float3 rayDir)
{
    DirectConstantBufferEntry directConstants = gConst.directConstants;
    float3 background = EvaluateSky(rayDir, directConstants.lightDir.xyz, directConstants.sunColor.rgb, directConstants.skyColor.rgb, directConstants.horizonColor.rgb);
    float2 hit = IntersectRayBox(rayOrigin, rayDir, directConstants.volumeMin.xyz, directConstants.volumeMax.xyz);

    if (hit.x > hit.y)
    {
        return background;
    }

    float t = max(hit.x, 0.0f);
    float tEnd = hit.y;
    float stepSize = (tEnd - t) / CLOUD_MARCH_STEPS;
    if (stepSize <= 0.0f)
    {
        return background;
    }

    float transmittance = 1.0f;
    float3 accumColor = 0.0f;

    [loop]
    for (uint stepIndex = 0; stepIndex < CLOUD_MARCH_STEPS; ++stepIndex)
    {
        float currentT = t + (stepIndex + 0.35f) * stepSize;
        float3 samplePos = rayOrigin + rayDir * currentT;
        float density = CloudDensityMLP<NUM_HIDDEN_LAYERS, HIDDEN_NEURONS>(
            samplePos,
            directConstants.time,
            directConstants.coverage,
            gMLPParams,
            rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
            rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets));

        if (density > 0.001f)
        {
            float lightDensity = CloudDensityMLP<NUM_HIDDEN_LAYERS, HIDDEN_NEURONS>(
                samplePos - directConstants.lightDir.xyz * 0.18f,
                directConstants.time,
                directConstants.coverage,
                gMLPParams,
                rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
                rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets));

            float height01 = saturate((samplePos.y - directConstants.volumeMin.y) / max(0.001f, directConstants.volumeMax.y - directConstants.volumeMin.y));
            float alpha = 1.0f - exp(-density * directConstants.densityScale * stepSize * directConstants.absorption);
            float3 sampleColor = ShadeCloudSample(
                density,
                lightDensity,
                height01,
                rayDir,
                directConstants.lightDir.xyz,
                directConstants.sunColor.rgb,
                directConstants.skyColor.rgb,
                directConstants.horizonColor.rgb,
                directConstants.densityScale);

            accumColor += transmittance * alpha * sampleColor;
            transmittance *= 1.0f - alpha;

            if (transmittance < 0.02f)
            {
                break;
            }
        }
    }

    return accumColor + transmittance * background;
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

    o_color = float4(RaymarchInference(rayOrigin, rayDir), 1.0f);
}
