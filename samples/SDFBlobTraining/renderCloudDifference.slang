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

float3 RenderSdfDirect(float3 rayOrigin, float3 rayDir, float jitter)
{
    DirectConstantBufferEntry dc = gConst.directConstants;
    float3 sky = EvaluateSky(rayDir, dc.lightDir.xyz, dc.sunColor.rgb, dc.skyColor.rgb, dc.horizonColor.rgb);
    float t = 0.08f + jitter * 0.10f;

    [loop]
    for (uint i = 0; i < NEURAL_MARCH_STEPS; ++i)
    {
        float3 p = rayOrigin + rayDir * t;
        float sdf = BlobSceneSdf(p, dc.time, dc.coverage);
        if (sdf < SDF_HIT_EPS)
        {
            if (dc.unlitAndPadding.x > 0.5f)
            {
                float3 unlit = ShadeBlobEmissive(p, dc.time, dc.coverage, dc.absorption);
                return CompositeSdfSurface(unlit, sky, t, dc.unlitAndPadding.y);
            }
            float3 n = EstimateSdfNormal(p, dc.time, dc.coverage);
            float3 lit = ShadeBlobSurface(
                p,
                n,
                rayDir,
                dc.lightDir.xyz,
                dc.time,
                dc.coverage,
                dc.sunColor.rgb,
                dc.skyColor.rgb,
                dc.horizonColor.rgb,
                dc.densityScale,
                dc.absorption,
                dc.unlitAndPadding.x);
            return CompositeSdfSurface(lit, sky, t, dc.unlitAndPadding.y);
        }

        t += SdfTraceStep(sdf);
        if (t > SDF_MAX_DISTANCE)
            break;
    }

    return sky;
}

float3 RenderSdfNeural(float3 rayOrigin, float3 rayDir, float jitter)
{
    DirectConstantBufferEntry dc = gConst.directConstants;
    float3 sky = EvaluateSky(rayDir, dc.lightDir.xyz, dc.sunColor.rgb, dc.skyColor.rgb, dc.horizonColor.rgb);
    float t = 0.08f + jitter * 0.10f;

    [loop]
    for (uint i = 0; i < CLOUD_MARCH_STEPS; ++i)
    {
        float3 p = rayOrigin + rayDir * t;
        float sdf = PredictSdfMLPSafe<NUM_HIDDEN_LAYERS, HIDDEN_NEURONS>(
            p,
            dc.time,
            dc.coverage,
            gMLPParams,
            rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
            rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets));

        if (sdf < NEURAL_HIT_EPS)
        {
            if (dc.unlitAndPadding.x > 0.5f)
            {
                float3 unlit = ShadeBlobEmissive(p, dc.time, dc.coverage, dc.absorption);
                return CompositeSdfSurface(unlit, sky, t, dc.unlitAndPadding.y);
            }
            float3 n = EstimateSdfNormal(p, dc.time, dc.coverage);
            float3 lit = ShadeBlobSurface(
                p,
                n,
                rayDir,
                dc.lightDir.xyz,
                dc.time,
                dc.coverage,
                dc.sunColor.rgb,
                dc.skyColor.rgb,
                dc.horizonColor.rgb,
                dc.densityScale,
                dc.absorption,
                dc.unlitAndPadding.x);
            return CompositeSdfSurface(lit, sky, t, dc.unlitAndPadding.y);
        }

        t += SdfTraceStepNeural(sdf);
        if (t > SDF_MAX_DISTANCE)
            break;
    }

    return sky;
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
    float jitter = lerp(0.10f, 0.92f, Hash12(input.pos.xy + jitterSeed));

    float3 directColor = RenderSdfDirect(rayOrigin, rayDir, jitter);
    float3 inferenceColor = RenderSdfNeural(rayOrigin, rayDir, jitter);
    float3 difference = abs(directColor - inferenceColor) * 4.0f;
    o_color = float4(saturate(difference + 0.02f), 1.0f);
}
