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

import CloudScene;

DECLARE_CBUFFER(DirectConstantBufferEntry, gConst, 0, 0);

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

float3 RenderSdfGroundTruth(float3 rayOrigin, float3 rayDir, float jitter)
{
    float3 lightDir = gConst.lightDir.xyz;
    float3 sky = EvaluateSky(rayDir, lightDir, gConst.sunColor.rgb, gConst.skyColor.rgb, gConst.horizonColor.rgb);

    float t = 0.08f + jitter * 0.10f;
    [loop]
    for (uint i = 0; i < CLOUD_MARCH_STEPS; ++i)
    {
        float3 p = rayOrigin + rayDir * t;
        float sdf = BlobSceneSdf(p, gConst.time, gConst.coverage);

        if (sdf < SDF_HIT_EPS)
        {
            if (gConst.unlitAndPadding.x > 0.5f)
            {
                float3 unlit = ShadeBlobEmissive(p, gConst.time, gConst.coverage, gConst.absorption);
                return CompositeSdfSurface(unlit, sky, t, gConst.unlitAndPadding.y);
            }
            float3 n = EstimateSdfNormal(p, gConst.time, gConst.coverage);
            float3 lit = ShadeBlobSurface(
                p,
                n,
                rayDir,
                lightDir,
                gConst.time,
                gConst.coverage,
                gConst.sunColor.rgb,
                gConst.skyColor.rgb,
                gConst.horizonColor.rgb,
                gConst.densityScale,
                gConst.absorption,
                gConst.unlitAndPadding.x);
            return CompositeSdfSurface(lit, sky, t, gConst.unlitAndPadding.y);
        }

        t += SdfTraceStep(sdf);
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
    float4 farPointH = mul(float4(ndc, 1.0f, 1.0f), gConst.viewProjectInverse);
    float3 farPoint = farPointH.xyz / farPointH.w;

    float3 rayOrigin = gConst.cameraPos.xyz;
    float3 rayDir = normalize(farPoint - rayOrigin);
    float jitterSeed = frac(gConst.time * 131.0f) * 71.0f;
    float jitter = lerp(0.10f, 0.92f, Hash12(input.pos.xy + jitterSeed));

    o_color = float4(RenderSdfGroundTruth(rayOrigin, rayDir, jitter), 1.0f);
}
