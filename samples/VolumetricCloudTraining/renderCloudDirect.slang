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

float3 RaymarchDirect(float3 rayOrigin, float3 rayDir, float jitter)
{
    float3 background = EvaluateSky(rayDir, gConst.lightDir.xyz, gConst.sunColor.rgb, gConst.skyColor.rgb, gConst.horizonColor.rgb);
    float2 hit = IntersectCloudLayer(rayOrigin, rayDir, gConst.volumeMin.y, gConst.volumeMax.y);

    if (hit.x > hit.y)
    {
        return background;
    }

    float t = max(hit.x, 0.0f);
    float tEnd = min(hit.y, t + 420.0f);
    if (tEnd <= t)
    {
        return background;
    }

    float transmittance = 1.0f;
    float3 accumColor = 0.0f;
    float stepSize = 1.35f;
    t += stepSize * jitter;

    [loop]
    for (uint stepIndex = 0; stepIndex < CLOUD_MARCH_STEPS; ++stepIndex)
    {
        if (t >= tEnd)
        {
            break;
        }

        float currentT = t;
        float3 samplePos = rayOrigin + rayDir * currentT;
        float density = CloudDensity(WorldToCloudFieldPos(samplePos), gConst.time, gConst.coverage);

        if (density > 0.001f)
        {
            float height01 = saturate((samplePos.y - gConst.volumeMin.y) / max(0.001f, gConst.volumeMax.y - gConst.volumeMin.y));
            float lightTransmittance = EstimateLightTransmittance(samplePos, gConst.lightDir.xyz, gConst.time, gConst.coverage, gConst.densityScale);
            float alpha = 1.0f - exp(-density * gConst.densityScale * stepSize * gConst.absorption);
            float3 sampleColor = ShadeCloudSample(
                density,
                1.0f - lightTransmittance,
                height01,
                rayDir,
                gConst.lightDir.xyz,
                gConst.sunColor.rgb,
                gConst.skyColor.rgb,
                gConst.horizonColor.rgb,
                gConst.densityScale);

            accumColor += transmittance * alpha * sampleColor;
            transmittance *= 1.0f - alpha;

            if (transmittance < 0.02f)
            {
                break;
            }
        }

        t += stepSize;
        stepSize *= 1.014f;
    }

    return accumColor + transmittance * background;
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
    float jitter = Hash12(input.uv + float2(gConst.time * 3.1f, gConst.time * 1.7f));

    o_color = float4(RaymarchDirect(rayOrigin, rayDir, jitter), 1.0f);
}
