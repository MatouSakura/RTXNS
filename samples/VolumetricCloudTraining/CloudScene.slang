/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

float Hash13(float3 p)
{
    return frac(sin(dot(p, float3(127.1f, 311.7f, 74.7f))) * 43758.5453123f);
}

float ValueNoise(float3 p)
{
    float3 cell = floor(p);
    float3 f = frac(p);
    f = f * f * (3.0f - 2.0f * f);

    float n000 = Hash13(cell + float3(0, 0, 0));
    float n100 = Hash13(cell + float3(1, 0, 0));
    float n010 = Hash13(cell + float3(0, 1, 0));
    float n110 = Hash13(cell + float3(1, 1, 0));
    float n001 = Hash13(cell + float3(0, 0, 1));
    float n101 = Hash13(cell + float3(1, 0, 1));
    float n011 = Hash13(cell + float3(0, 1, 1));
    float n111 = Hash13(cell + float3(1, 1, 1));

    float nx00 = lerp(n000, n100, f.x);
    float nx10 = lerp(n010, n110, f.x);
    float nx01 = lerp(n001, n101, f.x);
    float nx11 = lerp(n011, n111, f.x);
    float nxy0 = lerp(nx00, nx10, f.y);
    float nxy1 = lerp(nx01, nx11, f.y);
    return lerp(nxy0, nxy1, f.z);
}

float FBM(float3 p)
{
    float sum = 0.0f;
    float amplitude = 0.5f;

    [ForceUnroll]
    for (int octave = 0; octave < 4; ++octave)
    {
        sum += amplitude * ValueNoise(p);
        p = p * 2.03f + float3(13.7f, 7.1f, 5.3f);
        amplitude *= 0.5f;
    }

    return sum;
}

float CloudDensity(float3 p, float time, float coverage)
{
    float layerCenter = 0.45f;
    float layerHalfThickness = 0.38f;
    float heightMask = saturate(1.0f - abs(p.y - layerCenter) / layerHalfThickness);
    heightMask *= smoothstep(-0.05f, 0.12f, p.y);
    heightMask *= 1.0f - smoothstep(0.75f, 1.15f, p.y);

    float3 windPos = p;
    windPos.xz += float2(time * 0.55f, time * 0.28f);
    windPos.y *= 0.65f;

    float macroWaves = 0.5f
        + 0.25f * sin(windPos.x * 1.3f + time * 3.2f)
        + 0.2f * sin(windPos.z * 1.7f - time * 2.4f);

    float macroNoise = FBM(windPos * 1.25f);
    float broadNoise = FBM(windPos * 0.72f + 5.0f);
    float detailNoise = FBM(windPos * 3.1f + 11.0f);
    float erosionNoise = FBM(windPos * 5.4f + 37.0f);
    float wispyNoise = FBM(windPos * 8.0f + 53.0f);

    float densityShape = macroWaves * 0.28f + broadNoise * 0.52f + macroNoise * 0.62f + detailNoise * 0.26f;
    densityShape += wispyNoise * 0.08f * heightMask;
    densityShape -= erosionNoise * 0.22f;
    densityShape *= heightMask;

    float density = smoothstep(coverage - 0.22f, coverage + 0.18f, densityShape);
    density *= lerp(0.7f, 1.15f, saturate((p.y - 0.05f) / 0.65f));
    return saturate(density * heightMask);
}

float2 IntersectRayBox(float3 rayOrigin, float3 rayDir, float3 boxMin, float3 boxMax)
{
    float3 invRayDir = 1.0f / rayDir;
    float3 t0 = (boxMin - rayOrigin) * invRayDir;
    float3 t1 = (boxMax - rayOrigin) * invRayDir;
    float3 tMin3 = min(t0, t1);
    float3 tMax3 = max(t0, t1);

    float tNear = max(max(tMin3.x, tMin3.y), tMin3.z);
    float tFar = min(min(tMax3.x, tMax3.y), tMax3.z);
    return float2(tNear, tFar);
}

float3 EvaluateSky(float3 rayDir, float3 lightDir, float3 sunColor, float3 skyColor, float3 horizonColor)
{
    float skyAmount = saturate(rayDir.y * 0.5f + 0.5f);
    float3 sky = lerp(horizonColor, skyColor, skyAmount);
    float horizonGlow = pow(1.0f - saturate(abs(rayDir.y)), 5.0f);
    float sunDisk = pow(saturate(dot(rayDir, -lightDir)), 64.0f);
    float sunBloom = pow(saturate(dot(rayDir, -lightDir)), 10.0f);
    sky += horizonColor * horizonGlow * 0.2f;
    sky += sunColor * sunBloom * 0.12f;
    return sky + sunColor * sunDisk * 0.35f;
}

float3 ShadeCloudSample(
    float density,
    float lightDensity,
    float height01,
    float3 rayDir,
    float3 lightDir,
    float3 sunColor,
    float3 skyColor,
    float3 horizonColor,
    float densityScale)
{
    float forwardPhase = 0.35f + 0.65f * pow(saturate(dot(-rayDir, -lightDir)), 8.0f);
    float backwardPhase = 0.2f * pow(saturate(dot(rayDir, -lightDir)), 2.0f);
    float selfShadow = exp(-lightDensity * densityScale * 1.8f);
    float silverLining = pow(saturate(1.0f - density), 3.0f);

    float3 bodyColor = lerp(horizonColor, skyColor, saturate(height01 * 0.75f + 0.15f));
    bodyColor = lerp(bodyColor, sunColor, 0.25f * silverLining);

    float ambient = lerp(0.35f, 0.8f, height01);
    float direct = lerp(0.25f, 1.0f, selfShadow) * (forwardPhase + backwardPhase);
    return bodyColor * ambient + sunColor * direct * (0.45f + 0.15f * silverLining);
}
