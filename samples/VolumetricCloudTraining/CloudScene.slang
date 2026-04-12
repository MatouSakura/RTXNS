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

float Hash12(float2 p)
{
    return frac(sin(dot(p, float2(127.1f, 311.7f))) * 43758.5453123f);
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

static const float CLOUD_WORLD_TO_FIELD_XZ = 0.018f;
static const float CLOUD_WORLD_TO_FIELD_Y = 0.07f;
static const float CLOUD_WORLD_CENTER_Y = 24.0f;

float3 WorldToCloudFieldPos(float3 worldPos)
{
    return float3(
        worldPos.x * CLOUD_WORLD_TO_FIELD_XZ,
        (worldPos.y - CLOUD_WORLD_CENTER_Y) * CLOUD_WORLD_TO_FIELD_Y,
        worldPos.z * CLOUD_WORLD_TO_FIELD_XZ);
}

float CloudDensity(float3 p, float time, float coverage)
{
    float height01 = saturate(p.y * 0.5f + 0.5f);
    float flatBase = smoothstep(0.0f, 0.12f, height01);
    float softTop = 1.0f - smoothstep(0.58f, 1.0f, height01);
    float coreMask = saturate(1.0f - abs(height01 - 0.42f) * 1.55f);
    float heightMask = saturate(flatBase * softTop * (0.55f + coreMask));
    heightMask *= 1.0f - smoothstep(0.84f, 1.18f, abs(p.y));

    float timeScroll = time * 1.4f;
    float3 q = p;
    q.xz += float2(timeScroll * 0.45f, timeScroll * 0.18f);
    q.y *= 0.68f;

    float weatherA = FBM(float3(q.x * 0.34f + 9.0f, 0.0f, q.z * 0.34f + 5.0f));
    float weatherB = FBM(float3(q.x * 0.52f - 13.0f, q.y * 0.3f, q.z * 0.52f + 7.0f));
    float weather = saturate((weatherA * 0.78f + weatherB * 0.32f) - lerp(0.66f, 0.22f, coverage));

    float warpA = FBM(q * 0.42f + float3(11.0f, 2.0f, 17.0f));
    float warpB = FBM(q * 0.31f + float3(3.0f, 7.0f, 13.0f));
    q.xz += (float2(warpA, warpB) - 0.5f) * lerp(0.45f, 1.25f, weather);

    float baseShape = FBM(q * 0.95f);
    float bulkShape = FBM(q * 1.65f + float3(17.0f, 5.0f, 11.0f));
    float detailShape = FBM(q * 3.80f + float3(37.0f, 19.0f, 23.0f));
    float erosion = FBM(q * 7.50f + float3(61.0f, 41.0f, 29.0f));
    float wisps = FBM(q * 11.50f + float3(83.0f, 13.0f, 47.0f));

    float structure = weather * (baseShape * 0.72f + bulkShape * 0.34f);
    structure += detailShape * 0.16f * weather;
    structure += wisps * 0.05f * smoothstep(0.32f, 0.88f, height01);
    structure -= erosion * lerp(0.10f, 0.34f, height01);
    structure += 0.04f * sin(p.x * 0.55f + timeScroll * 1.3f) * sin(p.z * 0.47f - timeScroll * 0.9f);
    structure *= heightMask;

    float threshold = lerp(0.28f, 0.06f, coverage);
    float density = smoothstep(threshold, threshold + 0.24f, structure);
    density *= lerp(0.78f, 1.28f, smoothstep(0.18f, 0.58f, height01));
    density *= lerp(1.10f, 0.82f, smoothstep(0.54f, 1.0f, height01));
    return saturate(density * heightMask);
}

float2 IntersectCloudLayer(float3 rayOrigin, float3 rayDir, float layerMinY, float layerMaxY)
{
    if (abs(rayDir.y) < 1e-5f)
    {
        if (rayOrigin.y < layerMinY || rayOrigin.y > layerMaxY)
        {
            return float2(1.0f, -1.0f);
        }

        return float2(0.0f, 600.0f);
    }

    float t0 = (layerMinY - rayOrigin.y) / rayDir.y;
    float t1 = (layerMaxY - rayOrigin.y) / rayDir.y;
    return float2(min(t0, t1), max(t0, t1));
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

float EstimateLightTransmittance(
    float3 worldPos,
    float3 lightDir,
    float time,
    float coverage,
    float densityScale)
{
    float transmittance = 1.0f;
    float shadowStep = 8.0f;

    [ForceUnroll]
    for (int shadowStepIndex = 0; shadowStepIndex < 4; ++shadowStepIndex)
    {
        float stepLength = shadowStep * (1.0f + shadowStepIndex * 0.85f);
        float3 shadowPos = worldPos - lightDir * stepLength;
        float shadowDensity = CloudDensity(WorldToCloudFieldPos(shadowPos), time, coverage);
        transmittance *= exp(-shadowDensity * densityScale * 0.85f);
    }

    return transmittance;
}
