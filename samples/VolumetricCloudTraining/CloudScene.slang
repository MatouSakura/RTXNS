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

static const float CLOUD_WORLD_TO_FIELD_XZ = 0.028f;
static const float CLOUD_WORLD_TO_FIELD_Y = 0.22f;
static const float CLOUD_WORLD_CENTER_Y = 22.0f;

float3 WorldToCloudFieldPos(float3 worldPos)
{
    return float3(
        worldPos.x * CLOUD_WORLD_TO_FIELD_XZ,
        (worldPos.y - CLOUD_WORLD_CENTER_Y) * CLOUD_WORLD_TO_FIELD_Y,
        worldPos.z * CLOUD_WORLD_TO_FIELD_XZ);
}

float CloudHeightMask(float y)
{
    float bottom = smoothstep(-2.6f, -1.0f, y);
    float top = 1.0f - smoothstep(0.8f, 2.0f, y);
    float middle = 1.0f - saturate(abs(y + 0.25f) * 0.33f);
    return saturate(bottom * top * (0.70f + 0.30f * middle));
}

float FBMCloud(float3 p, float time, int octaves)
{
    float3 q = p - float3(0.0f, 0.1f, 1.0f) * (time * 1.10f);
    float sum = 0.0f;
    float amplitude = 0.5f;

    [unroll]
    for (int octave = 0; octave < 5; ++octave)
    {
        if (octave >= octaves)
        {
            break;
        }

        sum += amplitude * ValueNoise(q);
        q = q * 2.02f - float3(0.0f, time * 1.35f, 0.0f);
        amplitude *= 0.5f;
    }

    return sum;
}

float3 CloudDomainWarp(float3 p, float time)
{
    float3 q = p * 1.35f + float3(time * 0.15f, 0.0f, -time * 0.22f);
    float wx = FBMCloud(q + float3(7.2f, -3.6f, 1.4f), time * 0.70f, 3) - 0.5f;
    float wy = FBMCloud(q + float3(-5.3f, 4.8f, -2.2f), time * 0.80f, 3) - 0.5f;
    float wz = FBMCloud(q + float3(2.9f, 6.1f, 5.7f), time * 0.75f, 3) - 0.5f;
    return float3(wx, wy * 0.55f, wz);
}

float CloudDensityLod(float3 p, float time, float coverage, int octaves)
{
    float2 weatherUv = p.xz * 0.40f + float2(0.0f, time * 0.16f);
    float weather = FBM(float3(weatherUv, time * 0.06f));
    float weatherMask = smoothstep(lerp(0.74f, 0.42f, coverage), lerp(0.98f, 0.70f, coverage), weather);

    float3 warpedP = p + CloudDomainWarp(p, time) * 0.75f;
    float baseShape = FBMCloud(warpedP * 2.10f, time * 0.95f, max(3, octaves));
    float detail = FBMCloud(warpedP * 6.00f + float3(12.7f, 1.3f, -4.9f), time * 1.30f, min(4, octaves));
    float erosion = FBMCloud(warpedP * 10.0f + float3(-8.1f, 2.7f, 6.3f), time * 1.75f, min(3, octaves));

    float heightMask = CloudHeightMask(p.y);
    float shape = smoothstep(0.52f, 0.88f, baseShape + detail * 0.26f);
    shape *= weatherMask * heightMask;
    shape -= saturate((0.62f - erosion) * 0.35f) * (1.0f - heightMask * 0.65f);

    float density = saturate(shape);
    density = saturate(pow(density, 0.82f) * lerp(0.85f, 1.35f, coverage));
    return density;
}

float CloudDensity(float3 p, float time, float coverage)
{
    return CloudDensityLod(p, time, coverage, 5);
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
    float sunAmount = saturate(dot(rayDir, -lightDir));
    float horizonAmount = saturate(1.0f - abs(rayDir.y) * 2.6f);

    float3 zenith = skyColor * float3(0.62f, 0.70f, 0.90f);
    float3 lower = horizonColor * float3(1.04f, 0.92f, 0.82f);
    float3 sky = lerp(lower, zenith, smoothstep(0.0f, 0.95f, skyAmount));
    sky += horizonColor * horizonAmount * 0.16f;

    float sunBloom = pow(sunAmount, 10.0f);
    float sunDisk = pow(sunAmount, 220.0f);
    sky += sunColor * (sunBloom * 0.10f + sunDisk * 0.85f);
    return sky;
}

float CloudMarchStep(float distanceToSample)
{
    return max(0.30f, distanceToSample * 0.006f);
}

float CloudFog(float distanceToSample)
{
    float fieldDistance = distanceToSample * CLOUD_WORLD_TO_FIELD_XZ;
    return saturate(1.0f - exp(-0.16f * fieldDistance));
}

float CloudExtinction(float density, float densityScale, float absorption)
{
    return max(0.0f, density * densityScale * absorption * 0.90f);
}

float3 ShadeCloudSample(
    float density,
    float lightDensity,
    float distanceToSample,
    float stepSize,
    float3 rayDir,
    float3 lightDir,
    float3 background,
    float3 sunColor,
    float densityScale,
    float absorption)
{
    float phase = 0.30f + 1.05f * pow(saturate(dot(rayDir, -lightDir)), 7.0f);
    float lightTransmittance = exp(-lightDensity * densityScale * absorption * 1.25f);
    float extinction = CloudExtinction(density, densityScale, absorption);
    float edgeBoost = saturate(1.0f - density * 0.75f);
    float powder = 1.0f - exp(-density * 2.2f);

    float3 ambient = float3(0.36f, 0.45f, 0.58f);
    float3 scattering = sunColor * (0.35f + 0.65f * phase) * lightTransmittance * (0.45f + 0.55f * powder);
    float3 cloudTint = lerp(float3(1.08f, 1.02f, 0.94f), float3(0.66f, 0.72f, 0.80f), saturate(density * 1.35f));
    float3 localColor = cloudTint * (ambient + scattering * (0.85f + 0.35f * edgeBoost));

    localColor *= 0.70f + 0.30f * exp(-extinction * stepSize * 0.60f);
    localColor = lerp(localColor, background, CloudFog(distanceToSample) * 0.68f);

    float stepGlow = saturate(stepSize * 0.30f) * 0.05f;
    return localColor + sunColor * stepGlow * phase * edgeBoost;
}
