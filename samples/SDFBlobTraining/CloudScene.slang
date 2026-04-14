/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

float Hash12(float2 p)
{
    return frac(sin(dot(p, float2(127.1f, 311.7f))) * 43758.5453123f);
}

static const float3 SDF_DOMAIN_MIN = float3(-2.6f, -1.8f, -2.6f);
static const float3 SDF_DOMAIN_MAX = float3(2.6f, 1.8f, 2.6f);
static const float SDF_VALUE_CLAMP = 2.0f;
static const float SDF_HIT_EPS = 0.003f;
static const float SDF_MAX_DISTANCE = 40.0f;

bool IsInsideSdfDomain(float3 p)
{
    bool3 inRange = (p >= SDF_DOMAIN_MIN) & (p <= SDF_DOMAIN_MAX);
    return all(inRange);
}

float3 NormalizeSdfPos(float3 p)
{
    float3 extent = SDF_DOMAIN_MAX - SDF_DOMAIN_MIN;
    return (p - SDF_DOMAIN_MIN) / extent * 2.0f - 1.0f;
}

float NormalizeSdfCoverage(float coverage)
{
    return saturate((coverage - 0.2f) / 0.65f);
}

float SmoothMin(float a, float b, float k)
{
    float h = saturate(0.5f + 0.5f * (b - a) / k);
    return lerp(b, a, h) - k * h * (1.0f - h);
}

void GetBlobCenters(float time, out float3 c0, out float3 c1, out float3 c2)
{
    float t = time * 6.2831853f;
    c0 = float3(sin(t * 1.30f) * 0.85f, 0.20f + cos(t * 0.80f) * 0.22f, cos(t * 1.10f) * 0.75f);
    c1 = float3(cos(t * 1.10f + 1.2f) * 0.90f, -0.25f + sin(t * 1.40f) * 0.24f, sin(t * 1.35f + 0.6f) * 0.70f);
    c2 = float3(sin(t * 0.95f - 0.8f) * 0.78f, 0.05f + cos(t * 1.60f + 2.0f) * 0.20f, cos(t * 1.25f - 1.7f) * 0.92f);
}

float BlobBlend(float coverage)
{
    return lerp(0.12f, 0.70f, saturate(coverage));
}

float BlobSceneSdf(float3 p, float time, float coverage)
{
    float3 c0, c1, c2;
    GetBlobCenters(time, c0, c1, c2);

    float r0 = 0.74f + 0.07f * sin(time * 6.2831853f * 1.2f);
    float r1 = 0.66f + 0.08f * cos(time * 6.2831853f * 1.4f);
    float r2 = 0.71f + 0.06f * sin(time * 6.2831853f * 1.6f + 1.5f);
    float k = BlobBlend(coverage);

    float d0 = length(p - c0) - r0;
    float d1 = length(p - c1) - r1;
    float d2 = length(p - c2) - r2;

    float d = SmoothMin(d0, d1, k);
    d = SmoothMin(d, d2, k);

    // Slight squashing to get a "melted" look.
    d += (p.y + 0.35f) * 0.06f;
    return d;
}

float EncodeSdfForMLP(float sdfValue)
{
    return saturate((clamp(sdfValue, -SDF_VALUE_CLAMP, SDF_VALUE_CLAMP) + SDF_VALUE_CLAMP) / (2.0f * SDF_VALUE_CLAMP));
}

float DecodeSdfFromMLP(float encodedValue)
{
    return lerp(-SDF_VALUE_CLAMP, SDF_VALUE_CLAMP, saturate(encodedValue));
}

float3 EstimateSdfNormal(float3 p, float time, float coverage)
{
    const float eps = 0.0035f;
    return normalize(float3(
        BlobSceneSdf(p + float3(eps, 0, 0), time, coverage) - BlobSceneSdf(p - float3(eps, 0, 0), time, coverage),
        BlobSceneSdf(p + float3(0, eps, 0), time, coverage) - BlobSceneSdf(p - float3(0, eps, 0), time, coverage),
        BlobSceneSdf(p + float3(0, 0, eps), time, coverage) - BlobSceneSdf(p - float3(0, 0, eps), time, coverage)));
}

float SoftShadow(float3 ro, float3 rd, float time, float coverage)
{
    float result = 1.0f;
    float t = 0.02f;

    [unroll]
    for (int i = 0; i < 20; ++i)
    {
        float h = BlobSceneSdf(ro + rd * t, time, coverage);
        result = min(result, 14.0f * h / max(0.01f, t));
        t += clamp(h, 0.02f, 0.25f);
        if (t > 8.0f)
            break;
    }

    return saturate(result);
}

float3 EvaluateSky(float3 rayDir, float3 lightDir, float3 sunColor, float3 skyColor, float3 horizonColor)
{
    float skyAmount = saturate(rayDir.y * 0.5f + 0.5f);
    float sunAmount = saturate(dot(rayDir, -lightDir));
    float horizonAmount = saturate(1.0f - abs(rayDir.y) * 2.8f);

    float3 sky = lerp(horizonColor * 1.06f, skyColor * 0.90f, smoothstep(0.0f, 1.0f, skyAmount));
    sky += horizonColor * horizonAmount * 0.15f;
    sky += sunColor * (pow(sunAmount, 10.0f) * 0.12f + pow(sunAmount, 220.0f) * 0.80f);
    return sky;
}

float SdfTraceStep(float sdfValue)
{
    return clamp(abs(sdfValue) * 0.82f, 0.01f, 0.48f);
}

float SdfDistanceFog(float distanceToHit)
{
    return saturate(1.0f - exp(-0.035f * distanceToHit));
}

float3 ShadeBlobSurface(
    float3 position,
    float3 normal,
    float3 rayDir,
    float3 lightDir,
    float time,
    float coverage,
    float3 sunColor,
    float3 skyColor,
    float3 horizonColor,
    float specularCtrl,
    float emissiveCtrl)
{
    float3 L = normalize(-lightDir);
    float NdotL = saturate(dot(normal, L));
    float3 H = normalize(L - rayDir);
    float NdotH = saturate(dot(normal, H));
    float VdotN = saturate(dot(normal, -rayDir));

    float shadow = SoftShadow(position + normal * 0.01f, L, time, coverage);
    float specPower = lerp(20.0f, 120.0f, saturate(specularCtrl / 3.0f));
    float spec = pow(NdotH, specPower) * (0.16f + 0.55f * saturate(specularCtrl / 3.0f));
    float fresnel = pow(1.0f - VdotN, 4.0f);

    float tone = 0.50f + 0.50f * sin(time * 6.2831853f + position.x * 1.4f + position.z * 1.1f);
    float3 baseColor = lerp(float3(0.34f, 0.54f, 0.94f), float3(0.76f, 0.86f, 1.00f), tone);

    float surfaceSdf = abs(BlobSceneSdf(position, time, coverage));
    float edgeGlow = exp(-surfaceSdf * 8.0f) * saturate(emissiveCtrl / 3.0f);
    float3 emissive = float3(0.25f, 0.45f, 0.90f) * edgeGlow * (0.45f + 0.55f * tone);

    float3 lighting = baseColor * (0.18f + 0.82f * NdotL * shadow);
    lighting += sunColor * spec * shadow;
    lighting += skyColor * fresnel * 0.22f + horizonColor * fresnel * 0.05f;
    lighting += emissive;
    return lighting;
}
