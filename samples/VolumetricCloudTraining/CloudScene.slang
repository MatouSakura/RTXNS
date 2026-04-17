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

float ValueNoise2D(float2 p)
{
    float2 cell = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0f - 2.0f * f);

    float n00 = Hash12(cell + float2(0.0f, 0.0f));
    float n10 = Hash12(cell + float2(1.0f, 0.0f));
    float n01 = Hash12(cell + float2(0.0f, 1.0f));
    float n11 = Hash12(cell + float2(1.0f, 1.0f));

    float nx0 = lerp(n00, n10, f.x);
    float nx1 = lerp(n01, n11, f.x);
    return lerp(nx0, nx1, f.y);
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

float SignedNoise(float3 p)
{
    return ValueNoise(p) * 2.0f - 1.0f;
}

float FBM2D(float2 p)
{
    float sum = 0.0f;
    float amplitude = 0.5f;

    [unroll]
    for (int octave = 0; octave < 4; ++octave)
    {
        sum += amplitude * ValueNoise2D(p);
        p = p * 2.03f + float2(13.7f, 7.1f);
        amplitude *= 0.5f;
    }

    return sum;
}

float RayPlaneIntersect(float3 ro, float3 rd, float3 center, float3 normal)
{
    float denom = dot(normal, rd);
    if (abs(denom) > 0.0001f)
    {
        return dot(center - ro, normal) / denom;
    }

    return -1.0f;
}

float OpSmoothUnion(float d1, float d2, float k)
{
    float h = clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return lerp(d2, d1, h) - k * h * (1.0f - h);
}

static const float3 CLOUD_SIZE = float3(2.6f, 1.1f, 2.6f);
static const float3 CLOUD_HALF_SIZE = CLOUD_SIZE * 0.5f;
static const float3 CLOUD_BOX_MIN = -CLOUD_HALF_SIZE;
static const float3 CLOUD_BOX_MAX = CLOUD_HALF_SIZE;
static const float3 CLOUD_TRAIN_DOMAIN_MIN = CLOUD_BOX_MIN;
static const float3 CLOUD_TRAIN_DOMAIN_MAX = CLOUD_BOX_MAX;

float3 WorldToCloudFieldPos(float3 worldPos)
{
    return worldPos;
}

bool IsInsideCloudTrainingDomain(float3 p)
{
    bool3 inside = (p >= CLOUD_TRAIN_DOMAIN_MIN) & (p <= CLOUD_TRAIN_DOMAIN_MAX);
    return all(inside);
}

float CloudBoundaryFade(float3 p)
{
    float3 normAbs = abs(p / CLOUD_HALF_SIZE);
    float edge = max(normAbs.x, max(normAbs.y, normAbs.z));
    float fade = 1.0f - smoothstep(0.68f, 0.96f, edge);
    return saturate(fade * fade);
}

float CloudInteriorFade(float3 p)
{
    float3 norm = p / CLOUD_HALF_SIZE;
    float radial = length(norm * float3(0.90f, 1.20f, 0.90f));
    float fade = 1.0f - smoothstep(0.72f, 1.06f, radial);
    return saturate(fade);
}

float CloudDensity(float3 p, float time, float coverage)
{
    float3 normDist = p / CLOUD_HALF_SIZE;
    float topClip = smoothstep(-0.0f, 1.0f, normDist.y);
    float sideClipX = smoothstep(0.60f - topClip * 0.6f, 1.0f, abs(normDist.x));
    float sideClipZ = smoothstep(0.60f - topClip * 0.6f, 1.0f, abs(normDist.z));
    float bottomClip = smoothstep(-0.5f, -1.0f, normDist.y);

    float d1 = distance(normDist * float3(1.0f, 0.9f, 1.0f), float3(0.0f, -0.65f, 0.0f) * float3(1.0f, 0.8f, 1.0f));
    float d2 = distance(normDist * float3(1.6f, 1.0f, 1.6f), float3(-0.3f, 0.2f, 0.3f));
    float blendK = lerp(0.35f, 0.65f, saturate(coverage));
    float dens = 1.0f - OpSmoothUnion(pow(d1 * 1.1f, 1.2f), pow(d2 * 1.1f, 1.4f), blendK);

    float freq = 3.9f;
    float amp = 0.5f;
    float3 q = p + float3(1.0f, 0.0f, 2.0f);
    [unroll]
    for (int i = 1; i < 4; ++i)
    {
        float timeOffs = time * (1.0f + float(i)) * 2.5f;
        if (i == 1)
        {
            timeOffs = 0.0f;
        }

        float3 sineCoeff = float3(
            q.z * 2.4f * freq,
            q.x * 2.6f * freq + timeOffs,
            q.y * 2.8f * freq);
        float3 randOffs = sin(sineCoeff);
        dens += (abs(SignedNoise(q * freq + float3(i * 167) + randOffs * 0.022f * freq)) - 0.48f) * amp;
        freq *= 2.1f;
        amp *= 0.5f;
    }

    dens += lerp(-0.18f, 0.20f, saturate(coverage));
    float clipped = max(
        0.0f,
        dens * 0.8f -
            pow(sideClipX, 10.5f) * 0.4f -
            pow(sideClipZ, 10.5f) * 0.4f -
            pow(topClip, 13.5f) * 0.3f -
            pow(bottomClip, 25.0f) * 0.5f);

    float density = saturate((clipped - 0.055f) * 1.45f);
    density *= CloudBoundaryFade(p) * CloudInteriorFade(p);
    return density * density;
}

float2 IntersectAABB(float3 rayOrigin, float3 rayDir, float3 boxMin, float3 boxMax)
{
    float3 tMin = (boxMin - rayOrigin) / rayDir;
    float3 tMax = (boxMax - rayOrigin) / rayDir;
    float3 t1 = min(tMin, tMax);
    float3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return float2(tNear, tFar);
}

float3 EvaluateSky(float3 viewDir, float3 lightDir, float3 sunColor, float3 cameraPos)
{
    float3 sunDir = normalize(-lightDir);
    float skyLerp = smoothstep(-0.08f, 0.42f, viewDir.y);
    float horizonBand = exp(-abs(viewDir.y) * 18.0f);

    float3 zenithColor = float3(0.08f, 0.18f, 0.30f);
    float3 skyColor = float3(0.32f, 0.62f, 0.82f);
    float3 horizonColor = float3(0.92f, 0.88f, 0.82f);
    float3 groundColor = float3(0.43f, 0.50f, 0.60f);

    float3 sky = lerp(skyColor, zenithColor, saturate(viewDir.y * 0.5f + 0.5f));
    sky = lerp(horizonColor, sky, skyLerp);

    float sunGlow = pow(saturate(dot(viewDir, sunDir)), 8.0f);
    float sunDisk = smoothstep(0.995f, 0.9992f, dot(viewDir, sunDir));
    sky += sunColor * (sunGlow * 0.08f + sunDisk * 0.35f);
    sky += horizonColor * horizonBand * 0.08f;

    float groundBlend = smoothstep(-0.12f, 0.02f, viewDir.y);
    return lerp(groundColor, sky, groundBlend);
}

float CloudMarchStep(float distanceToSample)
{
    return 0.07f;
}

float CloudExtinction(float density, float densityScale, float absorption)
{
    float scale = 0.65f + densityScale * 0.35f;
    float absorb = max(0.15f, absorption);
    return max(0.0f, density * scale * absorb);
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
    float light = exp(-lightDensity * lightDensity * 12.0f);
    float ambientSoft = exp(-density * 5.0f) * 0.9f + 0.1f;
    float3 ambient = lerp(float3(0.15f, 0.25f, 0.55f), float3(0.58f, 0.85f, 1.02f), ambientSoft);
    float3 sampleColor = float3(0.7f, 0.85f, 1.0f) * (light * sunColor + ambient * 1.6f);

    float fog = saturate(1.0f - exp(-distanceToSample * 0.08f));
    return lerp(sampleColor, background, fog * 0.4f);
}

static const float CLOUD_TRACE_MAX_DISTANCE = 8.0f;
static const float CLOUD_DENSITY_HIT_THRESHOLD = 0.01f;
static const float CLOUD_TRANSMITTANCE_CUTOFF = 0.01f;
static const float CLOUD_LIGHT_PROBE_DISTANCE = 0.12f;

struct CloudRaymarchContext
{
    bool valid;
    float tStart;
    float tEnd;
    float3 background;
};

CloudRaymarchContext InitCloudRaymarch(
    float3 rayOrigin,
    float3 rayDir,
    float3 lightDir,
    float3 sunColor,
    float3 skyColor,
    float3 horizonColor,
    float3 boxMin,
    float3 boxMax)
{
    CloudRaymarchContext context;
    context.valid = false;
    context.tStart = 0.0f;
    context.tEnd = 0.0f;
    context.background = EvaluateSky(rayDir, lightDir, sunColor, rayOrigin);

    float2 hit = IntersectAABB(rayOrigin, rayDir, boxMin, boxMax);
    if (hit.x > hit.y)
    {
        return context;
    }

    float tStart = max(hit.x, 0.0f);
    float tEnd = min(hit.y, tStart + CLOUD_TRACE_MAX_DISTANCE);
    if (tEnd <= tStart)
    {
        return context;
    }

    context.valid = true;
    context.tStart = tStart;
    context.tEnd = tEnd;
    return context;
}

bool CloudHasVisibleDensity(float density)
{
    return density > CLOUD_DENSITY_HIT_THRESHOLD;
}

void CloudAccumulateStep(
    float density,
    float lightDensity,
    float distanceToSample,
    float stepSize,
    float3 rayDir,
    float3 lightDir,
    float3 background,
    float3 sunColor,
    float densityScale,
    float absorption,
    inout float transmittance,
    inout float3 accumColor)
{
    float extinction = CloudExtinction(density, densityScale, absorption);
    float alpha = saturate(1.0f - exp(-extinction * stepSize));
    float3 sampleColor = ShadeCloudSample(
        density,
        lightDensity,
        distanceToSample,
        stepSize,
        rayDir,
        lightDir,
        background,
        sunColor,
        densityScale,
        absorption);

    accumColor += transmittance * alpha * sampleColor;
    transmittance *= (1.0f - alpha);
}

bool CloudShouldTerminate(float transmittance)
{
    return transmittance < CLOUD_TRANSMITTANCE_CUTOFF;
}

float3 ComposeCloudColor(float3 accumColor, float transmittance, float3 background)
{
    return accumColor + transmittance * background;
}

float3 ApplyCloudPost(float3 color)
{
    return color;
}
