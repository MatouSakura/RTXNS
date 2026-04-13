/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#define INPUT_FEATURES 5
#define INPUT_NEURONS (INPUT_FEATURES * 6) // 6* from Frequency Encoding
#define OUTPUT_NEURONS 1

#define HIDDEN_NEURONS 64
#define NUM_HIDDEN_LAYERS 4
#define BATCH_SIZE (1 << 15)
#define BATCH_COUNT 96

#define LEARNING_RATE 0.0008f
#define COMPONENT_WEIGHTS float1(1.f)

#define NUM_TRANSITIONS (NUM_HIDDEN_LAYERS + 1)
#define NUM_TRANSITIONS_ALIGN4 ((NUM_TRANSITIONS + 3) / 4)
#define LOSS_SCALE 128.0

static const uint THREADS_PER_GROUP_TRAIN = 64;
static const uint THREADS_PER_GROUP_OPTIMIZE = 32;
static const uint THREADS_PER_GROUP_CONVERT = 64;
static const uint CLOUD_MARCH_STEPS = 144;

struct DirectConstantBufferEntry
{
    float4x4 viewProject;
    float4x4 viewProjectInverse;
    float4 cameraPos;

    float4 lightDir;
    float4 sunColor;
    float4 skyColor;
    float4 horizonColor;
    float4 volumeMin;
    float4 volumeMax;

    float time = 0;
    float coverage = 0;
    float densityScale = 0;
    float absorption = 0;
};

struct InferenceConstantBufferEntry
{
    DirectConstantBufferEntry directConstants;

    uint4 weightOffsets[NUM_TRANSITIONS_ALIGN4];
    uint4 biasOffsets[NUM_TRANSITIONS_ALIGN4];
};

struct TrainingConstantBufferEntry
{
    uint4 weightOffsets[NUM_TRANSITIONS_ALIGN4];
    uint4 biasOffsets[NUM_TRANSITIONS_ALIGN4];

    uint32_t maxParamSize;
    float learningRate;
    float currentStep;
    uint32_t batchSize;

    uint64_t seed;
    uint2 _pad = uint2(0, 0);
};
