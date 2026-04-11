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

struct UIData
{
    float sunIntensity = 3.0f;
    float coverage = 0.56f;
    float densityScale = 1.35f;
    float absorption = 1.75f;
    float time = 0.18f;
    bool animateTime = true;

    float trainingTime = 0.0f;
    uint32_t epochs = 0;

    bool reset = false;
    bool training = true;
    bool load = false;
    std::string fileName;
};
