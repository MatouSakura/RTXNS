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

#include <string>

struct UIData
{
    float sunIntensity = 1.45f;
    float coverage = 0.55f;
    float densityScale = 1.35f;
    float absorption = 1.20f;
    float time = 0.10f;
    bool animateTime = true;
    bool unlit = false;

    float trainingTime = 0.0f;
    uint32_t epochs = 0;

    bool reset = false;
    bool training = false;
    bool load = false;
    std::string fileName;

    float cameraMoveStep = 1.5f;
    float cameraHeightStep = 0.8f;
    int cameraMoveForward = 0;
    int cameraMoveRight = 0;
    int cameraMoveUp = 0;
    bool resetCamera = false;
    bool showNeuralView = false;
    bool showErrorView = false;
};
