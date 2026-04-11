/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <donut/app/UserInterfaceUtils.h>
#include <donut/app/imgui_renderer.h>
#include <imgui_internal.h>

#include "UIWidget.h"

UIWidget::UIWidget(UIData& uiData) : m_uiData(uiData)
{
}

UIWidget::~UIWidget()
{
}

void UIWidget::Draw()
{
    ImGui::SliderFloat("Sun Intensity", &m_uiData.sunIntensity, 0.2f, 8.f);
    ImGui::SliderFloat("Coverage", &m_uiData.coverage, 0.2f, 0.85f);
    ImGui::SliderFloat("Density Scale", &m_uiData.densityScale, 0.2f, 3.f);
    ImGui::SliderFloat("Absorption", &m_uiData.absorption, 0.4f, 3.f);
    ImGui::SliderFloat("Time", &m_uiData.time, 0.f, 1.f);
    ImGui::Checkbox("Animate Time", &m_uiData.animateTime);

    ImGui::Text("Epochs : %d", m_uiData.epochs);
    ImGui::Text("Training Time : %.2f s", m_uiData.trainingTime);

    if (ImGui::Button(m_uiData.training ? "Disable Training" : "Enable Training"))
    {
        m_uiData.training = !m_uiData.training;
    }

    if (ImGui::Button("Reset Training"))
    {
        m_uiData.reset = true;
    }

    if (ImGui::Button("Load Model"))
    {
        std::string fileName;
        if (donut::app::FileDialog(true, "BIN files\0*.bin\0All files\0*.*\0\0", fileName))
        {
            m_uiData.fileName = fileName;
            m_uiData.load = true;
        }
    }

    if (ImGui::Button("Save Model"))
    {
        std::string fileName;
        if (donut::app::FileDialog(false, "BIN files\0*.bin\0All files\0*.*\0\0", fileName))
        {
            m_uiData.fileName = fileName;
            m_uiData.load = false;
        }
    }
}

void UIWidget::Reset()
{
    m_uiData.epochs = 0;
    m_uiData.trainingTime = 0.0f;
}
