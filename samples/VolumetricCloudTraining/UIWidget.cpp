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

    ImGui::SeparatorText("Camera");
    ImGui::SliderFloat("Move Step", &m_uiData.cameraMoveStep, 0.02f, 2.0f, "%.2f");
    ImGui::SliderFloat("Height Step", &m_uiData.cameraHeightStep, 0.02f, 1.5f, "%.2f");
    ImGui::TextUnformatted("Mouse: L drag orbit, R drag pan, Wheel zoom");

    ImGui::PushButtonRepeat(true);
    if (ImGui::Button("Rise"))
    {
        m_uiData.cameraMoveUp += 1;
    }
    ImGui::SameLine();
    if (ImGui::Button("Fall"))
    {
        m_uiData.cameraMoveUp -= 1;
    }
    ImGui::PopButtonRepeat();

    const float arrowOffset = 34.0f;
    ImGui::PushButtonRepeat(true);
    ImGui::Dummy(ImVec2(arrowOffset, 0.0f));
    ImGui::SameLine();
    if (ImGui::ArrowButton("##cam_up", ImGuiDir_Up))
    {
        m_uiData.cameraMoveForward += 1;
    }

    if (ImGui::ArrowButton("##cam_left", ImGuiDir_Left))
    {
        m_uiData.cameraMoveRight -= 1;
    }
    ImGui::SameLine();
    if (ImGui::ArrowButton("##cam_down", ImGuiDir_Down))
    {
        m_uiData.cameraMoveForward -= 1;
    }
    ImGui::SameLine();
    if (ImGui::ArrowButton("##cam_right", ImGuiDir_Right))
    {
        m_uiData.cameraMoveRight += 1;
    }
    ImGui::PopButtonRepeat();

    if (ImGui::Button("Reset Camera"))
    {
        m_uiData.resetCamera = true;
    }

    ImGui::Text("Epochs : %d", m_uiData.epochs);
    ImGui::Text("Training Time : %.2f s", m_uiData.trainingTime);
    ImGui::Text("Training: %s", m_uiData.training ? "ON" : "OFF");

    if (ImGui::Button(m_uiData.training ? "Stop Training" : "Start Training"))
    {
        m_uiData.training = !m_uiData.training;
    }

    ImGui::Checkbox("Show Neural View", &m_uiData.showNeuralView);
    ImGui::Checkbox("Show Error View (Perf Cost)", &m_uiData.showErrorView);

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
