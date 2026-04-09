#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

if( TARGET implot )
    return()
endif()


if (NOT TARGET imgui)
    message(FATAL_ERROR "Implot requires imgui")
endif()

set(RTXNS_IMPLOT_SOURCE_DIR
    "${CMAKE_SOURCE_DIR}/external/implot"
    CACHE PATH "Local path to implot source")
set(RTXNS_IMPLOT_FETCH_URL
    "https://github.com/epezent/implot.git"
    CACHE STRING "URL used to fetch implot when no local source is present")
set(RTXNS_IMPLOT_FETCH_TAG
    "v0.17"
    CACHE STRING "Git tag or commit used to fetch implot")

if (EXISTS "${RTXNS_IMPLOT_SOURCE_DIR}/implot.cpp")
    set(IMPLOT_SOURCE_DIR "${RTXNS_IMPLOT_SOURCE_DIR}")
else()
    include(FetchContent)
    FetchContent_Declare(
        implot
        GIT_REPOSITORY ${RTXNS_IMPLOT_FETCH_URL}
        GIT_TAG ${RTXNS_IMPLOT_FETCH_TAG}
    )
    FetchContent_MakeAvailable(implot)
    set(IMPLOT_SOURCE_DIR "${implot_SOURCE_DIR}")
endif()

# Override Imgui build - we want a lean static library

set(implot_srcs
    ${IMPLOT_SOURCE_DIR}/implot.cpp
    ${IMPLOT_SOURCE_DIR}/implot.h
    ${IMPLOT_SOURCE_DIR}/implot_internal.h
    ${IMPLOT_SOURCE_DIR}/implot_items.cpp
)

add_library(implot STATIC ${implot_srcs})
set_target_properties(implot PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(implot PUBLIC "${IMPLOT_SOURCE_DIR}")
target_link_libraries(implot imgui)
