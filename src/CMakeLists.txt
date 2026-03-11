#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

set(LIBRARY_FILTER src)
if (RTXNS_BUILD_NEURAL_COMPONENTS)
    add_subdirectory(NeuralShading)
    add_subdirectory(Utils)
    add_subdirectory(NeuralShading_Shaders)
endif()

option(RTXNS_BUILD_PYTHON_BINDINGS "Build the headless Python renderer bindings" OFF)

if (RTXNS_BUILD_PYTHON_BINDINGS)
    add_subdirectory(PythonBindings)
endif()

