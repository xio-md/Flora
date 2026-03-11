if (NOT DONUT_WITH_VULKAN)
    message(FATAL_ERROR "RtxRenderPy currently requires DONUT_WITH_VULKAN=ON.")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development Development.Module)

set(RTXNS_LOCAL_PYBIND11_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/third_party/pybind11/include")
if (NOT EXISTS "${RTXNS_LOCAL_PYBIND11_INCLUDE_DIR}/pybind11/pybind11.h")
    message(FATAL_ERROR
        "pybind11 headers were not found at ${RTXNS_LOCAL_PYBIND11_INCLUDE_DIR}.")
endif()

message(STATUS "Using local pybind11 headers from ${RTXNS_LOCAL_PYBIND11_INCLUDE_DIR}")

Python_add_library(RtxRenderPy MODULE
    py_interface.cpp
    headless_pbr.cpp)

target_include_directories(RtxRenderPy
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${RTXNS_LOCAL_PYBIND11_INCLUDE_DIR}")

target_link_libraries(RtxRenderPy
    PRIVATE
        Python::Module
        donut_app
        donut_render)

set_target_properties(RtxRenderPy PROPERTIES
    OUTPUT_NAME "RtxRenderPy"
    PREFIX "")

foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(RtxRenderPy PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${config} "${RTXNS_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_${config} "${RTXNS_BINARY_DIR}")
endforeach()
