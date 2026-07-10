#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

#include <nvrhi/nvrhi.h>

namespace rtxns::shadow {

// Output of OMM CPU Bake — maps directly to NVRHI OpacityMicromapDesc fields
struct OMMBakeResult
{
    std::vector<uint8_t>  arrayData;          // per-microtriangle opacity bits
    std::vector<uint8_t>  descArray;          // ommCpuOpacityMicromapDesc[]
    std::vector<uint8_t>  indexBuffer;        // triangle → OMM index
    std::vector<uint8_t>  descHistogramData;  // VkMicromapUsageEXT[] (same layout as nvrhi::rt::OpacityMicromapUsageCount)
    std::vector<uint8_t>  indexHistogramData; // same layout, for BLAS attachment

    uint32_t descCount = 0;
    uint32_t descHistogramCount = 0;
    uint32_t indexCount = 0;
    uint32_t indexHistogramCount = 0;
    uint32_t indexFormat = 0; // omm::IndexFormat

    bool isValid() const { return !arrayData.empty() && !indexBuffer.empty(); }
};

// Per-geometry bake input
struct OMMBakeInput
{
    int meshIndex = -1;              // index into blasInputs[]
    unsigned int alphaTextureIndex = 0; // texture descriptor index

    // CPU-side alpha texture data (read back from GPU)
    std::vector<float> alphaPixels;  // single-channel float32 luminance
    uint32_t texWidth = 0;
    uint32_t texHeight = 0;

    // Geometry data (same as used for BLAS build)
    const void* indexData = nullptr;
    uint32_t indexCount = 0;
    uint32_t indexStride = 0; // 2 (U16) or 4 (U32)
    const void* uvData = nullptr;
    uint32_t uvStride = 0;

    float alphaCutoff = 0.5f;
    uint32_t subdivisionLevel = 5; // 4^5 = 1024 microtriangles per triangle
    uint32_t format = 2;           // omm::Format::OC1_4_State = 2
};

// CPU-side OMM baker using NVIDIA OMM SDK
class OMMBaker
{
public:
    OMMBaker();
    ~OMMBaker();

    // Bake OMM data for a single geometry input
    OMMBakeResult bake(const OMMBakeInput& input);

    // Read back an alpha texture from GPU to CPU for baking
    static bool readAlphaTexture(
        nvrhi::IDevice* device,
        nvrhi::ITexture* texture,
        uint32_t width,
        uint32_t height,
        std::vector<float>& outPixels);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace rtxns::shadow
