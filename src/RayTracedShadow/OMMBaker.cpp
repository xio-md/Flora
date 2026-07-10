#include "OMMBaker.h"

#include <omm.hpp>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace rtxns::shadow {

// =====================================================
// PIMPL wrapper
// =====================================================
class OMMBaker::Impl
{
public:
    Impl()
    {
        omm::BakerCreationDesc desc;
        desc.type = omm::BakerType::CPU;
        desc.messageInterface.messageCallback = [](omm::MessageSeverity, const char* msg, void*) {
            std::cout << "[OMM-SDK] " << msg << std::endl;
        };
        omm::Result res = omm::CreateBaker(desc, &baker);
        if (res != omm::Result::SUCCESS)
        {
            std::cerr << "[OMMBaker] Failed to create baker" << std::endl;
            baker = nullptr;
        }
    }

    ~Impl()
    {
        if (baker)
            omm::DestroyBaker(baker);
    }

    OMMBakeResult bake(const OMMBakeInput& input)
    {
        OMMBakeResult result;

        if (!baker)
        {
            std::cerr << "[OMMBaker] Baker not created" << std::endl;
            return result;
        }

        if (!input.alphaPixels.empty() && input.texWidth == 0)
        {
            std::cerr << "[OMMBaker] Invalid texture dimensions" << std::endl;
            return result;
        }

        // 1. Create texture
        omm::Cpu::TextureMipDesc mipDesc;
        mipDesc.width = input.texWidth;
        mipDesc.height = input.texHeight;
        mipDesc.textureData = input.alphaPixels.data();

        omm::Cpu::TextureDesc texDesc;
        texDesc.format = omm::Cpu::TextureFormat::FP32;
        texDesc.mipCount = 1;
        texDesc.mips = &mipDesc;

        omm::Cpu::Texture texHandle = nullptr;
        omm::Result res = omm::Cpu::CreateTexture(baker, texDesc, &texHandle);
        if (res != omm::Result::SUCCESS)
        {
            std::cerr << "[OMMBaker] CreateTexture failed" << std::endl;
            return result;
        }

        // 2. Setup bake parameters
        omm::Cpu::BakeInputDesc desc;
        desc.bakeFlags = omm::Cpu::BakeFlags::None;
        desc.texture = texHandle;
        desc.alphaCutoff = input.alphaCutoff;
        desc.alphaMode = omm::AlphaMode::Test;
        desc.runtimeSamplerDesc = {
            omm::TextureAddressMode::Wrap,
            omm::TextureFilterMode::Linear
        };
        desc.texCoordFormat = omm::TexCoordFormat::UV32_FLOAT;
        desc.texCoordStrideInBytes = input.uvStride;
        desc.texCoords = input.uvData;
        desc.indexBuffer = input.indexData;
        desc.indexCount = input.indexCount;
        desc.indexFormat = input.indexStride == 2
            ? omm::IndexFormat::UINT_16
            : omm::IndexFormat::UINT_32;
        // Use maxSubdivisionLevel for uniform subdivision (per-triangle subdivisionLevels is nullptr)
        desc.maxSubdivisionLevel = static_cast<uint8_t>(input.subdivisionLevel);
        desc.subdivisionLevels = nullptr; // use maxSubdivisionLevel for all triangles
        desc.format = static_cast<omm::Format>(input.format);
        desc.unknownStatePromotion = omm::UnknownStatePromotion::ForceOpaque;

        // 3. Bake
        omm::Cpu::BakeResult bakeResult = nullptr;
        res = omm::Cpu::Bake(baker, desc, &bakeResult);
        if (res != omm::Result::SUCCESS)
        {
            std::cerr << "[OMMBaker] Bake failed" << std::endl;
            omm::Cpu::DestroyTexture(baker, texHandle);
            return result;
        }

        // 4. Read back results
        const omm::Cpu::BakeResultDesc* bakeDesc = nullptr;
        res = omm::Cpu::GetBakeResultDesc(bakeResult, &bakeDesc);
        if (res != omm::Result::SUCCESS || !bakeDesc)
        {
            std::cerr << "[OMMBaker] GetBakeResultDesc failed" << std::endl;
            omm::Cpu::DestroyBakeResult(bakeResult);
            omm::Cpu::DestroyTexture(baker, texHandle);
            return result;
        }

        // Copy arrayData
        if (bakeDesc->arrayData && bakeDesc->arrayDataSize > 0)
        {
            result.arrayData.resize(bakeDesc->arrayDataSize);
            std::memcpy(result.arrayData.data(), bakeDesc->arrayData, bakeDesc->arrayDataSize);
        }

        // Copy descArray
        result.descCount = bakeDesc->descArrayCount;
        if (bakeDesc->descArray && result.descCount > 0)
        {
            size_t sz = result.descCount * sizeof(omm::Cpu::OpacityMicromapDesc);
            result.descArray.resize(sz);
            std::memcpy(result.descArray.data(), bakeDesc->descArray, sz);
        }

        // Copy descHistogram
        result.descHistogramCount = bakeDesc->descArrayHistogramCount;
        if (bakeDesc->descArrayHistogram && result.descHistogramCount > 0)
        {
            size_t sz = result.descHistogramCount * sizeof(omm::Cpu::OpacityMicromapUsageCount);
            result.descHistogramData.resize(sz);
            std::memcpy(result.descHistogramData.data(), bakeDesc->descArrayHistogram, sz);
        }

        // Copy indexBuffer
        result.indexCount = bakeDesc->indexCount;
        result.indexFormat = static_cast<uint32_t>(bakeDesc->indexFormat);
        size_t ibSize = 0;
        switch (bakeDesc->indexFormat)
        {
            case omm::IndexFormat::UINT_8:  ibSize = result.indexCount; break;
            case omm::IndexFormat::UINT_16: ibSize = result.indexCount * 2; break;
            case omm::IndexFormat::UINT_32: ibSize = result.indexCount * 4; break;
            default: break;
        }
        if (bakeDesc->indexBuffer && ibSize > 0)
        {
            result.indexBuffer.resize(ibSize);
            std::memcpy(result.indexBuffer.data(), bakeDesc->indexBuffer, ibSize);
        }

        // Copy indexHistogram
        result.indexHistogramCount = bakeDesc->indexHistogramCount;
        if (bakeDesc->indexHistogram && result.indexHistogramCount > 0)
        {
            size_t sz = result.indexHistogramCount * sizeof(omm::Cpu::OpacityMicromapUsageCount);
            result.indexHistogramData.resize(sz);
            std::memcpy(result.indexHistogramData.data(), bakeDesc->indexHistogram, sz);
        }

        std::cout << "[OMMBaker] Baked: " << result.descCount << " OMMs, "
                  << result.arrayData.size() << " bytes array, "
                  << result.indexCount << " indices" << std::endl;

        // Cleanup
        omm::Cpu::DestroyBakeResult(bakeResult);
        omm::Cpu::DestroyTexture(baker, texHandle);
        return result;
    }

    omm::Baker baker = nullptr;
};

// =====================================================
// Public interface
// =====================================================

OMMBaker::OMMBaker()
    : m_impl(std::make_unique<Impl>())
{}

OMMBaker::~OMMBaker() = default;

OMMBakeResult OMMBaker::bake(const OMMBakeInput& input)
{
    return m_impl->bake(input);
}

bool OMMBaker::readAlphaTexture(
    nvrhi::IDevice* device,
    nvrhi::ITexture* texture,
    uint32_t width,
    uint32_t height,
    std::vector<float>& outPixels)
{
    if (!device || !texture || width == 0 || height == 0)
        return false;

    // Create staging texture for readback (copy from source without modifying source state)
    nvrhi::TextureDesc stagingDesc;
    stagingDesc.width = width;
    stagingDesc.height = height;
    stagingDesc.format = nvrhi::Format::RGBA8_UNORM;
    stagingDesc.debugName = "OMMAlphaStaging";
    stagingDesc.initialState = nvrhi::ResourceStates::CopyDest;
    stagingDesc.keepInitialState = true;
    auto staging = device->createStagingTexture(stagingDesc, nvrhi::CpuAccessMode::Read);
    if (!staging)
        return false;

    // Use a separate command list to isolate the copy from the main render pipeline
    auto cmdList = device->createCommandList();
    cmdList->open();
    cmdList->copyTexture(staging, nvrhi::TextureSlice(), texture, nvrhi::TextureSlice());
    cmdList->close();
    device->executeCommandList(cmdList);
    // Ensure the GPU copy is complete before mapping the staging texture for readback
    device->waitForIdle();

    // Read back
    size_t rowPitch = 0;
    const auto* mapped = static_cast<const uint8_t*>(
        device->mapStagingTexture(staging, nvrhi::TextureSlice(),
            nvrhi::CpuAccessMode::Read, &rowPitch));

    if (mapped)
    {
        outPixels.resize(width * height);
        for (uint32_t y = 0; y < height; ++y)
        {
            for (uint32_t x = 0; x < width; ++x)
            {
                // RGBA8: alpha at byte offset 3
                outPixels[y * width + x] = mapped[y * rowPitch + x * 4 + 3] / 255.0f;
            }
        }
        device->unmapStagingTexture(staging);
        return true;
    }

    return false;
}

} // namespace rtxns::shadow
