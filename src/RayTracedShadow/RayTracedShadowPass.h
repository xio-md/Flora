#pragma once

#include "ShadowTypes.h"
#include <nvrhi/nvrhi.h>
#include <donut/engine/ShaderFactory.h>
#include <memory>
#include <vector>

namespace rtxns::shadow {

struct ShadowSceneResources
{
    nvrhi::BufferHandle instanceMetaBuffer;
    nvrhi::BufferHandle materialMetaBuffer;
    nvrhi::BufferHandle geometryMetaBuffer;
    nvrhi::BufferHandle combinedVertexBuffer;
    nvrhi::BufferHandle combinedIndexBuffer;
    std::vector<nvrhi::TextureHandle> alphaTextures;
    std::vector<ShadowInstanceMeta>  instanceMetas;
    std::vector<ShadowMaterialMeta>  materialMetas;
    std::vector<ShadowGeometryMeta>  geometryMetas;
    uint32_t instanceCount = 0;
    uint32_t materialCount = 0;
    uint32_t geometryCount = 0;
};

class RayTracedShadowPass
{
public:
    RayTracedShadowPass() = default;
    ~RayTracedShadowPass() = default;

    bool initialize(
        nvrhi::IDevice*                          device,
        donut::engine::ShaderFactory*             shaderFactory,
        uint32_t                                  width,
        uint32_t                                  height);

    /** Upload scene resources (instance/material/geometry metadata + combined VB/IB). */
    void setSceneResources(
        nvrhi::IDevice*                          device,
        const ShadowSceneResources&               resources);

    void renderShadow(
        nvrhi::ICommandList*                     commandList,
        nvrhi::rt::IAccelStruct*                 tlas,
        const ShadowConstants&                   constants,
        nvrhi::ITexture*                         depthTexture,
        nvrhi::ITexture*                         shadowOutput);

    void compositeShadow(
        nvrhi::ICommandList*                     commandList,
        nvrhi::ITexture*                         litColor,
        nvrhi::ITexture*                         shadowInput,
        nvrhi::ITexture*                         output,
        uint32_t                                  width,
        uint32_t                                  height);

    bool isValid() const { return m_rayQueryPipeline != nullptr; }

private:
    nvrhi::DeviceHandle                m_device;
    nvrhi::ShaderHandle                m_rayQueryShader;
    nvrhi::ShaderHandle                m_compositeShader;
    nvrhi::ComputePipelineHandle       m_rayQueryPipeline;
    nvrhi::ComputePipelineHandle       m_compositePipeline;
    nvrhi::BindingLayoutHandle         m_shadowBindingLayout;
    nvrhi::BindingSetHandle            m_shadowBindingSet;
    nvrhi::BindingLayoutHandle         m_compositeBindingLayout;
    nvrhi::BindingSetHandle            m_compositeBindingSet;
    nvrhi::BufferHandle                m_shadowConstantBuffer;
    nvrhi::SamplerHandle               m_sampler;

    // Scene metadata buffers (for alpha-tested shadows)
    nvrhi::BufferHandle                m_instanceMetaBuffer;
    nvrhi::BufferHandle                m_materialMetaBuffer;
    nvrhi::BufferHandle                m_geometryMetaBuffer;
    nvrhi::BufferHandle                m_combinedVertexBuffer;
    nvrhi::BufferHandle                m_combinedIndexBuffer;
    std::vector<nvrhi::TextureHandle>  m_alphaTextures;
    nvrhi::TextureHandle               m_fallbackTexture;
};

} // namespace rtxns::shadow
