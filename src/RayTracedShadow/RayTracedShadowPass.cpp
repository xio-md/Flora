#include "RayTracedShadowPass.h"
#include <donut/core/math/math.h>

#if DONUT_WITH_VULKAN
#include "shaders/shadow_rayquery_cs.spirv.h"
#include "shaders/shadow_composite_cs.spirv.h"
#endif

namespace rtxns::shadow {

bool RayTracedShadowPass::initialize(
    nvrhi::IDevice* device,
    donut::engine::ShaderFactory* shaderFactory,
    uint32_t width,
    uint32_t height)
{
    m_device = device;

    // ---- Create shaders from compiled SPIR-V blobs ----
#if DONUT_WITH_VULKAN
    {
        nvrhi::ShaderDesc shaderDesc(nvrhi::ShaderType::Compute);
        shaderDesc.entryName = "main";
        m_rayQueryShader = device->createShader(shaderDesc,
            g_shadow_rayquery_cs_spirv,
            sizeof(g_shadow_rayquery_cs_spirv));
        if (!m_rayQueryShader)
            return false;

        m_compositeShader = device->createShader(shaderDesc,
            g_shadow_composite_cs_spirv,
            sizeof(g_shadow_composite_cs_spirv));
        if (!m_compositeShader)
            return false;
    }
#else
    return false;
#endif

    // ---- Create sampler ----
    {
        nvrhi::SamplerDesc samplerDesc;
        samplerDesc.setAllFilters(true);
        samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);
        m_sampler = device->createSampler(samplerDesc);
    }

    // ---- Shadow ray query binding layout ----
    // t0: TLAS, b0: ShadowConstants, t1: depth, u0: shadow output
    // t2: instance meta, t3: material meta, t4: geometry meta
    // t5: combined vertex buffer, t6: combined index buffer
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.registerSpace = 0;

        layoutDesc.addItem(nvrhi::BindingLayoutItem::RayTracingAccelStruct(0));  // t0
        layoutDesc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(1));             // t1
        layoutDesc.addItem(nvrhi::BindingLayoutItem::Texture_UAV(0));             // u0
        layoutDesc.addItem(nvrhi::BindingLayoutItem::Sampler(0));                 // s0
        layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2));    // t2
        layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3));    // t3
        layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4));    // t4
        layoutDesc.addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(5));           // t5
        layoutDesc.addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(6));           // t6

        layoutDesc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(7).setSize(kMaxAlphaTextures));
        layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(263));  // t263

        m_shadowBindingLayout = device->createBindingLayout(layoutDesc);

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.CS = m_rayQueryShader;
        pipelineDesc.bindingLayouts = { m_shadowBindingLayout };
        m_rayQueryPipeline = device->createComputePipeline(pipelineDesc);

        if (!m_rayQueryPipeline)
            return false;
    }

    // ---- Composite binding layout ----
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.registerSpace = 0;

        layoutDesc.addItem(
            nvrhi::BindingLayoutItem::Texture_SRV(0));                     // t0
        layoutDesc.addItem(
            nvrhi::BindingLayoutItem::Texture_SRV(1));                     // t1
        layoutDesc.addItem(
            nvrhi::BindingLayoutItem::Texture_UAV(2));                     // u2

        m_compositeBindingLayout = device->createBindingLayout(layoutDesc);

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.CS = m_compositeShader;
        pipelineDesc.bindingLayouts = { m_compositeBindingLayout };
        m_compositePipeline = device->createComputePipeline(pipelineDesc);

        if (!m_compositePipeline)
            return false;
    }

    // ---- Shadow constants buffer ----
    {
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = sizeof(ShadowConstants);
        bufferDesc.structStride = sizeof(ShadowConstants);
        bufferDesc.debugName = "ShadowConstants";
        bufferDesc.initialState = nvrhi::ResourceStates::Common;
        bufferDesc.keepInitialState = true;
        bufferDesc.cpuAccess = nvrhi::CpuAccessMode::None;
        m_shadowConstantBuffer = device->createBuffer(bufferDesc);
        if (!m_shadowConstantBuffer)
            return false;
    }

    return true;
}

void RayTracedShadowPass::setSceneResources(
    nvrhi::IDevice* device,
    const ShadowSceneResources& resources)
{
    // Create dummy 16-byte buffers for slots that must be bound
    auto createDummyBuf = [device](const char* name) {
        nvrhi::BufferDesc d;
        d.byteSize = 16; d.structStride = 16; d.debugName = name;
        d.canHaveRawViews = true;
        d.initialState = nvrhi::ResourceStates::ShaderResource;
        d.keepInitialState = true;
        return device->createBuffer(d);
    };

    m_instanceMetaBuffer = resources.instanceMetaBuffer
        ? resources.instanceMetaBuffer : createDummyBuf("ShadowDummyInstanceMeta");
    m_materialMetaBuffer = resources.materialMetaBuffer
        ? resources.materialMetaBuffer : createDummyBuf("ShadowDummyMaterialMeta");
    m_geometryMetaBuffer = resources.geometryMetaBuffer
        ? resources.geometryMetaBuffer : createDummyBuf("ShadowDummyGeometryMeta");
    m_combinedVertexBuffer = resources.combinedVertexBuffer
        ? resources.combinedVertexBuffer : createDummyBuf("ShadowDummyVB");
    m_combinedIndexBuffer = resources.combinedIndexBuffer
        ? resources.combinedIndexBuffer : createDummyBuf("ShadowDummyIB");
    m_alphaTextures = resources.alphaTextures;
}

void RayTracedShadowPass::renderShadow(
    nvrhi::ICommandList* commandList,
    nvrhi::rt::IAccelStruct* tlas,
    const ShadowConstants& constants,
    nvrhi::ITexture* depthTexture,
    nvrhi::ITexture* shadowOutput)
{
    if (!m_rayQueryPipeline || !tlas)
        return;

    commandList->beginTrackingBufferState(m_shadowConstantBuffer, nvrhi::ResourceStates::Common);

    commandList->writeBuffer(m_shadowConstantBuffer, &constants, sizeof(ShadowConstants));
    commandList->setBufferState(m_shadowConstantBuffer, nvrhi::ResourceStates::ShaderResource);

    nvrhi::BindingSetDesc setDesc;
    setDesc.addItem(nvrhi::BindingSetItem::RayTracingAccelStruct(0, tlas));
    setDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(1, depthTexture));
    setDesc.addItem(nvrhi::BindingSetItem::Texture_UAV(0, shadowOutput));
    setDesc.addItem(nvrhi::BindingSetItem::Sampler(0, m_sampler));

    setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_instanceMetaBuffer));
    setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_materialMetaBuffer));
    setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_geometryMetaBuffer));
    setDesc.addItem(nvrhi::BindingSetItem::RawBuffer_SRV(5, m_combinedVertexBuffer));
    setDesc.addItem(nvrhi::BindingSetItem::RawBuffer_SRV(6, m_combinedIndexBuffer));

    // Texture alpha slots: bind real alpha textures when available, fallback otherwise.
    if (!m_fallbackTexture)
    {
        nvrhi::TextureDesc texDesc;
        texDesc.width = 1; texDesc.height = 1;
        texDesc.format = nvrhi::Format::RGBA8_UNORM;
        texDesc.debugName = "ShadowFallbackTex";
        texDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        texDesc.keepInitialState = true;
        m_fallbackTexture = m_device->createTexture(texDesc);
    }
    for (uint32_t i = 0; i < kMaxAlphaTextures; ++i)
    {
        nvrhi::ITexture* texture = i < m_alphaTextures.size() && m_alphaTextures[i]
            ? m_alphaTextures[i].Get()
            : m_fallbackTexture.Get();
        setDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(7, texture).setArrayElement(i));
    }
    setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(263, m_shadowConstantBuffer));

    m_shadowBindingSet = m_device->createBindingSet(setDesc, m_shadowBindingLayout);
    if (!m_shadowBindingSet)
    {
        nvrhi::Color clearWhite(1.0f, 1.0f, 1.0f, 1.0f);
        commandList->clearTextureFloat(shadowOutput, nvrhi::AllSubresources, clearWhite);
        return;
    }

    commandList->setTextureState(depthTexture, nvrhi::AllSubresources,
        nvrhi::ResourceStates::ShaderResource);
    commandList->setTextureState(shadowOutput, nvrhi::AllSubresources,
        nvrhi::ResourceStates::UnorderedAccess);
    for (auto& texture : m_alphaTextures)
    {
        if (texture)
            commandList->setTextureState(texture, nvrhi::AllSubresources,
                nvrhi::ResourceStates::ShaderResource);
    }
    commandList->commitBarriers();

    nvrhi::ComputeState state;
    state.pipeline = m_rayQueryPipeline;
    state.bindings = { m_shadowBindingSet };
    commandList->setComputeState(state);

    uint32_t groupsX = (uint32_t(constants.imageSize.x) + 7) / 8;
    uint32_t groupsY = (uint32_t(constants.imageSize.y) + 7) / 8;
    commandList->dispatch(groupsX, groupsY, 1);
}

void RayTracedShadowPass::compositeShadow(
    nvrhi::ICommandList* commandList,
    nvrhi::ITexture* litColor,
    nvrhi::ITexture* shadowInput,
    nvrhi::ITexture* output,
    uint32_t width,
    uint32_t height)
{
    if (!m_compositePipeline)
        return;

    nvrhi::BindingSetDesc setDesc;
    setDesc.addItem(
        nvrhi::BindingSetItem::Texture_SRV(0, litColor));
    setDesc.addItem(
        nvrhi::BindingSetItem::Texture_SRV(1, shadowInput));
    setDesc.addItem(
        nvrhi::BindingSetItem::Texture_UAV(2, output));

    m_compositeBindingSet = m_device->createBindingSet(setDesc, m_compositeBindingLayout);
    if (!m_compositeBindingSet)
        return;

    commandList->setTextureState(litColor, nvrhi::AllSubresources,
        nvrhi::ResourceStates::ShaderResource);
    commandList->setTextureState(shadowInput, nvrhi::AllSubresources,
        nvrhi::ResourceStates::ShaderResource);
    commandList->setTextureState(output, nvrhi::AllSubresources,
        nvrhi::ResourceStates::UnorderedAccess);
    commandList->commitBarriers();

    nvrhi::ComputeState state;
    state.pipeline = m_compositePipeline;
    state.bindings = { m_compositeBindingSet };
    commandList->setComputeState(state);

    uint32_t groupsX = (width + 7) / 8;
    uint32_t groupsY = (height + 7) / 8;
    commandList->dispatch(groupsX, groupsY, 1);
}

} // namespace rtxns::shadow
