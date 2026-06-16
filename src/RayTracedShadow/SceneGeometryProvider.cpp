#include "SceneGeometryProvider.h"
#include "RayTracedShadowPass.h"
#include <donut/engine/SceneGraph.h>
#include <donut/engine/SceneTypes.h>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <utility>

namespace rtxns::shadow {

static constexpr uint32_t c_SizeOfInterleavedVertex = sizeof(dm::float3) + sizeof(dm::float2); // 20 bytes

bool SceneGeometryProvider::hasGeometry(const donut::engine::SceneGraph& sceneGraph)
{
    for (const auto& mesh : sceneGraph.GetMeshes())
    {
        if (mesh->buffers && !mesh->IsCurve())
        {
            for (const auto& geom : mesh->geometries)
            {
                if (geom->type == donut::engine::MeshGeometryPrimitiveType::Triangles && geom->numIndices > 0)
                    return true;
            }
        }
    }
    return false;
}

std::vector<MeshBLASInput> SceneGeometryProvider::extractFromScene(const donut::engine::SceneGraph& sceneGraph)
{
    std::vector<MeshBLASInput> result;

    for (const auto& mesh : sceneGraph.GetMeshes())
    {
        if (!mesh->buffers || mesh->IsCurve())
            continue;

        auto* buffers = mesh->buffers.get();
        if (!buffers->vertexBuffer || !buffers->indexBuffer)
            continue;

        const auto& positionRange = buffers->getVertexBufferRange(
            donut::engine::VertexAttribute::Position);
        if (positionRange.byteSize == 0)
            continue;

        MeshBLASInput input;
        input.buffers = buffers;
        input.vertexBuffer = buffers->vertexBuffer;
        input.indexBuffer = buffers->indexBuffer;
        input.meshInfo = mesh.get();

        for (const auto& geom : mesh->geometries)
        {
            if (geom->type != donut::engine::MeshGeometryPrimitiveType::Triangles || geom->numIndices == 0)
                continue;

            GPUMeshDesc desc;
            desc.positionByteOffset = static_cast<uint32_t>(
                positionRange.byteOffset + (mesh->vertexOffset + geom->vertexOffsetInMesh) * sizeof(dm::float3));
            desc.indexByteOffset = (mesh->indexOffset + geom->indexOffsetInMesh) * sizeof(uint32_t);
            desc.vertexCount = geom->numVertices;
            desc.indexCount = geom->numIndices;
            desc.materialIndex = 0;
            desc.isTransparent = false;
            desc.isAlphaTested = false;

            if (geom->material)
            {
                desc.isAlphaTested =
                    geom->material->domain == donut::engine::MaterialDomain::AlphaTested ||
                    geom->material->domain == donut::engine::MaterialDomain::TransmissiveAlphaTested;
                desc.isTransparent =
                    geom->material->domain == donut::engine::MaterialDomain::AlphaBlended ||
                    desc.isAlphaTested ||
                    geom->material->domain == donut::engine::MaterialDomain::Transmissive ||
                    geom->material->domain == donut::engine::MaterialDomain::TransmissiveAlphaBlended;
            }

            input.geometries.push_back(desc);
        }

        if (!input.geometries.empty())
            result.push_back(std::move(input));
    }

    return result;
}

ShadowSceneResources SceneGeometryProvider::buildShadowSceneResources(
    nvrhi::IDevice* device,
    const donut::engine::SceneGraph& sceneGraph)
{
    ShadowSceneResources resources;

    // Collect CPU-side geometry data before FinishedLoading frees it
    std::vector<dm::float3> allPositions;
    std::vector<dm::float2> allTexcoords;
    std::vector<uint32_t>   allIndices;

    std::unordered_map<const donut::engine::Material*, uint32_t> materialToIndex;
    struct GeomInfo {
        uint32_t indexByteOffset;
        uint32_t vertexByteOffset;
        uint32_t texCoordByteOffset;
        uint32_t materialIndex;
        bool     isAlphaTested;
    };
    std::vector<GeomInfo> geoms;
    std::unordered_map<const donut::engine::MeshInfo*, std::pair<uint32_t, uint32_t>> meshGeometryRanges;

    auto addAlphaTexture = [&resources](const std::shared_ptr<donut::engine::LoadedTexture>& texture) -> uint32_t {
        if (!texture || !texture->texture)
            return kMaxAlphaTextures;

        for (uint32_t i = 0; i < resources.alphaTextures.size(); ++i)
        {
            if (resources.alphaTextures[i] == texture->texture)
                return i;
        }

        if (resources.alphaTextures.size() >= kMaxAlphaTextures)
            return kMaxAlphaTextures;

        uint32_t index = static_cast<uint32_t>(resources.alphaTextures.size());
        resources.alphaTextures.push_back(texture->texture);
        return index;
    };

    for (const auto& mesh : sceneGraph.GetMeshes())
    {
        if (!mesh->buffers || mesh->IsCurve())
        {
            continue;
        }

        auto& buffers = *mesh->buffers;
        uint32_t meshFirstGeometry = static_cast<uint32_t>(geoms.size());
        uint32_t meshGeometryCount = 0;

        for (const auto& geom : mesh->geometries)
        {
            if (geom->type != donut::engine::MeshGeometryPrimitiveType::Triangles || geom->numIndices == 0)
                continue;

            // Material index mapping
            uint32_t matIdx = 0;
            if (geom->material)
            {
                auto it = materialToIndex.find(geom->material.get());
                if (it == materialToIndex.end())
                {
                    matIdx = static_cast<uint32_t>(resources.materialMetas.size());
                    materialToIndex[geom->material.get()] = matIdx;

                    ShadowMaterialMeta matMeta;
                    matMeta.alphaCutoff = geom->material->alphaCutoff;
                    auto alphaTexture = geom->material->opacityTexture
                        ? geom->material->opacityTexture
                        : geom->material->baseOrDiffuseTexture;
                    matMeta.albedoTextureIndex = addAlphaTexture(alphaTexture);
                    matMeta.domain = static_cast<uint32_t>(geom->material->domain);
                    matMeta.flags = matMeta.albedoTextureIndex < kMaxAlphaTextures ? 1u : 0u;
                    resources.materialMetas.push_back(matMeta);
                }
                else
                {
                    matIdx = it->second;
                }
            }

            // Collect positions + texcoords interleaved first (determines vertex base offset)
            uint32_t geomVertStart = static_cast<uint32_t>(allPositions.size());
            uint32_t meshVertStart = mesh->vertexOffset + geom->vertexOffsetInMesh;
            for (uint32_t v = 0; v < geom->numVertices; ++v)
            {
                allPositions.push_back(buffers.positionData[meshVertStart + v]);
                allTexcoords.push_back(buffers.texcoord1Data.empty()
                    ? dm::float2(0.f, 0.f)
                    : buffers.texcoord1Data[meshVertStart + v]);
            }

            // Collect indices (rebase to combined vertex buffer)
            uint32_t geomIndexStart = static_cast<uint32_t>(allIndices.size());
            uint32_t meshIdxStart = mesh->indexOffset + geom->indexOffsetInMesh;
            for (uint32_t i = 0; i < geom->numIndices; ++i)
                allIndices.push_back(buffers.indexData[meshIdxStart + i] + geomVertStart);

            GeomInfo gi;
            gi.indexByteOffset = geomIndexStart * sizeof(uint32_t);
            gi.vertexByteOffset = geomVertStart * c_SizeOfInterleavedVertex;
            gi.texCoordByteOffset = geomVertStart * c_SizeOfInterleavedVertex + sizeof(dm::float3);
            gi.materialIndex = matIdx;
            gi.isAlphaTested = geom->material &&
                (geom->material->domain == donut::engine::MaterialDomain::AlphaTested ||
                 geom->material->domain == donut::engine::MaterialDomain::TransmissiveAlphaTested);
            geoms.push_back(gi);
            meshGeometryCount++;
        }

        if (meshGeometryCount > 0)
            meshGeometryRanges[mesh.get()] = { meshFirstGeometry, meshGeometryCount };
    }

    if (allPositions.empty())
        return resources;

    // Upload combined resources to GPU
    {
        // Interleave positions + texcoords: [pos(12) | tex(8)] per vertex
        std::vector<uint8_t> combinedVertices(allPositions.size() * c_SizeOfInterleavedVertex);
        for (size_t i = 0; i < allPositions.size(); ++i)
        {
            uint8_t* dst = combinedVertices.data() + i * c_SizeOfInterleavedVertex;
            memcpy(dst, &allPositions[i], sizeof(dm::float3));
            memcpy(dst + sizeof(dm::float3), &allTexcoords[i], sizeof(dm::float2));
        }

        nvrhi::BufferDesc vbDesc;
        vbDesc.byteSize = combinedVertices.size();
        vbDesc.isVertexBuffer = true;
        vbDesc.debugName = "ShadowCombinedVB";
        vbDesc.canHaveRawViews = true;
        resources.combinedVertexBuffer = device->createBuffer(vbDesc);

        nvrhi::BufferDesc ibDesc;
        ibDesc.byteSize = allIndices.size() * sizeof(uint32_t);
        ibDesc.isIndexBuffer = true;
        ibDesc.debugName = "ShadowCombinedIB";
        ibDesc.canHaveRawViews = true;
        ibDesc.format = nvrhi::Format::R32_UINT;
        resources.combinedIndexBuffer = device->createBuffer(ibDesc);

        auto cmdList = device->createCommandList();
        cmdList->open();
        cmdList->beginTrackingBufferState(resources.combinedVertexBuffer, nvrhi::ResourceStates::Common);
        cmdList->writeBuffer(resources.combinedVertexBuffer, combinedVertices.data(), combinedVertices.size());
        cmdList->setPermanentBufferState(resources.combinedVertexBuffer, nvrhi::ResourceStates::ShaderResource);
        cmdList->beginTrackingBufferState(resources.combinedIndexBuffer, nvrhi::ResourceStates::Common);
        cmdList->writeBuffer(resources.combinedIndexBuffer, allIndices.data(), allIndices.size() * sizeof(uint32_t));
        cmdList->setPermanentBufferState(resources.combinedIndexBuffer, nvrhi::ResourceStates::ShaderResource);
        cmdList->close();
        device->executeCommandList(cmdList);
        device->waitForIdle();
    }

    // Build per-geometry metadata
    for (const auto& gi : geoms)
    {
        ShadowGeometryMeta gm;
        gm.indexByteOffset = gi.indexByteOffset;
        gm.vertexByteOffset = gi.vertexByteOffset;
        gm.texCoordByteOffset = gi.texCoordByteOffset;
        gm.materialIndex = gi.materialIndex;
        resources.geometryMetas.push_back(gm);
    }

    // Build per-instance metadata
    // Walk scene graph in the same order used by TLAS instance construction.
    donut::engine::SceneGraphWalker walker(
        sceneGraph.GetRootNode().get());

    while (walker)
    {
        auto* node = walker.Get();
        if (!node) { walker.Next(false); continue; }

        auto leaf = node->GetLeaf();
        if (leaf)
        {
            auto mi = std::dynamic_pointer_cast<donut::engine::MeshInstance>(leaf);
            if (mi)
            {
                auto mesh = mi->GetMesh();
                if (mesh && mesh->buffers && !mesh->IsCurve())
                {
                    auto rangeIt = meshGeometryRanges.find(mesh.get());
                    if (rangeIt != meshGeometryRanges.end())
                    {
                        uint32_t firstGeometry = rangeIt->second.first;
                        uint32_t geometryCount = rangeIt->second.second;
                        bool hasAlphaTested = false;
                        for (uint32_t i = 0; i < geometryCount; ++i)
                            hasAlphaTested |= geoms[firstGeometry + i].isAlphaTested;

                        ShadowInstanceMeta im;
                        im.firstGeometryIndex = firstGeometry;
                        im.geometryCount = geometryCount;
                        im.hasAlphaTested = hasAlphaTested ? 1u : 0u;
                        im.pad = 0;
                        resources.instanceMetas.push_back(im);
                    }
                }
            }
        }
        walker.Next(true);
    }

    // Upload metadata buffers
    resources.instanceCount = static_cast<uint32_t>(resources.instanceMetas.size());
    resources.materialCount = static_cast<uint32_t>(resources.materialMetas.size());
    resources.geometryCount = static_cast<uint32_t>(resources.geometryMetas.size());

    {
        auto cmdList = device->createCommandList();
        cmdList->open();

        // Instance meta buffer
        if (!resources.instanceMetas.empty())
        {
            nvrhi::BufferDesc bufDesc;
            bufDesc.byteSize = resources.instanceMetas.size() * sizeof(ShadowInstanceMeta);
            bufDesc.structStride = sizeof(ShadowInstanceMeta);
            bufDesc.debugName = "ShadowInstanceMeta";
            bufDesc.canHaveRawViews = true;
            resources.instanceMetaBuffer = device->createBuffer(bufDesc);
            cmdList->beginTrackingBufferState(resources.instanceMetaBuffer, nvrhi::ResourceStates::Common);
            cmdList->writeBuffer(resources.instanceMetaBuffer, resources.instanceMetas.data(), bufDesc.byteSize);
            cmdList->setPermanentBufferState(resources.instanceMetaBuffer, nvrhi::ResourceStates::ShaderResource);
        }

        // Material meta buffer
        if (!resources.materialMetas.empty())
        {
            nvrhi::BufferDesc bufDesc;
            bufDesc.byteSize = resources.materialMetas.size() * sizeof(ShadowMaterialMeta);
            bufDesc.structStride = sizeof(ShadowMaterialMeta);
            bufDesc.debugName = "ShadowMaterialMeta";
            bufDesc.canHaveRawViews = true;
            resources.materialMetaBuffer = device->createBuffer(bufDesc);
            cmdList->beginTrackingBufferState(resources.materialMetaBuffer, nvrhi::ResourceStates::Common);
            cmdList->writeBuffer(resources.materialMetaBuffer, resources.materialMetas.data(), bufDesc.byteSize);
            cmdList->setPermanentBufferState(resources.materialMetaBuffer, nvrhi::ResourceStates::ShaderResource);
        }

        // Geometry meta buffer
        if (!resources.geometryMetas.empty())
        {
            nvrhi::BufferDesc bufDesc;
            bufDesc.byteSize = resources.geometryMetas.size() * sizeof(ShadowGeometryMeta);
            bufDesc.structStride = sizeof(ShadowGeometryMeta);
            bufDesc.debugName = "ShadowGeometryMeta";
            bufDesc.canHaveRawViews = true;
            resources.geometryMetaBuffer = device->createBuffer(bufDesc);
            cmdList->beginTrackingBufferState(resources.geometryMetaBuffer, nvrhi::ResourceStates::Common);
            cmdList->writeBuffer(resources.geometryMetaBuffer, resources.geometryMetas.data(), bufDesc.byteSize);
            cmdList->setPermanentBufferState(resources.geometryMetaBuffer, nvrhi::ResourceStates::ShaderResource);
        }

        cmdList->close();
        device->executeCommandList(cmdList);
        device->waitForIdle();
    }

    return resources;
}

} // namespace rtxns::shadow
