#include "AccelerationStructure.h"
#include "SceneGeometryProvider.h"
#include <donut/core/math/affine.h>
#include <donut/engine/SceneGraph.h>
#include <nvrhi/utils.h>
#include <unordered_map>

namespace rtxns::shadow {

bool AccelerationStructure::allocateAccelStructMemory(
    nvrhi::IDevice* device,
    nvrhi::rt::IAccelStruct* as,
    nvrhi::HeapHandle& heapOut)
{
    auto memReqs = device->getAccelStructMemoryRequirements(as);
    if (memReqs.size == 0)
        return true;

    nvrhi::HeapDesc heapDesc;
    heapDesc.capacity = memReqs.size;
    heapDesc.type = nvrhi::HeapType::DeviceLocal;
    heapDesc.debugName = "AccelStructHeap";

    heapOut = device->createHeap(heapDesc);
    if (!heapOut)
        return false;

    return device->bindAccelStructMemory(as, heapOut, 0);
}

std::vector<BuiltBLAS> AccelerationStructure::buildBLASes(
    nvrhi::IDevice* device,
    const std::vector<MeshBLASInput>& inputs)
{
    std::vector<BuiltBLAS> results(inputs.size());

    // Phase 1: create all BLASes first, then build them in a single command list
    struct PendingBLAS
    {
        uint32_t                    inputIndex;
        nvrhi::rt::AccelStructHandle handle;
        nvrhi::rt::AccelStructDesc   desc;
    };
    std::vector<PendingBLAS> pending;
    pending.reserve(inputs.size());

    std::vector<std::vector<nvrhi::rt::GeometryDesc>> allGeometryDescs;
    allGeometryDescs.reserve(inputs.size());

    for (uint32_t inputIndex = 0; inputIndex < inputs.size(); ++inputIndex)
    {
        const auto& input = inputs[inputIndex];
        std::vector<nvrhi::rt::GeometryDesc> geometryDescs;
        geometryDescs.reserve(input.geometries.size());

        for (const auto& geom : input.geometries)
        {
            nvrhi::rt::GeometryTriangles triangles;
            triangles.indexBuffer = input.indexBuffer;
            triangles.vertexBuffer = input.vertexBuffer;
            triangles.indexFormat = nvrhi::Format::R32_UINT;
            triangles.vertexFormat = nvrhi::Format::RGB32_FLOAT;
            triangles.indexOffset = geom.indexByteOffset;
            triangles.vertexOffset = geom.positionByteOffset;
            triangles.indexCount = geom.indexCount;
            triangles.vertexCount = geom.vertexCount;
            triangles.vertexStride = sizeof(dm::float3);

            nvrhi::rt::GeometryDesc geoDesc;
            geoDesc.geometryType = nvrhi::rt::GeometryType::Triangles;
            geoDesc.geometryData.triangles = triangles;
            if (!geom.isTransparent)
                geoDesc.setFlags(nvrhi::rt::GeometryFlags::Opaque);
            geometryDescs.push_back(geoDesc);
        }

        if (geometryDescs.empty())
        {
            allGeometryDescs.push_back({});
            continue;
        }

        nvrhi::rt::AccelStructDesc blasDesc;
        for (const auto& geo : geometryDescs)
            blasDesc.addBottomLevelGeometry(geo);
        blasDesc.setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
        blasDesc.setDebugName("BLAS");

        auto blas = device->createAccelStruct(blasDesc);
        if (!blas)
        {
            allGeometryDescs.push_back({});
            continue;
        }

        results[inputIndex] = { blas, nullptr };
        pending.push_back({ inputIndex, blas, blasDesc });
        allGeometryDescs.push_back(std::move(geometryDescs));
    }

    // Build all BLASes in a single command list
    if (!pending.empty())
    {
        auto cmdList = device->createCommandList();
        cmdList->open();
        for (size_t i = 0; i < pending.size(); ++i)
        {
            nvrhi::utils::BuildBottomLevelAccelStruct(cmdList, pending[i].handle, pending[i].desc);
        }
        cmdList->close();
        device->executeCommandList(cmdList);
        device->waitForIdle();
    }

    return results;
}

nvrhi::rt::AccelStructHandle AccelerationStructure::buildTLAS(
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    const std::vector<nvrhi::rt::InstanceDesc>& instances,
    bool isUpdate)
{
    nvrhi::rt::AccelStructDesc tlasDesc;
    tlasDesc.setTopLevelMaxInstances(instances.size());
    tlasDesc.setBuildFlags(
        nvrhi::rt::AccelStructBuildFlags::PreferFastTrace |
        nvrhi::rt::AccelStructBuildFlags::AllowUpdate);
    tlasDesc.setDebugName("TLAS");

    auto tlas = device->createAccelStruct(tlasDesc);
    if (!tlas)
        return nullptr;

    nvrhi::HeapHandle heap;
    if (!allocateAccelStructMemory(device, tlas, heap))
        return nullptr;

    auto flags = nvrhi::rt::AccelStructBuildFlags::PreferFastTrace;
    if (isUpdate)
        flags = flags | nvrhi::rt::AccelStructBuildFlags::PerformUpdate;

    commandList->buildTopLevelAccelStruct(tlas, instances.data(), instances.size(), flags);

    return tlas;
}

ShadowAccelStructures AccelerationStructure::buildStructures(
    nvrhi::IDevice* device,
    const std::vector<MeshBLASInput>& blasInputs,
    const std::vector<nvrhi::rt::InstanceDesc>& instances)
{
    ShadowAccelStructures result;
    result.blasList = buildBLASes(device, blasInputs);
    result.instances = instances;

    if (instances.empty() || result.blasList.empty())
        return result;

    // Build TLAS
    nvrhi::rt::AccelStructDesc tlasDesc;
    tlasDesc.setTopLevelMaxInstances(instances.size());
    tlasDesc.setBuildFlags(
        nvrhi::rt::AccelStructBuildFlags::PreferFastTrace |
        nvrhi::rt::AccelStructBuildFlags::AllowUpdate);
    tlasDesc.setDebugName("TLAS");

    result.tlas = device->createAccelStruct(tlasDesc);
    if (!result.tlas)
        return result;

    if (!allocateAccelStructMemory(device, result.tlas, result.tlasHeap))
    {
        result.tlas = nullptr;
        return result;
    }

    auto cmdList = device->createCommandList();
    cmdList->open();
    cmdList->buildTopLevelAccelStruct(
        result.tlas,
        instances.data(),
        instances.size(),
        nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
    cmdList->close();
    device->executeCommandList(cmdList);
    device->waitForIdle();

    result.built = true;
    return result;
}

void AccelerationStructure::updateTLAS(
    nvrhi::ICommandList* commandList,
    const ShadowAccelStructures& structures,
    const std::vector<nvrhi::rt::InstanceDesc>& instances)
{
    if (!structures.tlas || instances.empty())
        return;

    commandList->buildTopLevelAccelStruct(
        structures.tlas,
        instances.data(),
        instances.size(),
        nvrhi::rt::AccelStructBuildFlags::PreferFastTrace |
        nvrhi::rt::AccelStructBuildFlags::PerformUpdate);
}

std::vector<nvrhi::rt::InstanceDesc> AccelerationStructure::buildInstanceDescs(
    const donut::engine::SceneGraph& sceneGraph,
    const std::vector<BuiltBLAS>& blasList,
    const std::vector<MeshBLASInput>& blasInputs)
{
    std::vector<nvrhi::rt::InstanceDesc> instances;

    // Build mapping from MeshInfo* -> BLAS index
    std::unordered_map<const donut::engine::MeshInfo*, uint32_t> meshToBLAS;
    for (uint32_t i = 0; i < blasInputs.size(); ++i)
    {
        if (i < blasList.size() && blasList[i].handle && blasInputs[i].meshInfo)
        {
            meshToBLAS[blasInputs[i].meshInfo] = i;
        }
    }

    if (meshToBLAS.empty())
        return instances;

    // Walk scene graph
    donut::engine::SceneGraphWalker walker(
        const_cast<donut::engine::SceneGraphNode*>(sceneGraph.GetRootNode().get()));

    while (walker)
    {
        auto* node = walker.Get();
        if (!node)
        {
            walker.Next(false);
            continue;
        }

        auto leaf = node->GetLeaf();
        if (leaf)
        {
            auto meshInstance = std::dynamic_pointer_cast<donut::engine::MeshInstance>(leaf);
            if (meshInstance)
            {
                auto mesh = meshInstance->GetMesh();
                if (mesh && mesh->buffers && !mesh->IsCurve())
                {
                    auto it = meshToBLAS.find(mesh.get());
                    if (it != meshToBLAS.end())
                    {
                        uint32_t bi = it->second;
                        if (bi < blasList.size() && blasList[bi].handle)
                        {
                            const dm::affine3& a = node->GetLocalToWorldTransformFloat();

                            nvrhi::rt::InstanceDesc instance;
                            dm::affineToColumnMajor(a, instance.transform);
                            instance.setBLAS(blasList[bi].handle);
                            instance.setInstanceID(static_cast<uint32_t>(instances.size()));

                            instance.setInstanceContributionToHitGroupIndex(0);
                            bool hasTransparentGeometry = false;
                            bool hasAlphaTestedGeometry = false;
                            for (const auto& geom : blasInputs[bi].geometries)
                            {
                                hasTransparentGeometry |= geom.isTransparent;
                                hasAlphaTestedGeometry |= geom.isAlphaTested;
                            }
                            instance.setInstanceMask(hasAlphaTestedGeometry ? 0xFE : 0x01);
                            if (!hasTransparentGeometry)
                                instance.setFlags(nvrhi::rt::InstanceFlags::ForceOpaque);

                            instances.push_back(instance);
                        }
                    }
                }
            }
        }

        walker.Next(true);
    }

    return instances;
}

} // namespace rtxns::shadow
