#pragma once

#include "ShadowTypes.h"
#include "SceneGeometryProvider.h"
#include <donut/engine/SceneGraph.h>
#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include <vector>

namespace rtxns::shadow {

struct BuiltBLAS
{
    nvrhi::rt::AccelStructHandle handle;
    nvrhi::HeapHandle           heap;
};

struct ShadowAccelStructures
{
    std::vector<BuiltBLAS>             blasList;
    nvrhi::rt::AccelStructHandle       tlas;
    nvrhi::HeapHandle                  tlasHeap;
    std::vector<nvrhi::rt::InstanceDesc> instances;  // cached for updates
    bool                               built = false;
};

class AccelerationStructure
{
public:
    /**
     * Build BLAS for all mesh inputs.  One BLAS per MeshBLASInput.
     * The command list is executed immediately (begin/end/execute/wait).
     */
    static std::vector<BuiltBLAS> buildBLASes(
        nvrhi::IDevice*                        device,
        const std::vector<MeshBLASInput>&       inputs);

    /**
     * Build (or rebuild) the TLAS from a list of instances.
     * For subsequent frames, pass built=true and rebuild==false to update in place.
     */
    static nvrhi::rt::AccelStructHandle buildTLAS(
        nvrhi::IDevice*                        device,
        nvrhi::ICommandList*                   commandList,
        const std::vector<nvrhi::rt::InstanceDesc>& instances,
        bool                                   isUpdate = false);

    /**
     * Build or update a TLAS, storing the heap and handle together.
     */
    static ShadowAccelStructures buildStructures(
        nvrhi::IDevice*                        device,
        const std::vector<MeshBLASInput>&       blasInputs,
        const std::vector<nvrhi::rt::InstanceDesc>& instances);

    /**
     * Update TLAS instances (for animated frames).
     */
    static void updateTLAS(
        nvrhi::ICommandList*                   commandList,
        const ShadowAccelStructures&           structures,
        const std::vector<nvrhi::rt::InstanceDesc>& instances);

    /**
     * Build instance descriptions from the Donut scene graph.
     * Each SceneGraphNode with a MeshInstance becomes one TLAS instance.
     */
    static std::vector<nvrhi::rt::InstanceDesc> buildInstanceDescs(
        const donut::engine::SceneGraph&        sceneGraph,
        const std::vector<BuiltBLAS>&           blasList,
        const std::vector<MeshBLASInput>&       blasInputs);

    static bool allocateAccelStructMemory(
        nvrhi::IDevice* device,
        nvrhi::rt::IAccelStruct* as,
        nvrhi::HeapHandle& heapOut);
};

} // namespace rtxns::shadow
