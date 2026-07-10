#pragma once

#include "ShadowTypes.h"
#include "RayTracedShadowPass.h"
#include <donut/engine/SceneGraph.h>
#include <donut/engine/SceneTypes.h>
#include <nvrhi/nvrhi.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace rtxns::shadow {

struct MeshBLASInput
{
    const donut::engine::BufferGroup* buffers = nullptr;
    nvrhi::BufferHandle               vertexBuffer;
    nvrhi::BufferHandle               indexBuffer;
    std::vector<GPUMeshDesc>          geometries;
    const donut::engine::MeshInfo*    meshInfo = nullptr;

    // CPU-side geometry data snapshot (for OMM baking, saved before FinishedLoading clears them)
    std::vector<uint32_t>   cpuIndexData;
    std::vector<dm::float2> cpuTexcoordData;

    // OMM data (filled in after baking, before BLAS build)
    bool                               hasAlphaTestedGeometry = false;
    bool                               forceNonOpaque = false; // OMM stress test: disable Opaque flag
    nvrhi::rt::OpacityMicromapHandle   opacityMicromap;       // pre-built OMM array
    nvrhi::BufferHandle                ommIndexBuffer;        // triangle→OMM index buffer
    uint64_t                           ommIndexBufferOffset = 0;
    std::vector<nvrhi::rt::OpacityMicromapUsageCount> ommUsageCounts; // BLAS OMM usage counts
};

// CPU data cache for alpha-tested meshes, captured BEFORE Scene::FinishedLoading()
// releases BufferGroup::indexData / texcoord1Data. Keyed by MeshInfo pointer so
// the first-frame OMM baking loop can look up per-mesh geometry + material data
// even though the original CPU buffers have been freed by then.
struct OMMMeshCpuCacheEntry
{
    const donut::engine::MeshInfo* meshInfo = nullptr;

    // Per-mesh CPU geometry (copied element-by-element from BufferGroup before release)
    std::vector<uint32_t>   indexData;
    std::vector<dm::float2> texcoordData;

    // Material / texture info for OMM baking (first alpha-tested geometry in mesh)
    float    alphaCutoff = 0.5f;
    bool     hasAlphaTexture = false;
    std::shared_ptr<donut::engine::LoadedTexture> alphaTexture; // keeps GPU texture alive

    // Pre-readback alpha pixel data (populated in load_scene before FinishedLoading)
    std::vector<float> alphaPixels;
    uint32_t texWidth = 0;
    uint32_t texHeight = 0;
    bool     alphaReadBack = false; // true after alphaPixels populated
};

using OMMCpuCache = std::unordered_map<const donut::engine::MeshInfo*, OMMMeshCpuCacheEntry>;

class SceneGeometryProvider
{
public:
    static std::vector<MeshBLASInput> extractFromScene(const donut::engine::SceneGraph& sceneGraph);

    static bool hasGeometry(const donut::engine::SceneGraph& sceneGraph);

    /**
     * Build combined vertex/index buffers and per-instance metadata for alpha-tested shadow tracing.
     * Must be called AFTER Scene::Load() but BEFORE Scene::FinishedLoading() (CPU data still available).
     */
    static ShadowSceneResources buildShadowSceneResources(
        nvrhi::IDevice* device,
        const donut::engine::SceneGraph& sceneGraph);

    /**
     * Cache CPU-side index/texcoord data + material info for every alpha-tested mesh.
     * MUST be called AFTER Scene::Load() but BEFORE Scene::FinishedLoading(),
     * otherwise BufferGroup::indexData / texcoord1Data are already freed.
     * The returned cache is keyed by MeshInfo* and consumed by the first-frame
     * OMM baking loop (which runs after FinishedLoading).
     */
    static OMMCpuCache cacheAlphaTestedMeshData(
        const donut::engine::SceneGraph& sceneGraph);
};

} // namespace rtxns::shadow
