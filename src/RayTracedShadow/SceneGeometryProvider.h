#pragma once

#include "ShadowTypes.h"
#include "RayTracedShadowPass.h"
#include <donut/engine/SceneGraph.h>
#include <donut/engine/SceneTypes.h>
#include <nvrhi/nvrhi.h>
#include <memory>
#include <vector>

namespace rtxns::shadow {

struct MeshBLASInput
{
    const donut::engine::BufferGroup* buffers = nullptr;
    nvrhi::BufferHandle               vertexBuffer;
    nvrhi::BufferHandle               indexBuffer;
    std::vector<GPUMeshDesc>          geometries;
    const donut::engine::MeshInfo*    meshInfo = nullptr;
};

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
};

} // namespace rtxns::shadow
