#pragma once

#include <cstdint>
#include <donut/core/math/math.h>

namespace rtxns::shadow {

struct GPUMeshDesc
{
    uint32_t positionByteOffset;  // absolute byte offset into vertex buffer
    uint32_t indexByteOffset;     // absolute byte offset into index buffer
    uint32_t vertexCount;         // number of vertices
    uint32_t indexCount;          // number of index elements
    uint32_t materialIndex;       // for future alpha test / OMM
    bool     isTransparent;       // for future OMM / transparent shadows
    bool     isAlphaTested;       // true when geometry needs texture alpha testing
};

struct alignas(16) ShadowConstants
{
    dm::float3   sunDirection;
    float        sunJitter;
    dm::float4x4 invViewProj;
    dm::float4x4 invProj;
    dm::float4x4 invView;
    dm::float2   projParams;   // x = near, y = far
    dm::float2   imageSize;
    uint32_t     shadowEnabled;
    uint32_t     shadowRayMask;
};

// GPU-side per-instance metadata for alpha-tested shadow tracing
struct alignas(16) ShadowInstanceMeta
{
    uint32_t firstGeometryIndex; // first geometry for this TLAS instance
    uint32_t geometryCount;      // number of BLAS geometries for this instance
    uint32_t hasAlphaTested;     // 1 = any geometry needs alpha test
    uint32_t pad;
};

// GPU-side per-material metadata
struct alignas(16) ShadowMaterialMeta
{
    float    alphaCutoff;
    uint32_t albedoTextureIndex;  // index into texture descriptor array
    uint32_t domain;              // MaterialDomain enum
    uint32_t flags;               // 1 = has alpha texture
};

static constexpr uint32_t kMaxAlphaTextures = 256;

// GPU-side per-geometry metadata for UV reconstruction
struct alignas(16) ShadowGeometryMeta
{
    uint32_t indexByteOffset;     // byte offset into index buffer for this geometry's indices
    uint32_t vertexByteOffset;    // byte offset into vertex buffer for this geometry's vertices
    uint32_t texCoordByteOffset;  // byte offset into vertex buffer for UVs
    uint32_t materialIndex;       // index into ShadowMaterialMeta array
};

} // namespace rtxns::shadow
