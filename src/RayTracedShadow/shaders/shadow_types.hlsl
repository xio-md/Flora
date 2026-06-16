#pragma once

static const uint kMaxAlphaTextures = 256;

struct ShadowConstants
{
    float3  sunDirection;
    float   sunJitter;
    row_major float4x4 invViewProj;
    row_major float4x4 invProj;
    row_major float4x4 invView;
    float2  projParams;
    float2  imageSize;
    uint    shadowEnabled;
    uint    shadowRayMask;
};

struct ShadowInstanceMeta
{
    uint firstGeometryIndex;
    uint geometryCount;
    uint hasAlphaTested;
    uint pad;
};

struct ShadowMaterialMeta
{
    float alphaCutoff;
    uint  albedoTextureIndex;
    uint  domain;
    uint  flags;  // 1 = has alpha texture
};

struct ShadowGeometryMeta
{
    uint indexByteOffset;
    uint vertexByteOffset;
    uint texCoordByteOffset;
    uint materialIndex;
};
