#pragma pack_matrix(row_major)

#include "shadow_types.hlsl"

RaytracingAccelerationStructure    Scene : register(t0);
Texture2D<float>                   t_depth : register(t1);
RWTexture2D<float>                 u_shadow : register(u0);
SamplerState                       s_sampler : register(s0);

StructuredBuffer<ShadowInstanceMeta>  t_instanceMeta : register(t2);
StructuredBuffer<ShadowMaterialMeta>  t_materialMeta : register(t3);
StructuredBuffer<ShadowGeometryMeta>  t_geometryMeta : register(t4);
ByteAddressBuffer                     t_vertexData : register(t5);
ByteAddressBuffer                     t_indexData : register(t6);
Texture2D t_alphaTextures[kMaxAlphaTextures] : register(t7, space0);
StructuredBuffer<ShadowConstants>     t_shadowConstants : register(t263);

#define c_shadow t_shadowConstants[0]

static const uint c_SizeOfInterleavedVertex = 20; // float3(12) + float2(8) per vertex

float3 ReconstructWorldPosition(uint2 pixel, float depth)
{
    float2 pixelPosition = float2(pixel) + 0.5f;
    float2 uv = pixelPosition / c_shadow.imageSize;
    float4 clipPos = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f);
    float4 worldH = mul(clipPos, c_shadow.invViewProj);
    return worldH.xyz / worldH.w;
}

float2 GetTriangleUV(uint3 triIndices, float2 bary)
{
    float2 uv0 = asfloat(t_vertexData.Load2(triIndices.x * c_SizeOfInterleavedVertex + 12));
    float2 uv1 = asfloat(t_vertexData.Load2(triIndices.y * c_SizeOfInterleavedVertex + 12));
    float2 uv2 = asfloat(t_vertexData.Load2(triIndices.z * c_SizeOfInterleavedVertex + 12));
    return uv0 * (1.0f - bary.x - bary.y) + uv1 * bary.x + uv2 * bary.y;
}

bool AlphaTestCandidate(uint instanceID, uint geometryIndex, uint primitiveIndex, float2 barycentrics)
{
    if (instanceID >= kMaxAlphaTextures * 4096)
        return true; // accept unknown

    ShadowInstanceMeta im = t_instanceMeta[instanceID];
    if (im.hasAlphaTested == 0 || geometryIndex >= im.geometryCount)
        return true;

    ShadowGeometryMeta geom = t_geometryMeta[im.firstGeometryIndex + geometryIndex];
    ShadowMaterialMeta mat = t_materialMeta[geom.materialIndex];

    uint indexOffs = geom.indexByteOffset / 4 + primitiveIndex * 3;
    uint i0 = t_indexData.Load(indexOffs * 4);
    uint i1 = t_indexData.Load((indexOffs + 1) * 4);
    uint i2 = t_indexData.Load((indexOffs + 2) * 4);

    float2 uv = GetTriangleUV(uint3(i0, i1, i2), barycentrics);

    if ((mat.flags & 1u) == 0)
        return true;

    uint texIdx = mat.albedoTextureIndex;
    if (texIdx >= kMaxAlphaTextures)
        return true;

    float alpha = t_alphaTextures[NonUniformResourceIndex(texIdx)].SampleLevel(s_sampler, uv, 0).a;
    return alpha >= mat.alphaCutoff;
}

[numthreads(8, 8, 1)]
void main(uint3 idx : SV_DispatchThreadID)
{
    if (c_shadow.shadowEnabled == 0 ||
        idx.x >= (uint)c_shadow.imageSize.x ||
        idx.y >= (uint)c_shadow.imageSize.y)
        return;

    float depth = t_depth[idx.xy];
    if (depth >= 1.0f) { u_shadow[idx.xy] = 1.0f; return; }

    float3 wpos = ReconstructWorldPosition(idx.xy, depth);
    RayDesc ray;
    ray.Origin = wpos + c_shadow.sunDirection * 0.05f;
    ray.Direction = c_shadow.sunDirection;
    ray.TMin = 0.01f; ray.TMax = 1000.0f;

    // Opaque geometry auto-commits via GeometryFlags::Opaque in BLAS
    // Non-opaque (alpha-tested) geometry appears as candidates
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, c_shadow.shadowRayMask, ray);

    float shadow = 1.0f;
    uint iterCount = 0;
    const uint kMaxIter = 8;

    while (iterCount < kMaxIter && query.Proceed())
    {
        iterCount++;
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            uint instanceID = query.CandidateInstanceID();
            uint geometryIndex = query.CandidateGeometryIndex();
            uint primitiveIndex = query.CandidatePrimitiveIndex();
            float2 barycentrics = query.CandidateTriangleBarycentrics();

            if (AlphaTestCandidate(instanceID, geometryIndex, primitiveIndex, barycentrics))
            {
                query.CommitNonOpaqueTriangleHit();
            }
        }
    }

    // Check committed hit (opaque auto-commit or confirmed non-opaque)
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        shadow = 0.0f;

    u_shadow[idx.xy] = shadow;
}
