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

// ---------------------------------------------------------------------------
// Multi-sample sun jitter: each pixel shoots N rays with the sun direction
// perturbed by a small angular offset in the tangent plane. This produces a
// distance-dependent (contact-hardening) penumbra that matches the physical
// behaviour of an extended light source.
// ---------------------------------------------------------------------------
// SHADOW_SAMPLES is now runtime: c_shadow.shadowSamples (default 4)

// PCG-style hash: returns pseudo-random float in [0, 1)
float hash(uint n)
{
    n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return float(n & 0x7FFFFFFFu) / float(0x7FFFFFFFu);
}

float2 hash2(uint x, uint y)
{
    uint k = x * 1664525U + y * 1013904223U;
    return float2(hash(k), hash(k ^ 0xDEADBEEFu));
}

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

// Trace a single shadow ray with the given direction; returns 1.0f = lit, 0.0f = shadowed.
float TraceShadowRay(float3 origin, float3 direction)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.01f;
    ray.TMax = 1000.0f;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, c_shadow.shadowRayMask, ray);

    while (query.Proceed())
    {
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

    return (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0f : 1.0f;
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
    float3 sunDir = c_shadow.sunDirection;
    float jitter = c_shadow.sunJitter;

    // Fast path: no jitter → single ray (backward-compatible)
    if (jitter <= 0.0f)
    {
        u_shadow[idx.xy] = TraceShadowRay(wpos + sunDir * 0.05f, sunDir);
        return;
    }

    // Build orthonormal basis from sun direction for jitter in tangent plane.
    float3 tangent, bitangent;
    if (abs(sunDir.y) > 0.999f)
    {
        tangent = float3(1.0f, 0.0f, 0.0f);
        bitangent = float3(0.0f, 0.0f, -1.0f);
    }
    else
    {
        tangent = normalize(cross(float3(0.0f, 1.0f, 0.0f), sunDir));
        bitangent = cross(sunDir, tangent);
    }

    uint samples = max(1u, c_shadow.shadowSamples);
    float shadowAccum = 0.0f;
    for (uint s = 0; s < samples; ++s)
    {
        // Per-sample pseudo-random: pixel index seeded by sample index
        float2 rnd = hash2(idx.x + s * 137u, idx.y + s * 251u);

        // Uniform disk sampling in tangent plane → angular perturbation
        float theta = 6.2831853f * rnd.x; // 2*PI
        float r = sqrt(rnd.y) * jitter;
        float3 perturb = tangent * (r * cos(theta)) + bitangent * (r * sin(theta));
        float3 rayDir = normalize(sunDir + perturb);

        float3 origin = wpos + rayDir * 0.05f;
        shadowAccum += TraceShadowRay(origin, rayDir);
    }

    u_shadow[idx.xy] = shadowAccum / float(samples);
}
