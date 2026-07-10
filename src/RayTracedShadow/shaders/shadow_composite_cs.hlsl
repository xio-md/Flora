Texture2D<float4>  t_litColor : register(t0);
Texture2D<float>   t_shadow : register(t1);
RWTexture2D<float4> u_output : register(u2);

// Hejl-Burgess-Dawson filmic tonemap (same as Niagara)
float3 Tonemap(float3 c)
{
    float3 x = max(float3(0, 0, 0), c - 0.004f);
    return (x * (6.2f * x + 0.5f)) / (x * (6.2f * x + 1.7f) + 0.06f);
}

[numthreads(8, 8, 1)]
void main(uint3 pos : SV_DispatchThreadID)
{
    float4 color = t_litColor[pos.xy];
    float shadow = t_shadow[pos.xy];

    // Shadow ambient floor: lets ~5% sky/ambient through shadow regions
    // so that surface texture (cobblestone, etc.) remains visible.
    // GT shows ~40-50% brightness reduction, not total blackout.
    float shadowAmbient = 0.05f;
    float s = min(shadow + shadowAmbient, 1.0f);

    // Fixed exposure to bring HDR forward-pass output into tonemap range.
    // The forward pass with irradiance=2.5 produces values up to ~5-10 in HDR;
    // a fixed exposure of 0.03 maps this into the tonemap's useful range.
    float exposure = 0.15f;

    float3 shadowed = color.rgb * s * exposure;
    shadowed = Tonemap(shadowed);

    u_output[pos.xy] = float4(shadowed, color.a);
}
