Texture2D<float4>  t_litColor : register(t0);
Texture2D<float>   t_shadow : register(t1);
RWTexture2D<float4> u_output : register(u2);

// Niagara's Hejl-Burgess-Dawson filmic tonemap
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
    float peak = max(color.r, max(color.g, color.b));
    float exposure = clamp(0.2f / max(peak, 1.0e-4f), 0.0015f, 1.0f);

    // Niagara: shadowAmbient = 0.05
    shadow = min(shadow + 0.05f, 1.0f);
    // Keep HDR highlights in the tonemapper's useful range without crushing
    // lower-intensity synthetic test scenes.
    color.rgb *= shadow * exposure;
    color.rgb = Tonemap(color.rgb);
    u_output[pos.xy] = color;
}
