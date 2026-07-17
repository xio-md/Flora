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

    // The forward pass already contains Flora's ambient approximation.  Until
    // direct and ambient terms are split into separate buffers, retain a
    // meaningful floor here so RT visibility does not incorrectly black out
    // that ambient contribution in fully shadowed regions.
    float shadowAmbient = 0.25f;
    float s = min(shadow + shadowAmbient, 1.0f);

    // Shared cross-renderer reference display transform.  The Flora no-RT
    // control, Flora RT-shadow render, SAPIEN direct render and SAPIEN GI
    // render all use the same 0.60 exposure and HBD curve for the ReplicaCAD
    // four-way comparison.  This is presentation-only; lighting is unchanged.
    float exposure = 0.60f;

    float3 shadowed = color.rgb * s * exposure;
    shadowed = Tonemap(shadowed);

    u_output[pos.xy] = float4(shadowed, color.a);
}
