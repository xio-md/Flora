// Separable bilateral shadow blur (horizontal + vertical passes)
// Based on Niagara's shadowblur.comp.glsl
// c_shadow.blurDirection: 0 = horizontal, 1 = vertical

#include "shadow_types.hlsl"

Texture2D<float>   t_shadow : register(t0);
Texture2D<float>   t_depth : register(t1);
RWTexture2D<float> u_output : register(u0);
StructuredBuffer<ShadowConstants> t_shadowConstants : register(t263);

#define c_shadow t_shadowConstants[0]

// Minimal bilateral denoiser — multi-sample sun jitter already produces
// the contact-hardening penumbra. This pass only suppresses the residual
// noise from 4-sample Monte Carlo with a very tight 1-neighbour kernel.
static const int KERNEL = 1;

[numthreads(8, 8, 1)]
void main(uint3 pos : SV_DispatchThreadID)
{
    if (pos.x >= (uint)c_shadow.imageSize.x || pos.y >= (uint)c_shadow.imageSize.y)
        return;

    float shadow = t_shadow[pos.xy];
    float accumW = 1.0f;

    float depth = t_depth[pos.xy];

    // offsetMask: (1,0) for horizontal, (0,1) for vertical
    int2 offsetMask = (c_shadow.blurDirection == 0) ? int2(1, 0) : int2(0, 1);

    [loop]
    for (int sign = -1; sign <= 1; sign += 2)
    {
        int2 uvnext = int2(pos.xy) + sign * offsetMask;
        float dnext = t_depth[uvnext];
        float dgrad = abs(depth - dnext) < 0.1f ? dnext - depth : 0.0f;

        [loop]
        for (int i = 1; i <= KERNEL; ++i)
        {
            int2 uvoff = int2(pos.xy) + i * sign * offsetMask;
            // Very tight spatial weight: neighbour contributes ≤ 25 %
            float gw = exp2(-float(i * i) / 0.5f);
            float dv = t_depth[uvoff];
            // Depth weight: rejects samples across geometric edges
            float dw = exp2(-abs(dv - (depth + dgrad * float(i))) * 200.0f);
            float fw = gw * dw;

            shadow += t_shadow[uvoff] * fw;
            accumW += fw;
        }
    }

    u_output[pos.xy] = shadow / accumW;
}
