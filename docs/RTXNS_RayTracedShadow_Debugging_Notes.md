# RTXNS Ray-Traced Shadow 调试经验与注意事项

本文记录 Phase 1 ray-traced shadow pass 调试过程中遇到的实际问题、排查顺序和修复经验。目标读者是后续继续开发 RTXNS 阴影管线的工程师或代码代理，尤其适用于无法直接阅读渲染图像、只能依赖日志和像素统计的场景。

## 1. 先把问题拆成三层

不要直接从最终 `rt_shadow.png` 判断根因。RT shadow 的最终结果至少包含三层：

1. `depth -> world position` 重建是否正确。
2. RayQuery 输出的 `shadow mask` 是否正确。
3. Composite 是否正确把 `litColor * shadow` 写回。

每次只验证一层。否则很容易把颜色空间问题误判成 RayQuery 错误，或者把 mesh winding 问题误判成阴影方向错误。

## 2. 深度重建注意事项

Donut 的矩阵约定是 row-vector：

```hlsl
clip = world * View * Proj
world = clip * InvViewProj
```

在 HLSL 中应使用：

```hlsl
#pragma pack_matrix(row_major)

float2 pixelPosition = float2(pixel) + 0.5f;
float2 uv = pixelPosition / imageSize;
float4 clipPos = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f);
float4 worldH = mul(clipPos, invViewProj);
float3 wpos = worldH.xyz / worldH.w;
```

验证方式：

- 临时输出 `saturate((wpos.y + 2) / 7)`。地面应接近稳定中灰，对应 `y ~= 0`。
- 再分别输出 `wpos.x`、`wpos.z`，应随地面位置形成连续渐变。
- 只看到 `wpos.y` 正确还不够，`x/z` 坍缩也会导致所有 shadow ray 打到同一块几何。

## 3. Shadow Mask 先 raw 输出

Composite 可能引入颜色空间和纹理格式问题，所以调 RayQuery 时先让 composite 直接输出 shadow mask：

```hlsl
u_output[pos.xy] = float4(shadow.xxx, 1.0f);
```

预期：

- 白色：未遮挡，`shadow = 1`。
- 黑色：遮挡，`shadow = 0`。
- 如果 raw mask 只有局部变黑，但最终图整片变色，问题在 composite，不在 RayQuery。
- 如果 raw mask 整片变黑，再检查 receiver self-hit、ray direction、world position、instance mask。

没有图像能力时，用像素统计判断：

```powershell
python -c "from PIL import Image; import numpy as np; from pathlib import Path; p=Path(r'D:\RTXNS\output\shadow_test'); img=np.array(Image.open(p/'rt_shadow.ppm').convert('RGB')); print(img[...,0].min(), img[...,0].mean(), img[...,0].max(), np.unique(img[...,0])[:20])"
```

binary mask 正常时通常只应有接近 `0` 和 `255` 两类值，且黑色像素比例不应覆盖整个地面。

## 4. Receiver 和 Caster 要分离

Phase 1 的 ground + box 测试里，地面既是 receiver 又在 TLAS 里。如果 shadow ray 也追踪 ground，容易发生自遮挡。

当前做法：

- `ground` instance mask 设为 `0x01`。
- 其他 caster instance mask 设为 `0xFF`。
- shadow ray 使用 `TraceRayInline(..., 0xFE, ray)`，跳过 receiver-only ground。

注意：

- 这是 Phase 1 的最小可验证策略，不是完整材质系统。
- 后续更通用的方案应从 material/render flags 区分 caster/receiver，而不是依赖名字包含 `ground`。
- 如果新测试场景换了地面名字，需要同步更新 receiver 判定，否则可能回到自遮挡或无阴影。

## 5. Composite 颜色空间陷阱

这是本次最容易误判的问题之一。

原始 color target 是 `SRGBA8_UNORM`。如果把它复制到同样 `SRGBA8_UNORM` 的 SRV，再在 compute shader 中采样，shader 可能拿到的是 sRGB decode 后的 linear 值。UAV 写回不会自动做 sRGB encode，于是即使 `shadow = 1`，最终颜色也会整体变暗或偏色。

更糟的是，最终图可能表现为：

- 地面从灰色变成棕色。
- 盒子从橙色变成灰色。
- 非阴影区域也和 `no_shadow` 不一致。

当前 Phase 1 的稳定做法：

- `m_color_target` 保持 `SRGBA8_UNORM`，供 forward pass 正常渲染。
- `m_litColorSRV` 使用 `RGBA8_UNORM`，把已编码的颜色字节作为普通 UNORM 读取。
- `shadow_composite_cs.hlsl` 直接执行：

```hlsl
float4 color = t_litColor[pos.xy];
float shadow = t_shadow[pos.xy];
color.rgb *= shadow;
u_output[pos.xy] = color;
```

对 binary hard shadow (`0/1`) 来说，这能保证：

- `shadow = 1` 时颜色逐像素保持不变。
- `shadow = 0` 时变黑。

如果后续做 soft shadow、半影或 physically correct lighting，建议升级到 linear intermediate，例如 `RGBA16_FLOAT`，最后统一 tonemap/sRGB encode。不要在 `SRGBA8` SRV/UAV 上混合处理。

## 6. Mesh Winding 和 Normals 要一起修

测试盒子曾出现“中间像空的”的问题。根因不是 shadow pass，而是测试 mesh 的 triangle winding 和 outward normals 不一致。

只添加 per-face normals 不够。如果 winding 仍然反，渲染器按 winding 做背面剔除时，盒子会像一个朝内的盒壳。

当前 `_make_box()` 应满足：

- 每个面使用独立 4 个顶点。
- 每个面有一致的 outward normal。
- 三角形顺序和 outward normal 一致。
- 盒子位于 `y = 0..1`，而不是穿过地面 `y = -0.5..0.5`。

推荐三角形顺序：

```python
tri_base = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32)
```

如果盒子看起来缺面、空心、只有两片墙，先检查 winding，再检查 normals。

## 7. 无法看图时的数值验证清单

每次改 shadow pass 后，建议跑：

```powershell
$env:Path = "${env:LOCALAPPDATA}\Programs\Python\Python312;$env:Path"
$env:PYTHONPATH = "D:\RTXNS\bin\windows-x64"
python D:\RTXNS\tools\test_rt_shadow.py
```

然后用 Python 采样固定点：

```powershell
python -c "from PIL import Image; import numpy as np; from pathlib import Path; p=Path(r'D:\RTXNS\output\shadow_test'); no=np.array(Image.open(p/'no_shadow.ppm').convert('RGB')); rt=np.array(Image.open(p/'rt_shadow.ppm').convert('RGB')); pts=[(320,300),(100,300),(500,300),(320,240),(320,180)]; [print(pt, 'no', no[pt[1],pt[0]].tolist(), 'rt', rt[pt[1],pt[0]].tolist()) for pt in pts]"
```

判断标准：

- 远离阴影的 ground 点：`no == rt` 或仅有极小差异。
- 远离阴影的 box 点：`no == rt` 或仅有极小差异。
- 阴影区域 ground 点：`rt` 应明显更暗。
- 如果所有 ground 采样点都变暗，是 shadow mask 或 composite 全局错误。
- 如果所有颜色都偏色但 raw mask 正常，是 composite/format/sRGB 错误。
- 如果 box 本体缺面或像空壳，是 mesh winding/normals 错误。

也可以统计暗化覆盖比例：

```powershell
python -c "from PIL import Image; import numpy as np; from pathlib import Path; p=Path(r'D:\RTXNS\output\shadow_test'); no=np.array(Image.open(p/'no_shadow.ppm').convert('RGB')).astype(np.float32); rt=np.array(Image.open(p/'rt_shadow.ppm').convert('RGB')).astype(np.float32); mask=((no.sum(axis=2)>5)|(rt.sum(axis=2)>5)); dark=(no.mean(axis=2)-rt.mean(axis=2)); print('dark>30', int(((dark>30)&mask).sum()), 'ratio', float(((dark>30)&mask).sum()/mask.sum()))"
```

局部硬阴影应只占小比例；如果比例接近 `1.0`，说明整片可见区域都被暗化。

## 8. 推荐排查顺序

1. 确认 `no_shadow` 图里测试几何正确闭合，盒子不是空壳。
2. 临时输出 `wpos.y/x/z`，确认 world reconstruction。
3. 临时输出 raw shadow mask，确认 RayQuery 命中区域。
4. 恢复 composite，只验证 `shadow=1` 的非阴影区域是否逐像素保持原色。
5. 生成或查看 `shadow_diff.png`，确认暗化区域是否局部且方向合理。
6. 最后再调 bias、ray direction、caster/receiver mask。

## 9. 当前已知的 Phase 1 局限

- receiver/caster mask 依赖名字 `ground`，还不是通用材质系统。
- shadow factor 是 binary `0/1`，没有软阴影、半影或过滤。
- composite 采用 encoded-color 乘法，适合 binary hard shadow；软阴影阶段应改成 linear intermediate。
- `test_rt_shadow.py` 目前是专用验证场景，不应当被当作通用 mesh 生成器。

