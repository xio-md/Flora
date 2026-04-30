# DonutRender Week 9：材质与纹理支持矩阵

## 1. 本周目标

Week 9 的重点不是继续扩展增量更新路径，而是把当前材质系统从“只有常量因子可演示”推进到“能承载更真实 PBR 输入”的状态，优先补齐：

- `base color`
- `roughness`
- `metallic`
- `emissive`
- `opacity`

并且给出明确的支持边界和失败语义，避免未支持项静默退化。

## 2. 当前已执行的材质纹理

当前两条 Python 路径都已支持把以下 `ImageTexture` 真正写入材质：

- `DonutRenderPy -> GenesisStyleRenderer -> GLB`
- 直接 `GenesisStyleRenderer -> GLB`

| 输入字段 | 当前执行方式 | 备注 |
| --- | --- | --- |
| `kd` / `base color` | 输出到 `pbrMetallicRoughness.baseColorTexture` | 支持 raw `image_data`，也支持单独的 PNG/JPEG 直通 |
| `opacity` | 打包进 `baseColorTexture` 的 alpha 通道 | 当前需要 raw `image_data`，以便和 base color 合成 |
| `roughness` | 打包进 `metallicRoughnessTexture` 的 G 通道 | 当前需要 raw `image_data` |
| `metallic` | 打包进 `metallicRoughnessTexture` 的 B 通道 | 当前需要 raw `image_data` |
| `Light.emission` | 输出到 `emissiveTexture` | 支持 raw `image_data`，也支持单独的 PNG/JPEG 直通 |

对应因子语义也已同步调整：

- 当某个字段是“真正带图像内容的 `ImageTexture`”时，默认不再错误地把 `roughnessFactor / metallicFactor / emissiveFactor` 压回到旧的常量默认值；
- 若 `ImageTexture.scale` 存在，则继续作为纹理乘子保留；
- `opacity` 会驱动 `alphaMode = "BLEND"`，不再只依赖常量 alpha。

## 3. 当前回退与限制

本周刻意保持了清晰边界，下面这些项不会静默吞掉：

- `normal_map`
- `PlasticSurface.ks`
- `DisneySurface.eta`
- `DisneySurface.specular_tint`
- `DisneySurface.specular_trans`
- `DisneySurface.diffuse_trans`
- `MetalSurface.eta`
- `GlassSurface.kt`
- `GlassSurface.eta`

如果上述字段传入了真正带图像负载的 `ImageTexture`，当前会显式抛出 `UnsupportedFeatureError`。

此外，以下组合目前仍受限：

- `opacity` 纹理当前需要 raw `image_data`，因为它要被打包进 `baseColorTexture` alpha；
- `roughness / metallic` 纹理当前需要 raw `image_data`，因为 glTF 需要打成同一张 `metallicRoughnessTexture`；
- 若多张 raw 纹理分辨率不一致，当前会显式失败，而不是隐式 resize。

## 4. 本周验证入口

本周新增了两个直接可跑的产物：

- 示例：`samples/DonutRenderPyDemo/donut_render_material_texture_demo.py`
- smoke：`tools/donut_render/material_texture_smoke.py`

其中 smoke 会验证：

- 渲染链路能跑通；
- 输出 GLB 中确实存在：
  - `baseColorTexture`
  - `metallicRoughnessTexture`
  - `emissiveTexture`
  - `alphaMode = "BLEND"`

## 5. 对后续 Week 10+ 的意义

Week 9 完成后，材质路径至少已经从“只有常量参数”推进到“可执行的纹理化 PBR 子集”。这意味着后续如果继续下沉 `surface-only` 更新路径，已经不需要再先补一遍基础材质表达能力，而可以直接围绕：

- material handle 更新
- texture 资源生命周期
- `update_surface(...)` 的 native 增量同步

继续推进。
