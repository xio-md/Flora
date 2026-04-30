DonutRender Week 9 Report

## 本周完成

- 按照 `DonutPython_12Week_Plan.md` 第 9 周目标，把材质系统从“只有常量因子”的演示级状态推进到了“支持纹理化 PBR 子集”的后端级状态；
- 在 `python/donut_render_py/runtime.py` 中把 `ImageTexture` 真正接入后端材质执行链路，补齐：
  - `base color`
  - `roughness`
  - `metallic`
  - `emissive`
  - `opacity`
- 在 `python/rtxns_genesis_style/glb_builder.py` 中补齐 GLB 内嵌纹理写出能力，支持：
  - `images`
  - `textures`
  - `baseColorTexture`
  - `metallicRoughnessTexture`
  - `emissiveTexture`
- 在 `python/rtxns_genesis_style/renderer.py` 中同步补齐直接 `GenesisStyleRenderer` 路径的材质包装逻辑，使底层原型接口和 `DonutRenderPy` 高层接口获得一致的纹理能力；
- 新增可展示的纹理材质样例：
  - `samples/DonutRenderPyDemo/donut_render_material_texture_demo.py`
- 新增验证脚本：
  - `tools/donut_render/material_texture_smoke.py`
- 新增材质支持矩阵文档：
  - `docs/DonutRender_Week9_MaterialSupport.md`

## 本周验证

本周已本地验证通过：

- `python -m py_compile`
  - `python/donut_render_py/runtime.py`
  - `python/rtxns_genesis_style/glb_builder.py`
  - `python/rtxns_genesis_style/renderer.py`
  - `samples/DonutRenderPyDemo/donut_render_material_texture_demo.py`
  - `tools/donut_render/material_texture_smoke.py`
- `python tools/donut_render/backend_render_smoke.py`
  - 返回 `16384`
- `python tools/donut_render/material_texture_smoke.py`
  - 验证 GLB 中已生成：
    - `baseColorTexture`
    - `metallicRoughnessTexture`
    - `emissiveTexture`
    - `alphaMode = "BLEND"`
- `python tools/donut_render/api_lifecycle_smoke.py --quiet`
- `python tools/donut_render/scene_update_plan_smoke.py`
  - 生成 `.temp/scene_update_plan.json`
- `python samples/DonutRenderPyDemo/donut_render_material_texture_demo.py --width 96 --height 96`
  - 生成 `.temp/demo_outputs/donut_render_material_texture_demo/manifest.json`

## 本周结果

### 材质链路已经进入“可执行纹理化”阶段

Week 8 结束时，`surface-only` 仍然只具备常量参数层面的表达能力，`ImageTexture` 还没有真正进入后端材质执行链路。

本周完成后：

- `base color / roughness / metallic / emissive / opacity` 已可被真正写入 GLB 材质；
- `opacity` 不再只是常量 alpha，而是能驱动 `baseColorTexture` 的 alpha 通道；
- `roughness / metallic` 已能按 glTF 约定打包到 `metallicRoughnessTexture`；
- 同一套纹理材质能力同时覆盖：
  - `DonutRenderPy`
  - `GenesisStyleRenderer`

这意味着 Week 9 的交付已经不再只是“材质接口更像 LuisaRenderPy”，而是已经具备真实的纹理化 PBR 执行子集。

### 未支持项已转为显式失败，而不是静默退化

本周也同步收口了失败语义。对当前仍未正式执行的纹理字段，例如：

- `normal_map`
- 若干高阶 specular / transmission / eta 相关字段

现在若传入带真实图像负载的 `ImageTexture`，会显式抛出 `UnsupportedFeatureError`，避免调用方误以为功能已生效。

## 当前风险

- `surface-only` 仍然走 full rebuild，说明材质表达能力已经补齐，但 native `update_surface(...)` 还没有真正下沉；
- `roughness / metallic / opacity` 纹理当前主要面向 raw `image_data` 打包路径，编码文件直通还不适用于所有需要合成/打包的情况；
- `normal_map` 与更完整的 Disney/Glass/Metal 高阶纹理参数仍未执行；
- 当前 Week 9 的重点是“材质表达能力补齐”，不是“材质更新性能问题已经关闭”。

## 下周目标

- 进入 Week 10，继续围绕动态几何与长序列稳定性推进；
- 如果继续沿 `surface-only` 性能缺口推进，应优先评估 native `update_surface(...)` 的下沉优先级；
- 在保持 Week 8 增量路径与 Week 9 材质路径稳定的前提下，避免后续动态更新改动破坏当前材质纹理链路。

## 需要支持

- 一套稳定可复现的 Linux Vulkan-only 验证环境；
- 如果要继续降低 `surface-only` 的 rebuild 成本，需要尽早确定 native material handle / texture resource 生命周期的下沉方案。
