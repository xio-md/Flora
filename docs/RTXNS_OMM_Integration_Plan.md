# RTXNS 集成 Opacity Micromaps (OMM) 与光线追踪阴影 — 全局方案

## 1. 目标

在 RTXNS（Python → Donut → Vulkan 光栅化渲染器）中集成基于光线追踪的半透明阴影，借助 **Opacity Micromaps (OMM)** 硬件加速植被/叶片等透明材质的阴影评估。

### 1.1 参考实现

| 项目 | 路径 | 用途 |
|------|------|------|
| **Niagara** (zeux Vulkan 渲染器) | `D:\niagara\` | 完整的 OMM+RT 阴影管线参考实现 |
| **NVIDIA OMM SDK** (v1.9.1) | `D:\OMM\` | 官方 OMM 烘焙 SDK，CPU Baker 可脱离 Donut 使用 |

---

## 2. 现状分析

| 维度 | 当前状态 | 关键文件 |
|------|---------|---------|
| 渲染管线 | 纯光栅化 ForwardShadingPass，无阴影 | `D:\RTXNS\src\PythonBindings\headless_pbr.cpp:383-470` |
| 光线追踪 | 无 BLAS/TLAS/RT Pipeline/SBT | — |
| OMM | 无 | — |
| Donut RT | 框架不提供（README 明确由应用自行构建） | `D:\RTXNS\external\donut` (子模块未 init) |
| NVRHI RT API | 存在但未使用 (`nvrhi::rt::*`) | Donut 自带 |
| C++ 标准 | C++20 | `D:\RTXNS\CMakeLists.txt:18` |
| GPU 后端 | Vulkan 1.2+ | `D:\RTXNS\CMakeLists.txt:25` |

---

## 3. 总体架构（目标状态）

```
Python (GenesisStyleRenderer)
  │  D:\RTXNS\python\rtxns_genesis_style\renderer.py
  │
  ├── GlbSceneBuilder → 构建 GLB 场景
  │     D:\RTXNS\python\rtxns_genesis_style\glb_builder.py
  │
  ├── scene.load_scene() → C++ HeadlessPbrScene
  │     D:\RTXNS\src\PythonBindings\headless_pbr.cpp
  │
  └── scene.render_frame() →
        │
        ├── [现有] Rasterize ForwardShadingPass (albedo + depth)
        │     依赖: donut/render/ForwardShadingPass.h
        │
        └── [新增] RayTracedShadowPass
              ├── 1. BLAS 构建 (首次/脏更新)
              │     参考: D:\niagara\src\scenert.cpp:16-180 (buildBLAS)
              ├── 2. OMM 烘焙 (CPU, 首次/脏更新) [Phase 2]
              │     参考: D:\niagara\src\scene.cpp:861-1049 (buildSceneOmm)
              │     备选: D:\OMM\libraries\omm-lib\include\omm.h (OMM SDK C API)
              ├── 3. OMM AS 构建 [Phase 2]
              │     参考: D:\niagara\src\scenert.cpp:581-692 (buildOMM)
              ├── 4. TLAS 构建/更新 (每帧)
              ├── 5. TraceRay: 阴影光线 (每像素 1 spp)
              │     参考: D:\niagara\src\shaders\shadow.comp.glsl (GLSL RT shader)
              ├── 6. Shadow Denoise (可选双边模糊) [Phase 3]
              │     参考: D:\niagara\src\shaders\shadowblur.comp.glsl
              └── 7. 合成到 ForwardShadingPass 输出
```

---

## 4. 参考代码索引

### 4.1 Niagara 渲染器 (`D:\niagara\`)

| 文件 | 行号 | 用途 |
|------|------|------|
| `D:\niagara\src\scene.cpp` | `861-1049` | `buildSceneOmm()` — OMM 离线烘焙完整流程 |
| `D:\niagara\src\scene.cpp` | `20-22` | OMM 烘焙常量 (`kOmmSubdivisionScale`, `kOmmSubdivisionLevel`) |
| `D:\niagara\src\scene.cpp` | `836-858` | `normalizeIndicesForOMM()` — 索引旋转适配 meshoptimizer 格式 |
| `D:\niagara\src\scene.h` | `68-84` | `Mesh` 结构体 (含 `ommIndexData`, `ommIndexBase`) |
| `D:\niagara\src\scene.h` | `86-100` | `Geometry` 结构体 (含 `ommData`, `ommDescs`, `ommIndices`, `ommStates`) |
| `D:\niagara\src\scene.h` | `102-108` | `Camera` 结构体 |
| `D:\niagara\src\scenert.cpp` | `1-180` | `buildBLAS()` — BLAS 构建，含 OMM pNext 附着 |
| `D:\niagara\src\scenert.cpp` | `14` | `kBuildOMM` 标志 |
| `D:\niagara\src\scenert.cpp` | `64-83` | OMM 挂接到 BLAS 几何体 (pNext chain) |
| `D:\niagara\src\scenert.cpp` | `516` | Instance 标记: `FORCE_OPAQUE_BIT` vs `0` |
| `D:\niagara\src\scenert.cpp` | `581-692` | `buildOMM()` — Vulkan OMM AS 构建 |
| `D:\niagara\src\scenert.h` | `20` | `buildOMM()` 声明 |
| `D:\niagara\src\nagara.cpp` | `36-43` | 阴影/模糊/质量 全局开关变量 |
| `D:\niagara\src\nagara.cpp` | `359-374` | 键盘按键映射 (`F`/`B`/`Q`/`X`) |
| `D:\niagara\src\nagara.cpp` | `700-758` | 阴影管线创建 (shadowlqPipeline / shadowhqPipeline) |
| `D:\niagara\src\nagara.cpp` | `844` | OMM states 环境变量 (`OMM=4` 默认) |
| `D:\niagara\src\nagara.cpp` | `861-868` | `buildSceneOmm()` 调用 + OMMMIP 环境变量 |
| `D:\niagara\src\nagara.cpp` | `1096-1127` | OMM AS + ommIndex 缓冲区上传到 GPU |
| `D:\niagara\src\nagara.cpp` | `1771-1797` | 阴影 Ray Tracing dispatch |
| `D:\niagara\src\nagara.cpp` | `1791` | `sunJitter` 设置 (blur 时 1e-2，否则 0) |
| `D:\niagara\src\nagara.cpp` | `1815-1829` | 阴影双边模糊 dispatch |
| `D:\niagara\src\nagara.cpp` | `1885-1903` | 最终合成 (阴影 × 光照) |
| `D:\niagara\src\device.cpp` | `283-408` | OMM 特性启用 (`VK_KHR_opacity_micromap`) |
| `D:\niagara\src\config.h` | `1-52` | 全局配置常量 |
| `D:\niagara\src\shaders\shadow.comp.glsl` | `1-161` | 阴影光追主 shader (GLSL) |
| `D:\niagara\src\shaders\shadow.comp.glsl` | `26-35` | `ShadowData` 结构体 |
| `D:\niagara\src\shaders\shadow.comp.glsl` | `78-84` | `shadowTrace()` — 不透明阴影函数 |
| `D:\niagara\src\shaders\shadow.comp.glsl` | `86-123` | `shadowTraceTransparent()` — alpha test 透明阴影 |
| `D:\niagara\src\shaders\shadow.comp.glsl` | `143-151` | 太阳方向 jitter (gradient noise) |
| `D:\niagara\src\shaders\shadow.comp.glsl` | `153-160` | Quality 分支 (0=opaque OMM2, 1=transparent) |
| `D:\niagara\src\shaders\shadowblur.comp.glsl` | `1-64` | 可分离双边阴影模糊 |
| `D:\niagara\src\shaders\shadowblur.comp.glsl` | `3` | `#define BLUR 1` |
| `D:\niagara\src\shaders\shadowblur.comp.glsl` | `36` | 模糊核半宽: `KERNEL = 10` |
| `D:\niagara\src\shaders\shadowfill.comp.glsl` | `1-46` | 棋盘格补洞 shader |
| `D:\niagara\src\shaders\final.comp.glsl` | `1-80` | 最终合成 (阴影 ambient=0.05, sunIntensity=2.5) |
| `D:\niagara\src\shaders\mesh.h` | `1-` | GPU 侧 Mesh 结构 (含 `ommIndexData`, `lodRT`) |
| `D:\niagara\src\shaders\math.h` | `1-` | `gradientNoise()` 噪声函数 |
| `D:\niagara\src\scenecache.cpp` | `52-57` | `SceneCameraFile` 结构体 |
| `D:\niagara\src\scenecache.cpp` | `115-393` | 场景缓存 保存/加载 (含 OMM 数据持久化) |
| `D:\niagara\src\textures.cpp` | `262-381` | `decodeImageRGBA()` — DDS BCn 解码到 RGBA8 |
| `D:\niagara\extern\meshoptimizer\src\meshoptimizer.h` | `885-920` | `opacityMapMeasure/Rasterize/Compact` API |
| `D:\niagara\README.md` | `1-167` | 项目概述 + 全部 33 集 Stream 索引 |

### 4.2 NVIDIA OMM SDK (`D:\OMM\`)

| 文件 | 行号 | 用途 |
|------|------|------|
| `D:\OMM\libraries\omm-lib\include\omm.h` | `1-1206` | **C API** — 全部类型、枚举、结构体、函数声明 |
| `D:\OMM\libraries\omm-lib\include\omm.h` | `98-104` | `ommOpacityState` 枚举 (Transparent/Opaque/UO/UT) |
| `D:\OMM\libraries\omm-lib\include\omm.h` | `106-112` | `ommSpecialIndex` 枚举 (-1 ~ -4) |
| `D:\OMM\libraries\omm-lib\include\omm.h` | `114-122` | `ommFormat` 枚举 (OC1_2_State / OC1_4_State) |
| `D:\OMM\libraries\omm-lib\include\omm.h` | `124-134` | `ommUnknownStatePromotion` 枚举 (Nearest/ForceOpaque/ForceTransparent) |
| `D:\OMM\libraries\omm-lib\include\omm.h` | `282-287` | `ommCpuTextureFormat` 枚举 (UNORM8 / FP32) |
| `D:\OMM\libraries\omm-lib\include\omm.h` | `298-334` | `ommCpuBakeFlags` 枚举 (内部线程/特殊索引/去重等) |
| `D:\OMM\libraries\omm-lib\include\omm.h` | `340-356` | `ommCpuTextureMipDesc` 结构体 |
| `D:\OMM\libraries\omm-lib\include\omm.h` | `358-378` | `ommCpuTextureDesc` 结构体 |
| `D:\OMM\libraries\omm-lib\include\omm.h` | `380-460` | **`ommCpuBakeInputDesc`** — 核心烘焙输入结构体 |
| `D:\OMM\libraries\omm-lib\include\omm.h` | `492-530` | `ommCpuOpacityMicromapDesc` / `ommCpuBakeResultDesc` 输出结构体 |
| `D:\OMM\libraries\omm-lib\include\omm.h` | `532-544` | `ommCpuBlobDesc` 序列化结构体 |
| `D:\OMM\libraries\omm-lib\include\omm.h` | `568-594` | 全部 CPU Baker C API 函数声明 |
| `D:\OMM\libraries\omm-lib\include\omm.hpp` | `1-1088` | C++ namespace 封装 (inline 实现) |
| `D:\OMM\libraries\omm-lib\src\bake_cpu_impl.cpp` | `1-1990` | CPU Baker 完整14步管线实现 |
| `D:\OMM\libraries\omm-lib\src\bake_kernels_cpu.h` | `1-454` | **LevelLineIntersection** + 保守双线性算法 |
| `D:\OMM\libraries\omm-lib\src\bake_kernels_cpu.h` | `25-60` | `GetStateFromCoverage()` — 状态决策逻辑 |
| `D:\OMM\libraries\omm-lib\src\util\bird.h` | `1-184` | Bird Curve 数学 (微三角形索引/重心坐标) |
| `D:\OMM\libraries\omm-lib\src\defines.h` | `1-28` | `kMaxSubdivLevel = 12` |
| `D:\OMM\libraries\omm-lib\src\bake.cpp` | `1-480` | C API 入口点实现 |
| `D:\OMM\libraries\omm-lib\CMakeLists.txt` | `1-130` | 构建依赖: glm + stb + xxHash + lz4 (C++20) |
| `D:\OMM\support\tests\test_minimal_sample.cpp` | `1-160` | **最小使用示例** (CreateBaker → CreateTexture → Bake → GetResult) |
| `D:\OMM\docs\integration_guide.md` | `1-782` | SDK 集成指南 |
| `D:\OMM\docs\OMM_SDK_源码实现链路分析.md` | `1-758` | 中文源码级实现链路分析 |

---

## 5. 分阶段计划

### Phase 1: 基础光线追踪不透明阴影

**目标**: 构建完整的 Vulkan RT 管线底层设施，实现单光线硬阴影。

- BLAS + TLAS 构建
- 单光线 TraceRay 阴影查询
- 合成到现有 ForwardShadingPass 输出
- **不包含**: OMM、透明阴影、模糊、棋盘格、Python 绑定

> 详见: `D:\RTXNS\docs\RTXNS_Phase1_RayTracedShadow.md`

### Phase 2: OMM 烘焙与集成

**目标**: 集成 OMM 离线烘焙和 Vulkan OMM AS 构建。

- OMM CPU 离线烘焙（meshoptimizer 方案优先，OMM SDK 作为备选高精度方案）
- OMM AS 构建 (Vulkan `VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR`)
- BLAS OMM 附着 (pNext chain)
- Quality 0: OMM 2-state 强制不透明阴影

核心参考:
- `D:\niagara\src\scene.cpp:861-1049` — 烘焙流程
- `D:\niagara\src\scenert.cpp:581-692` — Vulkan OMM AS 构建
- `D:\niagara\src\scenert.cpp:64-83` — BLAS OMM pNext 附着
- `D:\OMM\support\tests\test_minimal_sample.cpp` — OMM SDK 最小示例

### Phase 3: 透明阴影 + 降噪

**目标**: 支持带透明度的阴影光线（植被透光效果）。

- Quality 1: Alpha test 透明阴影
- OMM 4-state 支持（Unknown Opaque / Unknown Transparent）
- 可选阴影双边模糊 + 棋盘格渲染

核心参考:
- `D:\niagara\src\shaders\shadow.comp.glsl:86-123` — `shadowTraceTransparent()`
- `D:\niagara\src\shaders\shadowblur.comp.glsl` — 双边模糊
- `D:\niagara\src\shaders\shadowfill.comp.glsl` — 棋盘格补洞
- `D:\niagara\src\nagara.cpp:153-160` — Quality 0 vs 1 分支逻辑

### Phase 4: Python 绑定与工具

**目标**: 在 Python 侧暴露 OMM/阴影控制。

- `GenesisStyleRenderer` 新增 OMM 开关、阴影质量参数
- 性能对比工具（OMM on/off, Quality 0/1, blur on/off）
- Bistro 场景 GLB 验证

核心参考:
- `D:\RTXNS\python\rtxns_genesis_style\renderer.py` — 现有 Python API
- `D:\RTXNS\src\PythonBindings\py_interface_donut_native.cpp` — 现有 pybind11 绑定

---

## 6. 关键风险与缓解

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| `external/donut` 子模块未初始化 (空目录) | **阻塞** | Phase 1 第一步: `git submodule update --init --recursive` |
| Donut/NVRHI RT API 文档稀少 | 中 | 参考 nvrhi 官方示例 + Niagara 的纯 Vulkan 实现 |
| Donut Scene 几何布局与 RT Buffer 格式不兼容 | 中 | Phase 1 先做最小 BLAS 验证 |
| Headless 模式下 Vulkan RT 扩展可用性 | 低 | 已确认 RTX 5080 + Vulkan 1.4 支持所有 RT 扩展 |
| C++ 编译时间增加（RT Shader 编译） | 低 | 仅新增 ~5 个 HLSL shader，增量编译可控 |

---

## 7. 预期产出

1. **C++ 库**: `rtxns_ray_traced_shadow` — 独立于 Donut 的 RT 阴影模块
2. **HLSL Shader**: ~5 个 RT/Compute shader
3. **Python 绑定扩展**: `GenesisStyleRenderer` 新参数
4. **文档**: Phase 1-4 详细方案文档
5. **验证场景**: Bistro GLB (`D:\niagara_bistro\bistro.gltf`) 的阴影对比图
