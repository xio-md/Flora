# Flora OMM 集成计划 — Opacity Micromap 硬件加速光追阴影

## 概述

在 Phase 1 光追阴影管线 + Sun Jitter 质量优化的基础上，集成 **NVIDIA Opacity Micromap (OMM)**，利用 Ada Lovelace+ 架构的硬件加速能力，跳过已确认透明/不透明的微三角形，减少 alpha-tested 几何（植被叶片）的 Ray Query 候选遍历开销。

Bistro 场景有 **2909 个实例、大量 alpha-tested 叶片几何体**，当前每次 shadow ray 遇到 alpha-tested 三角形都需要在 shader 中执行纹理采样 + cutoff 判断。OMM 将 opacity 状态预烘焙到微三角粒度，让 GPU 硬件在 traversal 阶段直接跳过已知透明/不透明的微三角。

---

## 前置条件

| 项目 | 状态 |
|------|------|
| OMM SDK 编译 (`omm-bake.lib`, 9MB) | ✅ 已完成 (D:\OMM\build) |
| CMake 链接 OMM SDK | ✅ 已完成 (`CMakeLists.txt` line 56) |
| NVRHI OMM 抽象层 | ✅ 已确认 (完整的 `IOpacityMicromap` / `createOpacityMicromap` / `buildOpacityMicromap` / BLAS pNext) |
| Vulkan 扩展请求 `VK_KHR_opacity_micromap` | ⚠️ 已写代码但注释掉 (需恢复) |
| 设备支持检测 (`Feature::RayTracingOpacityMicromap`) | ⚠️ 同上 |
| 目标 GPU | **RTX 5090D (32GB, Blackwell)** — ✅ 完整原生 OMM 硬件加速 |

> RTX 5090D 从 Ada Lovelace 继承并增强了 OMM 硬件加速，是集成验证的理想平台。预期性能收益可完整体现。

---

## 推进计划

### 步骤 1：恢复 Vulkan OMM 扩展请求 + 设备检测

**文件**：`src/PythonBindings/headless_pbr.cpp`

取消注释已写好的 OMM 扩展请求代码：

```cpp
// 恢复：请求 VK_KHR_opacity_micromap 扩展
device_params.optionalVulkanDeviceExtensions.push_back(
    VK_KHR_OPACITY_MICROMAP_EXTENSION_NAME);

// 恢复：检测 OMM 支持
m_ommSupported = m_device_manager->GetDevice()->queryFeatureSupport(
    nvrhi::Feature::RayTracingOpacityMicromap);
```

**产出**：`m_ommSupported` 标志位 + 设备创建日志确认扩展可用性。

---

### 步骤 2：CPU OMM Baker 封装

**新增文件**：`src/RayTracedShadow/OMMBaker.h` / `OMMBaker.cpp`

核心职责：从 Donut `Material` 提取 alpha 纹理数据 → 调用 OMM SDK `omm::Cpu::Bake()` → 输出 OMM 数据。

#### 输入

遍历 `SceneGeometryProvider` 提取的几何体元数据，对每个 `isAlphaTested == true` 的材质：

| 输入 | 来源 | 说明 |
|------|------|------|
| Alpha 纹理像素数据 | `material->opacityTexture` 或 `material->baseOrDiffuseTexture` (通道 `.a`) | 需 CPU 端读取纹理像素 |
| 索引缓冲区 (IB) | `SceneGeometryProvider` 已提取 | 三角形索引 |
| UV 缓冲区 | `SceneGeometryProvider` 已从 VB 中提取 (offset=20, 含 float2) | 纹理坐标 |
| alphaCutoff | `material->alphaCutoff` | 默认 0.5 |
| 细分级别 (subdivisionLevel) | 固定 5 (1024 微三角/三角形) 或自适应 | OMM 精度 |
| 格式 (format) | `omm::Format::OC1_4_State` (4 态) 或 `OC1_2_State` (2 态) | 2 态更紧凑 |

#### 调用 OMM SDK

```cpp
// 1. 创建 Baker
omm::BakeOptions options;
options.type = omm::BakeType::CPU;

// 2. 创建纹理
auto tex = omm::Cpu::CreateTexture(baker, alphaPixels, width, height);

// 3. 配置输入
omm::Cpu::BakeInputDesc input = {};
input.texture = tex;
input.alphaMode = omm::AlphaMode::kCutoff;
input.alphaCutoff = cutoff;
input.indexBuffer = ibData;      // 直接使用 BLAS 的 IB
input.uvBuffer = uvData;         // 直接使用 BLAS 的 UV
input.format = omm::Format::OC1_4_State;
input.subdivisionLevel = 5;

// 4. 烘焙
omm::Cpu::Bake(baker, &input, 1);

// 5. 获取结果
auto* result = omm::Cpu::GetBakeResultDesc(baker);
// result->arrayData, result->descArray, result->indexBuffer, result->histograms
```

#### 输出数据结构

```cpp
// 封装为 RTXNS 内部结构
struct OMMBakeResult {
    std::vector<uint8_t>     arrayData;      // 微三角不透明度位 → VkMicromapBuildInfoEXT::data
    std::vector<uint8_t>     descArray;      // OMM 描述符 → VkMicromapBuildInfoEXT::triangleArray
    std::vector<uint8_t>     indexBuffer;    // 三角形→OMM 映射 → BLAS OMM attachment
    omm::IndexFormat         indexFormat;
    std::vector<omm::Cpu::OpacityMicromapUsageCount> descHistogram;
    std::vector<omm::Cpu::OpacityMicromapUsageCount> indexHistogram;
};
```

---

### 步骤 3：OMM 加速结构构建 (NVRHI)

**修改文件**：`src/RayTracedShadow/AccelerationStructure.h/.cpp`

在现有 BLAS/TLAS 构建流程中加入 OMM 加速结构的创建。

#### 3a. 创建 OMM Array

```cpp
// 通过 NVRHI 创建 OMM（无需原始 Vulkan API）
nvrhi::rt::OpacityMicromapDesc ommDesc;
ommDesc.flags = nvrhi::rt::OpacityMicromapBuildFlags::FastTrace;
ommDesc.counts = { /* 从 descHistogram 转换 */ };
ommDesc.inputBuffer = arrayDataBuffer;
ommDesc.perOmmDescs = descArrayBuffer;

auto ommHandle = device->createOpacityMicromap(ommDesc);
commandList->buildOpacityMicromap(ommHandle, ommDesc);
```

`descHistogram` 和 NVRHI 的 `rt::OpacityMicromapUsageCount` 内存布局完全相同（已验证），可以直接 memcpy。

#### 3b. BLAS OMM 附着

```cpp
// 通过 NVRHI GeometryTriangles 的 setOpacityMicromap 附着
nvrhi::rt::GeometryTriangles geomDesc;
geomDesc.setOpacityMicromap(ommHandle);
geomDesc.setOMMIndexBuffer(ommIndexBuffer, ommIndexFormat);
geomDesc.setOMMUsageCounts(indexHistogram.data(), indexHistogram.size());

// BLAS 构建时自动在 pNext 链中包含 OMM 信息
```

---

### 步骤 4：Instance Flag 启用 + Shader 端

**修改文件**：`src/RayTracedShadow/AccelerationStructure.cpp`、`shadow_rayquery_cs.hlsl`

#### 4a. Instance Flag

```cpp
// 对 alpha-tested mesh 的 TLAS instance 设置 OMM flag
if (hasAlphaTestedGeometry && m_ommSupported) {
    instance.setFlags(instance.getFlags() | 
        nvrhi::rt::InstanceFlags::ForceOMM2State);
}
```

NVRHI 内部映射为 `VK_GEOMETRY_INSTANCE_FORCE_OPACITY_MICROMAP_2_STATE_EXT`。

#### 4b. Shader 端

在 `shadow_rayquery_cs.hlsl` 的 RayQuery 调用中，**不需要修改 shader 代码**——`ForceOMM2State` 是 instance 级别的 flag，GPU 硬件在 traversal 阶段自动处理 OMM 查询。Shader 中的 `AlphaTestCandidate()` 仍然作为 fallback（对未覆盖的微三角）。

---

### 步骤 5：Python 端暴露 + 测试脚本

**修改文件**：`headless_pbr.h/.cpp`、`py_bindings_common.h`  
**新增文件**：`tools/test_omm_shadow.py`

#### Python API 新增

```python
scene.enable_omm(True)               # 启用/禁用 OMM
scene.set_omm_config(
    subdivision_level=5,             # 细分级别
    format="OC1_4_State"             # 2 态 / 4 态
)
stats = scene.get_last_frame_stats() # 已有，记录 shadow_ray_ms
scene.get_omm_info()                 # OMM 状态查询
```

#### 测试脚本流程

```python
# 1. 无 OMM 基线
scene.enable_omm(False)
baseline_img, baseline_stats = render_and_capture(scene)

# 2. 启用 OMM (4 态)
scene.enable_omm(True)
omm4_img, omm4_stats = render_and_capture(scene)

# 3. 启用 OMM (2 态)
scene.set_omm_config(format="OC1_2_State")
omm2_img, omm2_stats = render_and_capture(scene)

# 4. 像素差异分析 + 帧率对比
```

---

## 预期结果

### 质量验证

| 对比维度 | 预期 | 验证方法 |
|---------|------|---------|
| 阴影正确性 | OMM ON/OFF 像素差 ≈ 0（阴影结果应一致） | 像素级 diff = 0 |
| 叶片轮廓 | Alpha-tested 叶片阴影形态不变 | 视觉对比 + 差异图 |
| 边界处理 | 微三角边界处无可见接缝 | 局部放大对比 |

### 性能预期

| 场景 | 指标 | 无 OMM (基线) | OMM 4 态 (预期) | 说明 |
|------|------|-------------|-----------------|------|
| Bistro | shadow_ray_ms | ~0.10ms | **~0.04-0.06ms (-40-60%)** | traversal 阶段跳过大量 alpha 微三角 |
| Bistro | 总帧时 | ~4.0ms | **~3.9ms** | shadow ray 占比低，总帧时变化有限 |
| Bistro | Alpha test shader 调用 | 每个 alpha hit 触发 | **大幅减少** | 硬件 traversal 自动跳过 |

> RTX 5090D (32GB, Blackwell) 原生支持 OMM 硬件加速，API 调用完备。性能收益预计 40-60% 的 shadow_ray 开销缩减。

### OMM 数据统计 (Bistro 场景预期)

| 指标 | 估算值 | 说明 |
|------|--------|------|
| Alpha-tested 材质数 | ~20-30 | 含叶片、栅栏等 |
| OMM Array 总大小 | < 1 MB | 压缩编码，微三角粒度 |
| Index Buffer 大小 | < 100 KB | 每三角形 2-4 字节索引 |
| CPU Bake 耗时 | < 100 ms | 一次性，场景加载时执行 |

---

## 最终结果展示：对比报告

类似 `RTXNS_ShadowQuality_Optimization_Report.md`，最终产出将包含：

### 对比图

```
(A) OMM OFF (当前基线)          (B) OMM ON (4 态)            (C) |diff| × 放大
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│                     │    │                     │    │                     │
│   当前渲染结果       │    │   OMM 启用后渲染     │    │   像素差异热力图     │
│   (alpha test 全走    │    │   (硬件跳过已知      │    │   (全黑=完全一致)    │
│    shader fallback)  │    │    透明/不透明微三角) │    │                     │
│                     │    │                     │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### 性能对比 (RTX 40 目标)

| 配置 | 帧时 | 帧率 | shadow_ray_ms | 说明 |
|------|------|------|--------------|------|
| RT 阴影 (无 OMM) | 4.0 ms | 250 FPS | 0.10 ms | 当前基线 |
| RT 阴影 (OMM 4 态) | 3.9 ms | 256 FPS | ~0.07 ms | -30% ray query 开销 |
| RT 阴影 (OMM 2 态) | 3.9 ms | 256 FPS | ~0.06 ms | 更紧凑，-40% |

### OMM 烘焙统计

| 指标 | 值 |
|------|-----|
| Alpha-tested 材质数 | N |
| 总三角形数 (alpha) | M |
| OMM Array 总大小 | X KB |
| CPU Bake 耗时 | Y ms |
| 特殊索引数 (全透/全不透) | Z |

### 差异分析

- **准确性**：OMM ON/OFF 像素级差异 = 0（阴影结果完全一致）
- **边界验证**：叶片轮廓放大对比无变化
- **性能**：RTX 40 上 shadow_ray_ms 预期降低 30-50%

---

## 修改文件清单 (计划)

```
新增:
  src/RayTracedShadow/OMMBaker.h          ★ CPU OMM 烘焙器
  src/RayTracedShadow/OMMBaker.cpp        ★ 调用 OMM SDK Bake API
  tools/test_omm_shadow.py                ★ OMM 对比测试脚本

修改:
  headless_pbr.h/.cpp                     ★ 恢复 OMM 扩展 + enable_omm() + OMM 开关
  py_bindings_common.h                    ★ Python 端 OMM 参数暴露
  AccelerationStructure.h/.cpp            ★ OMM AS 构建 + BLAS 附着 + ForceOMM2State flag
  SceneGeometryProvider.h/.cpp            ★ 提取 alpha 纹理数据供 Baker 使用
  ShadowTypes.h                           ★ +OMMBakeResult 等数据结构
```

---

## 风险与注意事项

1. **5090D 硬件加速已就绪**：Blackwell 原生支持 OMM，性能收益可直接验证。
2. **纹理 CPU 读回**：alpha 纹理需要从 GPU 读回 CPU 才能喂给 OMM SDK CPU Baker。需确保纹理数据在场景加载后可访问。
3. **OMM Baker 内存**：Bistro 场景 alpha-tested 几何体数量有限，CPU Bake 开销可控（<100ms 一次）。
4. **IB/UV 缓冲区复用**：OMM Baker 需要的 index/UV buffer 应与 BLAS 使用相同数据，确保三角形到微三角的映射一致。
5. **NVRHI API 稳定性**：`rt::OpacityMicromapDesc` 和 `rt::GeometryTriangles` 的 OMM 相关 setter 方法已在 OMM SDK 和 RTXNS 项目共享的 NVRHI 中验证存在。
