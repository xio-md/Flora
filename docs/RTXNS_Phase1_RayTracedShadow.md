# Phase 1: 基础光线追踪不透明阴影管线

## 1. 范围与边界

### Phase 1 做什么
- 为 RTXNS 构建完整的 Vulkan 光线追踪管线底层设施 (BLAS, TLAS, RT Pipeline, SBT)
- 实现单光线不透明硬阴影查询 (每像素 1 ray, `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH`)
- 阴影结果合成到现有 ForwardShadingPass 输出

### Phase 1 不做什么
- ❌ OMM 烘焙与集成 (Phase 2)
- ❌ 透明阴影 / Alpha test 阴影光线 (Phase 3)
- ❌ 阴影模糊、降噪、棋盘格 (Phase 3-4)
- ❌ Python 侧的阴影参数 (Phase 4)
- ❌ 多光线 / 软阴影 / 面积光源
- ❌ Host 侧环境变量读取

---

## 2. 前置依赖

### 2.1 必须完成的准备工作

```powershell
# 1. 拉取 Donut 子模块 (当前为空目录!)
cd D:\RTXNS
git submodule update --init --recursive

# 2. 验证 Donut 编译
cmake -S . -B build
cmake --build build --config Release --target DonutRenderPyNative
```

### 2.2 环境要求

| 依赖 | 版本/路径 | 状态 |
|------|----------|------|
| Vulkan SDK | `C:\VulkanSDK\1.4.350.0` | ✅ 已安装 |
| GPU | NVIDIA RTX 5080 (Vulkan 1.4) | ✅ 支持 RT + OMM |
| CMake | VS2022 自带 / 独立安装 | ✅ |
| Donut | `D:\RTXNS\external\donut` | ❌ **需要 git submodule update** |
| NVRHI (Donut 子模块) | `D:\RTXNS\external\donut\nvrhi\` | ❌ 随 Donut 拉取 |
| DXC (ShaderMake 自动下载) | 自动 | ✅ |

### 2.3 Vulkan RT 扩展 (Phase 1 需要)

```cpp
VK_KHR_acceleration_structure       // BLAS + TLAS
VK_KHR_ray_tracing_pipeline         // RT Pipeline
VK_KHR_deferred_host_operations     // AS 构建辅助
VK_KHR_buffer_device_address        // GPU 指针 (AS 构建需要)
VK_EXT_descriptor_indexing          // Bindless (Donut 已启用)
```

---

## 3. 新增文件清单

```
src/RayTracedShadow/
├── CMakeLists.txt                     # 新 static library: rtxns_ray_traced_shadow
├── RayTracedShadowPass.h              # 主 Pass: init/create/destroy/render
├── RayTracedShadowPass.cpp            # RT Pipeline + SBT + dispatch 实现
├── AccelerationStructure.h            # BLAS/TLAS 构建接口
├── AccelerationStructure.cpp          # AS 构建实现
├── SceneGeometryProvider.h            # 从 Donut Scene 提取几何接口
├── SceneGeometryProvider.cpp          # 遍历 Donut Scene → GPU Buffer
├── ShadowResources.h                  # 阴影专用资源 (image, buffer, desc)
├── ShadowResources.cpp
├── ShadowTypes.h                      # ShadowConstants, GPUMesh 结构体
└── shaders/
    ├── CMakeLists.txt                 # Shader 编译规则 (HLSL → SPIRV)
    ├── shadow_raygen.hlsl             # Ray Generation: 主阴影光线
    ├── shadow_rmiss.hlsl              # Ray Miss: 无遮挡 → 1.0
    ├── shadow_rchit.hlsl              # Closest Hit: 有遮挡 → 0.0
    ├── shadow_types.hlsl              # HLSL 侧共享结构体
    └── shadow_composite_cs.hlsl       # Compute: 将 shadow 图合成到 lit 图
```

---

## 4. 修改文件清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `D:\RTXNS\src\CMakeLists.txt` | 修改 | 添加 `add_subdirectory(RayTracedShadow)` |
| `D:\RTXNS\src\PythonBindings\CMakeLists.txt` | 修改 | 链接 `rtxns_ray_traced_shadow` |
| `D:\RTXNS\src\PythonBindings\headless_pbr.h` | 修改 | 新增 `enable_rt_shadows()`, `set_shadow_light()` |
| `D:\RTXNS\src\PythonBindings\headless_pbr.cpp` | 修改 | 构造/析构 `RayTracedShadowPass`，`render_frame()` 中调用 |
| `D:\RTXNS\src\PythonBindings\py_bindings_common.h` | 修改 | pybind11 绑定新方法 |
| `D:\RTXNS\python\rtxns_genesis_style\renderer.py` | 修改 | 新增 `enable_rt_shadows` 参数 (Phase 4 完整暴露) |

---

## 5. 数据结构设计

### 5.1 Host 侧: `ShadowTypes.h`

```cpp
// 对齐到 GPU 的 ShadowData (参考: D:\niagara\src\shaders\shadow.comp.glsl:26-35)
struct alignas(16) ShadowConstants {
    dm::float3 sunDirection;   // 太阳方向 (世界空间)
    float sunJitter;           // Phase 1: 0.0f (无 jitter)
    dm::float4x4 invViewProj;  // 从 depth 重建世界位置
    dm::float2 imageSize;      // 输出分辨率
    uint32_t shadowEnabled;    // 1 = 启用
    uint32_t pad;
};

// GPU 侧 Mesh 描述 (参考: D:\niagara\src\shaders\mesh.h 的 GPU Mesh 结构体)
struct GPUMesh {
    uint32_t vertexOffset;     // 顶点缓冲区中的字节偏移
    uint32_t indexOffset;      // 索引缓冲区中的字节偏移
    uint32_t indexCount;       // 三角形索引数
    uint32_t materialIndex;    // 材质索引 (Phase 1 仅用于 get opaque/transparent)
};
```

### 5.2 HLSL 侧: `shadow_types.hlsl`

```hlsl
// 与 C++ 侧 ShadowConstants 字节兼容
struct ShadowConstants {
    float3 sunDirection;
    float  sunJitter;
    float4x4 invViewProj;
    float2 imageSize;
    uint    shadowEnabled;
    uint    pad;
};

// Ray Payload (Phase 1: 仅 float)
struct ShadowPayload {
    float shadowFactor;  // 1.0 = 无遮挡, 0.0 = 完全遮挡
};

// 所有 shader 共享的 Bindless binding
[[vk::binding(0, 0)]] RaytracingAccelerationStructure Scene : register(t0);
[[vk::binding(1, 0)]] ConstantBuffer<ShadowConstants> c_shadow : register(b0);
[[vk::binding(2, 0)]] Texture2D<float> t_depth : register(t1);       // 从 raster pass 读取
[[vk::binding(3, 0)]] RWTexture2D<float> t_shadow : register(u0);   // R8_UNORM 输出
```

---

## 6. 渲染管线变更

### 6.1 当前 `render_frame()` 流程

> 参考: `D:\RTXNS\src\PythonBindings\headless_pbr.cpp:383-470`

```
1. Scene::Refresh                     // 蒙皮/动画
2. Clear color (black)
3. Clear depth (1.0)
4. ForwardShadingPass::PrepareLights  // 设置光照常量
5. RenderCompositeView (opaque)       // 光栅化不透明
6. RenderCompositeView (transparent)  // 光栅化半透明
7. Copy → staging → readback → RAM
```

### 6.2 改造后 `render_frame()` 流程

```
1. Scene::Refresh
2. [新增] Update TLAS instance transforms   // ★ 每帧更新
3. [新增] Clear shadow target (R8_UNORM 1.0)
4. [新增] Dispatch Shadow RT pass           // ★ TraceRay 阴影光线
   ├── 从 t_depth (现有 depth buffer) 重建世界位置
   ├── 向 sunDirection 发射单光线
   └── 写入 t_shadow (R8_UNORM)
5. [新增] Shadow composite compute pass     // ★ 将 shadow 乘到 lit 图像
6. [修改] ForwardShadingPass (opaque, 或后处理合成)
7. [修改] ForwardShadingPass (transparent)
8. Copy → staging → readback → RAM
```

**合成策略选择**: Phase 1 采用**后处理合成** (draw color → multiply by shadow)，
因为修改 Donut 的 `forward_ps.hlsl` 增加采样器绑定比较复杂。
具体做法：在 ForwardShadingPass 输出后，用一个 compute shader 将 `t_shadow` 乘到 color 输出的 RGB 通道。

---

## 7. 核心实现细节

### 7.1 设备初始化 (RT 扩展启用)

> 参考: `D:\niagara\src\device.cpp:283-408` (OMM/RT 特性启用)

```cpp
// headless_pbr.cpp 中 DeviceManager 创建后，启用 RT 扩展:
nvrhi::vulkan::DeviceHandle vkDevice = ...;

// 查询 RT 管线属性
VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipelineProps = {
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR
};
VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps = {
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
    &rtPipelineProps
};
VkPhysicalDeviceProperties2 props2 = {
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    &asProps
};
vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

// 关键限制参数 (RTX 5080 典型值):
// - rtPipelineProps.shaderGroupHandleSize = 32
// - rtPipelineProps.maxRayRecursionDepth >= 1
// - asProps.minAccelerationStructureScratchOffsetAlignment = 256
```

### 7.2 BLAS 构建 (`AccelerationStructure.cpp`)

> 参考: `D:\niagara\src\scenert.cpp:16-180` (buildBLAS)

```cpp
struct BuiltBLAS {
    nvrhi::BufferHandle vertexBuffer;      // GPU: 所有顶点位置 (float3)
    nvrhi::BufferHandle indexBuffer;       // GPU: 所有三角形索引 (uint32_t)
    nvrhi::rt::AccelSetHandle handle;      // Vulkan BLAS
};

BuiltBLAS buildBLAS(
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    const std::vector<GPUMesh>& meshes,
    const std::vector<dm::float3>& allVertices,
    const std::vector<uint32_t>& allIndices)
{
    // 1. 上传顶点/索引到 GPU
    auto vb = device->createBuffer(...);
    auto ib = device->createBuffer(...);

    // 2. 按 mesh 构建几何体描述
    std::vector<nvrhi::rt::GeometryDesc> geometryDescs;
    for (const auto& mesh : meshes) {
        nvrhi::rt::GeometryDesc geo;
        geo.type = nvrhi::rt::GeometryType::Triangles;
        geo.triangles.vertexBuffer = vb;
        geo.triangles.vertexOffset = mesh.vertexOffset;
        geo.triangles.vertexFormat = nvrhi::Format::RGB32_FLOAT;
        geo.triangles.vertexStride = sizeof(dm::float3);
        geo.triangles.vertexCount = /* 从 mesh 的 vertex range 计算 */;
        geo.triangles.indexBuffer = ib;
        geo.triangles.indexOffset = mesh.indexOffset;
        geo.triangles.indexFormat = nvrhi::Format::R32_UINT;
        geo.triangles.indexCount = mesh.indexCount;
        // Phase 2: geo.triangles.opacityMicromap = &ommAttachment;
        geometryDescs.push_back(geo);
    }

    // 3. 创建 BLAS
    nvrhi::rt::AccelSetDesc blasDesc;
    blasDesc.type = nvrhi::rt::AccelSetType::BottomLevel;
    blasDesc.geometryDescs = geometryDescs.data();
    blasDesc.geometryCount = geometryDescs.size();
    blasDesc.buildFlags =
        nvrhi::rt::BuildFlags::PreferFastTrace |
        nvrhi::rt::BuildFlags::AllowCompaction;  // 压缩可选项

    nvrhi::rt::AccelSetHandle blas = device->createAccelSet(blasDesc);

    // 4. 构建
    commandList->begin();
    commandList->buildAccelSet(blas);
    commandList->end();
    device->executeCommandList(commandList);

    return { vb, ib, blas };
}
```

### 7.3 TLAS 构建与更新 (`AccelerationStructure.cpp`)

```cpp
struct BuiltTLAS {
    nvrhi::BufferHandle instanceBuffer;
    nvrhi::rt::AccelSetHandle handle;
};

BuiltTLAS buildTLAS(
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    const BuiltBLAS& blas,
    const std::vector<donut::engine::DrawItem>& drawItems,
    bool isUpdate)  // Phase 1: 首次为 false，后续帧为 true
{
    // 1. 构建实例描述
    // 参考: D:\niagara\src\scenert.cpp:16-180 的实例循环
    // 参考: D:\niagara\src\scenert.cpp:516 的 instance.flags 设置
    std::vector<nvrhi::rt::InstanceDesc> instances;

    uint32_t customIndex = 0;
    for (const auto& item : drawItems) {
        nvrhi::rt::InstanceDesc instance;
        instance.bottomLevelAS = blas.handle;
        // Donut 的 transform 是 dm::float4x4 → 提取 3×4
        instance.transformRow0 = dm::float3(item.transform[0]);
        instance.transformRow1 = dm::float3(item.transform[1]);
        instance.transformRow2 = dm::float3(item.transform[2]);
        instance.instanceID = customIndex++;
        instance.instanceMask = 0xff;  // Phase 1: 全部可见
        // Phase 2: 非透明 mesh 设为 FORCE_OPAQUE_BIT
        //   instance.flags = item.isTransparent ? 0 :
        //       VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
        instance.hitGroupIndex = 0;  // 仅一个 hit group
        instances.push_back(instance);
    }

    // 2. 创建/更新 TLAS
    nvrhi::rt::AccelSetDesc tlasDesc;
    tlasDesc.type = nvrhi::rt::AccelSetType::TopLevel;
    tlasDesc.instanceDescs = instances.data();
    tlasDesc.instanceCount = instances.size();
    tlasDesc.buildFlags =
        nvrhi::rt::BuildFlags::PreferFastTrace |
        nvrhi::rt::BuildFlags::AllowUpdate;  // ★ 关键: 允许帧间更新

    nvrhi::rt::AccelSetHandle tlasHandle;
    if (isUpdate) {
        // 更新: 原地修改已分配的 TLAS
        // 注意: NVRHI 的 updateAccelSet 需要原 AS 以 AllowUpdate 标志创建
        commandList->updateAccelSet(tlasHandle, tlasDesc);
    } else {
        tlasHandle = device->createAccelSet(tlasDesc);
        commandList->buildAccelSet(tlasHandle);
    }

    return { instanceBuffer, tlasHandle };
}
```

### 7.4 RT Pipeline 创建 (`RayTracedShadowPass.cpp`)

```cpp
struct RayTracedShadowPipeline {
    nvrhi::rt::PipelineHandle pipeline;
    nvrhi::rt::ShaderTableHandle shaderTable;
    nvrhi::BindingSetHandle bindingSet;
};

RayTracedShadowPipeline createShadowPipeline(
    nvrhi::IDevice* device,
    donut::engine::ShaderFactory* shaderFactory)
{
    // 1. 编译/加载 shader (通过 Donut 的 ShaderFactory)
    //    ShaderMake 将 HLSL 转为 SPIRV
    auto rgen = shaderFactory->CreateShader("app/RTXNS/shaders/shadow_raygen.hlsl", "main", ...);
    auto rmiss = shaderFactory->CreateShader("app/RTXNS/shaders/shadow_rmiss.hlsl", "main", ...);
    auto rchit = shaderFactory->CreateShader("app/RTXNS/shaders/shadow_rchit.hlsl", "main", ...);

    // 2. Ray Tracing Pipeline
    nvrhi::rt::PipelineDesc pipelineDesc;
    pipelineDesc.maxRecursionDepth = 1;      // 阴影光线不需要递归
    pipelineDesc.maxPayloadSize = sizeof(float);  // ShadowPayload::shadowFactor
    pipelineDesc.maxAttributeSize = sizeof(float2); // 重心坐标

    pipelineDesc.rayGenShaders = { rgen };
    pipelineDesc.missShaders = { rmiss };
    pipelineDesc.hitGroups = {
        { .closestHitShader = rchit },
    };
    pipelineDesc.globalRootSignature = ...; // 见下方 Binding Layout

    auto pipeline = device->createRayTracingPipeline(pipelineDesc);

    // 3. Shader Binding Table (SBT)
    nvrhi::rt::ShaderTableDesc sbtDesc;
    sbtDesc.pipeline = pipeline;
    sbtDesc.rayGenShaderRecord = { rgen, /*data=*/nullptr, /*dataSize=*/0 };
    sbtDesc.missShaderRecords = {
        { rmiss, nullptr, 0 },
    };
    sbtDesc.hitGroupRecords = {
        { rchit, nullptr, 0 },
    };

    auto sbt = device->createShaderTable(sbtDesc);

    return { pipeline, sbt, /*bindingSet=*/... };
}
```

### 7.5 Shader Binding Layout

```
// set=0 (全局, Bindless — Donut 管理)
//   - TLAS (AccelerationStructure)
//   - 顶点/索引缓冲区
//   - 纹理 Samplers

// set=1 (per-pass, 本项目新增)
//   b0: ConstantBuffer<ShadowConstants>
//   t1: Texture2D<float> depth (from raster pass)
//   u0: RWTexture2D<float> shadowOutput (R8_UNORM)

// set=2 (per-material, Donut 管理)
//   - PBR 材质常量
//   - 纹理 Descriptors
```

> 注意: 需研究 Donut 的 DescriptorTableManager 如何管理 bindless，
> 以及如何在自定义 pass 中插入 descriptor set。
> 参考 Donut 示例中自定义 compute pass 的做法。

### 7.6 RT Shader: `shadow_raygen.hlsl`

> 参考: `D:\niagara\src\shaders\shadow.comp.glsl` (GLSL 版本)

```hlsl
// shadow_raygen.hlsl — Phase 1: 单光线不透明硬阴影
#include "shadow_types.hlsl"

[shader("raygeneration")]
void ShadowRayGen()
{
    uint2 pos = DispatchRaysIndex().xy;

    // 检查是否在有效区域内(处理非 8 对齐的情况，影射到 8×8 workgroup)
    if (pos.x >= c_shadow.imageSize.x || pos.y >= c_shadow.imageSize.y)
        return;

    // Step 1: 从 depth buffer 重建世界空间位置
    // 参考: D:\niagara\src\shaders\shadow.comp.glsl:136-141
    float depth = t_depth[pos];
    float2 uv = (float2(pos) + 0.5) / c_shadow.imageSize;
    float4 clip = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    float4 world = mul(c_shadow.invViewProj, clip);
    float3 wpos = world.xyz / world.w;

    // Step 2: 设置阴影光线
    RayDesc ray;
    ray.Origin = wpos + c_shadow.sunDirection * 1e-2;  // offset 避免自阴影
    ray.Direction = c_shadow.sunDirection;
    ray.TMin = 0.0;
    ray.TMax = 1e3;  // 1000 世界单位

    // Step 3: 发射光线
    // RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH → 命中即停止
    // RAY_FLAG_SKIP_CLOSEST_HIT_SHADER → 不执行 closest hit, 直接返回命中/未命中
    //   (Phase 1 使用后者更高效: 不需要 shader 执行，仅需要命中判定)
    ShadowPayload payload;
    payload.shadowFactor = 1.0;  // 默认无遮挡

    TraceRay(
        Scene,                                          // TLAS
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,       // 首次命中即停止
        0xff,                                           // instanceMask
        0,                                              // hitGroupIndex
        0,                                              // missShaderIndex
        0,                                              // missShaderIndex (2nd)
        ray,
        payload
    );

    t_shadow[pos] = payload.shadowFactor;
}
```

### 7.7 RT Shader: `shadow_rmiss.hlsl`

```hlsl
// shadow_rmiss.hlsl
#include "shadow_types.hlsl"

[shader("miss")]
void ShadowMiss(inout ShadowPayload payload)
{
    // 光线未命中任何几何体 = 无遮挡 → 保持 1.0
    // payload.shadowFactor 已在 raygen 中初始化为 1.0
}
```

### 7.8 RT Shader: `shadow_rchit.hlsl`

```hlsl
// shadow_rchit.hlsl — Phase 1: 任何命中 = 遮挡
#include "shadow_types.hlsl"

[shader("closesthit")]
void ShadowClosestHit(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attrib)
{
    // Phase 1: 任何命中 = 完全遮挡
    payload.shadowFactor = 0.0;

    // Phase 3 扩展:
    // - 读取 mesh material
    // - 如果有 alpha texture: 纹理采样 + alpha test (阈值 0.5)
    // - 如果有 OMM: 硬件自动处理，chit 仅在 Unknown 状态时触发
}
```

### 7.9 Shadow Composite CS: `shadow_composite_cs.hlsl`

```hlsl
// shadow_composite_cs.hlsl
// Phase 1: 后处理方案 — 将 shadow 图直接乘到 lit 图像上

Texture2D<float4> t_litColor : register(t0);  // ForwardShadingPass 输出
Texture2D<float>  t_shadow   : register(t1);
RWTexture2D<float4> u_output : register(u0);

[numthreads(8, 8, 1)]
void main(uint2 pos : SV_DispatchThreadID)
{
    float shadow = t_shadow[pos];

    // 简单合成: lit * shadow
    // 参考: D:\niagara\src\shaders\final.comp.glsl:65-73
    // Niagara 保留了 ambient floor: min(shadow + 0.05, 1.0)
    // Phase 1 先做最简单的乘法
    float4 color = t_litColor[pos];
    color.rgb *= shadow;

    u_output[pos] = color;
}
```

### 7.10 Shadow Pass Dispatch (`RayTracedShadowPass.cpp`)

```cpp
void RayTracedShadowPass::render(
    nvrhi::ICommandList* commandList,
    const dm::float3& sunDirection,
    const dm::float4x4& inverseViewProjection,
    const nvrhi::TextureHandle& depthTexture,    // 从 raster pass 获得
    const nvrhi::TextureHandle& shadowTexture,   // 目标 R8_UNORM
    uint32_t width,
    uint32_t height)
{
    // 1. 填充常量缓冲区
    ShadowConstants constants;
    constants.sunDirection = sunDirection;
    constants.sunJitter = 0.0f;  // Phase 1: 无 jitter
    constants.invViewProj = inverseViewProjection;
    constants.imageSize = dm::float2(float(width), float(height));
    constants.shadowEnabled = 1;

    // 2. 更新 Descriptor Set
    // Binding 1: constants (b0)
    // Binding 2: depth (t1)
    // Binding 3: shadow output (u0)
    // Binding 0: TLAS (通过 Donut bindless manager)
    commandList->writeBuffer(m_shadowConstantBuffer, &constants, sizeof(constants));
    commandList->setComputeBindings(m_shadowBindingSet); // 或 RT bindings

    // 3. Dispatch
    nvrhi::rt::DispatchRaysDesc dispatch;
    dispatch.pipeline = m_pipeline.pipeline;
    dispatch.shaderTable = m_pipeline.shaderTable;
    dispatch.width = width;
    dispatch.height = height;
    dispatch.depth = 1;

    commandList->dispatchRays(dispatch);

    // 4. 可选: Barrier 确保 shadowTexture 写入完成
    // (NVRHI 可能在 dispatchRays 后自动插入)
}
```

---

## 8. 与 Donut 的集成点

### 8.1 场景几何提取 (`SceneGeometryProvider.cpp`)

```cpp
// 从 Donut Scene 中提取所有 mesh 的几何数据，构建 BLAS 所需的 vertex/index buffer
// 参考 Donut 内部: donut/engine/Scene.h 的 GetMesh() / GetDrawItems()

struct SceneGeometry {
    std::vector<dm::float3> vertices;    // 所有 mesh 的顶点（合并）
    std::vector<uint32_t> indices;       // 所有 mesh 的索引（合并）
    std::vector<GPUMesh> meshes;         // 每个 mesh 的 range 描述
};

SceneGeometry extractSceneGeometry(const donut::engine::Scene* scene) {
    SceneGeometry result;
    // 遍历 Scene::GetDrawItems() 对应 mesh 的 geometry
    // 注意 Donut 内部可能使用 BufferGroup 管理几何数据
    // 需要找到正确的 vertex/index buffer 来源
    // 参考: Donut 的 GBufferFillPass 如何遍历几何体
    ...
}
```

### 8.2 HeadlessPbrScene 集成

```cpp
// D:\RTXNS\src\PythonBindings\headless_pbr.h 新增:
class HeadlessPbrScene {
    // ... 现有成员 ...
public:
    void enable_rt_shadows(bool enable);
    void set_shadow_light(
        const std::array<float, 3>& direction,
        const std::array<float, 3>& color,
        float irradiance);
private:
    std::unique_ptr<RayTracedShadowPass> m_rtShadowPass;
    bool m_rtShadowsEnabled = false;
    // ...
};
```

---

## 9. 参考资源完整列表

### 9.1 RT Pipeline 创建参考

| 来源 | 文件 | 内容 |
|------|------|------|
| Niagara | `D:\niagara\src\device.cpp:283-408` | Vulkan RT/OMM 特性启用 |
| Niagara | `D:\niagara\src\nagara.cpp:700-758` | 阴影管线创建 (specialization constants, 两套 variant) |
| Niagara | `D:\niagara\src\scenert.cpp:16-180` | BLAS 构建 (含几何体遍历、OMM 附着) |
| Niagara | `D:\niagara\src\scenert.cpp:100-180` | TLAS 构建 (实例循环、transform 设置) |
| NVRHI Doc | nvrhi 官方示例 | `nvrhi::rt::AccelSetDesc`, `nvrhi::rt::PipelineDesc` |

### 9.2 Shader 参考

| 来源 | 文件 | 行号 | 内容 |
|------|------|------|------|
| Niagara | `D:\niagara\src\shaders\shadow.comp.glsl` | `1-161` | 完整阴影 RT 主 shader (GLSL) |
| Niagara | `D:\niagara\src\shaders\shadow.comp.glsl` | `26-35` | ShadowData 结构体 |
| Niagara | `D:\niagara\src\shaders\shadow.comp.glsl` | `78-84` | `shadowTrace()` — 不透明光线 |
| Niagara | `D:\niagara\src\shaders\shadow.comp.glsl` | `136-141` | 从 depth 重建世界位置 |
| Niagara | `D:\niagara\src\shaders\shadow.comp.glsl` | `143-151` | Gradient noise jitter |
| Niagara | `D:\niagara\src\shaders\math.h` | 全文件 | `gradientNoise()` 实现 |
| Niagara | `D:\niagara\src\shaders\mesh.h` | 全文件 | GPU 侧 Mesh 结构体 |

### 9.3 Shadow 合成参考

| 来源 | 文件 | 行号 | 内容 |
|------|------|------|------|
| Niagara | `D:\niagara\src\shaders\final.comp.glsl` | `65-73` | 阴影 × 光照合成 (ambient floor=0.05) |
| Niagara | `D:\niagara\src\nagara.cpp` | `1885-1903` | Host 侧 ShadeData 填充 |

### 9.4 Donut 集成参考

| 来源 | 文件 | 内容 |
|------|------|------|
| RTXNS | `D:\RTXNS\src\PythonBindings\headless_pbr.cpp` | 现有渲染循环（需修改的关键文件） |
| RTXNS | `D:\RTXNS\src\PythonBindings\headless_pbr.cpp:177` | ForwardShadingPass 创建 |
| RTXNS | `D:\RTXNS\src\PythonBindings\headless_pbr.cpp:383-470` | render_frame() 完整实现 |
| RTXNS | `D:\RTXNS\src\PythonBindings\headless_pbr.cpp:100-205` | DeviceManager / NVRHI 初始化 |
| Donut | `external/donut/include/donut/engine/Scene.h` | Scene 类 API (拉取后) |
| Donut | `external/donut/include/donut/render/DrawStrategy.h` | DrawStrategy 遍历 |

---

## 10. 验证计划

### 10.1 单元验证

| 验证项 | 方法 | 预期 |
|--------|------|------|
| RT 扩展可用 | Vulkan validation layer 确认特性已启用 | 无错误 |
| BLAS 构建 | 无几何体场景 → 空 BLAS; 有几何体 → 正确大小 | `buildAccelSet` 成功 |
| TLAS 构建 | 单实例场景 → TLAS; 多实例场景 → TLAS | `createAccelSet` 成功 |
| RT Pipeline 创建 | 加载 HLSL shader → 创建 pipeline | `createRayTracingPipeline` 成功 |
| Shadow dispatch | 无几何体 → shadowTarget = 全白 (1.0) | readback 验证全部 ~255 |
| 遮挡验证 | 单平面 → 平面后方像素黑 (0.0)，前方白 (1.0) | readback 对比 |
| 无阴影回退 | `m_rtShadowsEnabled = false` | 输出与当前完全一致 |

### 10.2 性能验证

| 验证项 | 目标 |
|--------|------|
| BLAS 构建 | < 50ms 首次 |
| TLAS 更新 (每帧) | < 1ms |
| Shadow RT pass (1920×1080) | < 5ms |
| Composite pass | < 0.5ms |

### 10.3 回归验证

- `D:\RTXNS\samples\DonutRenderPyDemo\donut_render_demo_v0_5.py` 无变化
- `D:\RTXNS\samples\GenesisStylePy\genesis_style_example.py` 无变化 (RT 阴影默认关闭)
- readback 输出格式不变 (RGBA8, shape=[H, W, 4])

---

## 11. 时间估算

| 任务 | 时间 | 依赖 |
|------|------|------|
| Donut 子模块初始化 + 编译验证 | 0.5d | 无 |
| `SceneGeometryProvider` (Donut Scene → GPU Buffers) | 1d | Donut 编译通过 |
| `AccelerationStructure` (BLAS + TLAS) | 1.5d | SceneGeometryProvider |
| `ShadowResources` (image, buffer, desc 管理) | 0.5d | 无 |
| RT Shader 编写 (HLSL 4 个) | 0.5d | 无 |
| RT Pipeline + SBT 创建 | 1d | Shader 编译通过 |
| `RayTracedShadowPass::render()` 主流程 | 0.5d | Pipeline + TLAS |
| 集成到 `headless_pbr.cpp` render_frame | 0.5d | 全部上述 |
| 测试 + 修复 + 验证 | 1d | 集成完成 |
| **总计** | **~7d** | |

---

## 12. 风险与注意事项

1. **NVRHI RT API 封装层**: Donut 的 NVRHI 可能不完全暴露所有 Vulkan RT 功能。如果 NVRHI 的 RT 抽象不够用，可能需要直接使用 Vulkan API（类似 Niagara 的做法）。
2. **Donut 几何数据获取**: Donut 的 Scene 类内部可能使用 BufferGroup，顶点/索引数据可能不是连续布局。需要研究 Donut 如何将几何数据提供给 GBufferFillPass。
3. **Shader 编译**: Phase 1 需要 4 个 HLSL shader。需要配置 ShaderMake 或直接在 CMake 中添加编译规则将 HLSL 转为 SPIRV。
4. **Bindless 集成**: Donut 使用全局 descriptor heap (VK_EXT_descriptor_heap)。需要在自定义 pass 中正确接入 bindless 系统。
5. **Shadow 质量**: Phase 1 的单光线会产生锯齿状的硬阴影边缘。这是预期的，后续 Phase 会通过 OMM + blur 改善。
