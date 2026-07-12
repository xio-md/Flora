# RTXNS 并行渲染效率推进文档

> 日期: 2026-07-11，更新: 2026-07-12（附实测伸缩数据）
> 目标: 只关注并行渲染吞吐、GPU 利用率、显存效率和调度开销。不讨论渲染质量、材质/光照效果提升。

## 0. Week 1-3 完成状态与 ReplicaCAD 实测伸缩曲线

### 0.1 已完成

| Week | 内容 | 状态 |
|------|------|------|
| Week 1 | CameraDesc + RenderViewSlot + 多相机 API | ✅ |
| Week 2 | `render_frame_batch_v2()` 单 cmdList 批量 + 共享 BLAS/TLAS | ✅ |
| Week 3 | `submit_frame_batch()` / `read_frame_batch()` 异步 + EventQuery | ✅ |
| — | P0 Ring 占用保护 + 可配置 depth (K=4) + hash 校验 | ✅ |

### 0.2 ReplicaCAD 实际伸缩数据

测试场景: `Stage_v3_sc0_staging.glb`，RT 阴影 OFF，GPU: 单卡 Vulkan。**统一使用 async 端到端吞吐（submit+wait+read）**，submit-only 不反映观测生成速度。

**256×192:**

| N | sync cam-FPS | async cam-FPS | batch ms | per-cam FPS | async/sync |
|---|-------------|---------------|----------|-------------|------------|
| 1 | 3,410 | **5,924** | 0.17 | 5,924 | 1.74× |
| 2 | 5,294 | **6,326** | 0.32 | 3,163 | 1.20× |
| 4 | 6,228 | **8,767** | 0.46 | 2,192 | 1.41× |
| 8 | 5,295 | **7,885** | 1.01 | 986 | 1.49× |
| 12 | 6,666 | **9,635** | 1.25 | 803 | 1.45× |

**512×384（K=8 伸缩扫描）:**

| N | async cam-FPS | batch ms |
|---|---------------|----------|
| 4 | 1286 | 3.11 |
| async e2e N=4 vs sync batch N=4 加速: 1.26× |

**Ring 深度伸缩 (512×384, N=4 async e2e):**

| K | cam-FPS |
|---|---------|
| 2 | 1,333 |
| 4 | **1,557** |
| 8 | 1,481 |

### 0.3 关键发现（修订）

1. **N≥8 的 per-cam FPS 下降是 GPU 工作量随相机数线性增长的正常现象**，不是"单 cmdList 没拆开"造成的。同一 Graphics queue 上拆成多个 cmdList 不会让不同相机的 raster 真正并行——仍是顺序执行。
2. **P2（同 queue 多 cmdList）已否决**。NVRHI 确实支持 `executeCommandLists()` 批量提交，但同一 queue 的 cmdList 按提交顺序执行，不产生 GPU 并行。
3. **K=4 是 ring depth 最优值**，在此吞吐最高（1,557 cam-FPS @ 512×384）。K=2 限制流水深度，K=8 无额外收益。
4. **async/sync 比值**：N=1 时 1.96×（CPU 不等待 GPU），N 增大后逐渐降至 1.33×（readback 时间占比上升）。统一用端到端吞吐报告。
5. **`set_readback_ring_depth()` 切换深度时的 bug**：原先只改 `ringDepth` 字段，不重建已有相机的 `readbackRing` vector。切换 K=4→K=8 后访问越界 staging slot 导致 GPU 卡死。已修复：存在 pending token 时拒绝切换 + `waitForIdle()` 后批量重建所有 ring slot。

### 0.3.1 P2 受控否定实验

为区分“多 cmdList”与“多次 submit/query”两类开销，P2 实验将多个 micro-batch 录制成多个 cmdList，再用一次 `executeCommandLists(...)` 提交整个有序序列，仅设置一个 EventQuery。测试条件为 ReplicaCAD、256×192、RT shadow OFF、K=8、5 次交错 trial、每 trial 120 batch；multi-cmdList 与 single-cmdList 输出 hash 一致。

| 条件 | N=12 mb=4 | N=12 mb=2 | N=8 mb=4 | N=8 mb=2 |
|------|-----------|-----------|----------|----------|
| 两次独立受控套件的相对区间 | -1.1% ~ -11.9% | -13.7% ~ -15.7% | -1.6% ~ -13.2% | -16.9% ~ -22.6% |

去除重复 submit/query 后，`mb=4` 的损失有所缩小但没有稳定转正，`mb=2` 持续退化。因此负收益方向是真实的；旧实现只是放大了其幅度。正式路径保持单 cmdList，P2 接口仅用于实验或在提交前移除。

### 0.4 对后续规划的启示（修订）

- **不再追求多 cmdList 并行**。N≥8 吞吐退化的正确解法是降低每相机成本：readback 合批、binding/constant-buffer 复用、减少每相机 barrier。
- Week 4（资源池 + scheduler）能解决 VRAM 稳定性，配合 per-camera 成本优化可进一步提升 N=8/12 吞吐。
- 动态仿真阶段新增 transform/TLAS dirty 更新管线后，用"动态物体数 × 相机数"基准验证，不继续追逐 cmdList 方案。

---

## 1. 结论先行

当前 RTXNS 的低效率根因不是单个 pass 太慢，而是并行粒度和同步边界不对:

1. 当前多进程方案是 N 个独立 renderer 互相抢同一块 GPU。每个进程各自创建 Vulkan device、加载场景、构建 BLAS/TLAS、分配 render target 和 staging readback。它能把 GPU 利用率从单进程欠载拉高，但会带来 device 调度竞争和显存重复。
2. 当前 C++ 单帧路径是一个同步大函数。`HeadlessPbrScene::Impl::render_frame()` 从 scene refresh、raster、RT shadow、composite 到 readback 全部串在一次调用里，最后 `executeCommandList()` 后立即 `waitForIdle()`，CPU/GPU 没有跨帧重叠。
3. 当前资源模型是单相机模型。`m_view`、`m_camera`、`m_framebuffer_factory`、`m_color_target`、`m_depth_target`、`m_readback_target` 都是单份对象，无法在一次 renderer 调用中自然表达 N 个视角。

推荐路线: 先做“单进程、单 device、单 SceneGraph、批量多相机”的 batch renderer，再做异步 readback 和 timeline semaphore，最后考虑多场景/多 GPU。

最小可行目标不是立刻把全部 pass 并行到多队列，而是先把 Python 层的 N 次 `render_frame()` 合并成 C++ 层一次 `render_frame_batch()`，共享 scene、BLAS、shader、texture cache 和 command list 提交边界。

## 2. 当前 RTXNS 的关键瓶颈

### 2.1 同步边界过重

关键代码:

- `src/PythonBindings/headless_pbr.cpp:620`: `HeadlessPbrScene::Impl::render_frame()`
- `src/PythonBindings/headless_pbr.cpp:633`: 每帧创建并 open 一个 command list
- `src/PythonBindings/headless_pbr.cpp:1148`: close command list
- `src/PythonBindings/headless_pbr.cpp:1149`: `device->executeCommandList(command_list)`
- `src/PythonBindings/headless_pbr.cpp:1150`: `device->waitForIdle()`
- `src/PythonBindings/headless_pbr.cpp:1154`: `mapStagingTexture()` 同步读回

这意味着每一帧都强制 GPU 清空队列，再让 CPU 读 staging texture。即使单帧 GPU work 很小，也无法通过队列积累工作量提高占用。

推进判断:

- P0 阶段可以保留最终同步读回，但要把 N 个 camera 的工作合入一个 command list，降低 `waitForIdle()` 次数。
- P1 阶段把 readback 改成 ring-buffer staging，返回上一帧或前 K 帧结果，让 render submit 和 CPU copy 解耦。
- EventQuery 已完成稳态 `waitForIdle()` 的替代；同 queue 的 micro-cmdList 已通过 P2 实验否决，不再规划 timeline semaphore 作为该问题的解法。

### 2.2 单相机资源模型

关键代码:

- `src/PythonBindings/headless_pbr.cpp:637`: `m_framebuffer_factory->GetFramebuffer(m_view)`
- `src/PythonBindings/headless_pbr.cpp:1310`: 只创建一个 `FramebufferFactory`
- `src/PythonBindings/headless_pbr.cpp:1320`: `m_framebuffer_factory`
- `src/PythonBindings/headless_pbr.cpp:1321-1323`: 单份 color/depth/readback target
- `src/PythonBindings/headless_pbr.cpp:1324-1325`: 单份 `PlanarView` 和 `FirstPersonCamera`

推进判断:

把这些字段收进 `RenderViewSlot`:

```cpp
struct RenderViewSlot
{
    CameraDesc desc;
    donut::app::FirstPersonCamera camera;
    PlanarView view;

    std::shared_ptr<FramebufferFactory> framebufferFactory;
    nvrhi::TextureHandle colorTarget;
    nvrhi::TextureHandle depthTarget;
    nvrhi::TextureHandle litColorSRV;
    nvrhi::TextureHandle shadowTarget;
    nvrhi::TextureHandle shadowBlurTemp;
    nvrhi::TextureHandle compositeOutput;

    static constexpr uint32_t kReadbackLag = 2;
    std::array<nvrhi::StagingTextureHandle, kReadbackLag> readbackTargets;
    uint64_t frameSerial = 0;
    HeadlessPbrScene::FrameStats lastStats;
};
```

P0 先让 `std::vector<RenderViewSlot> m_views` 支持固定分辨率或同分辨率多相机。P1 再支持不同分辨率，避免一开始把资源池复杂化。

### 2.3 AS 构建和资源上传仍有多处 wait

关键代码:

- `src/RayTracedShadow/SceneGeometryProvider.cpp:248-249`: combined vertex/index buffer 上传后 `waitForIdle()`
- `src/RayTracedShadow/SceneGeometryProvider.cpp:356-357`: metadata buffer 上传后 `waitForIdle()`
- `src/RayTracedShadow/AccelerationStructure.cpp:130-131`: BLAS build 后 `waitForIdle()`
- `src/RayTracedShadow/AccelerationStructure.cpp:205-206`: TLAS build 后 `waitForIdle()`
- `src/RayTracedShadow/OMMBaker.cpp:225-227`: alpha texture copy 后 `waitForIdle()`
- `src/PythonBindings/headless_pbr.cpp:939`: OMM bake 流程中仍有 `waitForIdle()`
- `src/PythonBindings/headless_pbr.cpp:978-979`: 首帧 TLAS build 后 `waitForIdle()`

推进判断:

这些同步主要集中在加载或首帧建 AS，不是稳态 batch 的第一瓶颈。P0 阶段只保证 BLAS 在所有 camera 间共享，先不重写所有初始化同步。P1/P2 再把初始化上传合并到一个 setup command list，并通过 fence 等待一次。

### 2.4 RT shadow pass 的批处理注意点

关键代码:

- `src/RayTracedShadow/RayTracedShadowPass.cpp:226`: `m_shadowBindingSet` 会在 `renderShadow()` 内被覆盖
- `src/RayTracedShadow/RayTracedShadowPass.cpp:275`: `m_compositeBindingSet` 会在 `compositeShadow()` 内被覆盖
- `src/RayTracedShadow/RayTracedShadowPass.cpp:316`: `m_blurBindingSet` 会在 `blurShadow()` 内被覆盖
- `src/RayTracedShadow/RayTracedShadowPass.cpp:191` 和 `307`: 所有 camera 共用 `m_shadowConstantBuffer`

这在单相机同步路径里问题不明显，但在一个 command list 中连续录制多个 camera 时，绑定集 handle 和常量 buffer 的生命周期要非常小心。

推进判断:

- P0 批量录制时，先让每个 view slot 持有本帧 binding set 引用，至少保证 `executeCommandList()` 前不会释放。
- 更稳的做法是给 `RenderViewSlot` 分配独立 constant buffer，避免多个 camera 重写同一个 `m_shadowConstantBuffer`。
- `RayTracedShadowPass` 应逐步改成“pipeline/layout 共享，per-dispatch binding/constant 由调用方或 frame scratch 管理”的形态。

## 3. 成熟 renderer 可借鉴模式

### 3.1 SAPIEN: batch 是一等对象，不是多进程补丁

参考文件:

- `E:\cplus\SAPIEN\include\sapien\sapien_renderer\batched_render_system.h`
- `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cpp`
- `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cu`

关键设计:

1. `BatchedCamera` 显式持有一组 camera。见 `batched_render_system.h:26-54`，它不是通过 Python 多进程拼出来的，而是 C++ renderer 内部的 batch object。
2. 每个 batch 有自己的 command pool/command buffer、输出 buffer、timeline semaphore。见 `batched_render_system.h:43-53`。
3. `BatchedCamera::takePicture()` 会先等待上一帧 semaphore，再对所有 camera 调 render，最后提交一个 command buffer 并 signal timeline semaphore。见 `batched_render_system.cpp:127-148`。
4. `BatchedRenderSystem` 固定 scene version。创建 batch 后不允许随意增删 scene/camera，从而保证 GPU buffer layout 稳定。见 `batched_render_system.cpp:303-313`。
5. transform 和 camera 更新通过 GPU kernel 批量写入 buffer。见 `batched_render_system.cu:56-124`。
6. CUDA/Vulkan 外部同步用 timeline semaphore，而不是全局 `waitIdle()`。见 `batched_render_system.cpp:238-255` 和 `342-351`。

对 RTXNS 的迁移:

- 增加 `HeadlessPbrScene::create_camera_batch()` 或简单版 `add_camera()` + `render_frame_batch()`。
- batch 创建后限制结构性变更: 不允许在 batch 活跃期间重新 `load_scene()` 或改变 camera 数量，除非显式 `rebuild_batch()`。
- 相机更新先从 CPU 写 `PlanarView` 开始，后续再做 camera constant buffer 批量上传。
- 输出不要一开始就强制回 CPU。先提供 `render_frame_batch()` 返回 bytes，随后加 `render_frame_batch_async()` 和 `get_frame_batch()`。

### 3.2 LuisaRender: Pipeline 持有多 camera，CommandBuffer 管同步边界

参考文件:

- `E:\cplus\LuisaRender\src\base\pipeline.h`
- `E:\cplus\LuisaRender\src\base\pipeline.cpp`
- `E:\cplus\LuisaRender\src\base\integrator.cpp`
- `E:\cplus\LuisaRender\src\util\command_buffer.h`
- `E:\cplus\LuisaRender\src\base\geometry.cpp`
- `E:\cplus\LuisaRender\src\base\scene.cpp`

关键设计:

1. `Pipeline` 内部保存 `vector<Camera::Instance>`。见 `pipeline.h:84`、`pipeline.h:212-214`。
2. `Pipeline::create()` 一次构建 scene 的 cameras、geometry、environment、integrator。见 `pipeline.cpp:45-117`，尤其是 `pipeline.cpp:71-75`。
3. `scene_update()` 根据 dirty flag 局部更新 cameras/shapes/environment，然后 `clear_update()`。见 `pipeline.cpp:120-172` 和 `scene.cpp:60-70`。
4. `ProgressiveIntegrator::render()` 遍历所有 camera，在同一个 `CommandBuffer` 抽象下依次 prepare、render、download。见 `integrator.cpp:41-56`。
5. `CommandBuffer` 把命令记录、commit、synchronize 聚合到统一入口。见 `command_buffer.h:13-43`。
6. CPU 侧 transform 更新超过阈值后用 global thread pool 并行。见 `geometry.cpp:169-188`。

对 RTXNS 的迁移:

- 不要把 batch API 做成 Python 循环的包装。应在 `headless_pbr.cpp` 内复用一个 command list 遍历 view slots。
- 加 dirty state: scene dirty、camera dirty、target dirty、AS dirty、readback dirty。不要每帧默认刷新所有结构。
- 抽一个轻量 `RtxnsCommandRecorder` 或 `FrameGraphLite`，把 `open/close/execute/wait` 从业务 pass 中拿出来集中管理。
- 保持 `ForwardShadingPass`、`RayTracedShadowPass` 作为 pass 级对象共享，per-camera 的资源和 binding 由 view slot 或 frame scratch 持有。

## 4. 目标架构

### 4.1 P0 目标: 单 SceneGraph 多相机批渲染

```
RendererContext x 1
  DeviceManager x 1
  ShaderFactory x 1
  CommonRenderPasses x 1

HeadlessPbrScene x 1
  SceneGraph x 1
  TextureCache x 1
  ForwardShadingPass x 1
  RayTracedShadowPass x 1
  BLAS list x 1
  TLAS x 1
  RenderViewSlot[N]
    Camera/View x N
    Color/Depth/Shadow/Composite target x N
    Staging readback ring x N

render_frame_batch(camera_indices)
  open command list once
  Scene::Refresh once
  build/update shared TLAS once
  for each camera:
    render raster
    render shadow
    blur/composite
    copy output to its readback target
  close/execute once
  wait/map all requested outputs once
```

预期收益:

- Python 调用次数从 N 次降为 1 次。
- command list execute/wait 次数从 N 次降为 1 次。
- scene、texture、BLAS 不再按 worker 复制。
- 多相机工作量在单 device 队列里连续提交，减少多进程 Vulkan device 互相抢占。

### 4.2 P1 目标: 异步 readback 和 frame pipeline

```
Frame k:
  record render for batch k
  copy outputs to staging[k % K]
  signal fence/timeline value k
  return token k

Frame k-1:
  if fence ready:
    map staging[(k-1) % K]
    copy bytes to Python
```

预期收益:

- 去掉每帧 `waitForIdle()`，改成只等待需要读回的 staging slot。
- GPU 可连续吃 batch work，CPU 同时处理上一批图片。
- 对 RL/vectorized env 更友好，可以选择延迟一帧拿图。

### 4.3 P2 目标: ~~同一 Graphics queue 的多 cmdList~~ → 已否决，改为 per-camera 成本优化

**原实验假设**: 将相机拆为多个 cmdList 后，可在同一 Graphics queue 上提高吞吐。

**实测结论（2026-07-12）**: 同一 Graphics queue 上拆多个 cmdList 不会让不同相机的 raster 真正并行——Vulkan 按提交顺序执行同一 queue 的 cmd buffer。NVRHI 的 `executeCommandLists()` 已支持批量提交，但相机间仍是顺序渲染。真正的并行需要多个 VkQueue（如 graphics + compute + transfer）或 GPU-side multi-draw 合并，不属于当前阶段。

**修正方向**:
1. 降低每相机成本：readback 合批为一次 `mapStagingTexture`、binding set/constant-buffer 跨相机复用、减少 per-camera pipeline barrier。
2. 动态仿真阶段：仅在物体变换或场景结构变化时更新 TLAS（用 Donut 的 `HasPendingTransformChanges()` / `HasPendingStructureChanges()` 脏标记）。
3. 用"动态物体数 × 相机数"基准验证优化效果，不继续追逐 cmdList 拆分。

### 4.4 P3 目标: 多 scene / 多 GPU

多 GPU 不是替代 batch，而是 batch 之上的扩展:

- 每块 GPU 一个 `RendererContext`
- 每个 context 管一组 `HeadlessPbrScene` 或 batch
- Python 调度层按 device 分片
- 每个 device 内仍然使用 `render_frame_batch()`

## 5. 分阶段推进

本节分两层:

- **四周汇报版计划**: 每周都能形成一份进度汇报，强调“本周解决什么问题、产生什么收益、有什么可展示材料、下一周承接什么”。
- **工程细化 Phase**: 面向实际开发拆分，保留更细的代码任务、API 和验收条件。

### 5.1 四周汇报版计划总览

四周目标不是每周都追求完整架构闭环，而是每周都有可验证的增量收益:

| 周次 | 阶段主题 | 核心问题 | 阶段成果 | 汇报关键词 |
|------|----------|----------|----------|------------|
| Week 1 | 基准复现 + 多相机资源模型 | 当前并行效率低，但缺少稳定对照；renderer 内部仍是单相机结构 | 得到可复现 baseline，完成 `CameraDesc` / `RenderViewSlot` / 多相机 API 骨架 | 瓶颈量化、接口打通、兼容旧 API |
| Week 2 | 同步版 `render_frame_batch()` | Python 层 N 次调用、N 次 command submit/wait；BLAS/TLAS 不能按 batch 共享表达 | 单进程单 device 批量渲染多相机，scene/texture/BLAS 只加载一次 | 首个吞吐收益、显存收益、共享 AS |
| Week 3 | 异步 readback + 帧流水 | 每 batch 末尾仍 `waitForIdle()`，GPU/CPU 无跨帧重叠 | `submit_frame_batch()` / `read_frame_batch(token)`，staging ring，去掉稳态全局 wait | GPU 利用率提升、异步流水、延迟可控 |
| Week 4 | 资源池 + 调度器 + 总结报告 | batch size 增大后显存/readback/不同分辨率成为新瓶颈 | render target/readback pool、micro-batch scheduler、完整性能报告 | 稳定扩展、显存预算、最终对比 |

建议每周汇报固定包含五项:

1. **本周目标**: 用一句话说明解决的瓶颈。
2. **实现内容**: 列出关键代码文件和 API。
3. **量化收益**: FPS、GPU util、VRAM、CPU wall time、submit/wait 次数。
4. **可展示材料**: benchmark JSON、对比图、日志截图、示例脚本。
5. **下周计划**: 本周遗留风险如何进入下一阶段。

### 5.2 Week 1: 基准复现 + 多相机资源模型

**汇报标题建议**: RTXNS 并行渲染 Week 1 — 多进程瓶颈复现与多相机资源模型改造

#### 本周目标

把“并行效率低”从直觉变成可复现数据，并完成后续 batch renderer 的内部数据结构准备。Week 1 的重点不是立刻冲吞吐，而是建立可信 baseline 和不破坏旧 API 的多相机资源模型。

#### 计划工作

1. 修正 benchmark 工具路径和参数:
   - `tools/parallel_render_poc.py`
   - `tools/parallel_render_bottleneck.py`
   - 新增或预留 `tools/parallel_render_batch_benchmark.py`
2. 统一记录性能指标:
   - steady FPS
   - per-frame wall time
   - `FrameStats.total_ms`
   - GPU utilization
   - peak VRAM
   - command submit/wait 次数
3. 在 native renderer 中引入多相机结构:
   - `CameraDesc`
   - `RenderViewSlot`
   - `std::vector<RenderViewSlot> m_views`
4. 保持旧接口兼容:
   - 旧 `set_camera(...)` 写入 camera 0
   - 旧 `render_frame()` 继续渲染 camera 0
5. 增加初始 Python API:
   - `add_camera(...)`
   - `set_camera(index, ...)`
   - `camera_count`

#### 关键代码文件

- `src/PythonBindings/headless_pbr.h`
- `src/PythonBindings/headless_pbr.cpp`
- `src/PythonBindings/py_bindings_common.h`
- `python/rtxns_genesis_style/renderer.py`
- `tools/parallel_render_bottleneck.py`

#### 阶段成果

| 成果 | 说明 | 汇报价值 |
|------|------|----------|
| Baseline 数据表 | 复现 N=1/2/4/6/8 多进程吞吐、GPU util、VRAM | 说明当前瓶颈不是猜测 |
| 多相机 API 骨架 | `add_camera()` 能创建多个 camera slot | 说明 renderer 已从单相机向 batch 演进 |
| 旧 API 回归 | 原有 `render_frame()` 测试仍通过 | 说明改造风险可控 |
| 资源结构图 | 展示 `RenderViewSlot[N]` 替代单份 `m_camera/m_view/m_framebuffer` | 便于老师理解架构变化 |

#### 量化指标目标

Week 1 主要是基线和结构性进展，因此收益指标以“可测”和“不退化”为主:

| 指标 | Baseline | Week 1 目标 |
|------|----------|-------------|
| 单相机 `render_frame()` 输出 | 当前结果 | 像素/尺寸一致 |
| 单相机 FPS | 当前 N=1 | 不低于 baseline 95% |
| 多相机创建 | 无 | 支持 N 个 camera slot |
| 多进程 benchmark | 手动/路径固定 | 参数化、可复现 |
| 汇报材料 | 零散输出 | JSON + 图 + 表格 |

#### 可展示材料

- `output/parallel_render/bottleneck_results.json`
- `output/parallel_render/bottleneck_analysis.png`
- 新增 `output/parallel_render/week1_baseline_summary.md`
- 多相机资源结构图，可直接从本文档目标架构图整理。

#### 风险与下周承接

- Week 1 可能还没有明显 FPS 提升，这是正常的。本周核心收益是“让 batch 可实现”。
- 如果 `RayTracedShadowPass` 绑定集生命周期阻塞多相机录制，Week 2 优先处理 per-dispatch binding set。

### 5.3 Week 2: 同步版 `render_frame_batch()` + 共享 AS

**汇报标题建议**: RTXNS 并行渲染 Week 2 — 单进程单 Device 批量多相机渲染

#### 本周目标

实现第一个真正有吞吐收益的 batch renderer: Python 一次调用进入 C++，C++ 内部一次 command list 录制多个 camera，scene/texture/BLAS/TLAS 在 batch 内共享。

#### 计划工作

1. 从旧 `render_frame()` 拆出可复用录制函数:
   - `record_scene_refresh(command_list)`
   - `record_raster(command_list, RenderViewSlot&)`
   - `record_rt_shadow(command_list, RenderViewSlot&)`
   - `record_composite_and_copy(command_list, RenderViewSlot&)`
   - `readback_view(RenderViewSlot&)`
2. 实现同步版:
   - `render_frame(uint32_t camera_index)`
   - `render_frame_batch(const std::vector<uint32_t>& camera_indices)`
3. batch 内只做一次:
   - `Scene::Refresh(...)`
   - BLAS 首次构建
   - TLAS build/update
   - `executeCommandList(...)`
   - `waitForIdle()`
4. 修复 RT shadow batch 录制风险:
   - per-view shadow constant buffer
   - per-frame binding set scratch
   - 避免 `m_shadowBindingSet` / `m_compositeBindingSet` / `m_blurBindingSet` 被后续 camera 覆盖导致生命周期不清晰
5. Python 返回:
   - 第一版返回 `list[bytes]`
   - 同时返回 batch stats 或提供 `get_last_batch_stats()`

#### 关键代码文件

- `src/PythonBindings/headless_pbr.cpp`
- `src/PythonBindings/headless_pbr.h`
- `src/PythonBindings/py_bindings_common.h`
- `src/RayTracedShadow/RayTracedShadowPass.h`
- `src/RayTracedShadow/RayTracedShadowPass.cpp`
- `src/RayTracedShadow/AccelerationStructure.cpp`

#### 阶段成果

| 成果 | 说明 | 汇报价值 |
|------|------|----------|
| `render_frame_batch()` 可用 | 单次 API 返回多张图 | 标志从多进程并行转向真 batch |
| 单 device 共享资源 | 同一 SceneGraph/TextureCache/BLAS 被多个 camera 复用 | 直接解释显存收益来源 |
| 首个性能收益 | N=2/4 camera batch 对比旧单进程循环、多进程 POC | 可给老师展示量化提升 |
| batch stats | 每个 batch 的 raster/AS/shadow/readback 分解 | 后续优化有抓手 |

#### 量化指标目标

实际数字以测试为准，Week 2 汇报建议用“目标区间 + 实测结果”形式:

| 指标 | 当前多进程/单帧问题 | Week 2 目标 |
|------|--------------------|-------------|
| Python 调用次数 | N 次 `render_frame()` | 1 次 `render_frame_batch(N)` |
| command execute/wait | N 次 | 1 次 |
| BLAS 构建 | 多进程 N 份 | 单进程 1 份 |
| N=4 显存 | 近似 N×场景资源 | 约 1×场景资源 + 4×render target |
| N=4 batch 吞吐 | 旧单进程串行作为 baseline | 目标 2.5× 以上 |
| 画面一致性 | 单相机 baseline | 多 camera 输出尺寸/内容正确 |

#### 可展示材料

- `tools/parallel_render_batch_benchmark.py`
- `output/parallel_render/week2_batch_results.json`
- `output/parallel_render/week2_batch_vs_multiprocess.png`
- 四宫格相机渲染结果图: `camera_0/1/2/3.png`
- command submit/wait 日志摘要。

#### 风险与下周承接

- Week 2 仍然保留同步 readback，所以 GPU 利用率可能未达到理想值。
- 如果吞吐提升低于预期，优先分析 readback 和 per-camera binding set 创建开销，Week 3 通过异步 readback 继续解决。

### 5.4 Week 3: 异步 readback + 帧流水

**汇报标题建议**: RTXNS 并行渲染 Week 3 — 去除稳态 `waitForIdle` 与异步批渲染

#### 本周目标

解决 Week 2 同步 batch 的主要剩余瓶颈: batch 末尾仍等待 GPU 完全 idle，然后 CPU 读回图像。Week 3 要把“提交渲染”和“读取结果”拆开，让 GPU 渲染当前 batch 时 CPU 可以处理上一 batch。

#### 计划工作

1. 新增 readback ring:
   - 每个 `RenderViewSlot` 持有 K 个 staging texture
   - 推荐 K=2 或 K=3
2. 新增异步 API:
   - `BatchToken submit_frame_batch(indices)`
   - `bool is_batch_ready(token)`
   - `std::vector<std::vector<uint8_t>> read_frame_batch(token)`
   - 同步 `render_frame_batch()` 保留为便利函数
3. 用 fence/timeline value 管理 batch 完成状态:
   - steady-state render path 不再调用 `device->waitForIdle()`
   - 只在 `read_frame_batch(token)` 必要时等待对应 token
4. 增加 frame pipeline benchmark:
   - 同步 batch vs 异步 batch
   - batch size = 1/2/4/8
   - readback lag = 1/2/3
5. Python wrapper 支持:
   - 阻塞式 `render_cameras(cameras)`
   - 可选异步 `submit_cameras(cameras)` / `read_cameras(token)`

#### 关键代码文件

- `src/PythonBindings/headless_pbr.cpp`
- `src/PythonBindings/headless_pbr.h`
- `src/PythonBindings/py_bindings_common.h`
- 可选新增 `src/PythonBindings/frame_readback_ring.h`
- 可选新增 `src/PythonBindings/batch_render_stats.h`
- `python/rtxns_genesis_style/renderer.py`

#### 阶段成果

| 成果 | 说明 | 汇报价值 |
|------|------|----------|
| 异步 batch API | submit/read 分离 | 说明架构具备仿真/训练流水能力 |
| 去除稳态全局 wait | render path 不再每帧 `waitForIdle()` | 直接对应 GPU 欠载瓶颈 |
| readback ring | staging texture 轮转，避免读写同一资源 | 工程上更接近成熟 renderer |
| GPU 利用率提升 | 对比 Week 2 同步 batch | 可作为本周核心收益 |

#### 量化指标目标

| 指标 | Week 2 | Week 3 目标 |
|------|--------|-------------|
| steady-state `waitForIdle()` | 每 batch 1 次 | 0 次 |
| GPU util | 同步 batch 基线 | 明显提升，目标 50%+ |
| CPU/GPU 重叠 | 无 | CPU readback 上一 batch，GPU 渲染当前 batch |
| batch latency | 同步返回 | token 管理，可配置 lag |
| N=4/8 吞吐 | Week 2 baseline | 继续提升，目标接近多进程峰值或超过 |

#### 可展示材料

- 同步 vs 异步吞吐对比图
- GPU utilization 时间曲线
- readback lag 对吞吐/延迟影响表
- API 示例:

```python
token0 = scene.submit_frame_batch([0, 1, 2, 3])
token1 = scene.submit_frame_batch([0, 1, 2, 3])
images0 = scene.read_frame_batch(token0)
```

#### 风险与下周承接

- 异步 readback 可能引入资源生命周期和读脏数据风险，需要严格 token/fence 校验。
- Week 4 重点接资源池和调度器，避免异步后 batch size 增大导致显存尖峰。

### 5.5 Week 4: 资源池 + 调度器 + 四周总结

**汇报标题建议**: RTXNS 并行渲染 Week 4 — 批渲染资源池、显存预算与最终性能对比

#### 本周目标

把前三周的能力整理成稳定、可扩展、可汇报的系统: 支持 N=8/16 camera 的稳定运行，控制显存峰值，输出完整的四周性能收益报告。

#### 计划工作

1. 实现资源池:
   - `RenderTargetPool`
   - `ReadbackPool`
   - 按 `(width, height, format)` 复用
2. 实现 batch scheduler:
   - 同分辨率 camera 合批
   - 不同分辨率 camera 自动拆 micro-batch
   - 根据显存预算限制最大 batch size
3. 完善 benchmark:
   - 多进程 POC
   - 旧单进程串行
   - Week 2 同步 batch
   - Week 3 异步 batch
   - Week 4 资源池调度 batch
4. 输出最终四周汇报文档:
   - 性能收益表
   - 显存收益表
   - GPU utilization 对比
   - API/代码修改清单
   - 下一阶段规划: 多队列 timeline semaphore / 多 GPU 分片 / GPU buffer interop

#### 关键代码文件

- `src/PythonBindings/headless_pbr.cpp`
- `src/PythonBindings/headless_pbr.h`
- 可选新增 `src/PythonBindings/render_target_pool.h`
- 可选新增 `src/PythonBindings/batch_scheduler.h`
- `tools/parallel_render_batch_benchmark.py`
- `docs/RTXNS_Parallel_Render_Optimization_Weekly_Report.md`

#### 阶段成果

| 成果 | 说明 | 汇报价值 |
|------|------|----------|
| 资源池 | 降低重复 target/staging 分配 | 解释稳定性和显存控制 |
| micro-batch scheduler | 大规模 camera 自动分批 | 说明方案可扩展到训练/数据采集 |
| 四周性能总表 | baseline vs batch vs async vs scheduler | 最适合汇报的核心成果 |
| 后续路线 | timeline semaphore、多 GPU、CUDA/torch interop | 形成下一阶段研究计划 |

#### 量化指标目标

| 指标 | 当前多进程问题 | Week 4 目标 |
|------|----------------|-------------|
| N=8 显存 | 多进程约 N×场景资源 | 约 1×场景资源 + render targets |
| N=8 吞吐 | 多进程容易触顶/退化 | 稳定高于 Week 2/3 baseline |
| batch 稳定性 | 大 batch 容易显存尖峰 | scheduler 自动拆分 |
| 输出材料 | 单次实验图 | 四周完整报告 |

#### 可展示材料

- `output/parallel_render/week4_final_results.json`
- `output/parallel_render/week4_scaling_curve.png`
- `output/parallel_render/week4_vram_curve.png`
- `output/parallel_render/week4_gpu_util_curve.png`
- `docs/RTXNS_Parallel_Render_Optimization_Report.md`

#### 四周最终汇报建议结论

最终报告建议用如下结构:

1. **问题**: 多进程并行不是 renderer 内部并行，存在 GPU 欠载、device 调度竞争、显存重复。
2. **方法**: 参考 SAPIEN batch renderer 和 LuisaRender pipeline/command buffer，改为单 device 批量多相机。
3. **实现**: 多相机资源模型、同步 batch、异步 readback、资源池调度器。
4. **收益**: FPS、GPU util、VRAM、submit/wait 次数、BLAS 构建次数。
5. **后续**: 多队列 timeline semaphore、多 GPU 分片、GPU-native output。

### 5.6 每周汇报模板

可以直接复制下面模板作为周报骨架:

```markdown
# RTXNS 并行渲染优化周报 — Week X

## 概述

本周完成 ...，解决了 ...。核心收益是 ...。

| 指标 | 上周/基线 | 本周 | 收益 |
|------|-----------|------|------|
| 吞吐 FPS | | | |
| GPU util | | | |
| VRAM peak | | | |
| submit/wait 次数 | | | |
| BLAS 构建次数 | | | |

## 问题

本周针对的瓶颈是 ...

## 实现

- 修改文件:
- 新增 API:
- benchmark:

## 效果

性能表、图、示例输出。

## 风险

当前仍存在 ...

## 下周计划

下周将 ...
```

### 5.7 工程细化 Phase 0: 基准和保护线

目标: 固定性能指标，避免后续改造“感觉变快但不可复现”。

改动:

- 修正 `tools/parallel_render_bottleneck.py` 中硬编码路径。当前文件使用 `D:\RTXNS` 和 `D:\niagara_bistro`，如果当前工作目录是 `E:\cplus\RTXNS`，建议改成 argparse 参数或从 repo root 自动推导。
- 增加单进程 batch 对照脚本占位: `tools/parallel_render_batch_benchmark.py`。
- 指标统一输出 JSON: 吞吐 FPS、平均 GPU util、峰值 VRAM、per-frame wall time、`FrameStats`、readback bytes。

验收:

- 能复现现有多进程 N=1/2/4 结果。
- benchmark 记录 repo commit、GPU 名称、driver、分辨率、RT shadow 开关、shadow samples。

### 5.8 工程细化 Phase 1: 多相机资源模型

目标: `HeadlessPbrScene` 内部可以同时保存 N 个 view slot。

改动文件:

- `src/PythonBindings/headless_pbr.h`
- `src/PythonBindings/headless_pbr.cpp`
- `src/PythonBindings/py_bindings_common.h`

建议 API:

```cpp
struct CameraDesc
{
    std::array<float, 3> position;
    std::array<float, 3> target;
    std::array<float, 3> up;
    float fov_degrees = 60.0f;
    uint32_t width = 1024;
    uint32_t height = 768;
    float z_near = 0.1f;
    float z_far = 1000.0f;
};

uint32_t add_camera(const CameraDesc& desc);
void set_camera(uint32_t index, const CameraDesc& desc);
std::vector<uint8_t> render_frame(uint32_t camera_index);
std::vector<std::vector<uint8_t>> render_frame_batch(const std::vector<uint32_t>& camera_indices);
```

兼容策略:

- 旧的 `set_camera(...)` 继续写 camera 0。
- 旧的 `render_frame()` 等价于 `render_frame(0)`。
- Python 绑定先返回 `list[bytes]`，后续再优化成 contiguous buffer + shape metadata。

验收:

- 单 camera 结果和旧 API 一致。
- `add_camera()` 后不重新 `load_scene()`。
- N 个 camera 的 render targets 独立，不互相覆盖。

### 5.9 工程细化 Phase 2: 单 command list 批量录制

目标: 实现真正的 `render_frame_batch()`，而不是 Python/C++ 循环调用旧 `render_frame()`。

改动重点:

- 从 `render_frame()` 拆出:
  - `record_scene_refresh(command_list)`
  - `record_raster(command_list, RenderViewSlot&)`
  - `record_rt_shadow(command_list, RenderViewSlot&)`
  - `record_composite_and_copy(command_list, RenderViewSlot&)`
  - `readback_view(RenderViewSlot&)`
- `Scene::Refresh(command_list, frame_index)` 每 batch 调一次，不要每 camera 调一次。
- TLAS update 每 batch 调一次，不要每 camera 调一次。
- 每个 camera 只录制 view-dependent work。

特别注意:

- `RayTracedShadowPass` 当前会覆盖成员 binding set。批量录制时要把 per-dispatch binding set 保存在 frame scratch 中，或改为 view slot 持有。
- `m_shadowConstantBuffer` 当前共享。建议每 view slot 一个 shadow constant buffer，或者使用 per-frame dynamic constant buffer。

验收:

- N=1 batch 性能不低于旧 `render_frame()`。
- N=2/4 时，单进程 batch 的总吞吐明显高于单进程串行调用旧 API。
- command list execute/wait 次数可以通过日志验证为每 batch 一次。

### 5.10 工程细化 Phase 3: 共享 BLAS 和单次 TLAS 更新

目标: 确认几何 AS 只构建一次，并在所有 camera 间共享。

当前可复用点:

- `src/RayTracedShadow/AccelerationStructure.cpp:32`: `buildBLASes()` 已支持一次构建一组 BLAS。
- `src/RayTracedShadow/AccelerationStructure.cpp:212`: `updateTLAS()` 可在已有 TLAS 上更新。
- `src/PythonBindings/headless_pbr.cpp:694-999`: 当前已经把首次构建和后续 TLAS update 分开。

改动:

- 把 AS 状态从单帧逻辑中抽成 `ShadowAccelerationCache`。
- 明确 dirty 条件:
  - camera 改变: 不影响 BLAS/TLAS。
  - rigid transform 改变: 只需要 TLAS update。
  - mesh/material alpha-test 改变: 需要 rebuild scene shadow resources，可能 rebuild BLAS。
  - scene reload: 全部重建。
- 批量渲染中，AS update 放在 camera loop 之前。

验收:

- 渲染 N 个 camera 时 `blas_build_ms` 只在首 batch 非零。
- camera-only batch 中 `tlas_build_ms` 接近 0 或只出现一次。
- VRAM 不随 camera 数量按场景大小线性增长，只随 render target/readback target 增长。

### 5.11 工程细化 Phase 4: 异步 readback

目标: 从 `execute + waitForIdle + map` 改成 fence/timeline 控制的 readback ring。

改动文件:

- `src/PythonBindings/headless_pbr.cpp`
- 可能新增 `src/PythonBindings/frame_sync.h/.cpp`

建议 API:

```cpp
using BatchToken = uint64_t;

BatchToken submit_frame_batch(const std::vector<uint32_t>& camera_indices);
bool is_batch_ready(BatchToken token) const;
std::vector<std::vector<uint8_t>> read_batch(BatchToken token);
std::vector<std::vector<uint8_t>> render_frame_batch(const std::vector<uint32_t>& camera_indices);
```

其中 `render_frame_batch()` 是同步便利函数，内部调用 submit + wait + read。训练/仿真框架可使用异步 API。

验收:

- 同步 API 行为保持兼容。
- 异步 API 可以连续 submit K 个 batch 后再 read。
- GPU util 在 N=1 batch 场景明显提升，不再长期 1%。

### 5.12 工程细化 Phase 5: 调度器和资源池

目标: 控制显存和 CPU 开销。

改动:

- `RenderTargetPool`: 按分辨率和格式复用 color/depth/shadow/composite target。
- `ReadbackPool`: 按 `(width, height, format, lag)` 复用 staging texture。
- `BatchScheduler`: 根据显存预算把 camera_indices 分成多个 micro-batch。

策略:

- 默认 batch size 由 render target 显存估算决定。
- 同分辨率 camera 合批，不同分辨率先分组。
- 如果 RT shadow 开启，shadow target 和 blur temp 会让每 camera 显存翻倍，要纳入估算。

验收:

- N=8 camera 不出现显存尖峰。
- 不同分辨率 camera 可以渲染，但会被拆成多个 micro-batch。

### 5.13 远期探索: 多队列 timeline semaphore（非 P2 延续）

目标: 仅在存在明确的跨 queue 依赖（如 upload/copy 或 CUDA interop）并完成 per-camera 成本优化后评估；不用于重新尝试同 queue micro-cmdList。

参考:

- SAPIEN `batched_render_system.cpp:238-255`: Vulkan timeline semaphore 导出给 CUDA。
- SAPIEN `batched_render_system.cpp:342-351`: CUDA signal 后 Vulkan queue 等待。

RTXNS 里先不需要 CUDA interop，但 timeline semaphore 的思想可用于:

- upload/AS queue signal
- render queue wait upload/AS
- copy queue wait render
- CPU wait copy fence only when reading

验收:

- 无全局 `device->waitForIdle()` 出现在 steady-state render path。
- resize/reload/destroy 仍可使用 wait idle 作为安全边界。

## 6. 推荐的代码组织

### 6.1 保持现有 pass，先抽 view slot

第一轮不要重写 renderer。保留:

- `ForwardShadingPass`
- `RayTracedShadowPass`
- `SceneGeometryProvider`
- `AccelerationStructure`
- `OMMBaker`

先把 `headless_pbr.cpp` 中的单相机成员收敛为 `RenderViewSlot`，这是收益最高且风险可控的改动。

### 6.2 建议新增内部文件

可选新增:

- `src/PythonBindings/render_view_slot.h`: `CameraDesc`、`RenderViewSlot`、target resize helpers。
- `src/PythonBindings/frame_readback_ring.h`: staging texture ring 和 fence token。
- `src/PythonBindings/batch_render_stats.h`: batch 级统计。

如果希望少动 CMake，可以先都放在 `headless_pbr.cpp` 的匿名 namespace，稳定后再拆文件。

### 6.3 Python 层接口演进

第一阶段:

```python
cam0 = scene.add_camera(...)
cam1 = scene.add_camera(...)
images = scene.render_frame_batch([cam0, cam1])
```

第二阶段:

```python
token = scene.submit_frame_batch([0, 1, 2, 3])
# do simulation step / CPU preprocessing
images = scene.read_frame_batch(token)
```

Genesis-style wrapper:

- `python/rtxns_genesis_style/renderer.py:949`: 已有 `add_camera()` 概念，可映射到 native `add_camera()`。
- `python/rtxns_genesis_style/renderer.py:1046-1057`: 当前每次 render 都 `_apply_camera_desc()` 再 `render_frame()`，应增加 `render_cameras(cameras)` 批量入口。
- `python/donut_render_py/runtime.py:1518`: 当前 `Scene.render_frame(camera)` 是单 camera API，后续增加 `render_frames(cameras)`。

## 7. 参考文件清单

RTXNS 当前实现:

- `E:\cplus\RTXNS\src\PythonBindings\headless_pbr.cpp`: 当前 headless renderer 主体，重点看 `render_frame()`、target resize、AS 构建。
- `E:\cplus\RTXNS\src\PythonBindings\headless_pbr.h`: native API 声明。
- `E:\cplus\RTXNS\src\PythonBindings\py_bindings_common.h`: pybind11 绑定。
- `E:\cplus\RTXNS\src\RayTracedShadow\RayTracedShadowPass.cpp`: RT shadow / blur / composite compute dispatch。
- `E:\cplus\RTXNS\src\RayTracedShadow\AccelerationStructure.cpp`: BLAS/TLAS build/update。
- `E:\cplus\RTXNS\src\RayTracedShadow\SceneGeometryProvider.cpp`: shadow scene resources 上传。
- `E:\cplus\RTXNS\tools\parallel_render_poc.py`: 当前多进程吞吐 POC。
- `E:\cplus\RTXNS\tools\parallel_render_bottleneck.py`: 当前瓶颈分析工具。

SAPIEN 参考:

- `E:\cplus\SAPIEN\include\sapien\sapien_renderer\batched_render_system.h`: batch camera/render system 接口。
- `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cpp`: batch command buffer、timeline semaphore、camera batch takePicture。
- `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cu`: GPU 批量更新 object/camera transform。

LuisaRender 参考:

- `E:\cplus\LuisaRender\src\base\pipeline.h`: pipeline 持有多 camera、geometry、integrator。
- `E:\cplus\LuisaRender\src\base\pipeline.cpp`: create/update/render 分层。
- `E:\cplus\LuisaRender\src\base\integrator.cpp`: 遍历多 camera 的 render 入口。
- `E:\cplus\LuisaRender\src\util\command_buffer.h`: command buffer 聚合 commit/synchronize。
- `E:\cplus\LuisaRender\src\base\scene.cpp`: dirty flag 和 clear_update。
- `E:\cplus\LuisaRender\src\base\geometry.cpp`: CPU thread pool 更新 dynamic transforms。

## 8. 风险和规避

1. Binding set 生命周期风险  
   `RayTracedShadowPass` 现在把 binding set 存为成员，batch 内多次覆盖可能导致已录制命令引用的 handle 生命周期不清晰。规避: per-frame scratch 保存所有 binding set，执行完成后释放。

2. Constant buffer 覆盖风险  
   多 camera 连续写同一个 constant buffer，理论上同队列有顺序，但可读性和后续多队列风险很高。规避: 每 view slot 一个 constant buffer，或 frame allocator 分配 per-dispatch constant。

3. readback 吞吐风险  
   N 个 1024x768 RGBA8 输出约 N x 3 MB。N=8 每 batch 约 24 MB，CPU copy 和 Python bytes 创建可能成为新瓶颈。规避: 增加 contiguous batch buffer，或提供 GPU buffer/CUDA interop 路径。

4. resize 复杂度风险  
   不同 camera 分辨率会让 target pool 和 framebuffer cache 复杂化。规避: P0 只支持同分辨率 batch，不同分辨率自动拆 batch。

5. 场景动态更新风险  
   如果每帧 geometry 变化，BLAS 共享收益会下降。规避: 明确 dirty 分类，rigid transform 走 TLAS update，mesh topology 变化才 rebuild BLAS。

## 9. 建议里程碑

### M1: 同步 batch API

时间估计: 2-3 天  
成果:

- `add_camera()` / `set_camera(index)` / `render_frame_batch(indices)`
- 单 command list 批量录制
- 同步 readback
- benchmark 对比旧单进程循环和多进程 POC

成功标准:

- N=4 camera batch 总吞吐至少达到旧单进程 2.5x 以上。
- 显存峰值显著低于 4 进程方案。
- BLAS 首 batch 后不重复构建。

### M2: readback ring + async submit

时间估计: 2-4 天  
成果:

- `submit_frame_batch()` / `read_frame_batch(token)`
- staging ring
- fence/timeline value 管理

成功标准:

- steady-state render path 不再调用 `device->waitForIdle()`。
- 同步 API 与异步 API 结果一致。

### M3: batch scheduler + resource pool

时间估计: 2-3 天  
成果:

- render target/readback pool
- micro-batch 分组
- 显存预算控制

成功标准:

- N=8/16 camera 稳定运行。
- 显存随 camera 数量主要按 render target 增长，而不是按 scene 增长。

### M4: 多队列同步

时间估计: 3-5 天  
成果:

- upload/render/copy 阶段拆分
- timeline semaphore 或 NVRHI fence 封装
- 只在 reload/destroy 等生命周期边界 wait idle

成功标准:

- GPU util 对比 M1/M2 继续提升。
- 无 deadlock，无 staging 读脏数据。

## 10. 第一批具体任务

建议从以下小 PR 开始，避免一次改穿整个 renderer:

1. 新增 `CameraDesc` 和 `RenderViewSlot`，把 camera 0 迁入 slot，但旧 API 行为不变。
2. 把 `resize_targets()` 改成 `resize_targets(RenderViewSlot&, width, height)`。
3. 把 `set_camera()` 改成写 `m_views[0]`。
4. 抽出 `record_render_view(command_list, RenderViewSlot&)`，先只给 camera 0 调用。
5. 实现 `add_camera()` 和 `set_camera(index)`。
6. 实现同步版 `render_frame_batch()`，内部 open 一次 command list，loop views，execute/wait 一次。
7. 修 `RayTracedShadowPass` 的 per-dispatch binding set 生命周期。
8. 加 benchmark: 同一进程内 N 次旧 `render_frame()` vs 新 `render_frame_batch(N)` vs 多进程 POC。

这条路线的好处是每一步都能编译、能跑旧测试，并且每一步都把 renderer 往“成熟引擎里的 batch/pipeline 模型”推进。

## 11. 代码 Agent 执行手册

本节面向实际写代码的 Agent。目标是降低上手难度: 先按本节读文件、照着参考实现找对应关系，再按 micro-patch 开发。不要一上来重构整个 renderer。

### 11.1 开发前必须先读的文件

按这个顺序读，读完再改:

1. `E:\cplus\RTXNS\src\PythonBindings\headless_pbr.h`
   - 看 public API: `set_camera()`、`render_frame()`、`FrameStats`。
2. `E:\cplus\RTXNS\src\PythonBindings\py_bindings_common.h`
   - 看 pybind11 如何释放 GIL、如何把 `std::vector<uint8_t>` 转为 `py::bytes`。
3. `E:\cplus\RTXNS\src\PythonBindings\headless_pbr.cpp`
   - 重点入口:
     - `HeadlessPbrScene::Impl::set_camera()`，约 `headless_pbr.cpp:338`
     - `HeadlessPbrScene::Impl::render_frame()`，约 `headless_pbr.cpp:620`
     - `HeadlessPbrScene::Impl::resize_targets()`，约 `headless_pbr.cpp:1229`
     - 单相机成员区，约 `headless_pbr.cpp:1320-1325`
4. `E:\cplus\RTXNS\src\RayTracedShadow\RayTracedShadowPass.cpp`
   - 重点入口:
     - `renderShadow()`，约 `RayTracedShadowPass.cpp:179`
     - `compositeShadow()`，约 `RayTracedShadowPass.cpp:256`
     - `blurShadow()`，约 `RayTracedShadowPass.cpp:297`
   - 特别看 binding set 创建位置: `m_shadowBindingSet`、`m_compositeBindingSet`、`m_blurBindingSet`。
5. `E:\cplus\RTXNS\src\RayTracedShadow\AccelerationStructure.cpp`
   - 看 `buildBLASes()`、`updateTLAS()`，不要重复发明 AS 构建。
6. 参考实现:
   - `E:\cplus\SAPIEN\include\sapien\sapien_renderer\batched_render_system.h`
   - `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cpp`
   - `E:\cplus\LuisaRender\src\base\pipeline.h`
   - `E:\cplus\LuisaRender\src\base\pipeline.cpp`
   - `E:\cplus\LuisaRender\src\base\integrator.cpp`

### 11.2 参考实现到 RTXNS 的对应关系

| 参考项目 | 参考代码位置 | 设计点 | RTXNS 应该怎么做 |
|----------|--------------|--------|------------------|
| SAPIEN | `batched_render_system.h:42-53` | `BatchedCamera` 持有 camera 列表、输出 buffer、command buffer、timeline semaphore | RTXNS 用 `RenderViewSlot[N]` 持有 camera/view/target/readback，后续加 token/fence |
| SAPIEN | `batched_render_system.cpp:32-54` | batch 创建时检查 cameras 非空、分辨率一致，并分配 command buffer | RTXNS Week 2 先限制 batch 内同分辨率，减少 target pool 复杂度 |
| SAPIEN | `batched_render_system.cpp:127-148` | `takePicture()` 等上一帧 semaphore，渲染所有 camera，提交 copy command buffer，signal frame counter | RTXNS Week 3 的 `submit_frame_batch()` 可参考 token/frame counter 模式 |
| SAPIEN | `batched_render_system.cpp:303-313` | batch 创建后检查 scene version，不允许结构变更 | RTXNS batch 活跃后禁止 `load_scene()` 或 camera 数量变化，除非 `clear_cameras()` / `rebuild_batch()` |
| LuisaRender | `pipeline.h:84`, `pipeline.h:212-214` | Pipeline 内部保存 `_cameras`，暴露 `camera_count()` / `camera(i)` | RTXNS 增加 `m_views`、`camera_count()`、`view(i)` |
| LuisaRender | `pipeline.cpp:71-75` | pipeline create 时一次 build 所有 camera instance | RTXNS `add_camera()` 只创建 view slot，不重新 load scene |
| LuisaRender | `integrator.cpp:41-56` | 一个 `CommandBuffer` 内 loop 所有 camera | RTXNS `render_frame_batch()` 一个 `ICommandList` 内 loop 所有 camera |
| LuisaRender | `command_buffer.h:26-43` | command 记录和同步边界集中在 CommandBuffer | RTXNS 不要在每个 pass 内 execute/wait，集中在 batch submit 处 |

SAPIEN 可参考的最小逻辑:

```cpp
// SAPIEN pattern, simplified
for (auto& cam : mCameras)
    cam->getInternalRenderer().render(cam->getInternalCamera(), {}, {}, {}, {});

mFrameCounter++;
queue.submit(mCommandBuffer.get(), ..., mSemaphore.get(), mFrameCounter, ...);
```

LuisaRender 可参考的最小逻辑:

```cpp
// LuisaRender pattern, simplified
CommandBuffer command_buffer{&stream};
for (auto i = 0u; i < pipeline().camera_count(); i++) {
    auto camera = pipeline().camera(i);
    camera->film()->prepare(command_buffer);
    _render_one_camera(command_buffer, camera);
    camera->film()->download(command_buffer, pixels.data());
}
```

RTXNS 要落成的对应逻辑:

```cpp
// RTXNS target pattern
auto commandList = device->createCommandList();
commandList->open();

m_scene->Refresh(commandList, m_frame_index++);
recordOrUpdateSharedShadowAS(commandList);

for (uint32_t cameraIndex : cameraIndices) {
    auto& view = m_views.at(cameraIndex);
    recordRenderView(commandList, view);
}

commandList->close();
device->executeCommandList(commandList);
waitOrSignalBatchToken(...);
```

### 11.3 推荐数据结构骨架

第一版可以先放在 `headless_pbr.cpp` 的 `HeadlessPbrScene::Impl` 内部，稳定后再拆成头文件。

```cpp
struct CameraDesc
{
    std::array<float, 3> position{};
    std::array<float, 3> target{};
    std::array<float, 3> up{0.0f, 1.0f, 0.0f};
    float fov_degrees = 60.0f;
    uint32_t width = 1024;
    uint32_t height = 768;
    float z_near = 0.1f;
    float z_far = 1000.0f;
};

struct RenderViewSlot
{
    CameraDesc desc;
    donut::app::FirstPersonCamera camera;
    PlanarView view;

    uint32_t width = 0;
    uint32_t height = 0;
    float z_near = 0.1f;
    float z_far = 1000.0f;

    std::shared_ptr<FramebufferFactory> framebufferFactory;
    nvrhi::TextureHandle colorTarget;
    nvrhi::TextureHandle depthTarget;
    nvrhi::StagingTextureHandle readbackTarget;

    nvrhi::TextureHandle shadowTarget;
    nvrhi::TextureHandle shadowBlurTemp;
    nvrhi::TextureHandle compositeOutput;
    nvrhi::TextureHandle litColorSRV;

    // Week 2: keep per-dispatch binding sets alive until command list finishes.
    std::vector<nvrhi::BindingSetHandle> frameBindingScratch;

    HeadlessPbrScene::FrameStats lastStats{};
};
```

Week 3 再把 `readbackTarget` 扩展成 ring:

```cpp
struct ReadbackSlot
{
    nvrhi::StagingTextureHandle staging;
    uint64_t token = 0;
    bool hasPendingCopy = false;
};

std::array<ReadbackSlot, 3> readbackRing;
```

### 11.4 Micro-patch 顺序

每个 micro-patch 都要能编译，并能跑旧的单相机 smoke test。不要把 Week 1/2/3 混在一个巨大 patch 里。

#### Patch A: 只抽 `CameraDesc`，不改变行为

目标: 让旧 `set_camera(...)` 先转换成 `CameraDesc`，再调用内部 helper。

改动:

1. 在 `headless_pbr.h` 声明 `CameraDesc`，或先放在 `headless_pbr.cpp` 内部。
2. 新增 helper:

```cpp
CameraDesc makeCameraDesc(
    const std::array<float, 3>& position,
    const std::array<float, 3>& target,
    const std::array<float, 3>& up,
    float fovDegrees,
    uint32_t width,
    uint32_t height,
    float zNear,
    float zFar);
```

3. 旧 `set_camera(...)` 保持签名不变，只负责构造 desc。
4. 当前输出必须完全不变。

验收:

- `render_frame()` 仍能跑。
- `get_last_frame_stats()` 字段仍存在。
- 旧 Python 调用无需修改。

#### Patch B: 引入 `RenderViewSlot`，把 camera 0 迁入 slot

目标: 行为不变，但内部不再直接依赖单份 `m_camera/m_view/m_framebuffer_factory`。

改动:

1. 新增 `std::vector<RenderViewSlot> m_views;`
2. 构造函数里创建默认 camera 0:

```cpp
m_views.emplace_back();
set_camera( /* old default camera args */ );
```

3. 把 `resize_targets(width, height)` 改成:

```cpp
void resize_targets(RenderViewSlot& slot, uint32_t width, uint32_t height);
```

4. 把 `set_camera(...)` 内部改成:

```cpp
set_camera_desc(0, desc);
```

5. 新增:

```cpp
void set_camera_desc(uint32_t index, const CameraDesc& desc);
```

6. `render_frame()` 一开始取:

```cpp
auto& slot = m_views[0];
```

然后把本函数内的 `m_view`、`m_color_target`、`m_depth_target`、`m_framebuffer_factory`、`m_readback_target` 逐步替换为 `slot.view`、`slot.colorTarget` 等。

注意:

- 这一步先不要删除所有旧成员。可以先替换完再删除，降低编译错误定位难度。
- 如果编译错误太多，先只替换 `set_camera()` 和 `resize_targets()`，再替换 `render_frame()`。

验收:

- 单相机渲染结果和 Patch A 一致。
- resize 后仍能正确更新 framebuffer。
- 析构时能释放 slot 内资源。

#### Patch C: 添加多相机 API，但 `render_frame_batch()` 暂时可以内部循环同步

目标: 先把 Python API 和 native API 打通，让测试脚本能创建多个 camera。

改动:

1. `headless_pbr.h` 增加:

```cpp
uint32_t add_camera(...same args as set_camera...);
void set_camera_at(uint32_t index, ...same args...);
uint32_t camera_count() const noexcept;
std::vector<uint8_t> render_frame(uint32_t camera_index);
std::vector<std::vector<uint8_t>> render_frame_batch(const std::vector<uint32_t>& camera_indices);
```

2. 旧 `render_frame()` 保持:

```cpp
std::vector<uint8_t> render_frame()
{
    return render_frame(0);
}
```

3. 第一版 `render_frame_batch()` 可以临时写成循环调用 `render_frame(cameraIndex)`，但必须在文档/注释里标明这是临时桥接，不算性能完成。

验收:

- Python 能:

```python
ids = [scene.add_camera(...), scene.add_camera(...)]
imgs = scene.render_frame_batch(ids)
assert len(imgs) == 2
```

- Week 1 汇报可以展示 API 已打通，但不能宣称吞吐收益。

#### Patch D: 真正单 command list 的同步 batch

目标: Week 2 核心 patch。把临时循环版替换成一次 command list。

建议拆函数:

```cpp
void record_scene_once(nvrhi::ICommandList* commandList);
void record_shadow_as_once(nvrhi::ICommandList* commandList, HeadlessPbrScene::FrameStats& batchStats);
void record_render_view(nvrhi::ICommandList* commandList, RenderViewSlot& slot, HeadlessPbrScene::FrameStats& stats);
std::vector<uint8_t> readback_view(RenderViewSlot& slot);
```

`render_frame_batch()` 结构:

```cpp
std::vector<std::vector<uint8_t>> render_frame_batch(const std::vector<uint32_t>& indices)
{
    validate_camera_indices(indices);

    auto* device = m_context->device();
    auto commandList = device->createCommandList();
    commandList->open();

    m_scene->Refresh(commandList, m_frame_index++);
    record_shadow_as_once(commandList, m_lastBatchStats);

    for (auto index : indices) {
        auto& slot = m_views[index];
        slot.frameBindingScratch.clear();
        record_render_view(commandList, slot, slot.lastStats);
    }

    commandList->close();
    device->executeCommandList(commandList);
    device->waitForIdle();

    std::vector<std::vector<uint8_t>> outputs;
    outputs.reserve(indices.size());
    for (auto index : indices) {
        outputs.push_back(readback_view(m_views[index]));
        m_views[index].frameBindingScratch.clear();
    }
    return outputs;
}
```

关键要求:

- `m_scene->Refresh(...)` 只能在 batch 开头调用一次。
- 首帧 BLAS build 只能调用一次。
- 后续 TLAS update 只能调用一次。
- `executeCommandList()` 和 `waitForIdle()` 只能调用一次。

验收:

- 给 `executeCommandList` 和 `waitForIdle` 附近加临时 debug counter，N=4 batch 时计数为 1。
- `blas_build_ms` 只在首 batch 非零。
- `render_frame_batch([0])` 性能不低于旧 `render_frame()` 95%。

#### Patch E: 修 RT shadow pass 的 per-dispatch 状态

目标: 避免 batch 内多个 camera 连续调用 `renderShadow()` 时覆盖 pass 成员导致生命周期不明确。

当前风险:

- `RayTracedShadowPass.cpp:226`: `m_shadowBindingSet = ...`
- `RayTracedShadowPass.cpp:275`: `m_compositeBindingSet = ...`
- `RayTracedShadowPass.cpp:316`: `m_blurBindingSet = ...`
- `RayTracedShadowPass.cpp:191` 和 `307`: 写同一个 `m_shadowConstantBuffer`

建议改法:

1. 保留 pipeline/layout 在 `RayTracedShadowPass`。
2. 让调用方传入 scratch vector:

```cpp
void renderShadow(..., std::vector<nvrhi::BindingSetHandle>* keepAlive);
```

3. 创建 binding set 后:

```cpp
auto bindingSet = m_device->createBindingSet(setDesc, m_shadowBindingLayout);
if (keepAlive) keepAlive->push_back(bindingSet);
state.bindings = { bindingSet };
```

4. 常量 buffer 第一版可以仍共用，但更推荐让 `RenderViewSlot` 持有:

```cpp
nvrhi::BufferHandle shadowConstantBuffer;
```

然后 `renderShadow()` 接收 constant buffer 参数，或新增 `ShadowDispatchResources`。

验收:

- N=4 batch 连续渲染不会崩溃、不会出现后一个 camera 覆盖前一个 camera 输出。
- D3D/Vulkan validation 如果开启，不应报 binding/resource lifetime 相关错误。

#### Patch F: Week 3 异步 readback

目标: 不再在 steady-state `render_frame_batch()` 内直接 `waitForIdle()`。

建议 API:

```cpp
uint64_t submit_frame_batch(const std::vector<uint32_t>& indices);
bool is_frame_batch_ready(uint64_t token) const;
std::vector<std::vector<uint8_t>> read_frame_batch(uint64_t token);
```

第一版如果 NVRHI fence/timeline 不熟，可以先封装为 `PendingBatch`:

```cpp
struct PendingBatch
{
    uint64_t token = 0;
    std::vector<uint32_t> cameraIndices;
    // Fence/timeline handle goes here when available.
    bool submitted = false;
};
```

若底层 fence 接口暂时不清楚，宁可保守:

- `submit_frame_batch()` 只 submit，不 map。
- `read_frame_batch()` 内等待对应 work 完成。
- 不要读正在写入的 staging slot。

验收:

- 同步 `render_frame_batch()` 可由 `submit + wait/read` 实现。
- 异步路径连续 submit 2-3 个 batch 不崩溃。
- 输出顺序和 token 对应。

### 11.5 Python 绑定骨架

`py_bindings_common.h` 里按现有 `render_frame()` 风格加绑定。注意长时间渲染要释放 GIL。

建议接口:

```cpp
.def("add_camera",
    &rtxns::python::HeadlessPbrScene::add_camera,
    py::arg("position"),
    py::arg("target"),
    py::arg("up"),
    py::arg("fov_degrees"),
    py::arg("width"),
    py::arg("height"),
    py::arg("z_near") = 0.1f,
    py::arg("z_far") = 1000.0f)

.def("render_frame_batch",
    [](rtxns::python::HeadlessPbrScene& self, const std::vector<uint32_t>& indices)
    {
        auto frames = [&self, &indices]()
        {
            py::gil_scoped_release release;
            return self.render_frame_batch(indices);
        }();

        py::list out;
        for (const auto& pixels : frames)
            out.append(py::bytes(reinterpret_cast<const char*>(pixels.data()),
                                 static_cast<py::ssize_t>(pixels.size())));
        return out;
    },
    py::arg("camera_indices"))
```

Python smoke test skeleton:

```python
import sys
from pathlib import Path

sys.path.insert(0, r"E:\cplus\RTXNS\bin\windows-x64")
import DonutRenderPyNative as rr

rr.init(runtime_dir=r"E:\cplus\RTXNS", backend="vulkan")
scene = rr.create_scene()
scene.load_scene(r"...\bistro.gltf")

cam0 = scene.add_camera(position=[...], target=[...], up=[0,1,0],
                        fov_degrees=60, width=1024, height=768)
cam1 = scene.add_camera(position=[...], target=[...], up=[0,1,0],
                        fov_degrees=60, width=1024, height=768)

frames = scene.render_frame_batch([cam0, cam1])
assert len(frames) == 2
assert len(frames[0]) == 1024 * 768 * 4
rr.destroy()
```

### 11.6 Benchmark 工具骨架

新增 `tools/parallel_render_batch_benchmark.py`，它必须同时测三种模式:

1. `single_loop`: 单进程内循环调用旧 `render_frame()` N 次。
2. `native_batch`: 单进程内调用新 `render_frame_batch([0..N-1])`。
3. `multiprocess`: 复用当前 `parallel_render_poc.py` 结果或直接调用。

核心输出 JSON:

```json
{
  "mode": "native_batch",
  "num_cameras": 4,
  "frames": 100,
  "throughput_fps": 0.0,
  "avg_batch_ms": 0.0,
  "avg_gpu_util": 0.0,
  "peak_vram_mb": 0,
  "execute_count_per_batch": 1,
  "wait_count_per_batch": 1,
  "blas_build_count": 1
}
```

最小 benchmark 逻辑:

```python
def run_native_batch(scene, camera_ids, frames):
    scene.render_frame_batch(camera_ids)  # warmup
    t0 = time.perf_counter()
    for _ in range(frames):
        imgs = scene.render_frame_batch(camera_ids)
    dt = time.perf_counter() - t0
    return frames * len(camera_ids) / dt
```

Week 2 前不要过度优化 benchmark；先保证结果可信。

### 11.7 明确不要做的事

为了让执行 Agent 不跑偏，以下事情先不要做:

1. 不要一开始删除 `g_context`。当前瓶颈不是全局 context，而是缺少 batch object 和同步边界过重。`g_context` 可留到多 GPU 阶段再改。
2. 不要一开始做多队列 Vulkan timeline semaphore。先完成同步 batch，再异步 readback。
3. 不要重写 `ForwardShadingPass` 或 `RayTracedShadowPass` shader。并行优化主要在调度和资源组织。
4. 不要把多个进程的结果继续包装成“batch”。目标是 renderer 内部单 device batch。
5. 不要同时支持不同分辨率 batch。Week 2 先要求同分辨率，不同分辨率 Week 4 通过 scheduler 拆分。
6. 不要为追求性能牺牲旧 API。`render_frame()` 和旧 Python wrapper 必须持续可用。

### 11.8 每个 patch 的最低验收命令

根据本机环境调整路径，但每个 patch 至少跑这些:

```powershell
# 1. 构建
cmake --build E:\cplus\RTXNS\build --config Release

# 2. 原有单相机 smoke
python E:\cplus\RTXNS\tools\test_rt_shadow.py

# 3. RT shadow / blur 回归
python E:\cplus\RTXNS\tools\test_bistro_shadow.py

# 4. batch smoke
python E:\cplus\RTXNS\tools\parallel_render_batch_benchmark.py --mode native_batch --cameras 4 --frames 20
```

如果构建目录不同，先用 `Get-ChildItem E:\cplus\RTXNS\build` 确认。不要为了测试通过去改无关输出文件。

### 11.9 交付给老师的每周材料清单

代码 Agent 每周结束时至少生成:

| 文件 | 用途 |
|------|------|
| `output/parallel_render/weekX_results.json` | 原始性能数据 |
| `output/parallel_render/weekX_scaling.png` | 吞吐/加速比图 |
| `output/parallel_render/weekX_vram.png` | 显存曲线 |
| `output/parallel_render/weekX_samples.png` | 多 camera 输出拼图 |
| `docs/RTXNS_Parallel_Render_WeekX_Report.md` | 周报正文 |

周报正文必须包含:

- 本周解决的瓶颈
- 实现文件清单
- 新增 API
- 性能收益表
- 未解决风险
- 下一周计划

### 11.10 代码 Agent 的第一条任务提示词

可以直接把下面这段给执行 Agent:

```text
请在 E:\cplus\RTXNS 中做并行渲染 Week 1 的第一批改造。
先阅读 output\parallel_render_efficiency_roadmap.md 的第 11 节。
本次只做 Patch A-C:
1. 抽 CameraDesc；
2. 引入 RenderViewSlot，把 camera 0 迁入 slot，保持旧 render_frame 行为不变；
3. 增加 add_camera / set_camera(index) / camera_count / render_frame_batch(indices) 的 Python API。

注意:
- 不要删除 g_context；
- 不要做异步 readback；
- render_frame_batch 第一版可以临时循环调用 render_frame(cameraIndex)，但必须标注 TODO；
- 旧 set_camera 和旧 render_frame 必须兼容；
- 完成后运行单相机 smoke 和一个两相机 batch smoke。

重点文件:
- src/PythonBindings/headless_pbr.h
- src/PythonBindings/headless_pbr.cpp
- src/PythonBindings/py_bindings_common.h
- tools/parallel_render_batch_benchmark.py
```
