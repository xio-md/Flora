# RTXNS 并行渲染优化阶段汇报

> 汇报日期: 2026-07-11  
> 范围: 多进程瓶颈复现、多相机资源模型、同步 batch 渲染、异步 readback 帧流水  
> 场景: Bistro 1024x768  
> 目标: 提升并行渲染效率，减少 Python 调用、command submit、全局 GPU 等待和重复场景资源占用

## 概述

本阶段围绕 RTXNS 当前并行渲染效率偏低的问题，完成了从“多进程并行渲染 POC”到“单进程多相机 batch 渲染”的核心改造。原始方案中，每个 worker 各自创建 renderer、加载场景、构建 BLAS/TLAS，并在每帧 `render_frame()` 末尾执行全局 `waitForIdle()`。这会导致 GPU 欠载、进程间调度竞争和显存线性增长。

本次优化将渲染器内部从单相机结构扩展为多相机 slot 结构，并提供 `render_frame_batch()`、`submit_frame_batch()`、`is_batch_ready()`、`read_frame_batch()` 等接口。渲染提交从 N 次 Python 调用、N 个 command list、N 次 execute/wait，收敛为一次 batch 提交；同时通过 readback ring 和 NVRHI `EventQuery` 把“提交渲染”和“读取结果”解耦，消除稳态全局 `waitForIdle()`。

最终实验结果显示，异步 batch 路径在 N=3 cameras 下达到 **251 FPS**，相比同步 batch 基线 **140 FPS** 提升约 **79%**，相比首次异步实现的 **203 FPS** 再提升 **24%**。关键正确性问题也已修复: `is_batch_ready()` 从修复前错误地立即返回 `True`，变为提交后正确返回 `False`，说明 EventQuery 已标记到本次 command list 提交之后的队列时间点。

| 指标 | 改造前 / 中间状态 | 当前状态 | 收益 |
|------|------------------|----------|------|
| Python 调用 | N 次 `render_frame()` | 1 次 batch API | N -> 1 |
| command list | N 个 | 1 个 batch command list | N -> 1 |
| `Scene::Refresh` | N 次 | 1 次 | N -> 1 |
| `executeCommandList` | N 次 | 1 次 | N -> 1 |
| 稳态 `waitForIdle` | 每帧/每 batch 1 次 | 0 次 | 全局等待消除 |
| BLAS/TLAS | 多进程 N 份 | 单进程共享 1 份 | 显存和构建开销降低 |
| Week 2 同步 batch | 140 FPS (N=3) | 作为基线 | - |
| Week 3 初版异步 | 203 FPS | +45% vs 同步 batch | 初步流水化 |
| Week 3 修复后异步 | 251 FPS | +79% vs 同步 batch, +24% vs 初版异步 | 当前主结果 |
| `is_batch_ready` | 修复前 `True` (错误提前完成) | 修复后 `False` (正确未完成) | EventQuery 修复生效 |
| submit 耗时 | 6.8 ms/批 | 2.7 ms/批 | CPU 提交更轻 |

---

## 一、问题背景

### 1. 多进程 POC 的瓶颈

前期通过 `parallel_render_poc.py` 复现了多进程并行渲染方案。该方案虽然可以通过多个进程同时渲染不同相机，提高表观吞吐，但工程上存在三个主要瓶颈。

| 瓶颈 | 现象 | 原因 |
|------|------|------|
| GPU 欠载 | N=1 时 GPU 利用率很低 | `render_frame()` 内部同步等待，CPU/GPU 缺少流水 |
| 调度竞争 | N=4 后加速趋缓 | 多个独立 Vulkan device/queue 互相抢占 |
| 显存低效 | N=8 显存可到 15 GB 级别 | 每进程独立加载完整 Bistro 场景和加速结构 |

多进程 baseline 数据如下:

| N workers | 总吞吐 | 加速比 | per-worker FPS |
|-----------|--------|--------|----------------|
| 1 | 224 FPS | 1.00x | 224 |
| 2 | 302 FPS | 1.34x | 164 / 138 |
| 4 | 505 FPS | 2.25x | 92 - 197 |

这些数据说明，简单堆进程不能从根本上解决并行渲染效率问题。更合理的方向是参考成熟渲染器的 batch 思路: 一个 renderer 内部管理多相机视图，共享 scene、texture cache、BLAS/TLAS 和 command submit 边界。

### 2. 原始 RTXNS 内部结构限制

改造前，`HeadlessPbrScene::Impl` 内部使用单份相机和渲染目标:

```cpp
m_camera
m_view
m_color_target
m_depth_target
m_readback_target
m_shadowTarget
m_compositeOutput
```

这种结构只能自然表达一个视角。若要渲染 N 个 camera，只能在外层循环调用 `render_frame()`，导致每个视角重复刷新 scene、录制 command list、提交 GPU、等待 GPU 空闲和 CPU readback。

---

## 二、实现内容

### 1. 多相机资源模型

第一步将 renderer 内部从单相机成员变量扩展为多相机 slot。

核心新增结构:

```cpp
struct CameraDesc {
    std::array<float, 3> position;
    std::array<float, 3> target;
    std::array<float, 3> up;
    float fov_degrees;
    uint32_t width;
    uint32_t height;
    float z_near;
    float z_far;
};

struct RenderViewSlot {
    CameraDesc desc;
    FirstPersonCamera camera;
    PlanarView view;

    TextureHandle colorTarget;
    TextureHandle depthTarget;
    StagingTextureHandle readbackTarget;

    TextureHandle shadowTarget;
    TextureHandle shadowBlurTemp;
    TextureHandle compositeOutput;
    TextureHandle litColorSRV;

    std::array<ReadbackRingSlot, 2> readbackRing;
    uint32_t ringWriteIdx;
};
```

参考代码:

| 文件 | 位置 | 说明 |
|------|------|------|
| `src/PythonBindings/headless_pbr.h` | `CameraDesc`, `HeadlessPbrScene` API | 多相机接口声明 |
| `src/PythonBindings/headless_pbr.cpp` | `RenderViewSlot`, `m_views` | 多相机资源容器 |
| `src/PythonBindings/headless_pbr.cpp` | `set_camera_desc()` | 创建/更新相机 slot |
| `src/PythonBindings/headless_pbr.cpp` | `resize_slot_targets()` | 为每个 camera 分配 color/depth/readback/shadow 资源 |

新增 Python/C++ API:

```cpp
uint32_t add_camera(...);
void set_camera_at(uint32_t index, ...);
uint32_t camera_count() const noexcept;
std::vector<uint8_t> render_frame(uint32_t camera_index);
std::vector<std::vector<uint8_t>> render_frame_batch(const std::vector<uint32_t>& camera_indices);
```

旧 API 保持兼容:

```python
img = scene.render_frame()
```

默认仍然渲染 camera 0，不影响已有脚本。

### 2. 同步版 batch 渲染

第二步实现单 command list 批量多相机渲染。核心思想是把原 `render_frame()` 中的渲染步骤拆成可复用录制函数，使多个 camera 在同一个 command list 中依次录制。

同步 batch 路径:

```text
render_frame_batch_v2([0, 1, 2, 3])
  createCommandList()
  open()
  Scene::Refresh()                 1 次
  record_or_build_shadow_as()       1 次，共享 BLAS/TLAS
  for each camera:
      sync_and_record_view()        多次录制 raster/shadow/composite/copy
  close()
  executeCommandList()              1 次
  waitForIdle()                     1 次
  readback_slot()                   多个 camera 逐个读回
```

关键函数:

| 函数 | 作用 |
|------|------|
| `sync_and_record_view(cmdList, camera_index, use_ring)` | 将指定 camera slot 的 view/resource 同步到 legacy 渲染路径，并在传入 command list 上录制完整渲染 |
| `readback_slot(slot)` | 从单个 slot 的 staging texture 读回图像 |
| `render_frame_batch_v2(indices)` | Week 2 同步 batch 版本，保留为对照和 fallback |
| `record_or_build_shadow_as(cmdList)` | 当前共享 AS 构建/更新函数，供同步和异步 batch 共用 |

同步版 batch 的主要收益不是最终 FPS，而是打通了结构: Python 调用、scene refresh、command submit、AS 构建从 N 次收敛到 1 次。实验中 N=2 camera 从 277 FPS 提升到 345 FPS，提升约 25%。N=4 时提升较小，主要原因是末尾仍保留 `waitForIdle()`，同步 readback 成为绝对瓶颈。

| N cameras | 旧循环 | 同步 batch | 加速比 | 节省等待 |
|-----------|--------|------------|--------|----------|
| 1 | 643 FPS | 610 FPS | 0.95x | 0 |
| 2 | 277 FPS | 345 FPS | 1.25x | 100 次 |
| 4 | 201 FPS | 207 FPS | 1.03x | 300 次 |

### 3. 异步 readback 与帧流水

第三步将同步 batch 拆成 submit/read 两个阶段，避免每个 batch 末尾全局等待 GPU。

新增 API:

```cpp
uint64_t submit_frame_batch(const std::vector<uint32_t>& camera_indices);
bool is_batch_ready(uint64_t token) const;
std::vector<std::vector<uint8_t>> read_frame_batch(uint64_t token);
```

Python 使用方式:

```python
token0 = scene.submit_frame_batch([0, 1, 2])
token1 = scene.submit_frame_batch([0, 1, 2])

images0 = scene.read_frame_batch(token0)
images1 = scene.read_frame_batch(token1)
```

异步路径:

```text
submit_frame_batch()
  open command list
  Scene::Refresh()
  record_or_build_shadow_as()
  for camera:
      sync_and_record_view(use_ring=true)
  close
  executeCommandList()
  setEventQuery()              标记本次提交后的队列时间点
  store PendingBatch
  return token

read_frame_batch(token)
  find PendingBatch
  waitEventQuery(query)
  map staging ring texture
  copy pixels to CPU bytes
  erase PendingBatch
```

每个 camera slot 新增 readback ring:

```cpp
std::array<ReadbackRingSlot, 2> readbackRing;
uint32_t ringWriteIdx;
```

每次 submit 会把当前写入的 ring index 保存到 `PendingBatch::ringIndices`，read 时不再依赖当前 `ringWriteIdx` 反推槽位，避免读回错位。

### 4. Correctness 修复

异步路径初版暴露了几处关键正确性问题，本阶段已经完成主要修复。

| 级别 | 问题 | 修复 | 当前状态 |
|------|------|------|----------|
| P0 | `EventQuery` 放在 `executeCommandList` 前，导致 `is_batch_ready()` 可能提前返回 | `executeCommandList()` 后再 `setEventQuery()` | 已修复 |
| P1 | readback ring 读回槽位依赖当前 `ringWriteIdx`，连续提交后可能读错 | `PendingBatch` 保存每个 camera 的 `ringIndices` | 已修复读回定位 |
| P1 | batch 路径中 camera 0 legacy 状态可能沿用前一个 camera | `sync_and_record_view()` 对所有 camera index 无条件同步 slot 状态 | batch 主路径已修复 |
| P1 | async 路径首帧不构建 AS，RT shadow 可能直接 fallback | 抽出 `record_or_build_shadow_as()`，同步和异步 batch 共用 | 已修复 |
| P2 | OMM bake/cache 未并入 batch AS 构建 | 保留 TODO，计划通过预 bake cache 接入 | Week 4 处理 |

最新验证结果:

```text
is_batch_ready: False
```

该结果是关键正确性信号。修复前 query 标记在 submit 之前，刚提交后 `is_batch_ready()` 可能立即返回 `True`。修复后返回 `False`，说明 query 正确对应到本次 command list 提交之后的 GPU 队列进度。

---

## 三、实验结果

### 1. 同步 batch 到异步 batch 的收益

实验配置: Bistro 1024x768，RT 阴影关闭，N=3 cameras，20 批。

| 方案 | 吞吐 | submit 耗时 | 每批行为 | 说明 |
|------|------|-------------|----------|------|
| Week 2 同步 batch | 140 FPS | 嵌入同步调用 | execute 后 wait/read | 仍有全局等待 |
| Week 3 初版异步 | 203 FPS | 6.8 ms/批 | submit/read 分离 | +45%，但存在 EventQuery 提前完成 bug |
| Week 3 修复后异步 | 251 FPS | 2.7 ms/批 | query 正确标记提交后时间点 | +79% vs 同步 batch，+24% vs 初版异步 |

吞吐变化:

```text
同步 batch:      140 FPS
初版 async:      203 FPS
修复后 async:    251 FPS
```

收益解释:

1. `waitForIdle()` 被替换为 per-batch `waitEventQuery()`，不再清空整个 GPU 队列。
2. submit 阶段不再阻塞等待 readback，CPU 可以继续提交下一批或执行仿真/训练逻辑。
3. 多相机共享 scene refresh、AS build/update、texture cache 和 command submit 边界。
4. EventQuery 修复后，ready 状态与真实 GPU 完成点一致，避免错误地读取未完成 staging 资源。

### 2. 功能验证

当前验证覆盖:

```text
[1] Sync batch                  OK
[2] Async submit + read         OK
[3] is_batch_ready              False after submit, correct
[4] Pipelined submit/read       OK
[5] Shared AS build path        OK
[6] Release build               OK
```

构建验证:

```text
cmake --build E:\cplus\RTXNS\build --config Release
```

已通过，输出:

```text
DonutRenderPyNative.pyd
RtxRenderPy.pyd
```

### 3. 阶段性收益汇总

| 阶段 | 工作重点 | 阶段成果 | 可汇报收益 |
|------|----------|----------|------------|
| Week 1 | 多进程瓶颈复现 + 多相机资源模型 | `CameraDesc`、`RenderViewSlot`、`m_views`、多相机 API | 建立 baseline，确认多进程显存和调度瓶颈 |
| Week 2 | 单 command list 同步 batch | `sync_and_record_view()`、`render_frame_batch_v2()`、共享 AS | N=2 提升 25%，调用/提交/refresh N -> 1 |
| Week 3 | 异步 readback + 帧流水 | `submit_frame_batch()`、`is_batch_ready()`、`read_frame_batch()`、readback ring、EventQuery | N=3 达到 251 FPS，稳态 `waitForIdle` 为 0 |

---

## 四、当前代码结构

### 1. 主路径

当前公开同步 batch API:

```cpp
std::vector<std::vector<uint8_t>>
HeadlessPbrScene::render_frame_batch(const std::vector<uint32_t>& camera_indices)
{
    uint64_t token = m_impl->submit_frame_batch_impl(camera_indices);
    return m_impl->read_frame_batch_impl(token);
}
```

这意味着 `render_frame_batch()` 已经复用 Week 3 async submit/read 路径，只是对外表现为同步便利函数。训练、仿真或数据采集任务可以直接使用异步 API，手动控制 submit/read 间隔。

### 2. 关键代码参考

| 模块 | 文件 | 函数/结构 | 说明 |
|------|------|-----------|------|
| 多相机 API | `src/PythonBindings/headless_pbr.h` | `CameraDesc`, `add_camera`, `render_frame_batch` | 对外接口 |
| Python 绑定 | `src/PythonBindings/py_bindings_common.h` | `render_frame_batch`, `submit_frame_batch`, `read_frame_batch` | Python 可调用入口 |
| 多相机资源 | `src/PythonBindings/headless_pbr.cpp` | `RenderViewSlot`, `m_views` | 每 camera 独立 view/render target |
| 视角录制 | `src/PythonBindings/headless_pbr.cpp` | `sync_and_record_view()` | 在共享 command list 中录制单个 camera |
| 共享 AS | `src/PythonBindings/headless_pbr.cpp` | `record_or_build_shadow_as()` | batch 内共享 BLAS/TLAS build/update |
| 异步提交 | `src/PythonBindings/headless_pbr.cpp` | `submit_frame_batch_impl()` | 录制、execute、设置 EventQuery、保存 PendingBatch |
| 异步读回 | `src/PythonBindings/headless_pbr.cpp` | `read_frame_batch_impl()` | 等待 token、按 ring index 读回 |
| 状态追踪 | `src/PythonBindings/headless_pbr.cpp` | `PendingBatch` | 保存 token、camera indices、ring indices、query |

### 3. 与成熟渲染器思路的对应

本阶段实现与成熟渲染器的做法保持一致:

| 参考方向 | 成熟实现思路 | RTXNS 当前对应 |
|----------|--------------|----------------|
| SAPIEN batched render | 一次 takePicture 渲染多个 camera，使用 frame counter/semaphore 控制完成点 | `submit_frame_batch()` + `EventQuery` + token |
| LuisaRender command buffer | 在一个 command buffer 中集中记录多视角/多 pass，提交边界集中管理 | `render_frame_batch_v2()` / async command list 内 loop cameras |
| GPU readback pipeline | 多 staging buffer 轮转，避免 CPU map 等待当前 GPU 写入 | 每 camera `readbackRing[2]` |
| Shared scene resource | 多视角共享 scene、texture、AS | `m_views` 只保存 view/render targets，scene/AS 全局共享 |

---

## 五、风险与后续收尾

### 1. readback ring 的极端积压

当前 `PendingBatch` 已保存每个 camera 的 ring index，解决了“读回时根据当前 `ringWriteIdx` 反推导致读错槽位”的问题。下一步建议补充 ring slot backpressure: 当同一 camera 的同一 ring slot 仍被未读 token 占用时，应阻塞、报错、自动等待最早 token，或将 ring 深度从 2 扩展到 3/4。

这不是当前 N=3 正常流水测试的阻塞项，但会影响后续更深队列的稳定性。

### 2. 旧 `render_frame(index)` 的状态同步收口

batch 主路径中的 `sync_and_record_view()` 已对任意 camera index 无条件同步 slot 状态。后续建议把旧 `render_frame_for_index()` 也改为调用同一个 `sync_legacy_from_slot()` helper，避免 `render_frame(1)` 后再 `render_frame(0)` 时 legacy 状态残留。

### 3. OMM 与 async batch 的集成

当前 `record_or_build_shadow_as()` 已统一 batch 路径的 AS 构建，但 OMM bake/cache 逻辑仍保留在单相机 `render_frame()` 路径中。原因是 OMM bake 和 GPU OMM build 当前包含同步等待，与 async batch 的无全局等待目标冲突。

Week 4 建议采用预 bake cache 方式集成:

1. 在正式 batch 前离线或预热阶段完成 OMM bake。
2. batch AS build 只加载 cache 并上传 OMM buffer。
3. 避免 steady-state batch 路径中插入 `waitForIdle()`。

### 4. Benchmark 脚本路径参数化

部分 smoke/benchmark 脚本仍硬编码 `D:\RTXNS` 和 `D:\niagara_bistro`。后续建议统一改成命令行参数或自动从 repo root 推导，避免测到旧二进制或不同场景路径。

---

## 六、修改文件清单

### 核心代码

```text
src/PythonBindings/headless_pbr.h
  + CameraDesc
  + add_camera / set_camera_at / camera_count
  + render_frame(index) / render_frame_batch(indices)
  + submit_frame_batch / is_batch_ready / read_frame_batch

src/PythonBindings/headless_pbr.cpp
  + RenderViewSlot / m_views
  + set_camera_desc / resize_slot_targets / add_camera_slot
  + sync_and_record_view
  + record_or_build_shadow_as
  + render_frame_batch_v2
  + submit_frame_batch_impl / is_batch_ready_impl / read_frame_batch_impl
  + PendingBatch + readback ring

src/PythonBindings/py_bindings_common.h
  + 多相机 Python 绑定
  + 异步 batch Python 绑定
  + render/read 路径释放 GIL
```

### 测试和工具

```text
tools/smoke_batch_test.py
  多相机 API smoke test

tools/smoke_batch_v2.py
  同步 single-command-list batch 测试

tools/smoke_async.py
  异步 submit/read、is_batch_ready、流水测试

tools/bench_batch_real.py
  多 camera batch vs 旧循环 benchmark

tools/bench_batch_large.py
  N=1/2/4 多规模吞吐对比
```

### 展示材料

```text
output/parallel_render/parallel_render_results.json
output/parallel_render/throughput_compare.png
output/parallel_render/worker_*.png
docs/RTXNS_Parallel_Render_Week1_Report.md
docs/RTXNS_Parallel_Render_Week2_Report.md
docs/RTXNS_Parallel_Render_Week3_Report.md
```

![并行渲染吞吐对比](../output/parallel_render/throughput_compare.png)

---

## 七、下周计划

下一阶段建议定位为“并行渲染稳定化 + 调度器 + OMM 预烘焙接入”。

### 1. Ring backpressure 与更深流水

- 为每个 pending token 记录 `(cameraIndex, ringIndex)` 占用。
- submit 前检查目标 ring slot 是否仍 pending。
- 支持策略:
  - 简单版: 冲突时抛出异常，提示用户先 read。
  - 稳定版: 自动等待最早占用该 slot 的 token。
  - 性能版: ring 深度从 2 扩展为可配置 K。

### 2. BatchScheduler

- 同分辨率 camera 合并为一个 micro-batch。
- 不同分辨率 camera 自动拆分。
- 控制每批 readback bytes 和 staging texture 数量。
- 输出每批 submit time、GPU wait time、CPU copy time。

### 3. RenderTargetPool

- 按 `(width, height, format)` 复用 color/depth/shadow/composite target。
- 降低频繁 add/set camera 时的资源重建成本。
- 为 N=8/16 cameras 稳定运行做准备。

### 4. OMM 预 bake cache 接入 batch

- 复用已有 `load_omm_cache()` / `save_omm_cache()`。
- batch AS build 只消费已经 bake 好的 OMM 数据。
- 保证 steady-state async batch 不重新引入 `waitForIdle()`。

### 5. 完整性能报告

建议输出统一 benchmark 表:

| 方案 | N cameras | FPS | submit ms | wait ms | readback ms | VRAM | GPU util |
|------|-----------|-----|-----------|---------|-------------|------|----------|
| 多进程 POC | 1/2/4/8 | 待测 | - | - | - | 待测 | 待测 |
| 旧单进程循环 | 1/2/4/8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Week 2 同步 batch | 1/2/4/8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Week 3 异步 batch | 1/2/4/8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Week 4 scheduler | 1/2/4/8/16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

---

## 结论

本阶段已经完成并行渲染优化的核心路径: RTXNS 从“外部多进程并行”推进到“单 renderer 内多相机 batch + 异步 readback”。目前主路径可以用一次 batch 提交渲染多个 camera，共享 scene/AS，并通过 EventQuery 精确等待单个 batch，不再依赖全局 `waitForIdle()`。

量化收益上，N=3 cameras 的异步 batch 吞吐从同步 batch 的 140 FPS 提升到 251 FPS，提升约 79%；submit 耗时降到 2.7 ms；`is_batch_ready=False` 验证了 EventQuery 修复生效。该阶段已经具备向更大规模 N=8/16 camera、仿真/训练流水线和资源调度器继续扩展的基础。

