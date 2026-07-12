# RTXNS 动态具身智能仿真渲染方案

> 日期: 2026-07-12（修订版）
> 基于 [E:\cplus\RTXNS\output\parallel_render_efficiency_roadmap.md] 的 Week 1-4 基础上，补齐动态场景能力。
> **现状诚实声明**: RTXNS 当前覆盖 B=1, C>1, P=RGBA；离完整 B×C×P 具身智能仿真还需多个闭环。

---

## 0. P0 前置收尾：Ring 数据正确性

### 0.1 Bug 分析

**代码证据**:
```cpp
// headless_pbr.cpp:230 — ring 深度恒为 2
std::array<ReadbackRingSlot, 2> readbackRing;

// headless_pbr.cpp:1501-1502 — 无条件推进，无占用检查
m_views[idx].ringWriteIdx = (m_views[idx].ringWriteIdx + 1) % 2;
```

**触发路径**:
```python
# smoke_async.py:63-64 — 连续提交 20 次，无 read
for _ in range(F):
    tokens.append(scene.submit_frame_batch(indices))
```

**结论**: 提交 batch 10 时，ring slot 0 已被 batch 0/2/4/6/8 覆盖 5 次。`PendingBatch::ringIndices` 只记录了读哪个槽，不能阻止写入覆盖。**251 FPS 是"提交吞吐"，不是"正确多帧观测吞吐"。**

### 0.2 修复方案（优先级最高）

**数据结构变更** (`headless_pbr.h`):

```cpp
struct ReadbackRingSlot {
    nvrhi::StagingTextureHandle staging;
    uint64_t occupancyToken = 0;  // 0 = free, else = 占用此槽的 batch token
};

struct RenderViewSlot {
    // ...
    static constexpr uint32_t kDefaultRingDepth = 4;
    std::vector<ReadbackRingSlot> readbackRing;  // 可配置深度
    uint32_t ringWriteIdx = 0;
    uint32_t ringOccupied = 0;      // 当前占用槽数
};
```

**提交时占用检查** (`submit_frame_batch_impl`):

```cpp
// 检查是否所有 camera 的目标 ring slot 都空闲
bool all_free = true;
for (auto idx : indices) {
    auto& slot = m_views[idx];
    uint32_t target = slot.ringWriteIdx;
    if (slot.readbackRing[target].occupancyToken != 0)
        all_free = false;
}

// 方式 A: try_submit，返回 busy
if (!all_free) return 0;  // token=0 表示 busy

// 方式 B: 阻塞等待最早 batch（用于同步便利 API）
while (!all_free) {
    // waitEventQuery on earliest pending batch, then read & release
    // ...
}
```

**read 时释放占用** (`read_frame_batch_impl`):

```cpp
// 读回后释放 ring slot
for (size_t i = 0; i < found.cameraIndices.size(); i++) {
    auto idx = found.cameraIndices[i];
    auto ringIdx = found.ringIndices[i];
    m_views[idx].readbackRing[ringIdx].occupancyToken = 0;
    m_views[idx].ringOccupied--;
}
```

**可配置 ring 深度**:

```cpp
// Python API
void set_readback_ring_depth(uint32_t depth);  // 默认 4
```

### 0.3 验收测试

```python
# 应力测试: K+2 次连续 submit + 乱序 read + hash 校验
K = scene.get_readback_ring_depth()  # 4
hashes = []
tokens = []
for i in range(K + 2):
    # 每次提交前设置唯一的 marker（不同相机位置）
    scene.set_camera_at(0, position=[i, 0, 5], ...)
    token = scene.submit_frame_batch([0])
    if token == 0:
        print(f"  submit {i} returned BUSY (expected after K={K})")
    else:
        tokens.append((i, token))
        hashes.append(expected_hash_for_position(i))

for frame_id, token in tokens:
    img = scene.read_frame_batch(token)[0]
    actual_hash = hashlib.md5(img).hexdigest()
    assert actual_hash == hashes[frame_id], f"Frame {frame_id} data corrupted!"
```

**基准指标**:

| 指标 | 说明 |
|------|------|
| `submit_ms` | 仅 cmdList 录制 + execute 耗时 |
| `gpu_wait_ms` | `waitEventQuery` 阻塞时间 |
| `map_copy_ms` | `mapStagingTexture` + memcpy 耗时 |
| `total_ms` | 端到端总耗时 |
| `max_concurrent` | 最大在途 batch 数（≤ K） |

---

## 1. 核心发现：Donut SceneGraph 已是动态架构

**RTXNS 底层（Donut）已经具备完整的动态场景图能力**。每个 `SceneGraphNode` 有独立 TRS，`Refresh()` 合并层级变换，`UpdateInstance()` 写入 GPU `InstanceData` buffer，顶点着色器从 `t_Instances` StructuredBuffer 读取 `instance.transform` 做对象空间→世界空间变换。TLAS 的 `buildInstanceDescs()` 每帧从场景图读取最新 world transform。

**RTXNS 缺的不是底层架构，是对外暴露正确的控制接口。**

### 参考代码路径

| 组件 | 文件 | 行号 |
|------|------|------|
| `SceneGraphNode` TRS + 脏标记 | `E:\cplus\RTXNS\external\donut\include\donut\engine\SceneGraph.h` | 250-335 |
| `Refresh()` 层变换合并 | `E:\cplus\RTXNS\external\donut\src\engine\Scene.cpp` | 811-815 |
| `UpdateInstance()` → GPU buffer | `E:\cplus\RTXNS\external\donut\src\engine\Scene.cpp` | 1240-1248 |
| `HasPendingTransformChanges()` | `E:\cplus\RTXNS\external\donut\include\donut\engine\SceneGraph.h` | 558-559 |
| VS 读 `instance.transform` | vertex shader `buffer_loads` | — |
| `buildInstanceDescs()` → TLAS | `E:\cplus\RTXNS\src\RayTracedShadow\AccelerationStructure.cpp` | 228-316 |
| 已有 `update_node_transform()` | `E:\cplus\RTXNS\src\PythonBindings\headless_pbr.cpp` | 485-539 |

---

## 2. SAPIEN 参考基线

### 2.1 动态渲染调用链

```
scene.step()                                    # 物理步进
  └─ PhysxSystemCpu::step()
       ├─ PxScene::simulate() → fetchResults()
       └─ syncPoseToEntity() × N

scene.update_render()                           # 渲染同步
  └─ SapienRendererSystem::step()
       ├─ for body: node->setPosition/Rotation()
       ├─ for camera: camera->setTransform()
       └─ mScene->updateModelMatrices()         # 全量 mat4[N] → GPU SSBO

camera.take_picture()                           # 拍照
  └─ mRenderer->render(*mCamera, ...)
       ├─ upload camera UBO
       ├─ draw calls（读 SSBO 中的 model matrices）
       └─ signal timeline semaphore
```

### 2.2 参考代码路径

| 步骤 | 文件 | 行号 |
|------|------|------|
| 物理步进 | `E:\cplus\SAPIEN\src\physx\physx_system.cpp` | 241-253 |
| pose→Entity | `E:\cplus\SAPIEN\src\physx\rigid_component.cpp` | 29 |
| 渲染同步 | `E:\cplus\SAPIEN\src\sapien_renderer\sapien_renderer_system.cpp` | 180-197 |
| body→Node | `E:\cplus\SAPIEN\src\sapien_renderer\render_body_component.cpp` | 184-188 |
| camera 变换 | `E:\cplus\SAPIEN\src\sapien_renderer\camera_component.cpp` | 449-456 |
| camera 渲染 | `E:\cplus\SAPIEN\src\sapien_renderer\camera_component.cpp` | 124-131 |
| BatchedCamera | `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cpp` | 32-148 |
| CUDA transform kernel | `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cu` | 56-109 |
| CUDA-Vulkan 信号量 | `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cpp` | 238-255, 342-351 |

### 2.3 ManiSkill 仿真循环

```python
# E:\cplus\ManiSkill\mani_skill\envs\sapien_env.py
def step(action):
    _step_action(action)
      # sim_steps_per_control 次物理步进
      scene.step()
    get_obs()
      scene.update_render()     # 1 次
      sensor.capture() × N      # N 个相机
      sensor.get_obs() × N      # N 次读回
```

参考: `E:\cplus\ManiSkill\mani_skill\envs\scene.py`, `E:\cplus\ManiSkill\mani_skill\sensors\camera.py`

---

## 3. RTXNS 当前能力与缺口

### 3.1 已有（Week 1-3）

| 能力 | 路径 |
|------|------|
| 多相机 API | `add_camera()`, `set_camera_at()`, `camera_count()` |
| 单 cmdList 批量渲染 | `render_frame_batch_v2()` |
| 共享 BLAS/TLAS | `m_shadowAS` 全局唯一 |
| 异步 submit | `submit_frame_batch()` → EventQuery |
| 异步 readback | `read_frame_batch()` → `waitEventQuery` |

### 3.2 缺口（按优先级）

| 缺口 | 影响 | SAPIEN 对应 |
|------|------|-------------|
| **Ring 无占用保护** | 连续提交覆盖数据 | `BatchedCamera` timeline semaphore |
| 无批量 `set_transforms()` | 逐节点设变换 CPU 开销大 | `internalUpdate()` 循环 |
| TLAS 无条件每帧重建 | 静态场景也重建 InstanceDesc | `HasPendingTransformChanges()` |
| `update_node_transform()` 按 name 查找 | 动态物体多时 O(N²) | 稳定 Handle |
| 仅返回 RGBA8 | 无 depth/seg | 多 attachment + 独立 readback |
| 缺 ReplicaCAD 场景组装 | 只能加载单 .glb | JSON/URDF loader |
| 无物理引擎集成 | 无碰撞/动作步进 | PhysX via `PoseProvider` |
| 无多环境并行 | B=1 | `RenderSystemGroup` |

### 3.3 当前覆盖范围 vs 目标

```
目标: B environments × C cameras × P products
─────────────────────────────────────────────
当前: B=1, C>1, P=RGBA

缺失:
  B>1: 多场景并行调度
  P: Depth(R32_FLOAT), Semantic(instance/seg ID)
  物理: 碰撞、URDF 关节、动作步进
```

---

## 4. 架构修正（基于反馈）

### 4.1 flush 不应独立提交

**原方案问题**: `flush_scene_transforms()` 创建独立 cmdList → 额外 `executeCommandList`，增加同步点。

**正确做法**: transform 更新融入 `submit_frame_batch`，利用已有 cmdList。

```
set_node_transforms_batch(handles, matrices)
  → 只改 CPU SceneGraph，标记 transform dirty
  → 不触发任何 GPU 操作

submit_frame_batch(indices)         # 已有 cmdList
  → m_scene->Refresh(cmdList, ...)  # 已有（headless_pbr.cpp:1469）
     ├─ RefreshSceneGraph: 合并脏节点层级变换
     └─ RefreshBuffers: 上传 InstanceData → GPU
  → if (HasPendingTransformChanges())
       record_tlas_update(cmdList)  # 仅脏时更新，内联在同一个 cmdList
  → for camera: render + copy
  → executeCommandList
```

**SAPIEN 对应**:
```cpp
// E:\cplus\SAPIEN\src\sapien_renderer\sapien_renderer_system.cpp:180-197
// step() 也在同一个逻辑帧内处理完所有 transform → render
```

### 4.2 TLAS 按脏状态更新

**当前** (`headless_pbr.cpp:1440-1447`): BLAS 建好后仍每 batch 无条件重建 `InstanceDesc` + `updateTLAS`。

**Donut 已提供脏标记** (`SceneGraph.h:558-559`):
```cpp
bool HasPendingStructureChanges() const;  // 新增/删除实例、网格变化
bool HasPendingTransformChanges() const;  // 仅 TRS 变化
```

**改造 `record_or_build_shadow_as()`**:

| 脏状态 | 动作 |
|--------|------|
| 无变化 | 跳过 TLAS |
| 仅 `HasPendingTransformChanges()` | `buildInstanceDescs()` + `PerformUpdate` TLAS |
| `HasPendingStructureChanges()` | 重建 BLAS + TLAS + 资源表 |
| 材质/光照变化 | 不更新 TLAS |

### 4.3 set_node_transforms_batch 使用稳定 Handle

**当前** (`headless_pbr.cpp:485`): `update_node_transform(name)` → 遍历 `m_scene->GetSceneGraph()->GetMeshInstances()` 按名称匹配。动态物体多时 O(N²)。

**改造**: 加载阶段返回稳定的 `NodeHandle`（`uint32_t` 索引），批量 API 接收 `handles + float32[N, 4, 4]`:

```cpp
// 加载场景时建立映射
struct NodeHandle { uint32_t index; };
NodeHandle register_dynamic_node(const std::string& name);

// 批量更新（O(N) 直接索引）
void set_node_transforms_batch(
    const std::vector<NodeHandle>& handles,
    const std::vector<std::array<float, 16>>& transforms
);
```

### 4.4 Sensor 模型: Product 而非 Camera 类型

**原方案问题**: `rgb_cam/depth_cam/seg_cam` 视为三个普通相机。实际 depth/seg 需要不同的 attachment 和 shader，不应走普通 RGB pipeline。

**正确模型**:
```cpp
struct SensorRequest {
    uint32_t camera_index;  // 视角源
    std::set<Product> products;  // {Color, Depth, InstanceID, SemanticID}
};
```

| Product | 当前状态 | 需要 |
|---------|----------|------|
| `Color` (RGBA8) | ✅ | — |
| `Depth` (R32_FLOAT) | ❌ depth target 仅内部使用 | 线性化 D32→R32_FLOAT readback |
| `InstanceID` (R32_UINT) | ❌ | 新增 instance ID attachment |
| `SemanticID` (R32_UINT) | ❌ | ReplicaCAD 提供 `semantic_id`，需解析 JSON |

**ReplicaCAD 语义信息已就绪**:
- `E:\cplus\RTXNS\ReplicaCAD\configs\ssd\replicaCAD_semantic_lexicon.json` — 101 个语义类
- 每个 `*.object_config.json` 含 `semantic_id`

---

## 5. 实施计划

### 优先级排序

参考反馈:
> "先收尾并行渲染的 P0 正确性，然后以 ReplicaCAD 单环境、一个可移动刚体、两相机 RGB 为第一里程碑。"

```
优先级 0: Ring 数据正确性（2-3天）  ← 先做这个
优先级 1: ReplicaCAD 场景组装 MVP（2-3天）
优先级 2: 刚体运动脚本驱动验证（2-3天）
优先级 3: Depth / Semantic observation（2-3天）
优先级 4: PhysX PoseProvider + URDF 关节（3-5天）
优先级 5: B>1 多环境调度（3-5天）
优先级 6: persistent mapping / CUDA 零拷贝（后续按需）
```

### Milestone 0: Ring 正确性（当前阶段）

| 任务 | 内容 |
|------|------|
| 数据结构 | `ReadbackRingSlot.occupancyToken`, 可配置 depth K (默认4) |
| 提交逻辑 | `try_submit` 检查占用, 满时返回 busy |
| 读回逻辑 | `read_frame_batch` 释放 `occupancyToken` |
| 测试 | K+2 连续 submit, 乱序 read, hash 校验 |
| 基准 | 分项报告 submit/gpu_wait/map_copy/total |

### Milestone 1: ReplicaCAD 场景组装 MVP

**目标**: 解析 `scene_instance.json`，组合 stage .glb + 刚体 objects，建立 `ObjectHandle → RenderNode` 映射。

| 任务 | 参考 |
|------|------|
| 解析 `replicaCAD.scene_dataset_config.json` | `E:\cplus\RTXNS\ReplicaCAD\replicaCAD.scene_dataset_config.json` |
| 解析 `scene_instance.json` (stage + objects) | `E:\cplus\RTXNS\ReplicaCAD\configs\scenes\v3_sc0_staging_00.scene_instance.json` |
| 加载 stage .glb 作为背景 | `E:\cplus\RTXNS\ReplicaCAD\stages\*.glb` |
| 逐个加载 object .glb → add to scene graph | `E:\cplus\RTXNS\ReplicaCAD\objects\*.glb` |
| 建立 `NodeHandle` 映射表 | `name → SceneGraphNode* → index` |
| 应用 `object_instances[].translation/rotation` | JSON → TRS → `node->SetTransform()` |

**注意**: 第一版不加载 URDF 关节体，只处理 `motion_type=DYNAMIC` 的刚体 objects。

### Milestone 2: 脚本驱动刚体运动验证

```python
# 第一里程碑验证脚本
scene.load_replicacad_scene("v3_sc0_staging_00")
handles = scene.get_dynamic_node_handles()  # 返回所有 DYNAMIC object 的 handle

for step in range(100):
    # 程序化运动（无物理引擎）
    new_poses = compute_scripted_motion(step, handles)
    scene.set_node_transforms_batch(handles, new_poses)
    # 不调用独立 flush — submit 内部处理
    token = scene.submit_frame_batch([rgb_cam0, rgb_cam1])
    if token == 0:  # ring 满
        # 读回最老的 batch 释放 ring slot
        oldest = scene.get_oldest_pending_token()
        scene.read_frame_batch(oldest)
        token = scene.submit_frame_batch([rgb_cam0, rgb_cam1])
    images = scene.read_frame_batch(token)
    verify_motion(images, expected_poses)
```

### Milestone 3: Depth + Semantic observation

```cpp
// SensorRequest 驱动的多 product 渲染
struct SensorRequest {
    uint32_t camera_index;
    bool want_color;
    bool want_depth;      // → R32_FLOAT readback
    bool want_instance_id; // → R32_UINT readback
    bool want_semantic_id; // → R32_UINT, from ReplicaCAD JSON
};

// render_frame_batch → 对每个 product:
//   - Color: 已有 RGBA8 pipeline
//   - Depth: 添加 R32_FLOAT attachment, 线性化 depth target → copy → readback
//   - InstanceID: per-instance uint ID → 写入 attachment → readback
//   - SemanticID: 从 ReplicaCAD config 查表 → 写入 attachment
```

### Milestone 4: PhysX PoseProvider + URDF

**目标**: 最小化物理引擎集成——不重写整个 `BaseEnv`，而是定义 `PoseProvider` 接口。

```cpp
// 物理结果抽象层
struct PoseProvider {
    virtual void step(float dt) = 0;
    virtual std::vector<std::array<float, 16>> get_world_poses(
        const std::vector<NodeHandle>& handles) = 0;
};
```

- 初期实现: Python-side scripted motion → 通过 `set_node_transforms_batch` 传入
- SAPIEN 集成: 复用 `PhysxSystemCpu::step()` + `syncPoseToEntity()` → `getPose()` → matrix
- URDF: 每个 link 作为独立 `RenderBodyComponent`，关节运动 → link 独立变换

### Milestone 5+: B>1 多环境

当前 `g_context` 单例 (`headless_pbr.cpp`) 限制 B=1。需要:
- 每个环境共享 device/context，独立 scene instance
- Python 调度层按 env 分片 → batch render per env
- 多个 env 可在 GPU 队列上连续提交

---

## 6. 整合路线图

```
已完成                            P0 收尾                    Phase 5-7（修正）
════════                          ═══════                    ════════════════
Week 1: CameraDesc                Ring 占用保护              M1: ReplicaCAD 场景组装
       + RenderViewSlot           + occupancyToken           M2: 刚体运动验证
       + 多相机 API               + 可配置 depth K            M3: Depth + Semantic
Week 2: render_frame_batch        + K+2 应力测试             M4: PoseProvider + URDF
       + 共享 AS                  + 分项基准                 M5: B>1 多环境
Week 3: async submit                                               ...
       + readback ring
Week 4: 资源池 + scheduler
```

### 新增文件

```
src/PythonBindings/
  headless_pbr.h/.cpp        [修改] occupancy token, handle API, sensor model
  replicacad_loader.h/.cpp   [新增] ReplicaCAD JSON/GLB loader
  py_bindings_common.h       [修改]

python/rtxns_sim/
  base_env.py                [新增] 对标 SAPIEN BaseEnv（简化版）
  replicacad_env.py          [新增] ReplicaCAD 专用环境

tools/
  test_ring_correctness.py   [新增] P0 数据正确性验证
  test_replicacad_assembly.py[新增] 场景组装测试
  bench_replicacad_sim.py    [新增] 仿真基准
```

### 验收标准

1. **Ring 正确性**: K+2 次连续 submit 后乱序 read，每帧 hash 匹配预期
2. **场景组装**: `load_replicacad_scene()` 正确加载 stage + 刚体 objects
3. **动态渲染**: 脚本驱动运动反映在输出图像中
4. **TLAS 效率**: 静态物体/场景不触发 TLAS rebuild
5. **Sensor**: `SensorRequest(products={Color, Depth})` 返回正确的多张图像
