# Flora（RTXNS）动态具身仿真渲染与并行渲染推进方案

> 版本：2026-07-18
>
> 项目目录：`E:\cplus\Flora`
>
> 参考项目：`E:\cplus\SAPIEN`、`E:\cplus\Genesis`
>
> 数据集：`E:\cplus\Flora\ReplicaCAD`
>
> 文档目标：在现有多相机、批量提交和异步读回能力上，规划完整 ReplicaCAD 场景、动态刚体、URDF、多环境和 GPU Tensor 输出的实现路径，使 Flora 能作为 Genesis/Taichi 或未来自研物理模块的高吞吐渲染后端。

---

## 0. 执行摘要

### 0.1 最终目标

Flora 当前阶段只负责渲染，不负责物理求解。物理仿真可以由 Genesis、Taichi 或未来自研 GPU 物理模块完成，Flora 接收它们生成的刚体、URDF link 和相机位姿，然后输出观测 Tensor。

目标调用模型应接近 Genesis：

```python
renderer = flora.Renderer(device="cuda:0")
scene = renderer.create_scene_batch(num_envs=8)
scene.load_replicacad(r"E:\cplus\Flora\ReplicaCAD", "apt_0")
camera = scene.create_camera_batch(camera_descs)
scene.build()

# 位姿来自 Genesis/Taichi/Torch 或其他仿真器。
scene.bind_pose_source(rigid_poses, link_poses, camera_poses)

obs = camera.render(rgb=True, depth=True, segmentation=True)
rgb = obs.rgb.to_torch()       # [B, C, H, W, 4]
depth = obs.depth.to_torch()   # [B, C, H, W, 1]
instance = obs.instance.to_torch()
```

“比肩 SAPIEN”不只表示单相机 FPS 接近，而是至少满足以下五项：

1. 能完整加载 ReplicaCAD `scene_instance.json`，而不是只加载 stage GLB。
2. 能渲染普通家具、动态刚体和 URDF articulated object。
3. 能维护 `B` 个相互独立的环境和每环境 `C` 个相机。
4. 位姿从 GPU 输入，RGB/Depth/Segmentation 等结果以 GPU Tensor 输出。
5. 在相同场景、分辨率、输出产品和计时端点下，GPU-ready 吞吐达到 SAPIEN 的 90% 以上，冲刺达到或超过 SAPIEN。

### 0.2 当前结论

- 已完成的 single-cmdList、多相机异步路径是正确基础，应保留。
- 多 cmdList/micro-batch 在同一 Vulkan graphics queue 上不会产生相机级 GPU 并行，实测反而降低 11.5%～23.1%，该方向停止推进。
- 当前主要差距不是“再拆几个 cmdList”，而是缺少完整场景装配、独立多环境、GPU 位姿输入、GPU 输出和多模态 Sensor。
- 当前不应实现 PhysX `step()`、碰撞、质量和关节驱动；URDF 应作为渲染层级和外部 link pose 的消费者提前实现。
- 当前 Windows 环境无法得到有效的 SAPIEN `RenderSystemGroup` GPU 对照数据，最终性能比较必须在 Linux/CUDA 上执行。

### 0.3 推荐推进顺序

```text
2 天收尾当前并行渲染基线
        ↓
Iteration A（4 周）：ReplicaCAD 完整场景 + URDF + Sensor + Python 接口
        ↓
Iteration B（4 周）：SceneBatch + GPU Pose + GPU Output + SAPIEN 对标
        ↓
未来阶段：自研物理模块与渲染共享 GPU 状态
```

---

## 1. 范围与术语

### 1.1 当前阶段包含

- ReplicaCAD dataset config、stage config、object config、scene instance 和 semantic descriptor 解析。
- 普通静态/动态对象的可视资源装配。
- URDF link/joint/visual 层级和可视网格装配。
- 脚本、NumPy、Torch、Taichi/DLPack 等来源的批量位姿输入。
- `B environments × C cameras × P products` 的批量渲染。
- RGB、Depth、Instance ID、Semantic ID、Normal 产品。
- GPU 输出 Arena、显式 CPU 下载和异步背压。

### 1.2 当前阶段明确不包含

- 刚体碰撞检测和接触求解。
- 质量、惯量、摩擦、恢复系数等动力学计算。
- URDF 关节驱动、约束求解和力控制。
- 导航网格查询和路径规划。
- 为追求画质进行材质、阴影或降噪调优。

这些信息可以由加载器保留为元数据，但不得进入当前渲染主路径。

### 1.3 统一维度定义

| 符号 | 含义 |
|---|---|
| `B` | 独立仿真环境数，例如 1、2、4、8、16 |
| `C` | 每个环境的相机数 |
| `P` | 每相机请求的产品数，例如 Color、Depth、InstanceID |
| `N_rigid` | 每个环境普通刚体实例数 |
| `N_link` | 每个环境 URDF link 实例数 |

当前实现主要覆盖 `B=1, C>1, P=Color`。后续性能汇报必须明确写出 `B/C/P`，不能只报告模糊的“FPS”。

---

## 2. 已完成工作与实验结论

### 2.1 已完成能力

| 能力 | 当前实现 |
|---|---|
| 多相机描述和资源槽 | `CameraDesc`、`RenderViewSlot` |
| 多相机 API | `add_camera()`、`set_camera_at()`、`camera_count()` |
| 单 cmdList 批量渲染 | `render_frame_batch_v2()` / `submit_frame_batch_impl()` |
| 场景级共享 AS | batch 内共享 BLAS/TLAS 构建与更新 |
| 异步提交与完成查询 | `submit_frame_batch()` + `EventQuery` |
| 异步读回 | `read_frame_batch()` + readback ring |
| Ring 占用保护 | `occupancyToken`、可配置深度，busy 时返回 token 0 |
| Camera 状态同步 | 所有 camera index 无条件同步当前状态 |
| Async AS 初始化 | 同步和异步路径共享 `record_or_build_shadow_as()` |

关键代码入口：

- `E:\cplus\Flora\src\PythonBindings\headless_pbr.cpp:1335`：`sync_and_record_view()`
- `E:\cplus\Flora\src\PythonBindings\headless_pbr.cpp:1509`：`submit_frame_batch_impl()`
- `E:\cplus\Flora\src\PythonBindings\headless_pbr.cpp:1577`：提交后的 `setEventQuery()`
- `E:\cplus\Flora\src\PythonBindings\headless_pbr.cpp:1637`：readback ring map/copy/release

### 2.2 Ring 和 EventQuery 修复结果

| 指标 | 修复前 | 修复后 | 结论 |
|---|---:|---:|---|
| `is_batch_ready` | `True`（错误） | `False`（正确） | EventQuery 已移动到 execute 之后 |
| Async FPS | 203 | 251 | 提交路径约提升 24% |
| Ring 覆盖风险 | 存在 | 已修复 | PendingBatch 保存每相机 ring index，并以 token 占用 |
| Camera 0 状态 | 可能未恢复 | 无条件同步 | 多路径状态一致 |
| Async 首帧 AS | 可能缺失 | 共享构建逻辑 | 同步/异步一致 |

注意：251 FPS 是当时配置下的端到端结果，不能与后续不同分辨率、场景和输出端点的数据直接横向比较。

### 2.3 初始多相机伸缩实验

场景：ReplicaCAD `Stage_v3_sc0_staging.glb`；分辨率：128×96。该实验只加载 stage，并使用一个 SceneGraph 中的多相机，因此表示 `B=1, C=N`，不是 `B=N, C=1`。

| C | sync cam-FPS | async cam-FPS | batch ms | async per-camera FPS |
|---:|---:|---:|---:|---:|
| 1 | 4,465 | 7,417 | 0.13 | 7,417 |
| 2 | 4,688 | 8,107 | 0.25 | 4,054 |
| 4 | 7,594 | 8,489 | 0.47 | 2,122 |
| 8 | 8,347 | 10,688 | 0.75 | 1,336 |
| 12 | 6,929 | 10,377 | 1.16 | 865 |

可以得出的结论：

- 异步路径有效隐藏了一部分 CPU 等待和读回成本。
- 总 cam-FPS 在 `C=8` 左右见顶，之后吞吐不再增长。
- 每个 camera pass 仍在一个 cmdList 中顺序录制和执行，单相机平均成本随 C 近似线性累积。
- 这组数据不能证明已经具备独立多环境并行能力。

### 2.4 多 cmdList / micro-batch 对照实验

在同一构建和同一测试条件内，将相机拆为 2～6 个 cmdList 后得到：

| 模式 | C=12 cam-FPS | 相对 single | C=8 cam-FPS | 相对 single |
|---|---:|---:|---:|---:|
| single cmdList | 4,504 | baseline | 7,988 | baseline |
| `micro_batch=4` | 3,657 | -18.8% | 7,067 | -11.5% |
| `micro_batch=2` | 3,517 | -21.9% | 6,142 | -23.1% |

实验结论：

1. 同一个 `VkQueue` 上的多个 graphics command buffer 仍按 queue 顺序执行，不会让多个相机 render pass 自动并行。
2. 每个额外 cmdList 增加 `createCommandList/open/close/executeCommandList` 和 EventQuery 管理成本。
3. 拆分越细，固定提交与同步开销越明显。
4. Vulkan timeline semaphore 能表达跨 queue/跨 API 依赖，但不能让同一 graphics queue 上的相关 render pass凭空并行。
5. P2 多 cmdList 方向已被实验否定，`submit_frame_batch_ex` 已在并行渲染收尾中移除，生产异步路径固定为 single cmdList。

### 2.5 1280×720 当前路径对照实验

设备：RTX 3070 Laptop。以下数据用于辨别端点差异，不用于直接宣布 Flora 超过或落后 SAPIEN。

#### Flora async CPU RGBA8 readback

| 模式 | C=1 | C=2 | C=4 | C=8 |
|---|---:|---:|---:|---:|
| 无 RT shadow，e2e cam-FPS | 449.6 | 421.2 | 432.2 | 408.8 |
| RT shadow 8 samples，e2e cam-FPS | 374.7 | 390.1 | 397.3 | 378.5 |

#### SAPIEN 当前可运行端点

| 端点 | C=1 | C=2 | C=4 |
|---|---:|---:|---:|
| CPU float32 `Color` readback | 125.7 | 125.8 | 122.4 |
| 普通 `take_picture()`，不取输出 | 845.8 | 958.4 | 967.8 |

解释：

- Flora 表格包含 RGBA8 CPU readback；SAPIEN `take_picture()` 表格不包含输出读取，两者不是同一个端点。
- SAPIEN CPU `Color` 是 float32 输出，Flora 当前是 RGBA8，传输字节数和转换成本不同。
- 本机 SAPIEN `RenderSystemGroup` 在 Windows 创建 CUDA/Vulkan external semaphore 时进入 `OpaqueFd` 路径并发生 native crash；`get_picture_cuda()` 也无法形成可靠对照。
- 因此当前不能声称 Flora 已经比肩 SAPIEN，也不能从 CPU readback 数据推断 GPU batch 性能。

### 2.6 实验后的决策

停止投入：

- 继续拆分 graphics cmdList。
- 用更多 EventQuery 试图获得 GPU 并行。
- 只在单 stage、多相机、CPU readback 路径上刷 FPS。

继续投入：

- 完整 `scene_instance.json` 装配。
- `B>1` 独立环境状态和共享资产。
- GPU PoseSource。
- GPU OutputArena/DLPack。
- Sensor 多产品和严格同端点 SAPIEN 基准。

---

## 3. 当前代码分析

### 3.1 Flora C++ 渲染路径

当前 `submit_frame_batch_impl()` 的核心结构为：

```cpp
auto cmdList = device->createCommandList();
cmdList->open();

m_scene->Refresh(cmdList, m_frame_index++);
record_or_build_shadow_as(cmdList);

for (auto idx : indices)
    sync_and_record_view(cmdList, idx, ...);

cmdList->close();
device->executeCommandList(cmdList);
device->setEventQuery(query, nvrhi::CommandQueue::Graphics);
```

代码证据和影响：

| 位置 | 当前行为 | 后续处理 |
|---|---|---|
| `headless_pbr.cpp:492` | `update_node_transform(name)` 遍历 mesh instances 按名字查找 | build 时生成稳定 `NodeHandle`，运行时直接索引 |
| `headless_pbr.cpp:1509` | 一个 SceneGraph 内批量提交相机 | 保留 single-cmdList 基础，扩展为 SceneBatch 状态 |
| `headless_pbr.cpp:1562` | C 个相机逐个录制 | 第一阶段接受顺序录制；第二阶段评估 multiview/instancing |
| `headless_pbr.cpp:1643` | map staging texture 并 CPU copy | GPU fast path绕过 staging，CPU 下载变成显式 API |
| `headless_pbr.cpp:1977` | 全局共享 `g_context` | 它不等价于 `B=1`；可创建多个 scene，共享 context/device |
| `headless_pbr.cpp:2198` | `create_scene()` 从共享 context 创建 scene | 缺少的是跨 scene batch manager 和共享资产，不是 context 数量 |

旧方案中“`g_context` 单例限制 B=1”的说法需要删除。真正缺口是：

- 没有 `CompiledSceneBatch` 统一管理 B 个环境。
- 多 scene 之间没有明确的 Mesh/Texture/BLAS 资产共享层。
- 没有 `[B, N]` 的连续实例状态缓冲。
- 没有 `[B, C]` 的连续相机状态和输出布局。

### 3.2 RayTracedShadow 路径

`E:\cplus\Flora\src\RayTracedShadow\RayTracedShadowPass.cpp` 中仍存在运行期 BindingSet 创建：

- `:226`：shadow BindingSet
- `:275`：composite BindingSet
- `:316`：blur BindingSet

这些是 Week B4 的优化对象。当前阶段只要求功能正确，不应先围绕 RT shadow 做架构设计。与 SAPIEN 对标时，应先采用无 RT 或相同的简单阴影配置；RT shadow 作为单独产品线报告。

### 3.3 Flora Python 原型层

现有 `E:\cplus\Flora\python\rtxns_genesis_style\renderer.py` 已经提供 Genesis 风格原型：

- `:739`：`GenesisStyleRenderer`
- `:845`：`add_rigid_batch()`
- `:862`：`update_rigid_batch()`
- `:983`：`update_scene()`
- `:1029`：重新生成临时 `scene.glb`
- `:1134`：逐节点调用 C++ `update_node_transform()`

问题：

1. `add_rigid_batch()` 和 `update_rigid_batch()` 在 Python 按环境循环。
2. 拓扑变化通过生成一个临时 GLB 并重新 `load_scene()` 完成。
3. 增量更新依赖字符串名字，最终进入 C++ 线性扫描。
4. 输出仍以 NumPy/CPU 图像为中心。

处理原则：

- 保留现有公开对象和兼容调用，不新建第三套互不兼容的 Python 包。
- 将它定位为兼容/原型层，内部逐步转调 C++ `SceneBuilder` 和 `CompiledSceneBatch`。
- `build()` 之后禁止隐式重建 GLB；拓扑变更应明确报错或要求重新 build。
- `add_rigid_batch()` 最终只提交一份资产和 B 组状态，不复制 B 份 Python mesh 数据。

### 3.4 Genesis 的可复用设计

参考文件：

- `E:\cplus\Genesis\genesis\vis\visualizer.py:21`：Visualizer 只管理可视化，不拥有物理求解。
- `E:\cplus\Genesis\genesis\vis\visualizer.py:174`：所有静态几何和相机先 build。
- `E:\cplus\Genesis\genesis\vis\visualizer.py:207`：每帧从 solver 同步 visual states。
- `E:\cplus\Genesis\genesis\vis\batch_renderer.py:196`：将 rigid state 转为连续 Torch Tensor。
- `E:\cplus\Genesis\genesis\vis\batch_renderer.py:252`：`BatchRenderer`。
- `E:\cplus\Genesis\genesis\vis\batch_renderer.py:297`：`num_worlds=max(scene.n_envs, 1)`。
- `E:\cplus\Genesis\genesis\vis\batch_renderer.py:328`：一次请求 RGB/Depth/Segmentation/Normal。
- `E:\cplus\Genesis\genesis\vis\camera.py:365`：任一 camera 请求会统一触发 batch render，再切出当前 camera。

Flora 应复用的不是 Genesis 的物理代码，而是接口边界：

```text
静态 topology/build
        +
每帧 batched visual state
        +
一次 batch render 请求
        +
保持 env 维度的 GPU Tensor 输出
```

Genesis 的 batch renderer 当前限定 Linux x86-64/CUDA，这也说明最终 GPU interop 和 SAPIEN 对标应优先选择 Linux/CUDA。

### 3.5 SAPIEN 的可复用设计

参考文件：

- `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cpp:60`：相机输出拷贝到连续 GPU buffer。
- `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cpp:127`：`BatchedCamera::takePicture()`。
- `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cpp:260`：`setPoseSource()`。
- `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cpp:303`：CUDA 位姿更新。
- `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cpp:342`：timeline semaphore 通知。
- `E:\cplus\SAPIEN\src\sapien_renderer\batched_render_system.cu:56`：transform CUDA kernel。

重要认识：SAPIEN 的 camera render 本身也存在循环；它的主要优势并不是“每个 camera 在多个 graphics queue 上同时画”，而是：

- topology 固定后，位姿直接从 CUDA buffer 更新。
- 输出保留在 GPU 连续 buffer 中。
- CUDA/Vulkan 通过 timeline semaphore 协作。
- Python 侧不执行每物体、每相机的 CPU readback 循环。

因此 Flora 先做到 GPU state + GPU output，就可能接近 SAPIEN；真正的 single-pass multiview 可以放在此后优化。

---

## 4. ReplicaCAD 完整支持要求

### 4.1 本地数据统计

对 `E:\cplus\Flora\ReplicaCAD` 的实际扫描结果：

| 内容 | 数量 |
|---|---:|
| `*.scene_instance.json` | 91 |
| `*.object_config.json` | 92 |
| 普通对象实例总数 | 2,293 |
| articulated object 实例总数 | 540 |
| 含 articulated object 的场景 | 90/91（`empty_stage` 除外） |
| URDF | 12 |
| `.ao_config.json` | 5 |

代表场景 `apt_0`：113 个普通对象、6 个 articulated object。

### 4.2 Loader 解析链

必须严格按以下链路解析：

```text
replicaCAD.scene_dataset_config.json
  ├─ configs/stages/*.stage_config.json
  │    └─ render_asset -> stages/*.glb
  ├─ configs/objects/*.object_config.json
  │    ├─ render_asset -> objects/*.glb
  │    ├─ COM / scale
  │    └─ semantic_id
  ├─ configs/scenes/*.scene_instance.json
  │    ├─ stage_instance
  │    ├─ object_instances
  │    └─ articulated_object_instances
  ├─ urdf/<template>/<template>.urdf
  │    └─ link/joint/visual/mesh/origin
  └─ configs/ssd/replicaCAD_semantic_lexicon.json
```

不能将 `template_name` 简单拼接成固定 GLB 路径。应先建立模板注册表，再解析场景实例。

### 4.3 普通对象字段

Loader 至少处理：

- `template_name`
- `translation`
- `rotation`，ReplicaCAD 中为 quaternion `(w, x, y, z)`
- `uniform_scale` / `non_uniform_scale`
- `translation_origin`
- `motion_type`
- object config 的 `render_asset`
- object config 的 `COM`
- object config 的 `semantic_id`

坐标处理必须有单元测试：

```text
scene instance transform
    × translation_origin/COM correction
    × object template scale
    × glTF node local transform
    = final render world transform
```

禁止靠截图手调轴方向。stage config 中的 `up` 和 `front`、glTF 坐标约定、矩阵行列主序和 quaternion 顺序必须写入 golden test。

### 4.4 Articulated object 和 URDF

`articulated_object_instances` 至少处理：

- `template_name`
- `translation_origin`
- `fixed_base`
- `translation`
- `rotation`
- `uniform_scale`
- `motion_type`

URDF 第一版处理：

- `<link>` 名字和稳定 Link Handle。
- `<joint>` 的 parent/child、type、origin、axis 和 limit 元数据。
- `<visual>` origin、mesh filename、mesh scale 和 material。
- fixed、revolute、continuous、prismatic joint。
- 相对 URDF 文件目录解析 mesh。

URDF 第一版不处理：

- `<collision>` 几何上传。
- `<inertial>` 动力学计算。
- damping、friction、effort 的求解。

运行时优先接受外部提供的 link world pose。可以提供一个 CPU FK reference utility，仅用于 loader 验证和脚本动画，不能放进高性能渲染热路径。

### 4.5 缺失资产策略

dataset config 声明了 `../hab_fetch_1.0/robots/fetch_no_base.urdf`，本地目前不存在。Loader 应提供三种策略：

```python
scene.load_replicacad(..., missing_asset="error")  # 默认，明确失败
scene.load_replicacad(..., missing_asset="warn")   # 跳过并记录 manifest
scene.load_replicacad(..., missing_asset="skip")   # 仅测试使用
```

每次加载生成 `SceneLoadReport`：

```python
report = {
    "scene": "apt_0",
    "stage_count": 1,
    "object_instance_count": 113,
    "articulated_instance_count": 6,
    "link_count": ...,
    "unique_mesh_count": ...,
    "unique_texture_count": ...,
    "missing_assets": [],
    "warnings": [],
}
```

---

## 5. 目标架构

### 5.1 模块分层

```text
Python Simulation / Genesis Adapter / Taichi
             │
             │ StaticSceneDesc + PoseBatch + SensorRequest
             ▼
┌─────────────────────────────────────────────────────────┐
│ Flora Python Frontend                                   │
│ SceneBuilder -> CompiledSceneBatch -> SensorBatch       │
└─────────────────────────────────────────────────────────┘
             │ pybind11
             ▼
┌─────────────────────────────────────────────────────────┐
│ Flora C++ Runtime                                       │
│ AssetCache       : Mesh/Material/Texture/shared BLAS    │
│ SceneTopology    : stable node/link/instance handles    │
│ EnvStateBuffer   : [B, N_instance]                      │
│ CameraStateBuffer: [B, C]                               │
│ OutputArena      : [B, C, H, W, P]                      │
└─────────────────────────────────────────────────────────┘
             │ Vulkan/NVRHI + CUDA interop
             ▼
        DLPack / Torch / Taichi Tensor
```

### 5.2 静态拓扑与运行时状态分离

`SceneBuilder` 阶段允许：

- 加载/注册资产。
- 创建普通实例、URDF link 和 camera。
- 设置 instance/semantic ID。
- 确定 B、C、分辨率和输出产品。

`build()` 后：

- 冻结 Mesh、Material、节点数量和 handle 表。
- 创建共享 BLAS、实例状态缓冲和输出 Arena。
- 禁止隐式重新生成 GLB。
- 运行时只更新位姿、visibility/active mask、camera 和 SensorRequest。

### 5.3 稳定 Handle

建议 C++ 数据结构：

```cpp
struct AssetHandle    { uint32_t index; };
struct InstanceHandle { uint32_t index; };
struct LinkHandle     { uint32_t index; };
struct CameraHandle   { uint32_t index; };

struct SceneTopology {
    std::vector<AssetRecord> assets;
    std::vector<InstanceRecord> instances;
    std::vector<LinkRecord> links;
    std::vector<CameraRecord> cameras;
    std::unordered_map<std::string, uint32_t> nameToInstance;
};
```

名字只在 build 阶段解析一次。运行时 API 接收 handle 或与 topology 完全一致的连续数组，不再执行 `GetMeshInstances()` 名字扫描。

### 5.4 PoseBatch 合约

第一版 CPU API：

```cpp
struct PoseBatchView {
    const float* rigidPoses;   // [B, N_rigid, 7]
    const float* linkPoses;    // [B, N_link, 7]
    const float* cameraPoses;  // [B, C, 7]
    const uint8_t* activeMask; // [B, N_instance]
};
```

GPU API：

```python
scene.bind_pose_source(
    rigid_poses=rigid_tensor,    # DLPack-compatible CUDA tensor
    link_poses=link_tensor,
    camera_poses=camera_tensor,
    stream=cuda_stream,
)
```

约束：

- dtype 固定为 float32。
- quaternion 固定为 wxyz，对外文档不得混用 xyzw。
- 内存要求 contiguous；不满足时由 Python 显式转换，不在每帧 C++ 中静默复制。
- topology、B、N、C 必须与 build 时一致。
- GPU fast path 不允许 Python per-object loop。
- 同步使用 timeline semaphore/event，禁止 `waitForIdle()`。

### 5.5 Sensor 与输出合约

```cpp
enum class Product {
    ColorRGBA8,
    ColorRGBA16F,
    DepthR32F,
    InstanceR32U,
    SemanticR32U,
    NormalRGBA16F
};

struct SensorRequest {
    std::vector<CameraHandle> cameras;
    std::vector<Product> products;
};
```

GPU 输出布局：

| Product | dtype | shape |
|---|---|---|
| ColorRGBA8 | `uint8` | `[B,C,H,W,4]` |
| ColorRGBA16F | `float16` | `[B,C,H,W,4]` |
| Depth | `float32` | `[B,C,H,W,1]` |
| Instance | `uint32` | `[B,C,H,W,1]` |
| Semantic | `uint32` | `[B,C,H,W,1]` |
| Normal | `float16` | `[B,C,H,W,4]` |

推荐 API：

```python
obs = camera.render(rgb=True, depth=True, segmentation=True)
obs.rgb.to_dlpack()       # 零拷贝 fast path
obs.rgb.to_torch()        # 包装相同 GPU allocation
obs.rgb.numpy()           # 显式同步和下载 slow path
```

CPU readback ring 继续服务 `.numpy()` 和调试截图，但不能作为具身仿真主路径。

### 5.6 多环境实现分两步

第一步：API/状态批量化。

- 共享 AssetCache 和不可变 Mesh/Texture/BLAS。
- 每环境拥有独立实例位姿和 camera state。
- 可以暂时顺序录制 B×C render pass，先建立正确输出和 GPU 数据链。
- 一次 Python 调用、一次 frame submission、连续输出。

第二步：真正降低 B×C draw/recording 成本。

- texture array / layered output。
- Vulkan multiview（在管线和硬件支持时）。
- instance data 中增加 env/view 索引。
- indirect draw 或可复用 command buffer。
- raster 与 RT 分开设计；RT 可采用 per-env TLAS 或 descriptor-indexed TLAS，不能依赖有限 instance mask 隔离大量环境。

不要在第一步完成前直接实现复杂 multiview，否则难以区分场景装配、状态索引和 shader 错误。

---

## 6. 代码 Agent 执行规则

1. 每周先完成 correctness gate，再测性能；不允许用错误或缺资产画面报告高 FPS。
2. 不允许在 Python 中按 `B × N` 循环调用 `update_node_transform()` 作为最终实现。
3. 不允许动态帧重新生成或重新加载 GLB。
4. 不允许在 GPU fast path 中调用 `mapStagingTexture()`、`.cpu()`、`.numpy()` 或 `waitForIdle()`。
5. 不允许重新启用多 cmdList micro-batch 作为默认路径，除非新增独立 queue 且有新的 GPU trace 证据。
6. 不修改无关材质和阴影质量；性能基准先固定画质配置。
7. 每个新增批量 API 都必须有 B=1 兼容测试和 B=8 独立性测试。
8. 所有性能结果必须保存 JSON 原始数据，报告中的表格由 JSON 生成。

---

## 7. Gate 0：当前并行渲染收尾（2026-07-18 已完成）

完成状态：生产路径固定为 single cmdList，公开的 `submit_frame_batch_ex()` 已移除；Ring K+2 + 逆序 read + hash 测试通过；`frl_apartment_stage.glb` 的无 RT/RT 基线已写入 roadmap 和周报。

### 7.1 工作内容

- 将 single-cmdList 标为默认和已验证路径。
- 移除 `submit_frame_batch_ex(micro_batch_size)`，生产路径只保留 `submit_frame_batch()`。
- 固化 Ring/EventQuery/hash correctness test。
- 修正 benchmark 默认 ReplicaCAD 路径为 `E:\cplus\Flora\ReplicaCAD` 或仓库相对 `ReplicaCAD`。
- benchmark 增加场景、B/C/P、分辨率、输出端点、RT 配置和 commit hash 元数据。
- 将本节实验数据同步到周报和 roadmap。

### 7.2 目标文件

- `src/PythonBindings/headless_pbr.cpp`
- `src/PythonBindings/headless_pbr.h`
- `src/PythonBindings/py_bindings_common.h`
- `tools/test_ring_fix.py`
- `tools/bench_replicacad_parallel.py`
- `output/parallel_render_efficiency_roadmap.md`
- `docs/RTXNS_Parallel_Render_Weekly_Report_2026-07-11.md`

复现实验命令：

```powershell
cd E:\cplus\Flora

# Ring/EventQuery/legacy camera 数据正确性
E:\python\python.exe tools\test_ring_fix.py
E:\python\python.exe tools\test_ring_fix.py --rt-shadows

# ReplicaCAD 统一生产基准入口（single cmdList）
E:\python\python.exe tools\bench_replicacad_parallel.py --width 1280 --height 720 --camera-counts 1,4,8 --frames 30 --warmup 5 --global-warmup 20 --trials 5 --output output\bench_parallel\replicacad_single_cmdlist_720p.json
E:\python\python.exe tools\bench_replicacad_parallel.py --width 1280 --height 720 --camera-counts 1,4,8 --frames 30 --warmup 5 --global-warmup 20 --trials 5 --rt-shadows --output output\bench_parallel\replicacad_single_cmdlist_rt_720p.json
```

若某个脚本需要长时间运行，必须每完成一个配置就 flush 一条 JSON 记录，并输出当前 case、warmup/measure 进度和最近一次耗时，避免“长时间无返回”时无法区分正常运行和 native deadlock。

### 7.3 验收

- Ring depth K 下连续提交 K+2 次，超出部分稳定返回 busy。
- 乱序 read 后每帧 hash 与提交时 camera marker 一致。
- `is_batch_ready()` 在 GPU 未完成时为 false，完成后为 true。
- single/mb4/mb2 原始 JSON 可复现“micro-batch 无收益”结论。
- 运行 1000 次 submit/read 无覆盖、死锁和未释放 token。

### 7.4 可汇报成果

- 异步路径数据正确性闭环。
- 多 cmdList 方向以量化实验正式关闭。
- 后续所有动态仿真实验共享统一 benchmark 元数据。

---

## 8. Iteration A：完整场景与仿真渲染接口（四周）

### Week A1：ReplicaCAD Manifest 与 SceneDesc

> 完成状态（2026-07-18）：已完成。91/91 场景解析通过，2,293 个普通对象、540 个 articulated 实例、92 个 object config、5 个 stage config 和 12 个 URDF 均进入确定性 manifest；13 项单元测试及正式检查工具通过。

#### A1.1 目标

不急于渲染，先把 91 个场景全部解析成确定、可测试、与后端无关的 `SceneDesc`，解决路径、坐标、模板和缺失资源问题。

#### A1.2 建议文件

```text
python/donut_render_py/
  scene_desc.py                 # 静态数据结构
  replicacad.py                 # dataset/template/scene parser
  urdf.py                       # URDF 数据结构和解析器，A1 先完成 manifest

tests/
  test_replicacad_manifest.py
  test_replicacad_transforms.py

tools/
  inspect_replicacad_dataset.py # 输出覆盖率和缺失资源 JSON
```

#### A1.3 建议数据结构

```python
@dataclass(frozen=True)
class VisualAssetDesc:
    source_path: Path
    scale: tuple[float, float, float]

@dataclass(frozen=True)
class InstanceDesc:
    name: str
    template_name: str
    visual_asset: VisualAssetDesc
    world_pose_wxyz: np.ndarray
    motion_type: str
    semantic_id: int
    instance_id: int
    com: np.ndarray

@dataclass(frozen=True)
class SceneDesc:
    stage: InstanceDesc
    objects: tuple[InstanceDesc, ...]
    articulated: tuple[ArticulationDesc, ...]
    warnings: tuple[str, ...]
```

#### A1.4 实施步骤

1. 解析 dataset config 的路径注册表。
2. 扫描并缓存 stage/object/URDF 模板，不在每场景重复读 JSON/XML。
3. 解析 91 个 scene instance，分配确定的 instance ID。
4. 规范化所有路径并检查文件存在。
5. 写 quaternion、COM、scale 和矩阵顺序 golden test。
6. 输出 `replicacad_manifest_report.json`。

#### A1.5 验收标准

- 91/91 scene instance 解析成功；dataset registry 中缺失的外部 Fetch 依赖作为结构化 warning 单独记录。
- 普通对象实例统计为 2,293，articulated 实例统计为 540。原规划中的 541 为预估误差，当前数据集真实值为 540。
- `apt_0` 统计为 113 个普通对象和 6 个 articulated object。
- 所有本地 stage/object/URDF visual asset 均可解析到绝对路径。
- 相同输入多次解析产生完全相同的 handle/instance ID 顺序。
- 单元测试明确验证 wxyz 与矩阵行列主序。

#### A1.6 周报成果

| 指标 | 周初 | 周末目标 |
|---|---:|---:|
| 可解析完整场景 | 0 | 91 |
| 支持对象模板 | 仅手写 stage 路径 | 92 个 object config |
| articulated manifest | 0 | 540 个实例、12 个 URDF（已完成） |
| 资源缺失诊断 | 无 | 结构化 JSON 报告（已完成） |

本周汇报重点是“数据覆盖率从单 GLB 提升到完整 ReplicaCAD manifest”，不以 FPS 为主。

### Week A2：完整可视场景装配与 AssetCache

> 完成状态（2026-07-18）：已完成普通对象装配。`apt_0` 的 stage + 113 个普通对象已编译为 114 个顶层实例，81 个唯一 GLB 在 Donut model/graph 中去重；91/91 场景完成同进程原生加载、渲染和变换校验。`apt_0` 在 128×96、正确方向光、RT 8 samples、N=8 时达到 6,325 cam-FPS（stage-first/complete-first 共 6 trials 的顺序平衡中位数）。A2 同时修复异步 command buffer GC 和 RT binding set 重建问题，stage/普通对象场景各 10,000 batch 并包含一次热重载的压力测试通过。视觉完整场景仍依赖 A3 加载 `kitchen_counter` 等 6 个 URDF，否则其上的普通物体会看似悬空。

#### A2.1 目标

将 `SceneDesc` 编译成 Donut SceneGraph，完整显示 stage、普通家具和材质，并建立稳定 handle 和资产去重。

#### A2.2 建议文件

```text
src/PythonBindings/
  scene_asset_cache.h/.cpp      # Mesh/Material/Texture 缓存
  scene_builder.h/.cpp          # SceneDesc -> SceneGraph
  headless_pbr.h/.cpp           # 暴露 build/handle API
  py_bindings_common.h          # pybind11 数据结构

python/donut_render_py/
  runtime.py                    # create_scene_builder/load_replicacad/build

tests/
  test_replicacad_assembly.py
```

#### A2.3 实施步骤

1. 用规范化绝对路径和 import options 作为 AssetCache key。
2. 同一对象模板被多次实例化时共享 Mesh/Material/Texture。
3. 每个对象实例生成 `InstanceHandle`，保存 motion type、semantic ID 和 instance ID。
4. 正确应用 stage/object config、scene instance 和 glTF node transform。
5. 输出 scene AABB、实例数、unique asset 数、加载耗时和显存估算。
6. 保留现有临时 GLB 路径作 fallback，但新 ReplicaCAD 路径不得每帧生成 GLB。

#### A2.4 验收标准

- `apt_0` 中 stage + 113 个普通对象全部加载，位置和尺度通过 AABB/参考图验证。
- 重复对象模板只加载一份 GPU asset。
- 91 个场景完成 headless smoke render，无 native crash。
- 每场景实际实例数与 A1 manifest 一致。
- 随机抽取至少 20 个实例，最终 world transform 与 Python reference 误差小于 `1e-5`。
- 输出 `load_ms`、unique mesh/texture、实例数和峰值显存。

#### A2.5 周报成果

- 从“单 stage GLB”推进到“完整家具场景”。
- 展示 `apt_0`、一个 `v3_sc*_staging_*` 的全景和实例统计。
- 汇报 AssetCache 前后加载时间和显存；不要求预设提升比例，但必须提供对照数据。

#### A2.6 实际交付与结果（2026-07-18）

实际实现采用 Donut 原生 `models + graph` 作为第一版场景内 AssetCache：同一路径 GLB 只进入一次 model 表，多个实例共享底层 Mesh/Material/Texture。没有新增一套与 Donut 重复的 C++ mesh cache；跨 scene/跨 environment 的持久 GPU cache 保留到 Week B1。

关键交付文件：

```text
python/donut_render_py/donut_scene_compiler.py
tests/test_replicacad_assembly.py
tools/render_replicacad_complete_scene.py
tools/smoke_replicacad_complete_scenes.py
tools/bench_replicacad_complete_parallel.py
docs/RTXNS_ReplicaCAD_Assembly_Week_A2_Report.md
output/replicacad_complete/
```

验收结果：

| 指标 | A2 结果 |
|---|---:|
| `apt_0` 顶层实例 / 唯一 GLB | 114 / 81 |
| `apt_0` 原生 mesh / unique mesh / material | 136 / 103 / 90 |
| 91 场景 smoke | 91 / 91 |
| 普通对象原生覆盖 | 2,293 / 2,293 |
| 全量 transform 最大误差 | `8.41e-8` |
| 平均 / 最大加载 | 291.67 / 399.69 ms |
| 640×480 Raster / RT 8-sample | 812.5 / 412.0 FPS |
| 128×96 RT N=1/2/4/8 | 1,830 / 3,239 / 4,973 / 6,325 cam-FPS |
| 异步长稳态 | stage 10,000 + 热重载 + complete 10,000 batch 通过 |

完整结果和复现命令见 `docs/RTXNS_ReplicaCAD_Assembly_Week_A2_Report.md`。A2 收尾后，下一开发项切换到 Week A3；540 个 articulated instance 不计入 A2 静态普通对象完成率。

### Week A3：URDF 可视层级和动态位姿

#### A3.1 目标

支持 ReplicaCAD articulated object 的完整可视层级，在不接入物理引擎的情况下，由脚本或外部 link pose 驱动门、抽屉和冰箱等部件。

#### A3.2 建议文件

```text
python/donut_render_py/urdf.py
src/PythonBindings/urdf_scene_builder.h/.cpp
src/PythonBindings/scene_batch_state.h/.cpp
tests/test_urdf_loader.py
tests/test_pose_batch.py
tools/demo_replicacad_articulation.py
```

#### A3.3 第一版 API

```python
scene = renderer.create_scene_builder()
asset = scene.load_replicacad(dataset_root, "apt_0")
compiled = scene.build(num_envs=1)

rigid_handles = compiled.rigid_handles
link_handles = compiled.link_handles

compiled.set_rigid_poses(rigid_poses_np)  # [1, N_rigid, 7]
compiled.set_link_poses(link_poses_np)    # [1, N_link, 7]
```

#### A3.4 实施步骤

1. 解析 12 个 URDF 的 link/joint/visual。
2. 为每个 visual mesh 创建资产，为每个 link 创建稳定 handle。
3. 合并 articulated instance pose、URDF link pose 和 visual origin。
4. 实现 CPU `set_rigid_poses()` / `set_link_poses()`，一次传递连续数组。
5. C++ 直接索引 scene nodes，移除运行时名字查找。
6. 可选实现 Python CPU FK reference，用 joint position 生成 link pose，仅用于 demo/test。
7. transform dirty 时只更新 instance buffer/TLAS；禁止重新加载资产或 BLAS。

#### A3.5 验收标准

- `apt_0` 的 6 个 articulated object 全部可见。
- fixed/prismatic/revolute joint 的参考姿态正确。
- 冰箱门、柜门和抽屉脚本动画持续 1000 帧，无漂移、错 link 和资源增长。
- 每帧不出现 `load_scene()`、GLB 写盘或 BLAS rebuild。
- CPU batch API 与逐节点 reference 输出图像 hash 一致。
- 输出 pose update、SceneGraph refresh、TLAS update、render 四段耗时。

#### A3.6 周报成果

- 首次完成“外部状态驱动的动态 ReplicaCAD”。
- 提供静态/运动前后截图或短序列。
- 报告每帧动态对象数、link 数和位姿更新时间。

### Week A4：多模态 Sensor 与 Genesis 风格接口

#### A4.1 目标

以一个 camera render 请求返回对齐的 RGB、Depth、Instance、Semantic 和 Normal，并形成可供 Genesis/外部仿真器调用的稳定 Python API。

#### A4.2 建议文件

```text
src/PythonBindings/sensor_products.h/.cpp
src/PythonBindings/headless_pbr.h/.cpp
src/PythonBindings/py_bindings_common.h
python/donut_render_py/runtime.py
python/donut_render_py/objects.py
python/rtxns_genesis_style/renderer.py
tests/test_sensor_products.py
tools/demo_genesis_style_replicacad.py
```

#### A4.3 实施步骤

1. 定义 `SensorRequest`，camera 与 product 分离。
2. 输出线性 Depth R32F，并明确单位、near/far 和背景值。
3. 为普通实例和 URDF link 分配全局 instance ID。
4. 将 object config `semantic_id` 写入 instance/material 数据并输出 Semantic R32U。
5. 增加 Normal RGBA16F 或 R16G16B16A16_FLOAT 产品。
6. 先完成 CPU readback reference，GPU OutputArena 放到 Iteration B。
7. `GenesisStyleRenderer` 改为调用新 build/update/render API，保留兼容方法。

#### A4.4 验收标准

- 所有产品 shape/dtype 与文档一致。
- RGB、Depth、Instance、Semantic、Normal 的边缘在像素级对齐。
- 随机选择至少 20 个对象，中心像素 instance/semantic ID 与 manifest 一致。
- Depth 与相机几何 reference 误差在定义容差内。
- 一个外部 scripted simulator 只提供 PoseBatch，即可驱动场景并获取观测。
- 1000 帧多产品渲染无资源增长和 ID 变化。

#### A4.5 周报成果

| 能力 | Iteration A 前 | Week A4 后 |
|---|---|---|
| ReplicaCAD | 单 stage GLB | 完整 scene instance |
| 动态对象 | 少量名字更新 | 普通刚体 + URDF link batch pose |
| 产品 | RGBA8 | RGB/Depth/Instance/Semantic/Normal |
| Python 调用 | 渲染器测试 API | Genesis 风格 build/update/render |

---

## 9. Iteration B：SAPIEN 级并行渲染数据链（四周）

### Week B1：CompiledSceneBatch 与独立多环境

#### B1.1 目标

实现真正的 `B>1` 状态语义：同一 ReplicaCAD topology 复制为多个独立环境，共享不可变资产，每个环境的对象和相机位姿互不影响。

#### B1.2 建议文件

```text
src/PythonBindings/compiled_scene_batch.h/.cpp
src/PythonBindings/scene_batch_state.h/.cpp
src/PythonBindings/scene_asset_cache.h/.cpp
src/PythonBindings/py_bindings_common.h
tests/test_scene_batch_independence.py
tools/bench_replicacad_scene_batch.py
```

#### B1.3 初始实现要求

- B 个环境共享 Mesh、Texture、Material 和可共享的 BLAS。
- `EnvStateBuffer` 连续布局为 `[B, N_instance]`。
- `CameraStateBuffer` 连续布局为 `[B, C]`。
- 第一版允许 GPU 顺序执行 B×C pass，但 Python 只调用一次 submit。
- 输出必须保留 B 和 C 维度，不能展平后丢失映射。

#### B1.4 验收标准

- B=1/2/4/8 下使用不同对象和相机位姿，输出对应环境正确且无串扰。
- B=1 新 API 与旧单场景 API 图像 hash 一致。
- immutable asset 显存不随 B 重复增长；只允许 instance state、TLAS（若采用 per-env）和 output 按 B 增长。
- 一帧只执行一次 Python submit；禁止 Python env loop。
- 报告 B=1/2/4/8 的 build time、VRAM、CPU record、GPU render 和总吞吐。

#### B1.5 周报成果

- 首次从 `B=1,C>1` 推进到独立 `B>1,C>=1`。
- 提供每环境不同物体姿态的拼图及 hash 隔离测试。
- 给出共享资产后的显存增长曲线。

### Week B2：GPU PoseSource 与 Taichi/Torch 接口

#### B2.1 目标

外部仿真器直接在 GPU 上提供 PoseBatch，Flora 不经过 NumPy、名字查找或逐物体 Python 调用即可更新实例和相机状态。

#### B2.2 建议文件

```text
src/PythonBindings/gpu_pose_source.h/.cpp
src/PythonBindings/cuda_vulkan_interop.h/.cpp
src/PythonBindings/pose_update.cu        # 若采用 CUDA kernel
src/PythonBindings/py_bindings_common.h
python/donut_render_py/tensor.py
tests/test_gpu_pose_source.py
tools/bench_pose_update.py
```

#### B2.3 实施步骤

1. 先定义 DLPack 输入和生命周期规则。
2. 校验 device id、dtype、shape、stride 和 contiguous。
3. 将外部 pose buffer 绑定到 `CompiledSceneBatch`。
4. 使用 GPU kernel 转换 quaternion/position 到 renderer instance matrix。
5. 使用 timeline semaphore 或等价 event 表达 producer stream → Vulkan render 的依赖。
6. 不得每帧 import external memory；build/bind 时建立，帧间复用。
7. Linux 使用 OpaqueFd；Windows 如支持则实现 OpaqueWin32，不能硬编码 FD。

#### B2.4 验收标准

- CPU PoseBatch 与 GPU PoseSource 在相同输入下输出 hash 一致。
- B=8、完整 `apt_0` 连续 10,000 次 pose update 无泄漏和死锁。
- GPU fast path 中无 `.cpu()`、`.numpy()`、staging map 和 `waitForIdle()`。
- producer 使用非默认 CUDA stream 时同步正确。
- 报告 CPU batch、GPU batch 的 update p50/p95 和 CPU 占用。
- GPU update 后 CPU frame submission 时间不随 `B×N` 线性增加。

#### B2.5 Genesis/Taichi 接入示例

```python
# Taichi/Genesis 可先通过 Torch/DLPack 暴露状态。
rigid = simulator.rigid_poses_torch()  # [B, N_rigid, 7], CUDA
links = simulator.link_poses_torch()   # [B, N_link, 7], CUDA
cams = simulator.camera_poses_torch()  # [B, C, 7], CUDA

scene.bind_pose_source(rigid, links, cams, stream=torch.cuda.current_stream())
obs = sensors.render(rgb=True, depth=True)
```

### Week B3：GPU OutputArena 与零拷贝观测

#### B3.1 目标

让渲染结果保持在 GPU 中，以 DLPack/Torch Tensor 暴露；只有调用 `.numpy()` 时才进入现有 readback ring。

#### B3.2 建议文件

```text
src/PythonBindings/gpu_output_arena.h/.cpp
src/PythonBindings/cuda_vulkan_interop.h/.cpp
src/PythonBindings/sensor_products.h/.cpp
python/donut_render_py/tensor.py
tests/test_gpu_output_tensor.py
tools/bench_gpu_output.py
```

#### B3.3 实施步骤

1. build 时按产品分配连续、可导出的 GPU buffer。
2. camera render target 通过 GPU copy/compute pack 写入 `[B,C,H,W,...]`。
3. DLPack capsule 保存 allocation、shape、stride、dtype 和 owner 生命周期。
4. Tensor 消费者释放前，scene/output allocation 不得销毁。
5. 用 timeline value 表示对应 frame 已可被 CUDA/Torch 消费。
6. `.numpy()` 显式等待并下载，复用 occupancyToken/backpressure。
7. 未请求的产品不得执行 render/copy/pack。

#### B3.4 验收标准

- Torch 获取 CUDA Tensor，device pointer 在连续帧中稳定复用。
- GPU Tensor 路径无 CPU readback。
- B=1/2/4/8，RGB/Depth/Instance 输出 shape/dtype 正确。
- Torch kernel 可以直接消费输出并得到正确统计值。
- Tensor 生命周期覆盖异步渲染和下一帧提交，不发生提前复用。
- 分别报告 GPU-ready、Torch-ready、CPU NumPy e2e 三类吞吐。

#### B3.5 周报成果

- 从“异步 CPU 图像读回”推进到“GPU 仿真可直接消费的观测 Tensor”。
- 提供删除 readback 后的等待、传输字节数和吞吐变化。
- 明确 CPU 慢路径和 GPU 快路径的性能差距。

### Week B4：成本优化与 SAPIEN 对标

#### B4.1 目标

在正确的完整场景、独立环境和 GPU 端点上定位剩余瓶颈，并完成第一次可信 SAPIEN 对标。

#### B4.2 优化顺序

1. 缓存 RayTracedShadow 和 Sensor pass BindingSet。
2. 缓存静态 draw list/material binding。
3. 静态场景跳过 TLAS update，transform dirty 时只做 update。
4. 减少每 camera 常量和资源状态切换。
5. 评估可复用 command buffer/secondary command buffer，但不得假设它会提供 graphics queue 并行。
6. 用 GPU profiler 判断是否值得实现 multiview/instanced raster。
7. RT 与 raster 分开报告，先完成 raster/sensor parity。

#### B4.3 SAPIEN 公平对标配置

- 平台：Linux x86-64 + 同一 CUDA/Vulkan GPU。
- 场景：同一完整 ReplicaCAD `apt_0.scene_instance.json`。
- topology 和每帧 pose 完全一致。
- `B=1/2/4/8/16`，`C=1/4`。
- 分辨率：128×96、640×480、1280×720。
- 产品：先 ColorRGBA8；再 Color+Depth；再 Color+Depth+Segmentation。
- 分别记录 GPU submission、GPU-ready、Torch-ready 和 CPU readback。
- warmup 100 帧，计时至少 1000 帧，至少重复 5 组。
- 保存 p50、p95、标准差、VRAM、CPU utilization 和原始 JSON。

#### B4.4 最终验收门槛

正确性门槛：

- B=8 独立环境持续 1000 帧，无串扰、死锁、资源增长和 ID 错误。
- 所有输出产品通过 reference test。
- GPU fast path 无 CPU round-trip。

性能门槛：

- 相同端点下，B=8 GPU-ready 总吞吐达到 SAPIEN 的 90% 以上。
- stretch goal：达到或超过 SAPIEN。
- CPU record/submit 时间占端到端时间低于 10%，或给出 profiler 证据说明瓶颈在 GPU。
- B 增长时 immutable asset 显存保持近似常数。
- 若未达到目标，报告必须给出 GPU trace 中占比最高的三个 pass，而不是继续盲目拆 cmdList。

#### B4.5 周报成果

- 第一份端点严格对齐的 Flora vs SAPIEN 报告。
- 给出吞吐比、延迟、显存、CPU 开销和 modality cost。
- 明确下一阶段是 multiview、shader/pass 优化还是外部物理一体化。

---

## 10. 统一 Benchmark 规范

### 10.1 计时端点

| 名称 | 起点 | 终点 | 是否包含 CPU copy |
|---|---|---|---|
| `submit_only` | Python 调用 submit | command list 已提交 | 否 |
| `gpu_ready` | pose producer ready | Vulkan 完成输出 | 否 |
| `tensor_ready` | pose producer ready | CUDA/Torch 可安全消费 | 否 |
| `cpu_e2e` | pose 更新 | NumPy 图像可读 | 是 |

报告中不得只写“FPS”，必须写成例如：

```text
B=8, C=1, ColorRGBA8, 1280x720, tensor_ready = 742 cam-FPS
```

### 10.2 每次结果必须保存

```json
{
  "commit": "<git sha>",
  "gpu": "<device>",
  "os": "<os>",
  "driver": "<driver>",
  "backend": "vulkan",
  "scene": "apt_0.scene_instance.json",
  "scene_complete": true,
  "B": 8,
  "C": 1,
  "products": ["ColorRGBA8"],
  "resolution": [1280, 720],
  "rt_shadow": false,
  "endpoint": "tensor_ready",
  "warmup_frames": 100,
  "measure_frames": 1000,
  "p50_ms": 0.0,
  "p95_ms": 0.0,
  "cam_fps": 0.0,
  "vram_mb": 0.0
}
```

### 10.3 正确性基准

- 固定 camera/pose 的图像 hash。
- B 个环境使用明显不同的 marker/pose，检测跨环境串扰。
- Instance/Semantic 中心像素查表。
- Depth reference 平面/盒子测试。
- Ring K+2 submit 和乱序 read。
- 连续 10,000 帧资源和同步压力测试。

---

## 11. 风险与决策点

| 风险 | 影响 | 处理方式 |
|---|---|---|
| Windows CUDA/Vulkan external handle 差异 | SAPIEN 和 Flora interop 可能 native crash | Linux/CUDA 作为性能主平台；Windows 使用 OpaqueWin32 或 CPU fallback |
| Donut 资产对象无法跨 SceneGraph 共享 | B 增加时显存重复 | Week B1 前先验证资源 ownership；必要时统一 SceneGraph + batched instance buffer |
| URDF 坐标/scale/COM 组合错误 | 部件漂移或旋转轴错误 | A1/A3 使用 CPU reference 和 golden transform 测试 |
| GPU OutputArena 与 render target 格式不一致 | 需要额外 pack/copy | 先接受一次 GPU pack，profile 后决定直接渲染到 exportable buffer |
| TLAS 每帧更新过重 | 动态 RT 成为瓶颈 | raster parity 优先；RT 单独报告；按 dirty 和 per-env TLAS 评估 |
| multiview 改造过早 | shader/管线复杂且难调 | B1-B3 先完成数据链，B4 由 profiler 决策 |
| 不同渲染器输出格式不同 | 对标失真 | 固定产品、格式和计时端点，必要时双方都增加相同转换 |

---

## 12. 每周汇报模板

每周报告应固定包含：

### 12.1 本周目标

- 本周承诺的 capability 和 correctness gate。
- 对应代码文件和 API。

### 12.2 代码推进

| Commit/文件 | 改动 | 对外能力 |
|---|---|---|
| `<sha/path>` | `<summary>` | `<API/behavior>` |

### 12.3 正确性结果

- 数据覆盖率。
- 测试通过数。
- 截图/hash/ID/depth reference。
- 已知缺失资产和限制。

### 12.4 性能收益

| 指标 | 优化前 | 本周 | 收益 |
|---|---:|---:|---:|
| pose update p50 | | | |
| GPU-ready cam-FPS | | | |
| CPU e2e cam-FPS | | | |
| VRAM | | | |

### 12.5 结论与下周计划

- 哪个假设被验证或否定。
- 当前最大瓶颈及 profiler 证据。
- 下周的明确验收标准。

---

## 13. 下一步执行清单

### 立即执行

1. Gate 0 已完成：single-cmdList 基线、测试元数据和无 RT/RT correctness 已冻结。
2. 不再继续 micro-batch/multi-cmdList 优化。
3. Week A1 已完成：`ReplicaCADManifest/SceneDesc`、91 场景覆盖率、路径解析和 transform tests 已交付。
4. Week A2 已完成：`SceneDesc -> Donut SceneGraph`、场景内资产去重、91 场景原生 smoke 和完整场景并行基线已交付。
5. 下一项开发进入 Week A3：实现 URDF link/visual 层级、稳定 link handle 和 CPU `PoseBatch` reference API，先让 `apt_0` 的 6 个 articulated object 可见并可由外部位姿驱动。

### Iteration A 完成定义

- 完整 ReplicaCAD 场景可见。
- 普通刚体和 URDF link 可由外部 PoseBatch 动态驱动。
- RGB/Depth/Instance/Semantic/Normal 正确对齐。
- Python API 可按 Genesis 的 build/update/render 模式调用。

### Iteration B 完成定义

- B=8 独立环境共享资产并正确渲染。
- GPU PoseSource 和 GPU OutputArena 无 CPU round-trip。
- Linux/CUDA 上完成同端点 SAPIEN 对标，达到 90% 以上吞吐目标或给出明确 GPU profiler 瓶颈。

完成以上两轮后，Flora 才具备接入 Genesis/Taichi 和未来自研 GPU 物理模块的稳定渲染边界；在此之前，不应将物理求解代码直接耦合进渲染器。
