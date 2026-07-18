# Flora（RTXNS）Iteration A 阶段开发总结与暂停点

> 整理日期：2026-07-18
>
> 仓库：`https://github.com/xio-md/Flora.git`
>
> 分支：`main`
>
> 功能封存基线：`0ffb048`（完成 ReplicaCAD A4 多模态传感器与 Genesis 接口）
>
> 当前状态：Iteration A 已完成，项目阶段性暂停；未开始 Iteration B。

## 一、阶段目标与结论

Iteration A 的目标是让 Flora 从“能加载单个 GLB、输出 Color 的渲染测试程序”，推进为可以消费外部物理位姿、完整加载 ReplicaCAD、驱动 URDF 动态层级并输出具身智能多模态观测的渲染后端。

当前已经形成以下闭环：

```text
ReplicaCAD scene_instance.json
  -> manifest / template / URDF 解析
  -> Donut SceneGraph 编译与资产去重
  -> stage + 普通家具 + articulated visual 原生加载
  -> 外部 rigid/link pose 批量更新
  -> 单次 Scene::Refresh + 多相机渲染
  -> Color / Depth / Normal / Instance / Semantic SensorFrame
```

Iteration A 可以验收。当前实现适合作为 `B=1` 的动态具身视觉 reference 和 Iteration B 的正确性基线，但还不是 SAPIEN `RenderSystemGroup` 等价物。

---

## 二、阶段提交记录

| 阶段 | Commit | 主要交付 |
|---|---|---|
| 并行渲染收尾 | `96da791` | single-cmdList、异步 readback ring、ReplicaCAD 回归基线 |
| A1 | `082665f` | 91 个 ReplicaCAD 场景清单、模板、路径和 transform 解析 |
| A2 | `c0547db` / `cf7cb23` | 完整静态场景装配、资产去重、91 场景原生 smoke |
| A3 | `a36df2f` | URDF link/visual、稳定句柄、26 关节批量位姿和动态 TLAS |
| A4 | `0ffb048` | 五产品 Sensor、稳定标签、Genesis 风格接口和正式实验 |

---

## 三、当前已完成能力

### 1. 并行渲染基础

- `CameraDesc` 与独立 `RenderViewSlot` 支持一个场景内的多相机。
- batch 内只执行一次 `Scene::Refresh` 和一次场景级 BLAS/TLAS 构建或更新。
- Color 路径支持 `submit_frame_batch()`、EventQuery 和异步 `read_frame_batch()`。
- readback ring 默认深度 K=4，使用 occupancy token 防止未读槽位被覆盖。
- 超容量提交返回 busy，乱序读取使用提交时保存的 ring index。
- micro-batch/multi-cmdList 已由对照实验否定，同一 graphics queue 上拆分越细越慢，因此不再推进。

### 2. ReplicaCAD 完整场景

- 91 个 `scene_instance.json` 均可解析和编译。
- 全数据集包含 2,293 个普通对象和 540 个 articulated instance。
- articulated 数据覆盖 3,330 个 link 和 2,790 个 visual，遗漏数为 0。
- `apt_0` 包含 stage、113 个普通对象和 6 个 articulated object。
- `apt_0` 原生场景共有 171 个 mesh instance、127 个 unique mesh/geometry 和 111 个 material。

### 3. 动态对象与 URDF

- 支持 fixed、prismatic、revolute 和 continuous joint 的 reference FK。
- articulation 保留 `root -> link -> visual` SceneGraph 父子层级。
- 场景加载后建立稳定 node handle，逐帧不再按名称线性查找。
- `update_node_transforms_batch()` 在写入前完成全部校验，保证原子更新。
- `apt_0` 每批可更新 26 个 movable joint，原生 pose write 约 `0.02 ms/batch`。
- 动态帧只更新 SceneGraph world transform、instance desc 和 TLAS，不重新加载 GLB 或重建 BLAS。

### 4. 多模态 Sensor

| 产品 | Shape | Dtype | 定义 |
|---|---|---|---|
| Color | `[H,W,4]` | `uint8` | RGBA8 |
| Depth | `[H,W]` | `float32` | 相机光轴线性距离，单位米，背景 0 |
| Normal | `[H,W,3]` | `float32` | world-space unit shading normal |
| Instance | `[H,W]` | `uint32` | 稳定逻辑对象 ID，背景 0 |
| Semantic | `[H,W]` | `uint32` | ReplicaCAD semantic ID，unknown/background 为 0 |

- Depth/Normal/Instance 来自同一 GBuffer 几何覆盖并逐像素对齐。
- stage、普通对象和 articulated root 使用稳定非零 Instance ID。
- 同一个 articulation 的所有 link/visual 继承 root Instance ID，关节运动不改变身份。
- 原生标签 API 拒绝 ID 0、重复 node 和重复 Instance ID，并原子替换映射。
- Python `SensorFrame` 数组连续且拥有独立存储，不引用临时 pybind byte buffer。

### 5. Python 与 Genesis 风格接口

当前两层 Python API 均已接入：

```python
frame = renderer.render_sensor(camera)
frames = renderer.render_sensor_batch(cameras, products=("depth", "instance"))
```

- `rtxns_genesis_style.GenesisStyleRenderer` 支持 build/update/render 调用模式。
- `donut_render_py.Scene` 提供公开单相机和多相机 Sensor 包装。
- 外部模拟器可以提交 rigid/link matrix，再获取结构化 NumPy 观测。
- legacy Color 调用与 Sensor 调用交替时会复用既有 camera slot，不持续增加原生资源。

---

## 四、当前正式实验结果

### 1. 1280×720 多模态正确性

场景：`apt_0`；相机：`(3.5,2.0,3.5) -> (0,1,0)`；RT off。

| 指标 | 结果 |
|---|---:|
| 编译标签 | 120 |
| 可见 Instance | 61 |
| 可见 Semantic 值 | 27 |
| 有效几何像素 | 786,375 |
| Depth/Normal/Instance 掩码 | 全图完全一致 |
| Instance→Semantic | 20 个代表像素 + 全图 LUT 均正确 |
| 3 m Depth reference 误差 | `2.861e-6 m` |

结果图：`output/replicacad_a4/apt_0_multimodal_contact_sheet.png`。

### 2. 动态五产品吞吐

条件：128×96，`B=1`，`C=1/4/8`，每 batch 更新 26 个关节，RT off，同步 CPU readback，3 次中位数。

| C | Color-only cam-FPS | 五产品 cam-FPS | 五产品/Color |
|---:|---:|---:|---:|
| 1 | 1,773 | 807 | 45.5% |
| 4 | 3,671 | 1,212 | 33.0% |
| 8 | 4,082 | 1,177 | 28.8% |

五产品每相机 payload 为 344,064 bytes，是 Color-only 49,152 bytes 的 7 倍；同时增加 GBuffer 和 MaterialID pass。N=4 后吞吐接近平台，主要成本已转为 per-camera pass 和 CPU readback，而不是 pose write 或 Scene Refresh。

### 3. 1000 帧动态稳定性

| 检查项 | 结果 |
|---|---|
| 有效掩码错误 | 0 / 1000 |
| 未注册 Instance ID | 0 / 1000 |
| Semantic 映射错误 | 0 / 1000 |
| 恢复 pose 0 后五产品哈希 | 5 / 5 完全一致 |
| node handle | 380 -> 380 |
| 原生场景资源统计 | 前后完全一致 |
| RSS 增长 | `0.63 MiB` |

---

## 五、代码与文档入口

| 入口 | 作用 |
|---|---|
| `src/PythonBindings/headless_pbr.cpp` | 多相机、动态位姿、Sensor pass、readback 和标签映射 |
| `src/PythonBindings/py_bindings_common.h` | 原生 Python 绑定 |
| `python/donut_render_py/replicacad.py` | ReplicaCAD manifest 和 URDF 数据入口 |
| `python/donut_render_py/donut_scene_compiler.py` | SceneGraph、URDF 和 sensor label 编译 |
| `python/rtxns_genesis_style/renderer.py` | Genesis 风格兼容层 |
| `python/rtxns_genesis_style/sensor.py` | SensorFrame 合约和 NumPy 解码 |
| `docs/RTXNS_ReplicaCAD_Articulation_Week_A3_Report.md` | A3 动态关节报告 |
| `docs/RTXNS_ReplicaCAD_Multimodal_Week_A4_Report.md` | A4 多模态报告 |
| `output/RTXNS_Dynamic_Embodied_Sim_Plan.md` | A/B 两轮完整推进方案 |
| `output/replicacad_a4/` | A4 正式图像、指标、基准和压力结果 |

---

## 六、当前边界

当前明确未完成：

1. 只有 `B=1`，多个 camera 共享同一个环境状态。
2. 多产品路径为同步 CPU readback；只有 Color 有成熟异步 ring。
3. 位姿仍经过 CPU/pybind matrix，没有 GPU PoseSource。
4. 输出为 NumPy，没有 Torch/Taichi/DLPack GPU Tensor。
5. 不同环境之间尚无不可变 mesh/texture/material/BLAS 共享层。
6. articulated object 当前缺少可靠类别来源，Semantic 暂为 unknown 0。
7. 尚无 Vulkan GPU timestamp，当前分解主要是 CPU record 和端到端 wall time。
8. Windows 上尚未获得可用的 SAPIEN GPU-ready 同端点基准，不能宣称已经比肩 SAPIEN。
9. Flora 当前不负责碰撞、动力学或关节约束求解；物理状态由 Genesis/Taichi 或未来模块提供。

---

## 七、暂停与恢复约定

### 1. 暂停点

- Iteration A 已完成并封存。
- 不启动 Week B1，不改 SceneBatch、GPU Pose 或 GPU Output 数据链。
- A4 正式结果作为后续所有多环境和 GPU 输出改造的 correctness reference。
- 暂停期间不再围绕 RT 阴影画质、micro-batch 或更多 cmdList 做局部优化。

### 2. 恢复开发前的最小验证

```powershell
cmake --build E:\cplus\Flora\build-flora `
  --config Release --target DonutRenderPyNative

E:\python\python.exe -m unittest discover `
  -s E:\cplus\Flora\tests -p "test_*.py"

E:\python\python.exe E:\cplus\Flora\tools\test_ring_fix.py `
  --width 128 --height 96

E:\python\python.exe E:\cplus\Flora\tools\donut_render\genesis_multimodal_smoke.py
E:\python\python.exe E:\cplus\Flora\tools\donut_render\runtime_multimodal_smoke.py
```

预期结果：Release 构建通过、29 个单元测试通过、ring K=4 逆序读回通过、两条多模态 Python smoke 通过。

### 3. 下一次恢复入口

恢复开发时从 Iteration B / Week B1 开始，首要目标是：

```text
一个 compiled topology
  + B 组独立 rigid/link/camera 状态
  + 共享 immutable mesh/texture/material/BLAS
  + 环境隔离正确性测试
```

在 B1 完成前，不开始 GPU Tensor 性能宣传，也不将物理求解代码耦合进渲染器。
