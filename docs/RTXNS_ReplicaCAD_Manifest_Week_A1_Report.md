# RTXNS ReplicaCAD 完整场景解析周报 — Week A1

> 日期：2026-07-18
> 数据集：`ReplicaCAD/replicaCAD.scene_dataset_config.json`
> 范围：Dataset registry、SceneDesc、普通对象、URDF manifest、路径与变换正确性
> 本周不涉及：SceneGraph 装配、物理仿真、动态关节更新和渲染性能优化

## 概述

本周完成了 ReplicaCAD 完整场景数据层。此前 Flora 的 ReplicaCAD 示例只能直接加载 `frl_apartment_stage.glb`，得到房间墙体、地面和楼梯等静态 stage，无法读取 `scene_instance.json` 中定义的家具、普通刚体和 URDF articulated object。

当前新增 `ReplicaCADManifest` 和后端无关的 `SceneDesc`。解析器从 dataset config 出发，一次性缓存 stage、object、lighting、scene 和 URDF registry，再把 91 个 `scene_instance.json` 编译成稳定的场景描述。该描述不依赖 Donut SceneGraph、Vulkan 或物理引擎，可作为下一阶段场景装配、动态姿态更新和批量环境渲染的统一输入。

本周核心结果：

| 指标 | 周初 | 当前结果 | 状态 |
|---|---:|---:|---|
| 可解析完整场景 | 0 | **91/91** | 通过 |
| Stage template | 手工 GLB 路径 | **5/5** | 通过 |
| Object template | 0 | **92/92** | 通过 |
| 普通对象实例 | 0 | **2,293** | 通过 |
| URDF template | 0 | **12/12** | 通过 |
| Articulated 实例 | 0 | **540** | 通过 |
| URDF link / visual | 0 | **61 / 48** | 通过 |
| 场景所需 visual 缺失 | 未知 | **0** | 通过 |
| 单元测试 | 0 | **13/13** | 通过 |

原推进文档预估 articulated 实例为 541。全量读取当前数据集后确认真实值为 540：90 个非空场景各包含 6 个 articulated object，`empty_stage` 不包含 articulated object。因此本周将验收基线修订为 540，不通过伪造实例迎合旧数字。

---

## 一、问题背景

### 1. 只加载 Stage GLB 的限制

此前示例直接加载：

```text
ReplicaCAD/stages/frl_apartment_stage.glb
```

该文件只包含公寓建筑壳体。完整的 `apt_0` 实际由以下内容共同组成：

```text
frl_apartment_stage.glb
  + 113 个普通家具实例
  + 6 个 URDF articulated object
  + lighting / navmesh 引用
```

如果不解析 `apt_0.scene_instance.json`，渲染器无法知道家具模板、实例姿态、运动类型、语义 ID、冰箱和柜门等 URDF 引用。

### 2. 数据集中的不统一表示

真实数据中存在多种 handle 写法：

```text
stages/frl_apartment_stage
frl_apartment_stage
Stage_v3_sc0_staging
objects/frl_apartment_chair_05
```

解析器不能简单拼接字符串。本周实现了 canonical handle 和 alias registry，将带目录、不带目录和配置文件名统一解析到同一个模板，同时对重复 alias 做冲突检查。

### 3. 坐标和变换约定

ReplicaCAD scene instance 的 rotation 使用：

```text
[w, x, y, z]
```

本周明确固定以下约定：

- 四元数输入顺序为 `wxyz`。
- 四元数进入 `PoseDesc` 时归一化，零四元数直接拒绝。
- 变换矩阵以 row-major 形式表达，translation 位于最后一列。
- `translation_origin`、COM 和 visual scale 分开保留，不在解析阶段擅自合并。

---

## 二、实现内容

### 1. 后端无关 SceneDesc

新增 `python/donut_render_py/scene_desc.py`，主要数据结构包括：

| 数据结构 | 作用 |
|---|---|
| `PoseDesc` | 保存 translation、wxyz quaternion，并生成 row-major matrix |
| `VisualAssetDesc` | 保存绝对 visual/collision 路径和 scale |
| `InstanceDesc` | 保存 stage/普通对象实例、motion type、semantic ID、COM 和稳定 ID |
| `UrdfVisualDesc` | 保存 link visual mesh、origin xyz/rpy 和 scale |
| `UrdfJointDesc` | 保存 joint 类型、父子 link、axis 和 limit |
| `ArticulationDesc` | 保存 URDF 实例、fixed base、uniform scale 和 joint manifest |
| `SceneDesc` | 汇总 stage、objects、articulated、lighting、navmesh 和 warnings |

这些结构均为不可变 dataclass。A2 可以读取它们创建 Donut SceneGraph，未来物理模块也可从同一份描述建立刚体和关节系统。

### 2. ReplicaCADManifest

新增 `python/donut_render_py/replicacad.py`，主要流程为：

```text
dataset config
  -> 扫描 stage/object/scene/light/URDF registry
  -> 解析并缓存 5 + 92 + 91 + 7 + 12 个模板/配置
  -> canonical handle + alias registry
  -> parse_scene(handle)
  -> SceneDesc
```

解析器具有以下行为：

1. Stage、object 和 URDF visual 路径在内存中解析为绝对路径。
2. Object template JSON 和 URDF XML 只解析一次，多个场景复用缓存。
3. 每个场景使用确定性 ID：stage 为 0，普通对象从 1 开始，articulated object 紧随其后。
4. 相同输入重复解析产生相同 ID、顺序和 manifest digest。
5. 场景实际引用但无法解析的模板或 visual asset 作为硬错误。
6. Dataset registry 中未使用的外部缺失资源作为结构化 warning。

### 3. URDF Manifest

新增 `python/donut_render_py/urdf.py`，使用标准库 `xml.etree.ElementTree` 解析：

- link 名称；
- visual GLB 和相对路径；
- visual origin xyz/rpy；
- mesh scale；
- joint 类型、parent/child；
- joint origin、axis、lower/upper limit。

当前 12 个 URDF 共包含 61 个 link、49 个 joint 和 48 个 visual，所有 48 个本地 visual GLB 均存在。

### 4. 检查工具与报告

新增：

```text
tools/inspect_replicacad_dataset.py
output/replicacad_manifest_report.json
```

工具逐场景输出进度，避免长时间运行时无法判断是否卡死；报告使用原子写入，并将仓库内路径转换为相对路径，方便两台机器复现。

运行命令：

```powershell
cd E:\cplus\Flora
E:\python\python.exe tools\inspect_replicacad_dataset.py
```

---

## 三、验收结果

### 1. 全量覆盖率

```text
[01/91] parsed apt_0
...
[91/91] parsed v3_sc3_staging_20
Summary: scenes=91/91, objects=2293, articulated=540, warnings=22
PASS: ReplicaCAD manifest acceptance checks passed
```

报告确定性摘要：

```text
d03d00dc2f16d8eaaac47dddad5653a23e7fc06dedd3b3b44d94a590b0c107e3
```

摘要覆盖场景路径、灯光与 navmesh 引用、实例位姿与属性、visual/collision 资源、URDF visual 和 joint；路径统一转换为相对数据集根目录的 POSIX 形式，避免两台机器的盘符差异影响复现。外部同名 `scene_instance.json` 也不会覆盖 dataset registry 中规范场景的缓存。

### 2. apt_0 重点验证

| 项目 | 结果 |
|---|---:|
| Stage | `frl_apartment_stage` |
| 普通家具 | 113 |
| Articulated object | 6 |
| 普通对象 ID | 1–113 |
| Articulated ID | 114–119 |

6 个 articulated template 为：

```text
fridge
kitchen_counter
kitchenCupboard_01
chestOfDrawers_01
cabinet
door2
```

这证明之前使用的空公寓 stage 已具备装配完整家具与关节物体所需的数据描述。

### 3. Warning 分类

22 条 warning 均来自未被当前 91 场景使用的 registry 资源：

| Warning | 数量 | 说明 |
|---|---:|---|
| `missing_registry_path` | 1 | 外部 `hab_fetch_1.0` 未安装 |
| `missing_navmesh_asset` | 21 | config 注册了本地未提供的 `v3_sc4` navmesh |

关键验收指标：

```text
missing_required_visual_assets = 0
scene_warnings = 0
```

因此这些 warning 不阻塞 A2 的完整视觉场景装配。

### 4. 单元测试

新增：

```text
tests/test_replicacad_manifest.py
tests/test_replicacad_transforms.py
```

运行命令：

```powershell
E:\python\python.exe -m unittest discover -s tests -p "test_replicacad_*.py" -v
```

结果：

```text
Ran 13 tests
OK
```

覆盖内容包括 91 场景统计、资产存在性、稳定 ID、scene alias/cache、外部同名场景缓存隔离、未知场景拒绝、warning 分类、完整语义 manifest digest、`apt_0` 实例、wxyz quaternion、row-major matrix 和非法零四元数。

---

## 四、本周收益

本周收益不是 FPS，而是把 Flora 的 ReplicaCAD 支持从“只能手写一个 GLB 路径”推进到“完整数据集可确定性解析”：

1. 后续不需要为每个场景手工罗列家具路径和姿态。
2. 完整场景、物理模块和渲染模块可以共享同一份 `SceneDesc`。
3. 普通对象和 URDF 已获得稳定 instance ID，为 Instance/Semantic 输出建立基础。
4. 路径和变换错误在进入 GPU 渲染前即可由快速单元测试发现。
5. 两台机器可使用相同命令和相对路径报告复现实验。

---

## 五、限制与下一周计划

### 当前限制

- A1 只产生数据描述，尚未把 113 个家具加入 Donut SceneGraph。
- URDF 已解析 link、visual 和 joint manifest，但尚未计算实时 joint state。
- 未接入物理引擎，不会执行碰撞、重力或刚体积分。
- 当前图像仍由原 stage GLB 示例产生，完整 `apt_0` 可视化属于 A2。

### Week A2 计划

下一阶段实现 `SceneDesc -> Donut SceneGraph` 和 `AssetCache`：

1. 复用同一 GLB 资产，避免 113 个实例重复加载 mesh/texture。
2. 先装配 stage 和普通家具，生成完整 `apt_0` 对比图。
3. 为每个实例建立稳定 renderer handle。
4. 加载 URDF link visual，并按静态默认关节姿态装配。
5. 保留当前 `frl_apartment_stage.glb` 空场景性能基线，新增完整 `apt_0` 真实负载基线。

Week A2 的核心汇报目标是：同一个相机从空房壳画面升级为包含 113 个家具和 6 个 articulated object 的完整公寓画面。
