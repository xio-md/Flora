# Flora（RTXNS）引擎周报：ReplicaCAD 完整静态场景装配与并行渲染稳定性

> 汇报日期：2026-07-18（Iteration A / Week A2）
> 基线提交：`082665f`
> 硬件：NVIDIA GeForce RTX 3070 Laptop GPU（8 GB），驱动 610.74
> 数据集：`E:\cplus\Flora\ReplicaCAD`，重点场景 `apt_0`
> 本周范围：`SceneDesc -> Donut SceneGraph`、普通家具装配、资产去重、坐标验证、91 场景 smoke、完整场景并行吞吐与长稳态修复
> 不在本周范围：URDF articulated object、物理求解、多环境 SceneBatch、Depth/Segmentation GPU Tensor

## 概述

本周完成 ReplicaCAD Iteration A 的第二阶段。Flora 已从“只加载一个 stage GLB”推进到“解析并渲染完整静态 `scene_instance.json`”：`apt_0` 的 stage 和 113 个普通家具实例全部进入 Donut SceneGraph，81 个唯一 GLB 在模型表中去重，最终生成 114 个顶层渲染实例。Donut 原生统计为 136 个 mesh instance、103 个唯一 mesh、90 个唯一材质。

完整数据集验证从单一场景扩展到 91/91 个 scene instance。同一 renderer、同一 scene 对象连续加载并渲染全部场景，覆盖 2,293 个普通对象；每个场景均完成原生 world transform 校验，没有 native crash。全量最大变换误差为 `8.41e-8`，明显低于 `1e-5` 验收线。

在 A2 压测中还发现并修复了两个此前短基准没有暴露的并行渲染生命周期问题：异步路径未回收 NVRHI in-flight command buffer，以及 RT shadow 每相机每批次重复创建 binding set。修复后，“stage 连续 10,000 batch -> 热重载完整场景 -> 再连续 10,000 batch”通过；RT 阴影 binding 复用后，顺序平衡正式实验中的完整场景 N=8 吞吐为 6,325 cam-FPS。

| 本周指标 | A1 / 周初 | A2 / 周末 | 结果 |
|---|---:|---:|---|
| 可编译完整静态场景 | 0 | 91 | 完成 |
| `apt_0` 普通家具可见 | 0 / 113 | 113 / 113 | 完成 |
| `apt_0` 顶层渲染实例 | 1 个 stage | 114 | 完成 |
| `apt_0` 唯一 GLB | 手工单路径 | 81 | 场景内去重 |
| 全数据集普通对象 | 仅 manifest | 2,293 个原生装配验证 | 完成 |
| 91 场景原生 smoke | 0 / 91 | 91 / 91 | 完成 |
| 全量变换最大误差 | 未做原生对照 | `8.41e-8` | 优于 `1e-5` |
| N=8 完整场景吞吐 | 无完整场景数据 | 6,325 cam-FPS | 128x96、RT 8 samples |
| 长稳态异步提交 | 约 9,000 batch 后访问冲突 | 2 x 10,000 batch + 热重载通过 | P0 修复 |

---

## 一、问题与目标

### 1. A1 之后仍缺少什么

A1 已经能够把 ReplicaCAD 的 dataset config、stage/object config 和 91 个 scene instance 解析成确定性的 `SceneDesc`，但渲染端仍只接受单个 GLB。家具实例虽然存在于 Python manifest 中，却没有进入 Donut SceneGraph，因此不能回答以下问题：

- 同一模板出现多次时，是否只上传一份 Mesh/Material/Texture。
- ReplicaCAD 的 wxyz quaternion、scale、COM 和 `translation_origin` 是否与 Donut 最终 world transform 一致。
- 完整场景复杂度会让多相机吞吐下降多少。
- renderer 能否连续加载 91 个复杂场景，而不是只通过 JSON parser 测试。

### 2. A2 验收目标

1. `apt_0` 的 stage + 113 个普通对象全部可见。
2. 相同 GLB 只在 Donut `models` 表中出现一次，实例通过 scene graph 引用共享模型。
3. 至少抽查 20 个原生 world transform，误差小于 `1e-5`。
4. 91 个场景逐一完成编译、原生加载、渲染和读回。
5. 给出完整场景的加载、显存、单相机帧时和 N=1/2/4/8 并行吞吐。

---

## 二、实现内容

### 1. `SceneDesc -> Donut SceneGraph` 编译器

新增 [donut_scene_compiler.py](../python/donut_render_py/donut_scene_compiler.py)，将 A1 的后端无关描述编译为 Donut 原生 scene JSON：

```text
SceneDesc
  stage + ordinary objects
        |
        +-- 按规范化绝对资产路径去重
        |
        +-- Donut models：每个唯一 GLB 一项
        |
        +-- Donut graph：每个实例一个稳定 node
        |
        +-- *.manifest.json：instance/semantic/motion/transform 元数据
```

关键入口：

| 文件与位置 | 作用 |
|---|---|
| `python/donut_render_py/donut_scene_compiler.py:46` | `compose_instance_asset_matrix()`，组合 pose、scale、COM |
| `python/donut_render_py/donut_scene_compiler.py:92` | `CompiledDonutScene`，保存 Donut 文档和稳定实例表 |
| `python/donut_render_py/donut_scene_compiler.py:201` | `compile_donut_scene()`，完成模型去重和 graph 编译 |
| `python/donut_render_py/__init__.py` | 对外导出编译 API |
| `tools/render_replicacad_complete_scene.py` | 单场景编译、原生校验、渲染和指标输出 |

输出文件采用仓库相对路径，写入过程为临时文件 + 原子替换。相同输入重复编译得到字节一致的 JSON 和相同 digest，便于在两台开发机器间复现。

### 2. 资产去重与实例化

`apt_0` 有 113 个普通对象，但只引用 80 个唯一 object GLB；加上 stage 后，共 81 个唯一模型。编译器只在 `models` 中声明 81 次资源，每个对象节点通过 `model` 字段引用对应模型。Donut attach/clone 路径复制节点层级，但共享底层 `MeshInfo`、Geometry 和 Material 资源。

| `apt_0` 统计 | 数量 |
|---|---:|
| 顶层 stage | 1 |
| 普通对象实例 | 113 |
| 顶层渲染实例 | 114 |
| 唯一 GLB | 81 |
| 原生 mesh instances | 136 |
| 原生唯一 meshes / geometries | 103 / 103 |
| 原生唯一 materials | 90 |
| 唯一 vertices / indices | 313,457 / 392,739 |

顶层实例数和 mesh instance 数不同是正常的：一个 GLB 可以包含多个 mesh node。验收以“114 个顶层实例全部存在”和“模型表按 GLB 去重”为准。

### 3. 坐标、scale 与 COM

ReplicaCAD scene instance 的 quaternion 为 `(w, x, y, z)`，Donut scene JSON 需要 `(x, y, z, w)`。编译器在输出边界完成一次明确转换，不在运行时重复猜测顺序。

当 `translation_origin` 为 `COM` 或 `UNKNOWN` 时，输入平移表示对象质心位置。资产根节点平移按下式修正：

```text
T_asset = T_pose - R * (S * COM)
M_asset = Translate(T_asset) * Rotate(R) * Scale(S)
```

当 origin 为 `ASSET_LOCAL` 时，不做 COM 抵消。A2 同时修正了 A1 parser 对缺省 origin 的处理：实例缺省值先继承 scene/dataset config，仍未声明时保留为 `UNKNOWN`，不再错误默认为 `ASSET_LOCAL`。因此 A1 manifest digest 更新为 `d03d00dc2f16d8eaaac47dddad5653a23e7fc06dedd3b3b44d94a590b0c107e3`。

对应测试覆盖非零 COM、非单位 scale 和旋转组合。`apt_0` 随机抽取 20 个实例与原生节点矩阵对照，最大误差 `2.24e-7`；91 场景每场抽样后的全量最大误差 `8.41e-8`。

### 4. 原生诊断 API

为避免只看截图判断正确性，新增两个只读原生接口：

```python
matrix = scene.get_node_world_transform("instance_000042")
stats = scene.get_scene_stats()
```

参考实现：

| 文件与位置 | 作用 |
|---|---|
| `src/PythonBindings/headless_pbr.h:99` | world transform 和 `SceneStats` 声明 |
| `src/PythonBindings/headless_pbr.cpp:563` | 从 SceneGraph 读取节点最终矩阵 |
| `src/PythonBindings/headless_pbr.cpp:587` | 统计 mesh/material/geometry/vertex/index |
| `src/PythonBindings/py_bindings_common.h` | Python 只读绑定 |

这些 API 仅用于 build/smoke/诊断，不改变现有渲染接口。

---

## 三、稳定性问题与修复

### 1. 连续 `load_scene()` 的旧资源引用

初版全场景 smoke 在第 14 个场景附近出现 `VK_ERROR_DEVICE_LOST`。原因是 ForwardShadingPass binding cache、RT shadow AS 和 scene metadata 仍持有前一场景资源。新场景加载后，旧 binding 与新 buffer/TLAS 混用。

修复位于 `src/PythonBindings/headless_pbr.cpp:289` 的 `load_scene()`：

1. `waitForIdle()` 后先执行 NVRHI garbage collection。
2. 清空 ForwardShadingPass binding cache。
3. 释放旧 BLAS/TLAS、shadow metadata 和临时 binding。
4. 再 reset Scene、TextureCache 并加载新场景。

修复后同一 native scene 连续加载 91/91 场景通过。

### 2. 异步 command buffer 无限累积

长稳态测试在约 9,000 个 batch 后以 `0xC0000005` 访问冲突退出。NVRHI Vulkan backend 会把提交过的 `TrackedCommandBuffer` 保存在 in-flight list，只有应用显式调用 `runGarbageCollection()` 才会回收。原异步路径只 wait/reset EventQuery，没有 GC，因此列表和资源引用随 batch 数持续增长。

修复内容：

- `PendingBatch` 持有 `CommandListHandle`，直到 query 完成。
- `read_frame_batch()` 等待 query 后执行 `device->runGarbageCollection()`。
- ring 深度仍保持 4，不引入全局 `waitForIdle()`。

### 3. RT shadow binding set 每帧重建

原 `RayTracedShadowPass` 在每个 camera 的每个 batch 中创建：

- 1 个 shadow binding set；
- 1 个 composite binding set；
- 2 个 blur binding set。

N=8、10,000 batch 会产生约 32 万次 binding/descriptor 分配。新增 Donut `BindingCache` 后，绑定按 TLAS、scene buffer 和 camera target 组合缓存；场景资源切换时统一 `Clear()`。

参考代码：

| 文件与位置 | 作用 |
|---|---|
| `src/RayTracedShadow/RayTracedShadowPass.h:71` | pass 持有 `BindingCache` |
| `src/RayTracedShadow/RayTracedShadowPass.cpp:19` | 初始化缓存 |
| `RayTracedShadowPass.cpp:157` | 场景切换时清缓存 |
| `RayTracedShadowPass.cpp:233/283/325` | shadow/composite/blur 绑定复用 |

最终长稳态命令：

```powershell
E:\python\python.exe tools\bench_replicacad_complete_parallel.py `
  --mode both --order stage_first --cameras 8 --trials 1 `
  --batches 10000 --warmup-batches 0 `
  --output output\replicacad_complete\apt_0_parallel_rt8_stress.json
```

结果：stage 10,000 batch 和完整场景 10,000 batch 均通过，中间包含一次热重载，进程退出码为 0。

---

## 四、实验结果

### 1. 1280x720 单相机普通对象装配场景

配置：`apt_0`，CPU RGBA8 readback；相机和光照与原 Flora/SAPIEN 对比脚本一致：16:9、相机 `(3.5, 2.0, 3.5) -> (0, 1, 0)`、方向光 `(-0.4, -1.0, -0.6)`。RT 模式使用 direct light + ray-query shadow + blur。

| 指标 | Raster | RT shadow |
|---|---:|---:|
| Python 编译 | 567.14 ms | 576.93 ms |
| 原生加载 | 357.36 ms | 337.57 ms |
| 稳态帧时 | 3.240 ms | 5.849 ms |
| 稳态 FPS | 308.6 | 171.0 |
| 加载显存增量 | 558 MB | 558 MB |
| 20 个 transform 最大误差 | `2.24e-7` | `2.24e-7` |

修正光照方向后，RT 结果不再是全局曝光式压暗：79.4% 像素亮度变化不超过 3，19.6% 像素显著变暗超过 30，变化集中在左墙、桌椅、楼梯和植物的投影区域。

参考产物：

- `output/replicacad_complete/apt_0_complete_raster.png`
- `output/replicacad_complete/apt_0_complete_rt.png`
- `output/replicacad_complete/apt_0_raster_metrics.json`
- `output/replicacad_complete/apt_0_rt_metrics.json`

### 2. 128x96 多相机异步吞吐

配置：RT shadow 8 samples、ring depth 4、每个 case 500 batch warmup、1,000 batch 测量。分别执行 stage-first 和 complete-first 两种顺序，每种顺序 3 trials；合并 6 个 trial 后取中位数，抵消 GPU 动态频率、温度和先后顺序偏差。每次进程只进行一次场景切换，避免把高频 RT hot-reload 压测混入吞吐测量。

| Camera 数 N | Stage-only cam-FPS | 完整静态场景 cam-FPS | 完整 / Stage | 完整 batch ms |
|---:|---:|---:|---:|---:|
| 1 | 2,185 | 1,830 | 83.8% | 0.546 |
| 2 | 3,741 | 3,239 | 86.6% | 0.617 |
| 4 | 6,257 | 4,973 | 79.5% | 0.804 |
| 8 | 8,660 | 6,325 | 73.0% | 1.265 |

完整场景从 N=1 的 1,830 cam-FPS 增长到 N=8 的 6,325 cam-FPS，吞吐提升 3.46 倍。相比纯 stage，家具、材质和 136 个 mesh instance 带来约 13% 至 27% 的吞吐损失；N=8 保留 73.0%。

原生加载与设备级显存对照：

| 模式 | load ms | 显存增量 | mesh instances |
|---|---:|---:|---:|
| Stage-only | 62.77 | 82 MB | 20 |
| 完整静态场景 | 308.35 | 477 MB | 136 |

顺序平衡摘要：`output/replicacad_complete/apt_0_parallel_rt8_balanced.json`。原始 trial 分别位于 `apt_0_parallel_rt8_stage_first.json` 和 `apt_0_parallel_rt8_complete_first.json`；`apt_0_parallel_rt8_stress.json` 保留正确光照下的 2 x 10,000 batch 压力测试结果。

### 3. 91 场景全量 smoke

| 指标 | 结果 |
|---|---:|
| 请求 / 通过 | 91 / 91 |
| 普通对象 | 2,293 |
| 本阶段暂未装配 articulated instances | 540 |
| 平均原生加载 | 291.67 ms |
| 最大原生加载 | 399.69 ms |
| 最大 transform 误差 | `8.41e-8` |
| 总耗时 | 32.28 s |

正式结果：`output/replicacad_complete/all_scene_smoke.json`。

---

## 五、测试与复现

### 1. 构建

```powershell
cmake --build E:\cplus\Flora\build-flora --config Release --target DonutRenderPyNative
```

Release 构建通过。现有 C4819 和 `APIENTRY` 重定义 warning 不由本阶段引入。

### 2. 单元与生命周期测试

```powershell
E:\python\python.exe -m unittest discover -s tests -p "test_replicacad_*.py" -v
E:\python\python.exe tools\donut_render\api_lifecycle_smoke.py --quiet
```

- ReplicaCAD 单元测试：19/19 通过，包含 6 个 A2 assembly test。
- 原生 API lifecycle smoke：通过。
- Python `compileall`：通过。

### 3. 完整场景复现

```powershell
# 1280x720 raster（与历史 Flora/SAPIEN 对比图相同宽高比）
E:\python\python.exe tools\render_replicacad_complete_scene.py `
  --width 1280 --height 720 --warmup 5 --frames 10

# 1280x720 RT shadow
E:\python\python.exe tools\render_replicacad_complete_scene.py `
  --width 1280 --height 720 --warmup 5 --frames 10 --rt-shadows

# 91 个场景连续加载、渲染和变换校验
E:\python\python.exe tools\smoke_replicacad_complete_scenes.py

# 正式并行吞吐
E:\python\python.exe tools\bench_replicacad_complete_parallel.py `
  --mode both --order stage_first --trials 3 --batches 1000 --warmup-batches 500
E:\python\python.exe tools\bench_replicacad_complete_parallel.py `
  --mode both --order complete_first --trials 3 --batches 1000 --warmup-batches 500
```

---

## 六、当前边界

### 1. 本周明确未完成

- `apt_0` 的 6 个 URDF articulated object 尚未进入渲染场景。
- 这 6 个 URDF 中包含 `kitchen_counter`、`kitchenCupboard_01`、`fridge` 和 `chestOfDrawers_01` 等承载家具。其上的餐具、手包、鞋盒等普通对象已经按正确 world pose 加载，但因为承载家具缺失，当前 A2 图中会视觉悬空；因此本阶段图应称为“普通对象装配图”，不是视觉完整场景。
- 全数据集 540 个 articulated instance 只在 A1 manifest 中，未创建 link visual hierarchy。
- `motion_type`、instance ID 和 semantic ID 已进入 sidecar manifest，但尚未进入 GPU sensor 输出。
- 当前去重作用于单个编译场景；跨 scene、跨 environment 的持久 GPU AssetCache 仍需在 SceneBatch 阶段实现。
- Python 场景编译约 0.6 s，尚未做磁盘编译缓存；当前优先级低于 URDF 和动态位姿。
- 动态更新仍以单节点名字查询为主，不能作为大规模 PoseBatch 最终接口。
- RT 模式下“长时间渲染后连续多次交替 hot-reload”仍应作为独立生命周期测试；正式 benchmark 每个进程只切换一次场景。动态仿真的目标使用方式是 build/load 一次、持续更新 pose，不依赖每帧或高频重载场景。

### 2. A2 完成判断

A2 的核心完成定义是“完整普通家具场景可见、去重、变换正确、91 场景可稳定加载，并给出完整场景并行基线”。上述五项均已通过。因此建议 A2 在代码审查和提交后收尾，不继续在本阶段扩展传感器或物理功能。

---

## 七、下一周：A3 URDF 可视层级与动态位姿

下一步应进入 A3，而不是继续优化单 stage benchmark。建议按以下顺序推进：

1. 在 `python/donut_render_py/urdf.py` 完成 link、joint、visual、mesh scale 和 origin 的确定性解析。
2. 将一个 articulated instance 编译成稳定的 `root/link/visual` Donut 节点层级。
3. 为普通刚体和 URDF link 建立连续 handle 表，build 后不再按名字遍历。
4. 新增 `update_node_transforms_batch(handles, matrices)` CPU reference API。
5. 用 `apt_0` 的 6 个 articulated object 做静态可见性验收。
6. 用脚本驱动门/抽屉两个状态，验证 link pose 更新、TLAS update 和多相机渲染。
7. 扩展到 91 场景 540 个 articulated instance，输出覆盖率和缺失 mesh 报告。

A3 周报建议量化：可见 articulated 数、link/visual 数、动态更新对象数、pose update ms、TLAS update ms、更新前后截图，以及 N=1/4/8 相机下动态场景吞吐。

## 结论

本周不只是“让家具显示出来”。A2 建立了从 ReplicaCAD scene instance 到 Donut 原生 SceneGraph 的确定性编译链，完成 91 场景的原生验证，并用完整 `apt_0` 给出了第一份真实并行吞吐基线。压测进一步补上 NVRHI command buffer GC 和 RT binding cache 两个长稳态缺口，使现有异步 batch 从短跑可用提升到 2 x 10,000 batch 热重载可用。

下一阶段可以在这条稳定静态拓扑上增加 URDF link 和外部 PoseBatch，而不需要再次重写场景加载主链路。
