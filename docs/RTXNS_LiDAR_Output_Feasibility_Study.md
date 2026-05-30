# RTXNS LiDAR / 点云观测接入可行性调研与最小实验

## 1. 文档目的

本文档记录一次面向 `D:\xmd\RTXNS` 的 LiDAR / 点云观测接入调研与 sidecar smoke test。当前目标不是实现正式 LiDAR 渲染器，也不修改 C++ 渲染核心，而是在现有 Genesis 示例资产和 mesh 几何基础上，用轻量 Python 工具验证：

- 是否可以从当前场景几何生成 LiDAR-like 点云；
- 是否可以得到 range image / 距离图；
- 是否可以得到 BEV 可视化；
- 这条路线是否适合作为下周或后续阶段的数据出口实验。

## 2. LiDAR 在仿真器中的常见输出形式

### 2.1 Point cloud

点云通常是 LiDAR 最直接的输出形式，每个点至少包含三维坐标：

- `x, y, z`
- 可选 `intensity`
- 可选 `ring / channel`
- 可选 `timestamp`
- 可选 `semantic label`

在机器人、自动驾驶和仿真任务中，点云适合直接作为观测输入，也适合进一步投影成 BEV、range image 或 occupancy 表示。

### 2.2 Range image / 距离图

Range image 把 LiDAR 的扫描结构保留下来，通常可以理解为：

- 横轴：水平扫描角；
- 纵轴：激光线束 channel；
- 像素值：当前 ray 的 hit distance；
- 无命中区域：记为 0、inf、max range 或 mask。

它比无序点云更接近传感器原始扫描结构，后续也更适合做学习型感知输入。

### 2.3 BEV 可视化

BEV（bird's-eye view）通常把点云投影到地面平面，用于快速观察：

- 障碍物分布；
- 传感器位置；
- 有效扫描范围；
- 点云密度和遮挡情况。

本次实验的 BEV 只是调研可视化，不表示正式 occupancy grid 或语义地图。

### 2.4 可选扩展输出

成熟仿真器中的 LiDAR 还可能输出：

- `intensity`：与材质、入射角、距离衰减或传感器模型相关；
- `semantic label`：每个 hit point 对应的语义类别；
- `instance id`：每个 hit point 对应的对象实例；
- `timestamp`：用于运动畸变、rolling scan 或多传感器同步；
- `noise / dropout`：模拟真实传感器噪声和漏检。

RTXNS 当前没有这些正式输出，本次实验也不伪造这些字段。

## 3. 成熟方案调研

### 3.1 CARLA LiDAR

CARLA 提供面向自动驾驶仿真的 ray-casting LiDAR 传感器，常见能力包括：

- 输出点云；
- 配置 channels、range、rotation frequency、points per second；
- 支持语义 LiDAR；
- 与自动驾驶场景、车辆坐标系和传感器时间同步结合紧密。

CARLA 的优势是传感器系统完整，但它是大型仿真器，不适合作为 RTXNS 当前轻量 sidecar 实验依赖。

### 3.2 Isaac Sim RTX LiDAR

Isaac Sim 提供 RTX LiDAR / RTX sensor 路线，通常能利用 GPU / RTX 能力模拟传感器扫描，适合机器人和高保真传感器仿真。

它的优势是传感器模型更成熟、GPU 路径更强，但依赖体量很大，也会把实验重心从 RTXNS 当前渲染器转移到另一个完整仿真平台。本阶段不建议把 Isaac Sim 接入为 RTXNS 的运行依赖。

### 3.3 Open3D RaycastingScene

Open3D 的 `RaycastingScene` 可以对 triangle mesh 构建 ray intersection scene，适合做轻量 sidecar 实验：

- 输入 mesh；
- 构造 ray origins 和 directions；
- 输出每条 ray 的 `t_hit`；
- 由 `origin + direction * t_hit` 生成点云。

这条路线的优点是轻量、独立、不需要改 RTXNS C++ 渲染核心，适合在现有 mesh / GLB / OBJ 上快速验证 LiDAR-like 输出。

## 4. RTXNS 当前状态

根据当前仓库和阶段收口文档，RTXNS 已经具备：

- Python 可调用的离屏 RGBA 渲染链路；
- Genesis 风格的 Python scene / mesh 接入方式；
- 基于 Genesis 示例 OBJ 资产的展示链路；
- 当前渲染主路径会内部生成 GLB 并交给 Donut / Vulkan 后端渲染。

但当前尚未发现：

- 正式 LiDAR 输出；
- depth 输出导出到 Python；
- normal 输出导出到 Python；
- material buffer 输出导出到 Python；
- ray-based / GPU-based sensor API。

当前代码中存在 depth target 或渲染内部深度概念，但没有发现稳定的 Python readback API。因此本次优先选择“外部 mesh raycasting smoke test”，而不是直接改渲染器内核。

## 5. 三条接入路线比较

### 5.1 路线 A：Open3D sidecar mesh raycasting

做法：

- 从当前场景或 Genesis 资产拿到 `.obj / .glb / .ply`；
- 用 Open3D `RaycastingScene` 构建 mesh raycast scene；
- 生成 LiDAR-like rays；
- 输出点云、range image、BEV 和统计 JSON。

优点：

- 不改 C++；
- 能快速验证数据形态；
- 可复用当前 Genesis mesh 资产；
- 适合周报和最小实验。

局限：

- 不等于 RTXNS 正式 LiDAR；
- 暂不含 intensity、noise、motion distortion、semantic label；
- 如果与渲染器内部 scene 不共享资源，会存在几何同步问题。

### 5.2 路线 B：补 depth 输出后反投影点云

做法：

- 在 RTXNS 渲染链路中补 depth readback；
- 根据相机内参和深度图反投影成 camera point cloud；
- 可进一步生成局部 range-like 或 depth-derived 点云。

优点：

- 更贴近当前 renderer 输出；
- 对后续神经渲染数据出口也有价值；
- depth / normal / material buffer 都可以接到同一类 readback 框架中。

局限：

- 单相机 depth 不是 360 度 LiDAR；
- 与真实扫描式 LiDAR 的垂直线束、遮挡和时间结构不同；
- 需要修改或扩展渲染数据出口。

### 5.3 路线 C：RTXNS 内部实现 ray-based / GPU-based LiDAR sensor

做法：

- 在 RTXNS 内部增加 sensor abstraction；
- 用 CPU raycast、GPU ray query、ray tracing pipeline 或 compute acceleration 输出 LiDAR；
- 与 scene、material、instance、time step 统一管理。

优点：

- 最接近正式仿真器传感器；
- 可扩展 intensity、semantic label、运动畸变；
- 长期可以成为论文或系统贡献点。

局限：

- 工程量明显更大；
- 需要设计资源生命周期和传感器 API；
- 不适合作为当前周交付。

## 6. 推荐结论

当前推荐路线是：

1. 下周优先继续路线 A。
2. 路线 B 作为后续数据出口扩展，尤其是 depth / normal / material buffer readback。
3. 路线 C 作为长期研究方向，不作为当前周交付。

这条判断的核心原因是：RTXNS 当前还没有正式传感器输出，先用 Open3D sidecar 把点云、range image 和 BEV 的数据闭环跑通，比直接进入渲染核心更稳。

## 7. 小实验实现

新增脚本：

```text
tools/lidar/open3d_lidar_smoke.py
```

脚本能力：

- 支持 `.glb / .obj / .ply` mesh 输入；
- 默认优先使用相邻 Genesis 仓库中的 `duck.obj`；
- 如果 Genesis 资产不存在，则回退到仓库内 `external/donut/thirdparty/cgltf/fuzz/data/Box.glb`；
- 优先使用 Open3D `RaycastingScene`；
- 如果 Open3D 不可用，可回退到内置 numpy Moller-Trumbore ray-triangle fallback；
- 输出点云、range image、BEV 和统计 JSON。
- 默认使用 `object_orbit` 模式围绕孤立 mesh 做多站位 raycast，便于 smoke test 中得到可识别的物体点云；如果要模拟单个 360 度站位，可显式传入 `--scan-mode single`。

本次运行命令：

```text
C:\ProgramData\anaconda3\envs\isaacsim\python.exe tools\lidar\open3d_lidar_smoke.py
```

说明：当前默认 `python` 是 Python 3.13.9，Open3D 没有匹配 wheel。本次使用已有 Python 3.11 环境运行 Open3D 0.19.0；这不表示 RTXNS 依赖 Isaac Sim，只是使用了该环境里的 Python 解释器完成 smoke test。

## 8. 小实验结果

使用场景：

```text
D:\xmd\Genesis\genesis\assets\meshes\duck\duck.obj
```

运行结果：

- scan mode：`object_orbit`
- orbit poses：12
- ray 数量：393216
- channels：32
- horizontal_steps：1024
- 有效命中数：86538
- 有效命中比例：0.2200775146484375
- min range：337.7461853027344
- max observed range：607.3282470703125
- mean range：423.4248046875
- max range：829.9843750000001
- raycast backend：`open3d.RaycastingScene`
- mesh loader：`open3d.io`
- runtime：约 1.0 s

输出文件：

```text
D:\xmd\RTXNS\outputs\lidar_smoke\lidar_points.ply
D:\xmd\RTXNS\outputs\lidar_smoke\range_image.png
D:\xmd\RTXNS\outputs\lidar_smoke\bev.png
D:\xmd\RTXNS\outputs\lidar_smoke\bev_top.png
D:\xmd\RTXNS\outputs\lidar_smoke\bev_front.png
D:\xmd\RTXNS\outputs\lidar_smoke\bev_side.png
D:\xmd\RTXNS\outputs\lidar_smoke\bev_iso.png
D:\xmd\RTXNS\outputs\lidar_smoke\lidar_stats.json
```

结果说明：

- `lidar_points.ply` 是 Open3D raycast 命中的 LiDAR-like 点云，当前写出为带 RGB 属性的 ASCII PLY；
- `range_image.png` 是 channels x (horizontal_steps * orbit_poses) 的距离图，黑色表示无命中；
- `bev.png` 是保留扫描半径圈的 XZ 俯视可视化；
- `bev_top.png` 是按点云包围盒自适应缩放的 XZ 俯视轮廓，更适合放进汇报；
- `bev_front.png` 是 XY 前视投影，可观察垂直线束对物体外形的采样；
- `bev_side.png` 是 ZY 侧视投影，可观察鸭子头部和身体的侧向轮廓；
- `bev_iso.png` 是简单等轴测投影，是当前最适合直观看出鸭子形状的汇报图；
- 早期单站 360 度测试只有 167 个 hit，PLY 看起来像几条横向切片；当前默认改成多站位 object scan，是为了让孤立 mesh 的 smoke test 更容易验证几何是否正确。

当前局限：

- 不是 RTXNS 正式 LiDAR；
- 暂不含 intensity；
- 暂不含 noise / dropout；
- 暂不含 motion distortion；
- 暂不含 semantic label；
- 暂未与 RTXNS 内部 scene resource 共享生命周期；
- 当前 range 单位沿用 mesh 原始单位，没有做物理米制标定。

## 9. 是否值得继续

路线 A 值得继续。它已经证明：在不改 RTXNS C++ 核心的前提下，可以从当前 Genesis mesh 资产生成点云、range image 和 BEV，可作为 LiDAR 数据出口调研的最小闭环。

但它应该继续被称为：

```text
Open3D sidecar LiDAR-like smoke test
```

而不是：

```text
RTXNS native LiDAR sensor
```

## 10. 下一步建议

建议下一步按下面顺序推进：

1. 把 sidecar 输入从单个 OBJ 扩展到当前 demo 生成的临时 GLB 或 manifest 中记录的 scene asset。
2. 给脚本增加 sensor pose presets，例如 `camera_like / top_down / ego_vehicle_like`。
3. 输出更完整的 range image mask，区分 no-hit 和 clipped-by-max-range。
4. 如果后续补 depth readback，再做路线 B 的 depth-to-point-cloud 对照实验。
5. 长期再评估是否需要在 RTXNS 内部实现真正 ray-based / GPU-based LiDAR sensor。
