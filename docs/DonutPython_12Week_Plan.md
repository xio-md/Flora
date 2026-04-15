# Donut Python 后端 12 周交付计划

## 1. 文档目的

本文档用于指导未来 12 周内的开发、周报与阶段性交付，目标不是“单纯做一个可运行 Demo”，而是把当前基于 RTXNS + Donut 的 Python 原型(在D:\xmd\RTXNS)，逐步推进为一个可替代 `D:\xmd\Genesis\D:\xmd\Genesis\ext\LuisaRender` 的 Python 渲染后端。

这个后端的定位是：

- 对齐 `LuisaRenderPy` 的模块组织方式与 Python 调用习惯；
- 主要承担 Genesis 中“离屏高质量渲染\类光追渲染”这一层职责；
- 不替代 `D:\xmd\Genesis\D:\xmd\Genesis\vis\rasterizer.py` 现有光栅化路径；
- 为下一阶段基于 Slang 的神经渲染\神经着色接入预留统一运行时、资源生命周期和 Python API 结构。

因此，这 12 周的目标应理解为：

- 第一个三个月完成 Donut 后端的“LuisaRender 化”；
- 下一个三个月在同一后端基础上接入 Slang 神经渲染能力。

## 2. 当前边界与正确目标

### 2.1 不应把目标定义错

当前工作的重点不是“适配整个 Genesis 可视化系统”，而是替代下面这条链路：

- `D:\xmd\Genesis\D:\xmd\Genesis\ext\LuisaRender\src\apps\CMakeLists.txt`
- `D:\xmd\Genesis\D:\xmd\Genesis\ext\LuisaRender\src\apps\py_interface.cpp`
- `D:\xmd\Genesis\D:\xmd\Genesis\ext\LuisaRender\src\apps\py_scene.h`
- `D:\xmd\Genesis\D:\xmd\Genesis\vis\raytracer.py`

Genesis 里真正对应的两条渲染路径是：

- 光栅化路径：`D:\xmd\Genesis\D:\xmd\Genesis\vis\rasterizer.py`
- 高质量离屏\光追路径：`D:\xmd\Genesis\D:\xmd\Genesis\ext\LuisaRender` + `D:\xmd\Genesis\D:\xmd\Genesis\vis\raytracer.py`

这意味着：

- 未来 Donut Python 后端的主要替代对象是 `LuisaRenderPy`，不是 `rasterizer.py`；
- Genesis 原有光栅化 API 可以继续保留；
- Donut Python 后端需要重点对齐的是 `LuisaRenderPy` 的 Python 模块层级、Scene 更新语义、CMake\pybind11 组织方式；
- “类 Genesis 仿真器调用”仍然重要，但它只是上层适配逻辑，不是本阶段唯一核心。

### 2.2 当前原型已经完成了什么

当前仓库内已经有一个可运行的原型基线：

- Python 扩展模块入口：`D:\xmd\RTXNS\src\PythonBindings\py_interface.cpp`
- Headless Vulkan 离屏 PBR 后端：`D:\xmd\RTXNS\src\PythonBindings\headless_pbr.cpp`
- Python 侧原型适配层：`D:\xmd\RTXNS\python\rtxns_genesis_style\renderer.py`
- 极简 GLB 场景导出器：`D:\xmd\RTXNS\python\rtxns_genesis_style\glb_builder.py`
- Demo：`D:\xmd\RTXNS\samples\GenesisStylePy\genesis_style_example.py`

当前已经验证通过的能力包括：

- headless Vulkan 初始化；
- Python 调用 C++ 模块完成离屏 PBR 渲染；
- 多帧 `update_scene()` \ `render_camera()` 风格调用；
- 刚体、粒子等基本更新路径；
- 生成多帧图像序列。

### 2.3 当前原型还不具备最终交付意义

虽然原型能跑，但它离“替代 LuisaRender 模块”还有明显距离：

- 当前更新路径仍主要依赖 Python 侧重建临时 GLB，再调用 `load_scene()`；
- 还没有形成与 `LuisaRenderPy` 对齐的对象层级和接口兼容策略；
- 还没有形成与 `src\apps\CMakeLists.txt` 相似的正式模块构建组织；
- 资源生命周期、对象句柄、错误语义没有正式冻结；
- 材质、纹理、相机、环境、形状更新能力还不完整；
- 缺少 Linux Vulkan-only 验证、内存稳定性与基准测试；
- 还没有为下一阶段 Slang 神经渲染预留清晰的统一后端接口。

## 3. 12 周后的最终交付定义

12 周结束时，交付物不应只是一个 Demo，而应满足以下条件：

1. 形成一个正式的 Python 模块，构建方式与 `LuisaRenderPy` 类似，采用 `pybind11 + CMake` 输出可导入模块。
2. 提供一个与 `LuisaRenderPy` 相近的顶层 API：
   - `init(...)`
   - `create_scene()`
   - `destroy()`
3. 提供一个与 `PyScene` 语义相近的 Scene API：
   - `init(...)`
   - `update_environment(...)`
   - `update_emission(...)`
   - `update_subsurface(...)`
   - `update_surface(...)`
   - `update_shape(...)`
   - `update_camera(...)`
   - `update_scene(...)`
   - `render_frame(...)`
4. 至少完成 PBR 方向的核心能力：
   - 形状更新；
   - 材质更新；
   - 相机更新；
   - 环境光\方向光；
   - 离屏 RGB\RGBA 输出。
5. 对 Genesis 上层来说，未来可以用接近 `raytracer.py` 的方式切换到底层实现，而不必重写整个可视化系统。
6. 为后续 Slang 神经渲染扩展保留统一上下文、资源管理和 Python 接口风格。

## 4. 对标对象与接口兼容要求

### 4.1 对标 `LuisaRenderPy` 的组织方式

根据 `D:\xmd\Genesis\D:\xmd\Genesis\ext\LuisaRender\src\apps\CMakeLists.txt`，LuisaRender 当前的 Python 模块构建方式是：

- 使用 `pybind11_add_module(LuisaRenderPy py_interface.cpp)`；
- 模块是标准 Python 扩展；
- CMake 负责查找 Python 与 pybind11；
- 模块直接链接到底层渲染库。

Donut 后端在三个月内应尽量形成类似结构：

- `pybind11_add_module(DonutRenderPy ...)` 或等价命名；
- 独立 CMake 目标；
- 明确区分：
  - 核心运行时库；
  - Python 绑定层；
  - Demo\样例。

### 4.2 对标 `LuisaRenderPy` 的 Python 层级

根据 `D:\xmd\Genesis\ext\LuisaRender\src\apps\py_interface.cpp`，LuisaRender 当前 Python API 不只是 `Scene + render_frame`，而是完整对象体系，包括：

- Transform
- Texture
- Light
- Subsurface
- Surface
- Shape
- Film \ Filter
- Camera
- Environment
- Integrator
- Spectrum
- Render
- Scene

因此 Donut 后端不应长期停留在“仅支持 `load_scene()` 的 Python 包装器”状态，而应逐步形成一套 Luisa 风格但 Donut 实现的对象模型。

### 4.3 与 Genesis 上层的真实关系

本阶段的接口兼容优先级应为：

1. 先对齐 `LuisaRenderPy` 的对象层级和 Scene 更新语义；
2. 再对齐 `raytracer.py` 所需的调用顺序；
3. 不接管 `rasterizer.py`；
4. 为未来神经渲染扩展预留接口，而不是一开始直接实现神经训练。

## 5. 开发策略

整个 12 周按三个阶段推进：

- 第 1 个月：完成 LuisaRender 对标分析、模块边界整理、Python API 草案冻结。
- 第 2 个月：完成 Scene 更新链路重构，逐步从“GLB 重建”转向“对象级增量更新”。
- 第 3 个月：完成正式可交付版本，补齐材质\动态更新\跨平台验证\文档。

每周都必须满足两个目标：

- 有可向老师\老板汇报的阶段性成果；
- 有真实降低交付风险的工程推进。

## 6. 周报模板

建议每周都采用同一模板：

### 周报结构

- 本周完成
- 本周验证
- 当前风险
- 下周目标
- 需要支持

### 示例

- 本周完成：完成 LuisaRenderPy 接口梳理，并冻结 DonutRenderPy 的第一版模块边界。
- 本周验证：Windows Vulkan 下 headless PBR 多帧 Demo 可稳定输出。
- 当前风险：Scene 更新仍依赖全量 GLB 重建，性能无法直接满足仿真器高频更新。
- 下周目标：整理对象层级，建立 Surface\Shape\Camera 兼容接口草案。
- 需要支持：Linux 机器或容器用于 Vulkan-only 验证。

## 7. 面向代码 Agent 的执行规则

为了方便后续使用代码 Agent 协同开发，每周任务建议拆成标准格式：

### 任务模板

- 目标
- 范围
- 预期修改文件
- 验收标准
- 演示或证据

### Agent 执行规则

- 每周只聚焦一个主主题，不在同一周混合过多方向。
- 每周至少有一个可 review 的产物：
  - 代码；
  - 文档；
  - Demo；
  - 测试；
  - benchmark。
- 所有任务必须和“替代 LuisaRender 模块”这一主线直接相关。
- 不做与当前阶段无关的大范围 UI、编辑器或完整神经训练系统扩展。

### 每周任务完成判定

一项周任务只有在满足以下条件时才算完成：

1. 代码或文档已落到仓库；
2. 有明确验证步骤；
3. 可在周报里用 3-5 句话讲清楚；
4. 明确降低了一个交付风险。

## 8. 12 周详细安排

## 第一阶段：接口对标与基础设施收口（第 1-4 周）

### 第 1 周：LuisaRender 对标分析与范围冻结

本周目标：

- 明确 Donut 后端要替代的真实对象与边界；
- 冻结“本阶段不替代 rasterizer，只替代 LuisaRender 模块”的范围。

主要任务：

- 梳理 `py_interface.cpp` 暴露出的 Python 对象体系；
- 梳理 `raytracer.py` 真正依赖的 Scene 更新顺序；
- 梳理 `src\apps\CMakeLists.txt` 的 Python 模块构建方式；
- 产出《LuisaRender -> DonutRender 接口映射草案》。

预期产出：

- 一份接口映射文档；
- 一份模块边界文档；
- 一份当前原型与目标后端之间的缺口列表。

可交给代码 Agent 的任务：

- 统计 LuisaRenderPy 暴露类和方法；
- 生成第一版接口映射表；
- 检查当前 Python 原型中缺失的对象层级。

验收标准：

- 能清楚说明未来 Donut 后端的替代对象是谁；
- 能列出必须兼容的 LuisaRenderPy API 子集；
- 能把“本阶段不做什么”写清楚。

周报重点：

- 当前原型只证明技术可行，真正目标是替代 LuisaRender 绑定层。

### 第 2 周：第一版 Python API 冻结

本周目标：

- 冻结 DonutRenderPy 的第一版 Python API 草案。

主要任务：

- 设计与 LuisaRender 接近的对象层：
  - Texture
  - Surface
  - Shape
  - Camera
  - Environment
  - Render \ Scene
- 明确第一版支持项与延后项；
- 明确错误语义、初始化顺序与销毁顺序。

预期产出：

- `DonutRenderPy` API 草案文档；
- 第一版对象层级图；
- 当前原型包装层需要重构的清单。

可交给代码 Agent 的任务：

- 基于当前 `renderer.py` 重构数据结构；
- 起草对象类壳和异常信息；
- 生成 API 文档初稿。

验收标准：

- 有完整 API 草案；
- 能明确说明哪些接口将对齐 LuisaRenderPy；
- 能说明哪些功能在 PBR 阶段先简化。

周报重点：

- 正在冻结接口，避免后续 Slang 阶段接入时大规模返工。

### 第 3 周：CMake 与 Python 模块组织收口

本周目标：

- 让 Donut Python 后端的工程组织更接近 LuisaRender 的模块化结构。

主要任务：

- 调整 CMake，使核心运行时、Python 绑定、Demo 分层清晰；
- 明确模块命名、输出路径、依赖关系；
- 文档化 Windows 构建流程；
- 规划 Linux Vulkan-only 构建要求。

预期产出：

- Donut Python 后端的 CMake 组织说明；
- Windows 构建说明；
- Linux 构建准备清单。

可交给代码 Agent 的任务：

- 清理 `src\CMakeLists.txt` 与 `src\PythonBindings\CMakeLists.txt`；
- 生成构建说明文档初稿；
- 补充构建检查脚本或说明。

验收标准：

- 工程结构清晰；
- Python 模块的构建方式可复现；
- 与 LuisaRender 当前构建方式的对应关系被写清楚。

周报重点：

- 当前工作从“本地能跑”转向“后端工程可维护”。

### 第 4 周：第一个月阶段性 Demo 与评审

本周目标：

- 完成第一个月的阶段性可汇报版本。

主要任务：

- 整理当前 Demo；
- 固化一条代表性多帧渲染流程；
- 形成《第一个月评审文档》，明确下月要解决的关键问题。

预期产出：

- Demo v0.1；
- 一份第一个月阶段评审文档；
- 一份明确的第二阶段任务清单。

可交给代码 Agent 的任务：

- 清理 Demo 代码；
- 统一示例命名与输出格式；
- 生成评审文档初稿。

验收标准：

- 有可运行、可展示的阶段 Demo；
- 有清晰问题列表；
- 下个月的工作重点明确。

第 1 个月交付：

- LuisaRender 对标分析
- DonutRenderPy API 草案 v0.1
- 模块化 CMake 组织说明
- 多帧 PBR Demo v0.1
- 第一个月评审文档

## 第二阶段：Scene 更新链路与 Luisa 风格对象化（第 5-8 周）

### 第 5 周：资源生命周期与对象句柄设计

本周目标：

- 建立 DonutRenderPy 的资源生命周期模型。

主要任务：

- 定义 Context、Scene、Texture、Surface、Shape、Camera 的所有权；
- 明确 Python 持有对象和 C++ 持有对象之间的关系；
- 明确 `destroy\reset` 行为；
- 补充对象误用检测。

预期产出：

- 生命周期文档；
- 第一版句柄\对象关系说明；
- 一批销毁和异常路径检查。

可交给代码 Agent 的任务：

- 清理 `initialize\create_scene\destroy` 路径；
- 给对象层补充无效状态检查；
- 起草生命周期设计文档。

验收标准：

- 生命周期与对象所有权写清楚；
- 明显误用可以干净失败；
- 后续增量更新有可靠对象基础。

周报重点：

- 从 Demo 逻辑切换到后端级生命周期管理。

### 第 6 周：Scene 脏状态分类与更新策略

本周目标：

- 把当前“全量重建”路径拆成可度量的更新类别。

主要任务：

- 定义至少以下脏状态：
  - 相机更新；
  - 变换更新；
  - 几何更新；
  - 材质更新；
  - 环境更新；
- 让包装层知道何时可跳过全量重建；
- 记录每类更新耗时。

预期产出：

- 更新路径文档；
- dirty-state 代码实现；
- 一份小场景 benchmark。

可交给代码 Agent 的任务：

- 在 `renderer.py` 中补齐 dirty-state 分类；
- 加入计时日志；
- 跑小型基准测试并汇总。

验收标准：

- 各类更新在代码中有明确标识；
- camera-only 等路径不再走全量场景重建；
- 有可展示的更新耗时数据。

周报重点：

- 开始把“能运行”推进到“适合仿真循环”。

### 第 7 周：对齐 LuisaRender Scene API 的增量更新设计

本周目标：

- 设计 DonutRenderPy 的对象级增量更新接口。

主要任务：

- 以 `PyScene::update_environment\update_surface\update_shape\update_camera\update_scene\render_frame` 为对标；
- 设计 Donut 后端对应接口；
- 决定哪些部分先保持 Python 层包装，哪些部分应下沉到 C++。

预期产出：

- 一份《增量更新接口设计文档》；
- 一份 `LuisaRenderPy Scene API -> DonutRenderPy Scene API` 对照表。

可交给代码 Agent 的任务：

- 提取 `py_interface.cpp` 中 Scene API；
- 生成对照表；
- 起草 C++\Python 分层方案。

验收标准：

- 新 Scene API 设计文档完整；
- 明确第一批要实现的真增量更新接口；
- 有清晰的实现优先级。

周报重点：

- 进入真正替代 LuisaRenderPy Scene 更新层的设计阶段。

### 第 8 周：第一批真增量更新落地

本周目标：

- 落地第一批真正的对象级更新能力。

主要任务：

- 优先实现：
  - camera-only 真增量更新；
  - rigid transform 更新；
  - 或 light\environment 更新；
- 对比全量重建与增量更新的行为差异；
- 固化第二个月 Demo。

预期产出：

- 第一批真增量更新代码；
- 一份 benchmark 说明；
- Demo v0.5。

可交给代码 Agent 的任务：

- 修改 C++ 绑定层，增加增量更新入口；
- 同步 Python 侧包装逻辑；
- 生成对比测试脚本。

验收标准：

- 至少有一类更新不再依赖全量重建；
- benchmark 可说明收益；
- 第二个月能展示“更新链路正在 LuisaRender 化”。

第 2 个月交付：

- 生命周期模型
- dirty-state 更新系统
- Scene 增量更新接口设计
- 第一批真增量更新实现
- 更新耗时 benchmark

## 第三阶段：可交付后端化与未来神经渲染预留（第 9-12 周）

### 第 9 周：材质与纹理支持补齐

本周目标：

- 把材质系统从“演示级”推进到“后端级”。

主要任务：

- 补齐 base color \ roughness \ metallic \ emissive \ opacity 支持；
- 尝试支持纹理驱动的 PBR 参数；
- 明确当前不支持项与回退策略。

预期产出：

- 材质支持矩阵；
- 至少一个带纹理示例；
- 材质回退说明。

可交给代码 Agent 的任务：

- 扩展 `glb_builder.py` 与材质包装逻辑；
- 增加材质测试；
- 产出材质支持表文档。

验收标准：

- 材质支持范围清晰；
- 有可展示的纹理化示例；
- 未支持项不会静默失败。

周报重点：

- 材质能力开始接近 LuisaRender 上层真正需要的范围。

### 第 10 周：动态几何与长序列稳定性

本周目标：

- 让 deformable \ particles 等动态内容可稳定运行。

主要任务：

- 强化 deformable 更新路径；
- 强化 particles 更新路径；
- 跑更长帧序列验证稳定性；
- 补回归测试。

预期产出：

- 动态几何验证文档；
- 100 帧以上测试记录；
- 若干回归样例。

可交给代码 Agent 的任务：

- 编写长序列测试脚本；
- 编写 deformable\particles regression case；
- 汇总稳定性结果。

验收标准：

- 动态几何能稳定更新；
- 长帧序列无明显错误或崩溃；
- 结果可直接用于周报与月报。

周报重点：

- 后端开始具备仿真循环中动态更新的可靠性。

### 第 11 周：Linux Vulkan-only 验证与交付风险清理

本周目标：

- 关闭跨平台交付风险。

主要任务：

- 验证 Linux Vulkan-only 构建；
- 排查 Windows 假设；
- 记录稳定性与内存行为；
- 收口工程依赖与文档。

预期产出：

- Linux 验证文档；
- 平台差异说明；
- 一份交付前风险清单。

可交给代码 Agent 的任务：

- 整理 Linux 构建说明；
- 扫描明显平台特定逻辑；
- 汇总依赖与运行环境要求。

验收标准：

- Linux Vulkan-only 验证有记录；
- 已知平台风险被明确写出；
- 交付前的主要风险可讲清楚。

周报重点：

- 从“本机原型”推进到“可对外交付的后端”。

### 第 12 周：最终交付收口与下阶段接口预留

本周目标：

- 形成第一个三个月的正式交付。

主要任务：

- 整理 README、QuickStart、API 文档；
- 固化最终 Demo；
- 写《最终交付说明》；
- 补写下一阶段 Slang 神经渲染扩展接口预留说明。

预期产出：

- API v1.0 文档；
- 最终演示脚本；
- 最终交付说明；
- 神经渲染阶段预留说明。

可交给代码 Agent 的任务：

- 补最终文档；
- 清理样例；
- 整理 API 一览表；
- 汇总限制与后续路线。

验收标准：

- Donut Python 后端可作为“LuisaRender 替代方向”的第一阶段成果进行汇报；
- API、Demo、文档、限制项都完整；
- 能自然衔接下一个三个月的 Slang 神经渲染工作。

第 3 个月交付：

- DonutRenderPy API v1.0
- Scene 增量更新后的可用版本
- 材质\纹理\动态更新能力说明
- Linux Vulkan-only 验证记录
- 最终交付说明
- 下阶段神经渲染接口预留说明

## 9. 每月汇报重点

### 第一个月汇报重点

- 完成 LuisaRender 绑定层对标分析；
- 冻结 DonutRenderPy 的对象模型与 Python API 草案；
- 证明当前方向可以替代 `LuisaRenderPy`，而不是只是在做另一个 Demo。

### 第二个月汇报重点

- Scene 更新链路开始从“全量重建”走向“对象级增量更新”；
- 后端正在逼近 LuisaRender 的 Scene API；
- 仿真器高频调用的主要瓶颈被识别并开始解决。

### 第三个月汇报重点

- Donut Python 后端已经形成可交付版本；
- 具备替代 `ext\LuisaRender` 第一阶段职责的基础能力；
- 下一阶段可以直接在此基础上接入 Slang 神经渲染，而不必重做运行时。

## 10. 这 12 周内不应扩展的方向

为了保证三个月交付，本阶段不应主动扩展到以下方向：

- 完整路径追踪器重写；
- 全量替代 Genesis rasterizer；
- 完整 UI\编辑器系统；
- 完整神经训练框架；
- 大规模资产管线改造；
- 与当前主线无关的性能微优化。

这些内容要么属于下一阶段，要么应在主线交付完成后再展开。

## 11. 下一个三个月的预览：Slang 神经渲染阶段

完成本阶段后，下一个三个月可以围绕以下目标展开：

1. 在同一 Python 后端中加入 Slang 推理入口；
2. 增加神经材质\神经着色模式；
3. 增加训练步 `train_step(...)` 与 checkpoint 逻辑；
4. 实现 PBR 与神经渲染的统一 Scene 与资源管理。

这一阶段的前提正是本阶段完成：

- 统一 Python 模块形式；
- 统一 Scene 更新接口；
- 统一资源生命周期；
- 统一 CMake 与运行时结构。

## 12. 每周必须留下的交付痕迹

每周结束时，至少应留下以下 5 类痕迹：

- 一段简短书面总结；
- 一个代码 diff 或文档 diff；
- 一个可验证结果；
- 一个明确的下周目标；
- 一个明确的剩余风险。

如果一周结束时缺少这五项中的多项，就说明这一周虽然忙，但还不适合作为正式周报材料。
