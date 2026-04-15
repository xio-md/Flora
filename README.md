# 仿真器渲染后端（Donut / Vulkan + Python）

本仓库服务于实验室在搭的 **Agent + 具身智能仿真平台**：覆盖任务生成、RAG、Skill、视觉反馈与自动化评测等环节。这里维护的是其中的 **GPU 离屏渲染后端** 以及 **供 Python 直接调用的原生扩展接口**，方便把高质量画面接进仿真闭环（例如与 Python + Taichi 驱动的环境进程并排跑）。

实现上我们借用了 NVIDIA 开源 **[RTXNS](https://github.com/NVIDIA-RTX/RTXNS)** 的工程布局与 **Donut** 框架，但仓库目标 **不是** 复刻 RTXNS 的神经着色教程或发行说明；上游那套交互式 C++ 神经样本已从树里拿掉，避免和本课题混淆。若你需要完整的 RTXNS 样本与训练链路，请直接对照上游仓库。

中长期目标（模块边界、与 `LuisaRenderPy` 的对齐节奏、接入 Genesis 类仿真器时的职责划分）写在 **[docs/DonutPython_12Week_Plan.md](docs/DonutPython_12Week_Plan.md)**，实现细节以代码与 `docs/` 下设计稿为准。

## 本仓库提供什么

- **DonutRenderPyNative**：无头 Vulkan 路径的 pybind11 扩展，供 `python/donut_render_py`、`python/rtxns_genesis_style` 等包调用。
- **Python 示例**：`samples/DonutRenderPyDemo`、`samples/GenesisStylePy`、`samples/HeadlessPbrPy` 以及共享的 `samples/python_demo_common.py`。
- **`tools/donut_render/`**：导入检查、场景 API、增量帧与性能类小脚本，用于回归和本地对拍。

产物默认落在 `bin/windows-x64` 或 `bin/linux-x64`（与 CMake 里 `RTXNS_BINARY_DIR` 一致）。

## 环境与依赖

- **CMake** 3.10 以上（建议 3.24+）
- **带 C++ 桌面组件的 VS2022**（Windows）或 **Ninja + 对应编译器**（Linux）
- **Python**：与最终运行脚本相同的解释器，且已安装开发头文件/库（CMake 会 `find_package(Python ... Development.Module)`）
- **NumPy**（运行 Python 示例与包）
- **Vulkan**：显卡需支持协作向量相关扩展（具体以 Donut/Vulkan 要求为准），并安装匹配的 [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)

默认配置 **不** 打开上游 RTXNS 的神经着色库（`RTXNS_BUILD_NEURAL_COMPONENTS=OFF`），因此一般 **不需要** 配置 `SHADERMAKE_SLANG_PATH`。只有当你显式打开该开关、自行接 Slang 神经渲染时，才需要按上游方式准备 Slang/DXC 工具链。

## 获取代码

必须拉全子模块，否则 Donut 依赖不完整：

```bash
git clone --recursive <仓库地址>
cd <仓库目录>
```

若已克隆：

```bash
git submodule update --init --recursive
```

## 构建（推荐：默认即编 Python 扩展）

在仓库根目录：

```powershell
cmake -S . -B build
cmake --build build --config Release --target DonutRenderPyNative
```

编出例如 `bin/windows-x64/DonutRenderPyNative.pyd`（Linux 为 `.so`）。若仍要兼容旧模块名，配置时加上 `-DRTXNS_BUILD_LEGACY_RTX_RENDER_PY=ON`，会额外生成 `RtxRenderPy`。

可选：单独建目录、关掉其他开关（与旧文档里的「最小 Donut 配置」等价）：

```powershell
cmake -S . -B build\donut-py `
  -DRTXNS_BUILD_DONUT_RENDER_PYTHON=ON `
  -DRTXNS_BUILD_NEURAL_COMPONENTS=OFF
cmake --build build\donut-py --config Release --target DonutRenderPyNative
```

## 运行 Python 示例

把仓库里的 `python` 目录加入 `PYTHONPATH`，并用 `--module-dir` 指向编好的原生模块目录（或在你的集成代码里传入等价的 `module_dir` / `RuntimeOptions`）。

```powershell
cd <仓库根目录>
$env:PYTHONPATH = "$PWD\python"
python samples\DonutRenderPyDemo\donut_render_demo_v0_1.py --module-dir "$PWD\bin\windows-x64"
```

多帧与清单文件示例：

```powershell
python samples\DonutRenderPyDemo\donut_render_demo_v0_5.py --module-dir "$PWD\bin\windows-x64" --output-dir "$PWD\.temp\demo_out"
```

Linux 下把路径改成 `bin/linux-x64` 并使用 `/` 即可。

## 目录结构（节选）

| 路径 | 说明 |
|------|------|
| `python/` | 对外 Python 包（含 `donut_render_py`、`rtxns_genesis_style`） |
| `src/PythonBindings/` | 原生模块源码（headless PBR、pybind 入口） |
| `samples/` | Python 演示与共用脚本（已无上游 C++ 神经样本） |
| `external/donut` | Donut 子模块 |
| `docs/` | 设计说明与排期；英文的 `QuickStart.md` 仍保留作工具链参考，开头注明了本 fork 的差异 |

## 上游与许可

Slang、Donut、Vulkan 等上游项目请各自遵循其许可与引用要求。本树内仍可见 RTXNS 版权头的文件，代表其来源；使用与分发请以 **[LICENSE.md](LICENSE.MD)** 及上游条款为准。

历史 BibTeX 条目（若你发表工作仍需引用 NVIDIA RTXNS 原始 SDK）可参考上游 README；本 README 不再展开产品级宣传文案。
