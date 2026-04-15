# RTX Neural Shading（RTXNS）

RTX Neural Shading（也常写作 RTX Neural Shaders）面向希望在 Windows 或 Linux 上把机器学习接入图形管线的开发者。仓库里包含从简单推理到网络训练的多套示例，并配套 Slang 着色语言；在 Windows 上可走 DirectX 预览版 Agility SDK 路径，在 Windows/Linux 上也可走 Vulkan 协作向量（Cooperative Vector）扩展。

若你主要关心神经着色样本与训练流程，默认需要先熟悉神经网络与图形基础。若你只想用 **Donut 无头 Vulkan 后端** 在 Python 里出图、跑小场景，可以只看下面「Donut Python 后端」一节，那一支对神经样本的依赖可以关到最小。

## 环境与依赖

### 通用

- [CMake 3.24.3](https://github.com/Kitware/CMake/releases/download/v3.24.3/cmake-3.24.3-windows-x86_64.msi) 或更高（根 `CMakeLists.txt` 写的是 3.10，实际样本与工具链按文档以 3.24 级为准较稳妥）
- [Slang](https://shader-slang.com/tools/)（版本由构建系统拉取或指定，见下文）
- **Python**：构建原生扩展时需要本机已安装对应版本的 Python 开发头文件与库（`find_package(Python ... Development.Module)`）

### Windows

- Visual Studio 2022（带 C++ 桌面开发）
- 若启用 DX12 协作向量预览：DirectX 预览 Agility SDK、DXC、以及支持 Shader Model 6.9 预览的驱动（详见仓库内英文说明与 [QuickStart](docs/QuickStart.md)）

### Linux

- Ninja 等常用构建工具链

### Vulkan（Windows / Linux）

- 显卡需支持 `VK_NV_cooperative_vector`（文档上常见下限为 RTX 20 系）
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)（例如 1.3.296.0）与较新的 Game Ready / Studio 驱动

带 `*` 的包体多数会在 CMake 配置阶段自动拉取，不必单独再装一遍。

## 获取代码

子模块必须拉全，否则 Donut 与第三方对不上：

```bash
git clone --recursive <你的仓库地址>
cd RTXNS
```

若已经克隆过但子模块是空的：

```bash
git submodule update --init --recursive
```

## 构建 C++ 样本（神经着色主线）

1. 建构建目录并配置（生成器按本机习惯替换）：

```bash
cmake -S . -B build
```

2. 若要在 Windows 上打开 DX12 协作向量预览路径，加上：

```bash
cmake -S . -B build -DENABLE_DX12_COOP_VECTOR_PREVIEW=ON
```

3. 编译 Release（Windows 示例）：

```bash
cmake --build build --config Release
```

4. 可执行文件默认落在 `bin/windows-x64` 或 `bin/linux-x64`（由顶层 `RTXNS_BINARY_DIR` 决定）。样本启动时按各程序说明附加 `-dx12` 或 `-vk`。

更细的步骤、样本列表与驱动说明见 **[docs/QuickStart.md](docs/QuickStart.md)**（英文）。

> 从 v1.0.0 升到 v1.1.0 若遇 CMake 缓存问题，可先删掉 `build` 再重新配置（见原发行说明）。

## 构建 Donut Python 后端（`DonutRenderPyNative`）

这一支给 `python/donut_render_py` 等包加载原生模块，走 **Vulkan + Donut**，不要求你同时打开神经样本开关。

在仓库根目录执行（PowerShell 下路径按你的习惯改）：

```powershell
cmake -S . -B build\donut-py `
  -DRTXNS_BUILD_DONUT_RENDER_PYTHON=ON `
  -DRTXNS_BUILD_NEURAL_COMPONENTS=OFF `
  -DRTXNS_BUILD_SAMPLES=OFF
```

若仍需要旧的 `RtxRenderPy` 模块名做兼容，可再加：

```powershell
  -DRTXNS_BUILD_LEGACY_RTX_RENDER_PY=ON
```

只编模块时通常只需要：

```powershell
cmake --build build\donut-py --config Release --target DonutRenderPyNative
```

编出文件在 **`bin/windows-x64`**（Linux 上为 `bin/linux-x64`），例如：

- `DonutRenderPyNative.pyd`（Windows）或对应 `.so`（Linux）
- 若开了兼容选项，还有 `RtxRenderPy.pyd` / `RtxRenderPy.so`

> 若你同时打开 `RTXNS_BUILD_NEURAL_COMPONENTS` 或 `RTXNS_BUILD_SAMPLES`，CMake 会要求设置 `SHADERMAKE_SLANG_PATH` 等工具链变量，详见配置阶段报错提示。

## 运行 Python 示例

1. 安装好 **NumPy**（与构建时用的 Python 为同一解释器）。
2. 把仓库里的 `python` 目录加到 `PYTHONPATH`，这样 `donut_render_py`、`rtxns_genesis_style` 等包能被找到。
3. 运行时让程序能找到原生模块：默认会到仓库下的 `bin/windows-x64` 或 `bin/linux-x64`；若你改输出目录，对示例脚本传入 `--module-dir`，对 `donut_render_py` API 则传入对应的 `module_dir` / `RuntimeOptions`。

示例脚本在 **`samples/DonutRenderPyDemo`**，例如：

```powershell
cd <仓库根目录>
$env:PYTHONPATH = "$PWD\python"
python samples\DonutRenderPyDemo\donut_render_demo_v0_1.py --module-dir "$PWD\bin\windows-x64"
```

带多帧输出与清单文件的较新示例：

```powershell
python samples\DonutRenderPyDemo\donut_render_demo_v0_5.py --module-dir "$PWD\bin\windows-x64" --output-dir "$PWD\.temp\demo_out"
```

`tools/donut_render` 下还有一些用于性能与行为自检的小脚本，可按需直接 `python xxx.py` 运行；多数支持 `--module-dir` 与仓库根路径参数。

在 Linux 上可把 `PYTHONPATH` 指到仓库里的 `python` 目录，`--module-dir` 指向 `bin/linux-x64`，命令形式与上面类似，只是把路径分隔符换成 `/`。

## 仓库结构（节选）

| 目录 | 说明 |
|------|------|
| [assets](assets) | 样本资源 |
| [docs](docs) | 技术说明与快速上手（部分为英文） |
| [samples](samples) | C++ / Python 示例 |
| [external/donut](external/donut) | Donut 框架 |
| [external](external) | 其他第三方依赖 |
| [src](src) | 库代码、Python 绑定源码 |
| [python](python) | 面向 Python 用户的包（含 Donut 无头渲染封装） |

## 延伸阅读

- [Slang 用户指南](https://shader-slang.com/slang/user-guide/)
- [SlangPy 文档](https://slangpy.readthedocs.io/en/latest/)
- [Vulkan `VK_NV_cooperative_vector`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_NV_cooperative_vector.html)
- [Donut 上游](https://github.com/NVIDIAGameWorks/donut)

## 联系与引用

问题可走 GitHub Issues；其他事宜可联系 `rtxns-sdk-support@nvidia.com`。

研究引用可使用：

```bibtex
@online{RTXNS,
   title   = {{{NVIDIA}}\textregistered{} {RTXNS}},
   author  = {{NVIDIA}},
   year    = 2025,
   url     = {https://github.com/NVIDIA-RTX/RTXNS},
   urldate = {2025-02-03},
}
```

## 许可

见 [LICENSE.md](LICENSE.MD)
