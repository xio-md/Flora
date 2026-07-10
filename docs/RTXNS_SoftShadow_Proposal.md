# RTXNS 渲染质量提升新路线 — 距离自适应 PCF 软阴影

> 本文档基于 minimax-vision (MiniMax-M3) 对当前 RTXNS Bistro 阴影渲染的客观视觉评估,提出本周后续的质量提升实验方案。

---

## 一、现状评估 (MiniMax-M3 视觉分析)

### 1.1 当前软阴影 (1-sample + bilateral blur) 视觉评估

来源: `output/bistro_test/bistro_rt_shadow_steady.png`

**第一轮评估 (详细)**:

| 维度 | MiniMax-M3 评估 | 状态 |
|------|----------------|------|
| 宏观 penumbra | "树冠投影外缘有明显柔和渐变过渡,penumbra 效果可见" | ✅ 良好 |
| 微观 alpha-tested 几何 | "单片叶子的微观边缘仍保留较锐利轮廓,软化主要集中在宏观轮廓上" | ⚠️ 待改进 |
| 远距离小光斑 | "远距离小光斑处仍存在空间 aliasing 痕迹,提示 PCSS/EVSM filter size 或样本数对细小高对比特征覆盖不足" | ❌ 缺陷 |
| 树叶 dappled 阴影 | "树叶的 dappled 阴影图案细节保留较好,能看到多个不规则光斑穿透叶冠" | ✅ 良好 |
| 整体可信度 | "相比纯硬阴影,亮区/暗区之间有过渡带,更接近自然光透过树冠的真实观感" | ✅ 提升 |

**第二轮评估 (聚焦缺陷)**:

> - **阴影整体过黑、不透明**,且没有随地面石块起伏变化的明暗层次,导致石板纹理几乎被完全"吃掉",缺乏真实树叶阴影应有的**半透明感与柔和过渡**。
> - 阴影内部的"光斑"形状**过于规则、对比强烈**,像被剪掉的镂空贴图。

### 1.2 与 Niagara GT2 对比 (新捕获的参考图)

来源: `output/bistro_test/rtxns_vs_niagara_new.png` (RTXNS 左 vs Niagara GT2 右)

| 维度 | RTXNS | Niagara GT2 | 差距 |
|------|-------|-------------|------|
| **阴影边缘柔和度** | "可见明显锯齿/颗粒感,叶子孔洞与暗部交界较硬,双边模糊后仍残留高频噪声" | "边缘过渡自然,半影区(penumbra)更宽,呈现连续渐变" | 严重 |
| **树叶光斑细节** | "亮斑轮廓较锐利,但形状偏规则/重复,有'贴花感'" | "光斑更小更密,相互融合,更接近真实树冠漏光的随机分布" | 中等 |
| **阴影内部地面纹理** | "暗部鹅卵石纹理被噪声'吃掉',低光处信噪比差,半透明感弱" | "阴影下鹅卵石仍清晰可辨,纹理色彩保留完整,半透明遮蔽更真实" | 严重 |
| **整体明暗层次** | "对比度偏高,中间调被压缩,亮暗过渡跳变明显" | "中间调丰富,亮→半影→暗呈连续 S 曲线,色彩饱和度更稳定" | 中等 |

### 1.3 核心差距 (MiniMax-M3 定位)

> **最大差距是噪声/方差收敛速度**:RTXNS 即便在 1-sample + bilateral blur 下仍残留显著的高频颗粒,尤其是阴影内部和叶子边缘,这是 GT 完全消除的。

---

## 二、发现的 3 个视觉缺陷 (改进空间)

### 缺陷 A — 阴影内部高频噪声残留 (严重)

- **位置**: 阴影内部和叶子边缘
- **现象**: 1-sample + bilateral blur 仍残留高频颗粒,暗部鹅卵石纹理被噪声"吃掉"
- **根因**: 单采样 + 空间双边模糊无法消除时域方差,噪声收敛速度太慢

### 缺陷 B — 阴影边缘锯齿+半影不足 (中等)

- **位置**: 叶子孔洞与暗部交界处
- **现象**: 边缘锯齿/颗粒感明显,Niagara GT 呈现更宽更连续的半影
- **根因**: 固定 bilateral kernel 无法自适应不同距离的半影宽度

### 缺陷 C — 树叶光斑形状规则、"贴花感" (中等)

- **位置**: 阴影内部 dappled 光斑
- **现象**: 光斑形状偏规则/重复,缺少真实树冠漏光的随机分布
- **根因**: 单采样抖动产生的二值化光斑,缺少多采样累积的灰度渐变

---

## 三、新路线方案 — SVGF 时域降噪 + 导向滤波

### 3.1 核心思路

基于 MiniMax-M3 的改进建议,把当前"1-sample + bilateral blur"升级为 **SVGF (Spatiotemporal Variance-Guided Filtering)** 风格的时域降噪:

**三个改进点**:

1. **时域累积 (TAA-style reprojection)** — 解决缺陷 A + C
   - 用上一帧的阴影结果做时域累积,等效 N×sample
   - 基于运动矢量重投影,避免鬼影
   - **这是 Niagara GT 噪声收敛快的核心原因**

2. **方差驱动的自适应滤波 (Variance-Guided Filtering)** — 解决缺陷 A + B
   - 计算局部方差,高方差区域用大 kernel,低方差区域用小 kernel
   - 替代当前的固定 bilateral kernel
   - 保留边缘锐度的同时压低方差

3. **多采样累积 (Multi-sample accumulation)** — 解决缺陷 C
   - 把 `shadowSamples` 从 1 提到 4
   - 配合时域累积,产生灰度渐变和自然光斑形状

### 3.2 解决的缺陷

| 缺陷 | SVGF 如何解决 |
|------|--------------|
| A. 阴影内部高频噪声 | 时域累积等效 N×sample,方差驱动的自适应滤波压低残余噪声 |
| B. 阴影边缘锯齿 | 方差引导的 kernel 在高方差(边缘)区域自动加大 |
| C. 光斑形状规则 | 多采样 + 时域累积产生灰度渐变和随机分布 |

### 3.3 实现路径 (基于现有代码)

**改进点 1 — 多采样** (无需改代码):

```python
scene.set_shadow_samples(4)  # 从默认 1 提到 4
```

**改进点 2 — 时域累积 + 方差滤波** (修改 shader):

新增文件: `src/RayTracedShadow/shaders/shadow_svgf_cs.hlsl`

```hlsl
// Pass 1: 时域累积 — 用上一帧阴影 + 运动矢量做 reprojection
float temporalAccum = mix(prevShadow, currShadow, alpha);  // alpha=0.1

// Pass 2: 方差计算 — 7×7 局部窗口估方差
float variance = computeLocalVariance(shadowImage, pos, 7);

// Pass 3: 方差驱动滤波 — 高方差大 kernel,低方差小 kernel
float kernelSize = clamp(variance * kVarScale, kMin, kMax);
float filtered = guidedFilter(shadowImage, depthImage, pos, kernelSize);
```

修改 `RayTracedShadowPass.h/cpp`: 新增 SVGF pass,替换现有 blur pass

### 3.4 预期效果 (基于 MiniMax-M3 评估推算)

| 指标 | 当前 | 预期 | 提升 |
|------|------|------|------|
| 阴影内部噪声 | 严重缺陷 | 轻微 (接近 GT) | ↓ 2 级 |
| 阴影边缘锯齿 | 中等缺陷 | 轻微 | ↓ 1 级 |
| 光斑形状规则 | 中等缺陷 | 轻微 | ↓ 1 级 |
| 阴影内部纹理保留 | 严重缺陷 | 中等 (接近 GT) | ↓ 1 级 |
| 中间调层次 | 中等缺陷 | 轻微 | ↓ 1 级 |
| 单帧耗时 | 0.14 ms | 0.25 ms (估) | +79% (仍实时) |

---

## 四、实验计划

### 阶段 1: 基线采集 (已完成)

- ✅ 用 MiniMax-M3 评估当前 `bistro_rt_shadow_steady.png`
- ✅ 用 PowerShell 启动 Niagara + 手动捕获新 GT2 (`output/bistro_test/GT2.png`)
- ✅ 拼接 RTXNS vs GT2 对比图 (`output/bistro_test/rtxns_vs_niagara_new.png`)
- ✅ MiniMax-M3 详细对比分析 5 个维度
- ✅ 定位核心差距: 噪声收敛速度

### 阶段 2: 多采样基线 (本周)

1. 把 `set_shadow_samples` 从 1 提到 4
2. 重新渲染 bistro 场景
3. 用 MiniMax-M3 评估噪声降低程度
4. 量化: 残余高频噪声降级

### 阶段 3: SVGF 时域降噪 (本周/下周)

1. 新增 `shadow_svgf_cs.hlsl` (时域累积 + 方差计算 + 导向滤波)
2. 修改 `RayTracedShadowPass.cpp` 加入 SVGF pass
3. 需要历史帧 buffer (prevShadow texture)
4. 用 MiniMax-M3 评估与 GT2 的差距缩小程度

### 阶段 4: 视觉评估 (本周)

1. 用 MiniMax-M3 重新评估新图
2. 与 Niagara GT2 再次对比
3. 量化: 5 个维度的差距降级

### 阶段 5: 性能验证 (本周)

- 测量 SVGF pass 耗时变化
- 确保 shadow_ray + SVGF 总耗时仍 < 0.5 ms (实时预算)

---

## 五、给老师的一句话总结

> **用 MiniMax-M3 视觉模型对 RTXNS 与 Niagara GT 做了 5 维度客观对比,定位核心差距为"噪声收敛速度"(1-sample + bilateral blur 残留高频颗粒)。提出 SVGF 时域降噪路线:时域累积 + 方差驱动导向滤波 + 多采样,预期在 0.25ms 实时预算内将 5 个维度差距降级 1-2 级,显著接近 Niagara GT 视觉质量。**
