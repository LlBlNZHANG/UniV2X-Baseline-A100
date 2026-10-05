# STCV-Occ V2-0：Metric-aligned Cooperation Utility 定义与 Oracle 验证总结

## 1. 阶段定位

STCV-Occ V2-0 的目标不是训练新的预测器，而是首先重新定义：

> **一个路侧新增 Occupancy patch 的“合作价值”应当如何度量。**

STCV-Occ V1 使用：

\[
U_R^{V1}=BA_R-HA_R
\]

其中：

- \(BA_R\)：Beneficial Addition，接受路侧信息后新增的正确占用格；
- \(HA_R\)：Harmful Addition，接受路侧信息后新增的错误占用格。

并根据：

\[
y_R=\mathbb{1}[U_R^{V1}>0]
\]

将合作价值压缩成二分类标签。

V2-0 重点解决 V1 Utility 定义中的两个问题：

1. \(BA-HA\) 默认 BA 和 HA 对最终评价指标具有完全对称的影响；
2. \(BA-HA\) 只是一种局部启发式计数，并不严格对应最终 Occupancy IoU 的实际边际变化。

因此，本阶段主要完成：

\[
\boxed{
BA-HA
\rightarrow
\text{Metric-aligned Marginal Cooperation Utility}
}
\]

本阶段**没有训练 STCV-v2 网络**。

---

# 2. V1 Utility 数据审计

## 2.1 Candidate 定义

现有 G3 数据中，首先定义 GT 与路侧几何有效域：

\[
M^{valid}=M^{GT}\land M^{warp}
\]

然后定义路侧新增 Occupancy：

\[
ADD=(\neg O_v)\land O_i\land M^{valid}
\]

其中：

- \(O_v\)：Ego Occupancy binary prediction；
- \(O_i\)：aligned infrastructure Occupancy binary prediction。

进一步划分：

\[
BA=ADD\land GT
\]

\[
HA=ADD\land(\neg GT)
\]

因此：

\[
ADD=BA+HA
\]

每个 candidate P4 patch 的原始 V1 Utility 为：

\[
U_R^{V1}=BA_R-HA_R
\]

对应标签：

\[
y_R=\mathbb{1}[BA_R-HA_R>0]
\]

---

## 2.2 数据一致性审计

Train split 共包含：

\[
86,143
\]

个 candidate P4 patches。

一致性检查结果：

```text
ADD != BA + HA          : 0
utility != BA - HA      : 0
label != 1[utility > 0] : 0
```

说明现有 V1 数据严格符合理论定义。

Train split 中：

\[
N_{positive}=12,720
\]

正样本比例：

\[
14.77\%
\]

负样本或零 Utility patch：

\[
73,423
\]

占：

\[
85.23\%
\]

说明候选路侧新增区域中，**负合作现象明显占主导**。

---

# 3. V1 Utility Magnitude 分布

原始 V1 Utility：

\[
U_R=BA_R-HA_R
\]

统计结果：

\[
\operatorname{mean}(U)=-4.280
\]

\[
\operatorname{median}(U)=-4
\]

范围：

\[
U\in[-16,16]
\]

由于 P4 patch 为：

\[
4\times4
\]

最多包含 16 个 cell，因此 Utility 存在明显的 magnitude 差异。

若简单归一化：

\[
u_R=\frac{BA_R-HA_R}{16}
\]

则：

\[
u_R\in[-1,1]
\]

Train 数据中：

\[
31.66\%
\]

的 patch 满足：

\[
|u_R|\ge0.5
\]

同时约：

\[
31.65\%
\]

满足：

\[
|u_R|\le0.125
\]

说明数据中同时存在：

- 弱合作价值 patch；
- 强合作价值 patch。

因此，将所有正 Utility 压缩为标签 1、所有负 Utility 压缩为标签 0，会丢失大量价值强度信息。

这一结果为后续 V2-1：

\[
\boxed{
Binary\ Classification
\rightarrow
Magnitude\text{-}aware\ Utility\ Learning
}
\]

提供了数据依据。

---

# 4. Horizon 风险审计

Train split 各 Horizon 的统计如下：

| Horizon | Patches | Positive Rate | ADD | BA | HA | WAR |
|---|---:|---:|---:|---:|---:|---:|
| \(t_0\) | 18,022 | 0.1688 | 111,241 | 24,222 | 87,019 | 0.7823 |
| \(t_1\) | 18,446 | 0.1682 | 120,979 | 25,962 | 95,017 | 0.7854 |
| \(t_2\) | 18,448 | 0.1430 | 128,613 | 23,068 | 105,545 | 0.8206 |
| \(t_3\) | 17,008 | 0.1270 | 124,066 | 19,293 | 104,773 | 0.8445 |
| \(t_4\) | 14,219 | 0.1250 | 99,072 | 15,094 | 83,978 | 0.8476 |

其中：

\[
WAR=\frac{HA}{ADD}
\]

可以看到 Train split 中，随着预测时域向未来延伸：

\[
WAR:
0.7823
\rightarrow
0.8476
\]

总体呈上升趋势。

Positive rate 则从：

\[
16.88\%
\]

下降至：

\[
12.50\%
\]

说明远期 Occupancy 中负合作风险总体更高。

但是该趋势在 Val/Test split 中并非严格单调，因此最终没有采用：

\[
\beta_t=f(t)
\]

这种人为 horizon risk weighting。

Horizon 继续作为 predictor 的输入特征，由模型自主学习其与合作价值之间的关系。

---

# 5. Distance 风险审计

不同距离区间中的 WAR 存在明显差异。

例如：

\[
WAR_{0-5m}=0.9618
\]

而：

\[
WAR_{50-55m}=0.7545
\]

但 patch-level Spearman 相关性仅为：

\[
\rho(distance,WAR)=-0.0818
\]

同时：

\[
\rho(distance,U)=0.0791
\]

虽然 \(p\) 值很小，但主要原因是样本规模较大，实际相关强度较弱。

因此无法支持简单的：

\[
\text{Risk}=f(distance)
\]

单调函数假设。

最终决定：

> **不将 distance 作为人工 Utility weighting，而继续把 spatial position 保留在 predictor feature 中。**

---

# 6. 固定 HA Cost \(\beta\) 审计

进一步测试：

\[
U_R=BA_R-\beta HA_R
\]

其中：

\[
\beta\in\{1.0,1.1,1.25,1.5,2.0\}
\]

结果：

| \(\beta\) | Positive Ratio | Flip vs V1 | V1 Positive Kept |
|---:|---:|---:|---:|
| 1.00 | 0.1477 | 0.0000 | 1.0000 |
| 1.10 | 0.1477 | 0.0000 | 1.0000 |
| 1.25 | 0.1445 | 0.0031 | 0.9789 |
| 1.50 | 0.1361 | 0.0116 | 0.9218 |
| 2.00 | 0.1229 | 0.0247 | 0.8325 |

随着 \(\beta\) 增大，越来越多 V1 positive patches 被拒绝。

但 \(\beta>1\) 缺乏直接的官方 metric 依据，因此最终没有采用人为 cost-sensitive：

\[
BA-\beta HA
\]

作为 V2 主 Utility。

---

# 7. 官方 Occupancy IoU 定义审计

UniV2X 官方 Occupancy IoU 对每个类别累计：

\[
TP,\quad FP,\quad FN
\]

并计算：

\[
IoU=
\frac{TP}
{TP+FP+FN}
\]

官方评价两个 BEV 范围：

\[
30m\times30m
\]

以及：

\[
100m\times100m
\]

其中 Occupied class 为最终论文中关注的主要 Occupancy IoU。

这一官方评价形式为 V2 Utility 的重新定义提供了直接依据。

---

# 8. Metric-aligned Marginal Cooperation Utility

对于某个 sample 和 horizon，记 Ego Occupancy 当前统计量为：

\[
TP,\quad FP,\quad FN
\]

定义：

\[
D=TP+FP+FN
\]

则 Ego 当前 Occupancy IoU 为：

\[
IoU^{ego}
=
\frac{TP}{D}
\]

对于某个 candidate patch \(R\)，若接受路侧新增 Occupancy，其中包含：

\[
BA_R
\]

个 beneficial cells，以及：

\[
HA_R
\]

个 harmful cells。

接受后：

\[
TP'=TP+BA_R
\]

\[
FP'=FP+HA_R
\]

\[
FN'=FN-BA_R
\]

所以新的 IoU 分母：

\[
TP'+FP'+FN'
=
D+HA_R
\]

因此接受 patch \(R\) 后：

\[
IoU_R^{accept}
=
\frac{TP+BA_R}
{D+HA_R}
\]

于是定义 V2 的 Metric-aligned Marginal Cooperation Utility：

\[
\boxed{
U_R^{V2}
=
\Delta IoU_R
=
\frac{TP+BA_R}
{TP+FP+FN+HA_R}
-
\frac{TP}
{TP+FP+FN}
}
\]

该 Utility 直接衡量：

> **接受一个 infrastructure Occupancy patch 后，相对于 Ego-only Occupancy，foreground IoU 的实际局部边际变化。**

---

# 9. \(\Delta IoU\) Utility 的决策性质

有：

\[
\Delta IoU_R>0
\]

等价于：

\[
\frac{TP+BA_R}{D+HA_R}
>
\frac{TP}{D}
\]

整理得到：

\[
BA_RD>TP\cdot HA_R
\]

进一步：

\[
BA_R>
\frac{TP}{D}HA_R
\]

因为：

\[
\frac{TP}{D}
=
IoU_{ego}
\]

所以：

\[
\boxed{
\Delta IoU_R>0
\iff
BA_R>IoU_{ego}\cdot HA_R
}
\]

与 V1：

\[
BA_R>HA_R
\]

相比，新 Utility 不再假设 BA 和 HA 对最终 IoU 的影响完全对称。

其 harmful-addition cost 会随着当前 Ego Occupancy 状态自适应变化。

因此：

\[
\boxed{
U_R^{V2}
}
\]

具有明显更强的 metric alignment。

---

# 10. \(\Delta IoU_{100}\) Oracle Audit

Oracle audit 仅使用 G3 Train scenes：

\[
13\text{ scenes}
\]

共：

\[
361\text{ valid samples}
\]

对应：

\[
1805
\]

个 sample-horizons。

Candidate P4 patches：

\[
86,143
\]

没有使用 Val/Test 对 Utility 设计进行反向选择。

结果如下：

| Strategy | 30m IoU | 100m IoU |
|---|---:|---:|
| Ego | 0.268100 | 0.244324 |
| Official OR | 0.226811 | 0.239787 |
| V1 \(BA-HA\) Oracle | 0.326363 | 0.296963 |
| **V2 \(\Delta IoU_{100}\) Oracle** | **0.332924** | **0.303466** |

相对于 V1 P4 Oracle：

\[
\boxed{
0.326363
\rightarrow
0.332924
}
\]

30m absolute gain：

\[
\boxed{
+0.006561
}
\]

100m：

\[
\boxed{
0.296963
\rightarrow
0.303466
}
\]

absolute gain：

\[
\boxed{
+0.006503
}
\]

即大约提高：

\[
+0.66
\]

和：

\[
+0.65
\]

个 IoU 百分点。

---

# 11. Patch 决策变化

V1 Oracle 接受：

\[
12,720
\]

个 candidate patches。

Acceptance rate：

\[
14.77\%
\]

Metric-aligned \(\Delta IoU_{100}\) Oracle 接受：

\[
19,743
\]

个 candidate patches。

Acceptance rate：

\[
22.92\%
\]

两种 Utility 共有：

\[
7,023
\]

个 patch 决策发生变化。

占 candidate 总数：

\[
8.15\%
\]

并且：

\[
19,743-12,720=7,023
\]

说明新的 Metric-aligned Utility 基本表现为：

> **保留 V1 正合作 patch，同时识别出额外一批 V1 因 \(BA\le HA\) 而拒绝、但实际上仍然能够改善 Occupancy IoU 的 candidate patches。**

这正是 V2 Utility 相对于 V1 的主要改进来源。

---

# 12. Multi-scale \(\Delta IoU\) Utility 审计

为了进一步验证是否应该同时考虑 30m 和 100m 官方 Occupancy metric，测试：

\[
\boxed{
U_R^{MS}
=
\alpha\Delta IoU_R^{30}
+
(1-\alpha)\Delta IoU_R^{100}
}
\]

其中：

\[
\alpha
\in
\{0,0.25,0.5,0.75,1.0\}
\]

对应：

- \(\alpha=0\)：纯 \(\Delta IoU_{100}\)
- \(\alpha=1\)：纯 \(\Delta IoU_{30}\)

Oracle 结果如下：

| \(\alpha\) | 30m IoU | 100m IoU | Mean |
|---:|---:|---:|---:|
| **0.00** | **0.332924** | **0.303466** | **0.318195** |
| 0.25 | 0.332292 | 0.303341 | 0.317816 |
| 0.50 | 0.332233 | 0.303320 | 0.317776 |
| 0.75 | 0.332225 | 0.303305 | 0.317765 |
| 1.00 | 0.332311 | 0.252676 | 0.292494 |

最佳结果为：

\[
\boxed{\alpha=0}
\]

即纯：

\[
\boxed{
\Delta IoU_{100}
}
\]

---

# 13. Multi-scale 相对于单尺度 \(\Delta IoU_{100}\) 的变化

相对于：

\[
\alpha=0
\]

加入 30m Utility 后：

### \(\alpha=0.25\)

\[
30m:-0.000632
\]

\[
100m:-0.000125
\]

### \(\alpha=0.50\)

\[
30m:-0.000690
\]

\[
100m:-0.000147
\]

### \(\alpha=0.75\)

\[
30m:-0.000698
\]

\[
100m:-0.000161
\]

纯 30m Utility：

\[
\alpha=1
\]

导致：

\[
100m:-0.050790
\]

明显下降。

因此：

\[
\boxed{
\text{Multi-scale Utility 未进一步改善 Oracle 上限}
}
\]

最终不作为 V2 主方法。

---

# 14. Multi-scale 失败原因分析

\(\Delta IoU_{30}\) 的统计显示：

\[
527/1805
\]

个 sample-horizons 在 30m 区域内：

\[
TP+FP+FN=0
\]

即 occupied IoU denominator 为 0。

而 100m 范围仅有：

\[
17/1805
\]

个 zero-denominator horizons。

同时：

\[
\operatorname{median}
|\Delta IoU_{30}|
=
0
\]

而：

\[
\operatorname{median}
|\Delta IoU_{100}|
=
0.0012976
\]

说明 30m Utility 对大量 candidate patches 不提供有效边际信号。

因此单独或组合使用 30m Utility 会引入稀疏、局部且不稳定的监督。

相反：

\[
\Delta IoU_{100}
\]

具有更完整的全局 Occupancy context，而且实验表明：

> 即使 Utility 只根据 100m 全 BEV 定义，也能够同时提升 30m 和 100m 的最终 Occupancy IoU。

---

# 15. V2-0 最终 Utility 定义

经过：

1. V1 \(BA-HA\) 审计；
2. Horizon risk 审计；
3. Distance risk 审计；
4. Fixed \(\beta\) cost-sensitive 审计；
5. Metric-aligned \(\Delta IoU\) 推导；
6. V1 vs \(\Delta IoU\) Oracle；
7. Multi-scale \(\Delta IoU\) Oracle；

最终 V2-0 固定：

\[
\boxed{
U_R^{V2}
=
\Delta IoU_R^{100}
}
\]

即：

\[
\boxed{
U_R^{V2}
=
\frac{TP+BA_R}
{TP+FP+FN+HA_R}
-
\frac{TP}
{TP+FP+FN}
}
\]

其中 \(TP,FP,FN\) 基于：

\[
100m\times100m
\]

full-BEV Occupancy scope，在每个 sample、每个 horizon 下计算。

---

# 16. V2-0 最终结论

STCV-Occ V1 使用：

\[
U_R^{V1}=BA_R-HA_R
\]

作为 patch-level cooperation utility。

V2-0 将其升级为：

\[
\boxed{
U_R^{V2}
=
\Delta IoU_R^{100}
}
\]

从：

> **基于正确/错误新增格数量差的启发式 Utility**

升级为：

> **直接与最终 Occupancy IoU 对齐的边际合作价值。**

Oracle 实验表明：

\[
30m:
0.326363
\rightarrow
0.332924
\]

\[
100m:
0.296963
\rightarrow
0.303466
\]

分别获得：

\[
\boxed{
+0.006561
}
\]

和：

\[
\boxed{
+0.006503
}
\]

的 absolute IoU gain。

因此：

\[
\boxed{
\text{V2-0 Metric-aligned Utility Definition：PASS}
}
\]

---

# 17. 后续 V2 总体路线

V2 后续方法链固定为：

\[
\boxed{
\text{Metric-aligned Marginal Utility}
\rightarrow
\text{Magnitude-aware Utility Estimation}
\rightarrow
\text{Planning Task-consistency Validation}
\rightarrow
\text{Uncertainty-aware Adaptive Selection}
}
\]

对应：

```text
V2-0
Metric-aligned Utility Definition
ΔIoU100
PASS

        ↓

V2-1
Magnitude-aware Utility Learning
直接预测连续合作价值
NEXT

        ↓

V2-2
Planning Task-consistency Audit
验证更可靠的 Occupancy 是否改善下游 Planning

        ↓

V2-3
Uncertainty-aware Adaptive Decision
由固定阈值升级为价值-不确定性联合决策

        ↓

V2-4
Final Official Evaluation
```

---

# 18. 论文中“方法”部分建议写法

## 18.1 问题描述

可以写：

> 现有车路协同 Occupancy 融合通常直接接受经过空间对齐的路侧预测，但路侧新增 Occupancy 并不总能改善车端预测，其中错误新增信息可能进一步降低融合结果的可靠性。为此，需要在无 Ground Truth 的推理阶段估计不同路侧新增区域的实际合作价值，并仅接受具有正边际收益的信息。

---

## 18.2 V1 问题引出

可以写：

> A straightforward cooperation utility can be defined as the difference between beneficial and harmful additions, i.e., \(BA_R-HA_R\). However, this formulation implicitly assigns equal and symmetric contributions to beneficial and harmful occupancy cells, while its decision boundary is not directly aligned with the final occupancy evaluation metric.

中文可写：

> 一种直接的合作价值定义为候选区域内有益新增格与有害新增格数量之差 \(BA_R-HA_R\)。然而，该定义默认有益新增与有害新增对最终预测性能具有对称贡献，且其决策边界并未与 Occupancy 的最终评价指标直接对齐。

---

## 18.3 V2 Utility 定义

推荐正文写法：

> 为使合作价值与 Occupancy 评价目标保持一致，本文从 foreground IoU 的边际变化出发重新定义路侧候选区域的合作价值。对于候选 patch \(R\)，记车端 Occupancy 的当前统计量为 \(TP\)、\(FP\) 和 \(FN\)。接受该路侧区域后，其中的 \(BA_R\) 将 false-negative 转化为 true-positive，而 \(HA_R\) 将引入新的 false-positive。因此，接受前后的 foreground IoU 分别表示为
>
> \[
> IoU^{ego}
> =
> \frac{TP}{TP+FP+FN}
> \]
>
> 和
>
> \[
> IoU_R^{accept}
> =
> \frac{TP+BA_R}
> {TP+FP+FN+HA_R}.
> \]
>
> 据此，定义 metric-aligned marginal cooperation utility：
>
> \[
> U_R
> =
> IoU_R^{accept}
> -
> IoU^{ego}.
> \]
>
> 当 \(U_R>0\) 时，表示接受该路侧候选区域能够提高当前 Occupancy IoU；反之则表示该信息会产生负合作效应。

---

# 19. 论文中“实验”部分建议写法

推荐建立如下表格：

| Utility Strategy | 30m IoU | 100m IoU |
|---|---:|---:|
| Ego | 26.81 | 24.43 |
| Official OR | 22.68 | 23.98 |
| \(BA-HA\) Patch Oracle | 32.64 | 29.70 |
| **Metric-aligned \(\Delta IoU\) Oracle** | **33.29** | **30.35** |

注意：

这里如果转换成百分制，应写为：

\[
0.332924
\rightarrow
33.29
\]

不要混用 0–1 和百分制。

论文中可以描述：

> Compared with the original \(BA-HA\) patch oracle, the proposed metric-aligned utility improves the 30 m and 100 m occupancy IoU by 0.66 and 0.65 percentage points, respectively, demonstrating that directly modeling the marginal change of the evaluation metric provides a more accurate criterion for identifying beneficial cooperative occupancy regions.

中文：

> 与基于 \(BA-HA\) 的原始 P4 Oracle 相比，基于边际 IoU 的合作价值分别将 30 m 和 100 m Occupancy IoU 提升 0.66 和 0.65 个百分点，说明与评价指标直接对齐的边际价值能够更准确地识别真正有益的路侧合作区域。

---

# 20. Multi-scale 实验在论文中如何处理

Multi-scale 不建议作为最终方法。

但可以作为消融实验或补充实验保留。

可以写：

> We further examined whether explicitly combining the official 30 m and 100 m evaluation ranges could provide a stronger cooperation utility. However, incorporating the 30 m marginal IoU did not further improve the oracle performance. This is mainly because the near-field metric is sparse for a considerable number of sample-horizon pairs, resulting in insufficient marginal utility signals. Therefore, the 100 m full-BEV marginal IoU is adopted as the final cooperation utility.

中文：

> 进一步地，本文尝试联合 30 m 与 100 m 官方评价范围构造多尺度合作价值。然而，引入 30 m 边际 IoU 后 Oracle 性能未进一步提高。分析发现，部分样本时刻在 30 m 范围内不存在有效前景 Occupancy，使得局部边际价值较为稀疏。因此，最终采用基于 100 m 全 BEV 范围计算的边际 IoU 作为合作价值监督。

建议将 Multi-scale 表格放在：

```text
Ablation Study
```

或：

```text
Supplementary Experiments
```

而不是方法正文。

---

# 21. V2-0 论文创新点建议表述

推荐表述为：

> **Metric-aligned marginal cooperation utility.**
> Different from heuristic beneficial-harmful counting, we formulate cooperative occupancy selection as a marginal utility estimation problem and quantify the value of each infrastructure-added region through its induced change in foreground Occupancy IoU. This provides a task-aligned and magnitude-aware supervision signal for subsequent cooperation value learning.

中文版本：

> **评价指标对齐的边际合作价值建模。** 针对传统有益/有害新增格计数无法直接反映融合结果实际性能变化的问题，本文将路侧 Occupancy 选择建模为边际合作价值估计问题，通过候选区域引起的 foreground Occupancy IoU 变化量量化其实际合作收益，为后续连续合作价值学习提供与任务目标直接对齐的监督信号。

注意当前阶段还没有真正完成：

> magnitude-aware predictor

因此 V2-0 阶段单独总结时，不要提前声称模型已经实现 magnitude-aware learning。

---

# 22. 当前可以使用的核心数据

## Dataset / Split

```text
Total valid samples: 549
Total scenes: 21

Train scenes: 13
Val scenes: 4
Test scenes: 4
```

V2-0 Oracle 只使用：

```text
Train valid samples: 361
Train sample-horizons: 1805
Candidate P4 patches: 86143
```

---

## V1 Utility

```text
Positive patches: 12720
Positive ratio: 14.77%

Utility mean: -4.280
Utility median: -4
Utility range: [-16, 16]
```

---

## V1 Oracle

```text
30m IoU:  0.326363
100m IoU: 0.296963
```

---

## V2 DeltaIoU100 Oracle

```text
Accepted patches: 19743
Acceptance rate: 22.92%

30m IoU:  0.332924
100m IoU: 0.303466
```

---

## Improvement over V1 Oracle

```text
30m:
+0.006561
≈ +0.66 percentage points

100m:
+0.006503
≈ +0.65 percentage points
```

---

## Decision Difference

```text
Changed patch decisions:
7023 / 86143

Flip ratio:
8.15%
```

---

## Multi-scale Result

```text
alpha=0.00:
30m  = 0.332924
100m = 0.303466
Mean = 0.318195

alpha=0.25:
30m  = 0.332292
100m = 0.303341

alpha=0.50:
30m  = 0.332233
100m = 0.303320

alpha=0.75:
30m  = 0.332225
100m = 0.303305

alpha=1.00:
30m  = 0.332311
100m = 0.252676
```

最终：

\[
\boxed{\alpha=0}
\]

对应：

\[
\boxed{\Delta IoU_{100}}
\]

为最佳 Oracle 设计。

---

# 23. V2-0 最终一句话总结

> STCV-Occ V2-0 将 V1 基于 \(BA-HA\) 的启发式二值合作判据升级为与 Occupancy foreground IoU 直接对齐的边际合作价值 \(\Delta IoU_{100}\)，并通过 Train-only P4 Oracle 实验证明该定义相较 V1 Oracle 在 30 m 与 100 m Occupancy IoU 上分别获得 0.66 和 0.65 个百分点的提升；进一步的 Multi-scale Utility 审计未获得额外收益，因此最终采用 full-BEV metric-aligned marginal utility 作为 STCV-v2 后续连续合作价值学习的监督基础。

---

# 24. V2-0 阶段状态

```text
V2-0 Metric-aligned Utility Definition

V1 BA-HA audit                  PASS
Magnitude distribution audit   PASS
Horizon risk audit              PASS
Distance risk audit             PASS
Fixed-beta audit                PASS
DeltaIoU derivation             PASS
DeltaIoU100 Oracle              PASS
Multi-scale Oracle audit        PASS
Final utility selection         PASS

FINAL:
U_v2 = DeltaIoU100

STATUS:
V2-0 COMPLETE
```

下一阶段：

\[
\boxed{
\text{V2-1 Magnitude-aware Utility Learning}
}
\]

目标：

\[
X_R
\longrightarrow
\widehat{\Delta IoU_R}
\]

即从：

\[
\boxed{
\text{“该 patch 是否有价值？”}
}
\]

进一步升级为：

\[
\boxed{
\text{“该 patch 的实际合作价值有多大？”}
}
\]