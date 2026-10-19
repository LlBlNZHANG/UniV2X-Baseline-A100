# STCV-Occ V2-1：Magnitude-aware Cooperative Utility Learning

> **阶段状态：主线完成，可冻结**
>
> **最终采用方案：**
>
> \[
> \boxed{
> U_R^{V2}=\Delta IoU_{R}^{100}
> }
> \]
>
> \[
> \boxed{
> \lambda_{\mathrm{sign}}=0.5,\qquad
> L_{\mathrm{mag}}=\mathrm{SmoothL1}
> }
> \]
>
> **主模型参数量：495,490**
>
> **最终主 checkpoint：**
>
> `G0-G1-G2/V2/V2-1_magnitude_learning/checkpoints/main_lambda_0p50/best.pth`
>
> **说明：**
>
> - V2-1A：连续 Utility target 审计；
> - V2-1B：Magnitude-aware Utility Predictor 训练；
> - V2-1C：Occupancy fusion-level validation；
> - V2-1D：Magnitude loss 改进消融，NO-GO；
> - V2-1E：Set-level IoU 对齐探索，仅获得边际收益，不纳入最终主模型；
> - Test split 在 V2-1 模型选择阶段保持封闭，尚未用于最终模型选择。

---

# 1. 研究目的

STCV-Occ V1 使用二值 Utility：

\[
U_R^{V1}=BA_R-HA_R,
\]

其中：

- \(BA_R\)：接受路侧 Occupancy 后，在区域 \(R\) 内新增的正确占用单元；
- \(HA_R\)：接受路侧 Occupancy 后，在区域 \(R\) 内新增的错误占用单元。

V1 根据：

\[
U_R^{V1}>0
\]

判断某一 P4 patch 是否值得接受。

该定义可以判断“有益 / 有害”，但存在两个不足：

1. \(BA\) 与 \(HA\) 被赋予固定、对称的单位代价，与最终 Occupancy IoU 的实际评价形式并不完全一致；
2. predictor 只学习二值标签，无法表达不同候选区域对最终 Occupancy 指标的贡献幅度。

因此 V2-1 的目标是：

> 将合作价值从简单的二值有益性判断升级为与 Occupancy IoU 对齐的连续边际效用估计，使模型不仅判断“是否值得合作”，还能够估计“合作能够带来多大的实际价值”。

---

# 2. V2 Utility 定义

## 2.1 Ego Occupancy 状态

对于每个 sample、每个 horizon，定义 Ego Occupancy 的 occupied-class 统计量：

\[
TP,\quad FP,\quad FN.
\]

令：

\[
D=TP+FP+FN.
\]

当前 Ego occupied-class IoU 为：

\[
IoU_{\mathrm{ego}}
=
\frac{TP}{D}.
\]

对于候选 patch \(R\)，接受路侧新增 Occupancy 后：

- \(BA_R\) 个单元由 FN 转化为 TP；
- \(HA_R\) 个单元新增为 FP。

因此接受该 patch 后：

\[
TP'=TP+BA_R,
\]

\[
FN'=FN-BA_R,
\]

\[
FP'=FP+HA_R.
\]

新的 occupied-class IoU 为：

\[
IoU_R'
=
\frac{TP+BA_R}{D+HA_R}.
\]

于是定义 V2 的连续合作价值：

\[
\boxed{
U_R^{V2}
=
\Delta IoU_R^{100}
=
\frac{TP+BA_R}{D+HA_R}
-
\frac{TP}{D}
}
\]

其中统计范围采用完整 100 m BEV。

---

## 2.2 Utility 正负条件

由：

\[
\Delta IoU_R>0
\]

可得：

\[
BA_R D > TP\cdot HA_R,
\]

即：

\[
\boxed{
BA_R>
IoU_{\mathrm{ego}}HA_R
}
\]

因此 V2 不再人为使用固定：

\[
BA_R>\beta HA_R
\]

作为价值标准，而是让 HA 的实际代价由当前 Ego Occupancy 状态自动决定。

这使 Utility 与当前 Occupancy 质量具有自适应关系。

---

## 2.3 与 V1 的关系

由于：

\[
IoU_{\mathrm{ego}}\leq1,
\]

如果：

\[
BA_R>HA_R,
\]

则必然：

\[
BA_R>
IoU_{\mathrm{ego}}HA_R.
\]

因此：

\[
\boxed{
U_R^{V1}>0
\Rightarrow
U_R^{V2}>0
}
\]

即 V1-positive 是 V2-positive 的子集。

V2 的核心变化不是简单放宽阈值，而是依据最终 Occupancy IoU 对候选区域重新定义边际合作价值。

---

# 3. V2-1A：Magnitude Target Audit

## 3.1 数据规模

沿用 G3 完全相同的 scene-level split：

| Split | Candidate patches | V2 Positive | Positive ratio |
|---|---:|---:|---:|
| Train | 86,143 | 19,743 | 22.92% |
| Val | 18,555 | 4,876 | 26.28% |
| Test | 14,750 | 3,188 | 21.61% |

原始 V1 Train positive：

\[
12,720/86,143=14.77\%.
\]

V2 Train positive：

\[
19,743/86,143=22.92\%.
\]

新增：

\[
19,743-12,720=7,023
\]

个 V2-positive patch。

对应：

\[
\boxed{
8.15\%
}
\]

的 Train candidate 由 V1 non-positive 转变为 V2 positive。

同时审计结果满足：

\[
\boxed{
V1+,\ V2-\;=0
}
\]

与理论推导一致。

---

## 3.2 连续 Target

训练使用：

\[
\boxed{
u_R=100\times\Delta IoU_R^{100}
}
\]

注意：

> 乘以 100 只是数值尺度变换，用于改善训练数值范围；理论上的 Utility 仍然是 \(\Delta IoU_R^{100}\)，不能把 ×100 描述成新的理论定义。

Train target：

\[
std(u)=0.523137.
\]

Val target：

\[
std(u)=0.798943.
\]

Test target：

\[
std(u)=0.616545.
\]

Train 中：

\[
P(u>0)=22.92\%.
\]

说明连续 Utility 具有明显类别不平衡和长尾特征。

---

## 3.3 Zero-target 特殊现象

审计发现部分：

\[
\Delta IoU=0
\]

并不代表新增 Occupancy 是“无害”的。

尤其 Test 中存在较多：

\[
BA=0,\qquad HA>0
\]

但：

\[
\Delta IoU=0
\]

的候选区域。

原因是部分 sample-horizon 的 Ego occupied-class：

\[
TP=0.
\]

此时接受纯 HA patch 前后：

\[
IoU=0,
\]

因此边际 IoU 数值仍然为 0。

这意味着：

> 单独使用连续回归 target，无法明确要求模型拒绝所有纯 harmful addition。

因此 V2-1 最终采用：

\[
\boxed{
\text{Magnitude regression}
+
\text{Sign auxiliary supervision}
}
\]

的双任务学习方式。

---

# 4. V2-1B：Magnitude-aware Utility Predictor

## 4.1 输入特征

为保证与 V1 的比较公平，V2-1 沿用完全相同的 127-D 输入特征：

\[
X_R\in\mathbb R^{127}.
\]

包括：

### Local features：80-D

P4 patch 内：

\[
[P_v,\ P_i,\ P_i-P_v,\ |P_i-P_v|,\ M_{\mathrm{warp}}]
\]

共：

\[
5\times4\times4=80.
\]

### Horizon：5-D

5 个 Occupancy horizon 的 one-hot 编码。

### Spatial：2-D

候选 patch 中心位置归一化坐标：

\[
(u,v).
\]

### Neighborhood：40-D

8 邻域 × 5 个通道均值：

\[
8\times5=40.
\]

总计：

\[
80+5+2+40
=
\boxed{127}.
\]

---

## 4.2 网络结构

主模型：

\[
127
\rightarrow512
\rightarrow512
\rightarrow256
\rightarrow128.
\]

包含：

- Linear；
- LayerNorm；
- GELU；
- Dropout = 0.1；
- 512-D residual block。

共享表示之后设置两个 head：

### Magnitude head

\[
128\rightarrow1
\]

预测：

\[
\hat u_R
\approx
100\Delta IoU_R^{100}.
\]

### Sign head

\[
128\rightarrow1
\]

预测：

\[
P(U_R^{V2}>0).
\]

总参数量：

\[
\boxed{495,490}.
\]

该模型仍属于轻量级 Utility Predictor，相对于 UniV2X 主干计算量较小。

---

# 5. 损失函数

主模型采用：

\[
L
=
L_{\mathrm{mag}}
+
\lambda_{\mathrm{sign}}L_{\mathrm{sign}}.
\]

其中：

\[
L_{\mathrm{mag}}
=
SmoothL1(\hat u_R,u_R),
\]

\[
L_{\mathrm{sign}}
=
BCEWithLogits.
\]

Sign loss 使用：

\[
pos\_weight
=
\frac{N_{\mathrm{negative}}}
{N_{\mathrm{positive}}}
=
3.363217.
\]

训练设置：

| 参数 | 设置 |
|---|---:|
| Optimizer | AdamW |
| LR | \(1\times10^{-3}\) |
| Weight decay | \(1\times10^{-4}\) |
| Batch size | 2048 |
| Dropout | 0.1 |
| SmoothL1 beta | 1.0 |
| Gradient clipping | 5.0 |
| Seed | 2026 |
| Max epochs | 60 |
| Early stopping patience | 10 |

---

# 6. \(\lambda_{\mathrm{sign}}\) 消融

分别测试：

\[
\lambda_{\mathrm{sign}}
\in
\{0,\ 0.25,\ 0.5,\ 1.0\}.
\]

Validation predictor-level 结果：

| \(\lambda_{\mathrm{sign}}\) | MAE ↓ | RMSE ↓ | Pearson ↑ | Pred Std | AUROC ↑ | AUPRC ↑ | F1 ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.349916 | 0.773972 | 0.252050 | 0.228112 | — | — | — |
| 0.25 | 0.357796 | 0.762870 | 0.297491 | **0.236328** | 0.718978 | 0.474717 | 0.506727 |
| 0.50 | 0.354367 | 0.762655 | 0.298428 | 0.225122 | 0.722780 | 0.473433 | 0.514498 |
| 1.00 | **0.349234** | **0.761532** | **0.307313** | 0.203924 | **0.729327** | **0.480572** | **0.520930** |

从纯 predictor-level 指标看：

\[
\lambda_{\mathrm{sign}}=1.0
\]

表现最好。

但最终模型选择不根据 proxy metric，而根据实际 Occupancy fusion-level validation。

---

# 7. V2-1C：Occupancy Fusion-level Validation

## 7.1 决策规则

V2-1 主决策只使用 magnitude head：

\[
\boxed{
a_R=
\mathbf 1[\hat U_R>0]
}
\]

融合：

\[
\boxed{
O_R^{V2}
=
O_{v,R}
\lor
(a_R\land O_{i,R})
}
\]

即：

- Ego Occupancy 永远保留；
- predictor 只控制是否接受路侧新增 Occupancy；
- Sign head 仅作为训练阶段 auxiliary supervision；
- 推理阶段不直接依赖 sign threshold。

这一设计保证 V2-1 的逻辑为：

\[
X_R
\rightarrow
\widehat{\Delta IoU_R}
\rightarrow
\mathbf1[
\widehat{\Delta IoU_R}>0
].
\]

---

## 7.2 Validation 完整 Occupancy 结果

Validation：

\[
96\text{ samples},
\]

共：

\[
480\text{ horizons},
\]

其中：

\[
18,555
\]

个 candidate patches。

Dataset-level foreground IoU：

| Strategy | 30m IoU | 100m IoU | Mean |
|---|---:|---:|---:|
| Ego | 0.315315 | 0.336759 | 0.326037 |
| Official OR | 0.229482 | 0.330526 | 0.280004 |
| V1 Oracle | 0.328446 | 0.396584 | 0.362515 |
| **V2 \(\Delta IoU\) Oracle** | **0.330727** | **0.403358** | **0.367042** |

首先说明：

\[
\boxed{
V2\ Oracle > V1\ Oracle
}
\]

100m：

\[
0.396584
\rightarrow
0.403358,
\]

绝对提升：

\[
\boxed{+0.006774}.
\]

这说明 metric-aligned Utility 本身比 V1 的 \(BA-HA\) 具有更高的理论选择上限。

---

## 7.3 不同 \(\lambda_{\mathrm{sign}}\) 的真实融合结果

使用：

\[
\hat U_R>0
\]

作为统一决策规则。

| \(\lambda_{\mathrm{sign}}\) | Accept rate | 30m IoU | 100m IoU | Mean |
|---:|---:|---:|---:|---:|
| 0.00 | 26.01% | **0.312229** | 0.359365 | **0.335797** |
| 0.25 | 34.71% | 0.309688 | 0.358976 | 0.334332 |
| **0.50** | 33.18% | 0.309799 | **0.360369** | 0.335084 |
| 1.00 | 32.21% | 0.306893 | 0.360131 | 0.333512 |

由于 V2 Utility 的主要定义目标是：

\[
\Delta IoU_{100},
\]

因此 V2-1 的主 hyperparameter selection criterion 固定为：

\[
\boxed{
Validation\ 100m\ Occupancy\ IoU
}
\]

最终选择：

\[
\boxed{
\lambda_{\mathrm{sign}}=0.5
}
\]

而不是 predictor-level proxy 指标最高的 \(\lambda=1.0\)。

---

# 8. V2-1 最终主结果

最终采用：

\[
\boxed{
\lambda_{\mathrm{sign}}=0.5
}
\]

以及：

\[
\boxed{
SmoothL1
}
\]

对应 Validation：

\[
Ego_{100m}=0.336759,
\]

\[
OfficialOR_{100m}=0.330526,
\]

\[
\boxed{
V2Learned_{100m}=0.360369
}
\]

\[
V2Oracle_{100m}=0.403358.
\]

---

## 8.1 相对 Ego

绝对提升：

\[
0.360369-0.336759
=
\boxed{+0.023610}.
\]

相对提升约：

\[
\frac{0.023610}{0.336759}
\approx
\boxed{7.01\%}.
\]

---

## 8.2 相对 Official OR

绝对提升：

\[
0.360369-0.330526
=
\boxed{+0.029843}.
\]

相对提升约：

\[
\frac{0.029843}{0.330526}
\approx
\boxed{9.03\%}.
\]

注意：

> 上述结果属于当前 Validation split 上的 Occupancy fusion-level validation，不是最终官方 Test benchmark，也不是 Planning downstream performance。

---

## 8.3 Oracle Headroom

V2 Oracle：

\[
0.403358.
\]

Learned：

\[
0.360369.
\]

剩余：

\[
0.403358-0.360369
=
\boxed{0.042989}.
\]

因此 predictor 仍存在明显 Oracle gap。

这说明：

> V2 Utility 定义具有较高潜力，而当前性能瓶颈主要来自 Utility estimation，而非 Oracle selection rule 本身。

---

# 9. Magnitude Shrinkage

最终主模型 Validation：

\[
std(U)=0.798943,
\]

而：

\[
std(\hat U)=0.225122.
\]

因此预测 magnitude 存在明显收缩：

\[
\boxed{
std(\hat U)\ll std(U)
}
\]

模型能够学习 Utility 的方向和相对排序，但对长尾高价值/高风险候选区域的绝对幅度估计仍偏保守。

这一现象进一步通过 V2-1D loss audit 进行验证。

---

# 10. V2-1D：Magnitude Loss Ablation

为了缓解 magnitude shrinkage，额外测试：

1. MSE；
2. Weighted-SmoothL1。

Weighted-SmoothL1：

\[
L_{\mathrm{mag}}
=
\frac{
\sum_i
w_i
SmoothL1(\hat u_i,u_i)
}{
\sum_iw_i
},
\]

其中：

\[
w_i
=
1+
\min
\left(
\frac{|u_i|}
{Q_{90}^{train}},
1
\right).
\]

---

## 10.1 Predictor-level 结果

| Loss | MAE ↓ | RMSE ↓ | Pearson ↑ | Pred Std | AUROC ↑ | AUPRC ↑ | F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **SmoothL1** | **0.354367** | 0.762655 | 0.298428 | 0.225122 | 0.722780 | 0.473433 | 0.514498 |
| MSE | 0.361137 | 0.758212 | 0.317045 | 0.264931 | 0.714718 | 0.473807 | 0.501219 |
| Weighted-SmoothL1 | 0.358539 | **0.758084** | **0.317692** | **0.267868** | **0.725685** | **0.474446** | **0.518019** |

Weighted-SmoothL1 确实使：

\[
std(\hat U):
0.225122
\rightarrow
0.267868,
\]

并提高 Pearson。

但 proxy metric 改善不能直接作为采用依据。

---

## 10.2 Occupancy fusion-level 结果

| Loss | Accept rate | 30m IoU | 100m IoU | Mean |
|---|---:|---:|---:|---:|
| **SmoothL1** | 33.18% | **0.309799** | **0.360369** | **0.335084** |
| MSE | 38.56% | 0.304270 | 0.358086 | 0.331178 |
| Weighted-SmoothL1 | 38.07% | 0.306619 | 0.359680 | 0.333149 |

相对于 SmoothL1：

### MSE

\[
\Delta IoU_{30m}=-0.005530,
\]

\[
\Delta IoU_{100m}=-0.002283.
\]

### Weighted-SmoothL1

\[
\Delta IoU_{30m}=-0.003181,
\]

\[
\Delta IoU_{100m}=-0.000690.
\]

因此：

\[
\boxed{
\text{V2-1D = NO-GO}
}
\]

结论：

> 单纯扩大预测 magnitude 的动态范围，并不能保证实际 Occupancy fusion 改善；Utility regression proxy metric 与最终集合式选择结果之间仍存在 objective gap。

最终继续采用普通 SmoothL1。

---

# 11. V2-1E：Set-level IoU-aligned Learning Exploration

## 11.1 动机

patch-wise training 独立优化：

\[
SmoothL1(\hat U_i,U_i),
\]

但推理时实际一次接受的是一组候选：

\[
S=
\{R_i:\hat U_i>0\}.
\]

因此尝试定义 soft acceptance：

\[
a_i
=
\sigma
\left(
\frac{\hat U_i}{T}
\right)
\]

并构造 Set-level IoU：

\[
IoU_{\mathrm{soft}}
=
\frac{
TP+\sum_i a_iBA_i
}{
D+\sum_i a_iHA_i
}.
\]

Set loss：

\[
L_{\mathrm{set}}
=
-
100
\left(
IoU_{\mathrm{soft}}
-
IoU_{\mathrm{ego}}
\right).
\]

最终：

\[
L
=
L_{\mathrm{mag}}
+
0.5L_{\mathrm{sign}}
+
\lambda_{\mathrm{set}}L_{\mathrm{set}}.
\]

---

## 11.2 Group Dataset

以一个：

\[
\text{sample-horizon}
\]

作为一个 candidate set。

| Split | Candidate patches | Non-empty groups | Mean candidates/group |
|---|---:|---:|---:|
| Train | 86,143 | 1,721 | 50.05 |
| Val | 18,555 | 438 | 42.36 |
| Test | 14,750 | 410 | 35.98 |

Candidate ordering 和原 V2-1 数据完全一致：

\[
\boxed{\text{PASS}}
\]

---

## 11.3 Group-aware Control

首先令：

\[
\lambda_{\mathrm{set}}=0.
\]

从原 V2-1 checkpoint 继续 group-aware refinement。

最终最优仍为：

\[
\boxed{epoch=0}.
\]

说明：

> 单纯改变 batching / refinement 过程本身并不能改善模型。

该 control 排除了 group-aware batching 对结果的混杂影响。

---

## 11.4 Set-IoU Result

测试：

\[
\lambda_{\mathrm{set}}=0.02,
\qquad
T=0.25.
\]

完整 Validation epoch sweep：

| Epoch | 30m IoU | 100m IoU | Mean | Accept rate |
|---:|---:|---:|---:|---:|
| 0 | 0.309799 | 0.360369 | 0.335084 | 33.18% |
| 1 | 0.311565 | 0.359466 | 0.335515 | 32.66% |
| 2 | 0.307203 | 0.359029 | 0.333116 | 40.83% |
| **3** | **0.309827** | **0.360884** | 0.335355 | 36.61% |
| 4 | 0.309357 | 0.360215 | 0.334786 | 36.43% |
| 5 | 0.311284 | 0.360349 | **0.335816** | 33.93% |

按照预先固定的主指标：

\[
Validation\ 100m\ IoU,
\]

epoch 3 最优：

\[
0.360369
\rightarrow
0.360884.
\]

绝对提升：

\[
\boxed{+0.000515}.
\]

30m：

\[
0.309799
\rightarrow
0.309827,
\]

变化：

\[
\boxed{+0.000027}.
\]

---

## 11.5 V2-1E 结论

Set-level metric alignment 提供了方向一致的提升，但幅度有限：

\[
\boxed{
+0.000515\ 100m\ IoU
}
\]

仅相当于：

\[
0.0515
\]

个 IoU 百分点。

因此当前不将 Set-IoU 纳入最终主模型，以避免为了极小 Validation 增益增加：

- group-aware training；
- temperature \(T\)；
- \(\lambda_{\mathrm{set}}\)；
- set-level optimization；
- 更复杂的训练与模型选择流程。

因此：

\[
\boxed{
\text{V2-1E = exploratory / marginal improvement}
}
\]

而非最终主方法。

---

# 12. V2-1 最终冻结方案

最终 V2-1 采用：

\[
\boxed{
U_R^{V2}
=
\Delta IoU_R^{100}
}
\]

Predictor：

\[
\boxed{
127\text{-D}
\rightarrow
0.495M\ Dual-head\ MLP
}
\]

训练目标：

\[
\boxed{
L
=
SmoothL1(\hat u,u)
+
0.5L_{\mathrm{sign}}
}
\]

推理：

\[
\boxed{
a_R=
\mathbf1[\hat U_R>0]
}
\]

融合：

\[
\boxed{
O_R^{V2}
=
O_{v,R}
\lor
(a_R\land O_{i,R})
}
\]

最终主 checkpoint：

`checkpoints/main_lambda_0p50/best.pth`

最终主结果：

\[
\boxed{
100m\ IoU=0.360369
}
\]

Validation：

\[
OfficialOR=0.330526
\]

\[
Ego=0.336759
\]

\[
V2Learned=0.360369
\]

\[
V2Oracle=0.403358.
\]

---

# 13. V2-1 的核心实验结论

## 结论一：V2 Utility 定义比 V1 更合理

V1 Oracle：

\[
0.396584
\]

V2 Oracle：

\[
0.403358.
\]

说明与 Occupancy IoU 对齐的边际 Utility 提供了更高的理论选择上限。

---

## 结论二：连续 Utility 是可学习的

最终 predictor 在 held-out Validation scene 上：

\[
Pearson=0.298428
\]

并且 sign auxiliary task：

\[
AUROC=0.722780,
\]

\[
AUPRC=0.473433.
\]

说明从推理时可获得的 soft Occupancy 特征中，确实可以学习 candidate patch 的合作价值。

---

## 结论三：预测 Utility 能真正改善 Occupancy fusion

100m：

\[
OfficialOR:
0.330526
\]

\[
V2Learned:
0.360369.
\]

绝对提升：

\[
\boxed{+0.029843}.
\]

这说明：

> 路侧 Occupancy 并非始终具有正合作价值；通过显式估计候选区域的 marginal cooperative utility，可以避免部分 harmful infrastructure additions，并改善最终稠密 Occupancy fusion。

---

## 结论四：更好的 regression proxy 不一定意味着更好的 fusion

MSE / Weighted-SmoothL1 提高了：

- Pearson；
- prediction variance；
- 部分 classification proxy metric；

但最终 Occupancy IoU 下降。

因此：

\[
\boxed{
\text{utility prediction metric}
\neq
\text{final fusion quality}
}
\]

论文中应强调最终方法选择必须由实际 Occupancy fusion performance 验证。

---

# 14. 论文中的方法定位

V2-1 不应该表述为：

> “提出了一个更大的 MLP。”

真正的方法贡献是：

\[
\boxed{
\text{Metric-aligned Cooperation Utility Estimation}
}
\]

以及：

\[
\boxed{
\text{Magnitude-aware Utility Learning}
}
\]

核心逻辑：

\[
\text{Soft occupancy evidence}
\rightarrow
\text{Marginal utility estimation}
\rightarrow
\text{Selective cooperation}.
\]

0.495M MLP 只是 Utility estimator 的具体实现，而不是主要创新本身。

---

# 15. 与 SQSA 的论文统一关系

第一项工作 SQSA 解决：

\[
\boxed{
\text{谁和谁对应？}
}
\]

即稀疏 Agent Query / instance-level cooperative reliability。

第二项工作 STCV-Occ 解决：

\[
\boxed{
\text{已经空间对齐以后，哪些路侧 Occupancy 值得加入？}
}
\]

即稠密 scene-level cooperative reliability。

因此论文整体可以统一为：

\[
\boxed{
\text{Multi-level Reliable Cooperative Fusion}
}
\]

具体为：

\[
\text{Sparse instance association reliability}
+
\text{Dense occupancy cooperation reliability}.
\]

---

# 16. 论文写作推荐表述

## 16.1 方法动机

可写：

> 现有车路协同 Occupancy 融合通常默认空间对齐后的路侧预测具有一致的正向合作价值，但前期分析表明，路侧新增 Occupancy 中同时存在能够弥补车端漏检的 beneficial additions 与引入额外误检的 harmful additions。固定 OR 融合无法区分二者，因而可能导致合作后 Occupancy 质量下降。为此，本文进一步从合作边际价值角度建模路侧信息可靠性。

---

## 16.2 Utility 定义

可写：

> 与基于正确新增单元和错误新增单元数量差值的启发式效用不同，本文依据 occupied-class IoU 的变化推导候选区域的边际合作价值，将其定义为接受该区域路侧 Occupancy 前后的 IoU 增量。该定义能够根据当前车端 Occupancy 状态自适应衡量 beneficial addition 与 harmful addition 对最终评价指标的不同影响。

---

## 16.3 Magnitude-aware Learning

可写：

> 为进一步保留不同候选区域之间合作价值幅度的差异，本文将二值有益性判断扩展为连续边际效用估计。网络以车端与路侧 soft Occupancy 概率、空间位置、预测时间步及邻域上下文作为输入，同时学习连续效用回归和效用符号判别。其中，连续回归用于估计合作收益大小，辅助符号监督则用于增强模型对正负合作价值边界的判别能力。

---

## 16.4 Selective Fusion

可写：

> 推理阶段无需真实 Occupancy 标注，仅依据预测边际效用决定是否接受对应区域的路侧新增 Occupancy。对于预测效用为正的区域执行合作融合，否则保持车端 Occupancy 不变，从而形成 Ego-preserving selective cooperative fusion。

---

# 17. 当前可以声称的内容

可以写：

> Validation 实验表明，与固定 OR 融合相比，学习得到的合作效用能够更加有效地筛选路侧 Occupancy 信息，并提高 100 m 范围内的 Occupancy IoU。

可以写：

> Metric-aligned Utility Oracle 高于原 BA-HA Oracle，说明新的边际效用定义具有更高的理想选择上限。

可以写：

> Sign auxiliary supervision 能够提升连续 Utility representation 的学习质量，但最终权重需要依据实际 Occupancy fusion performance，而非单一 proxy metric 选择。

可以写：

> MSE 和 magnitude-weighted loss 虽能够改善连续效用预测的部分统计指标，但未转化为更高的 Occupancy IoU，说明 utility regression quality 与最终集合式融合效果之间存在一定目标偏差。

---

# 18. 当前不能过度声称的内容

目前不要写：

> “取得统计显著提升。”

尚未进行统计显著性检验。

不要写：

> “全面优于 UniV2X。”

当前 V2-1C 是固定 scene split 上的 Validation fusion-level experiment，不是最终完整官方 Test benchmark。

不要写：

> “Occupancy 改善必然提升 Planning。”

该问题正在 V2-2 Planning Task-consistency Audit 中单独验证。

不要写：

> “解决了 uncertainty-aware decision。”

当前决策仍然是：

\[
\hat U>0.
\]

真正的 uncertainty-aware adaptive decision 属于后续 V2-3。

不要把：

\[
0.360884
\]

作为当前最终主模型结果。

该结果来自 V2-1E exploratory Set-IoU refinement，仅比主模型提高：

\[
0.000515,
\]

因此当前主方法仍采用：

\[
\boxed{0.360369}.
\]

---

# 19. 当前主要局限

### 1. Utility prediction 仍存在明显 Oracle gap

\[
0.360369
\quad vs\quad
0.403358.
\]

说明仍有大量合作价值尚未被准确预测。

### 2. Magnitude shrinkage

\[
std(\hat U)=0.225122
\ll
std(U)=0.798943.
\]

当前模型对极端 Utility magnitude 的估计偏保守。

### 3. 输入仍是人工组织的 127-D 特征

虽然具有明确语义和较低计算成本，但可能限制 Utility 的可预测上限。

### 4. 当前仍采用固定 zero-crossing decision

\[
a_R=\mathbf1[\hat U_R>0].
\]

没有显式考虑预测不确定性和风险。

### 5. 尚未验证 Planning task consistency

当前只能证明：

\[
\text{Occupancy fusion improves}.
\]

尚不能证明：

\[
\text{Planning improves}.
\]

---

# 20. 后续工作

V2-1 完成后，研究主线不再继续调 regression loss。

后续按照：

\[
\boxed{
\text{V2-2 Planning Task-consistency Audit}
}
\]

验证：

\[
\text{Improved Occupancy}
\Rightarrow?
\text{Improved Planning}.
\]

随后进入：

\[
\boxed{
\text{V2-3 Uncertainty-aware Adaptive Decision}
}
\]

重点解决：

\[
\hat U>0
\]

这一固定决策规则的局限，通过预测置信度、风险或动态 acceptance policy 进一步提高合作可靠性。

最后进行：

\[
\boxed{
\text{V2-4 Final Official Evaluation}
}
\]

在方法和超参数完全冻结后，再使用 Test / official evaluator 完成最终测试。

---

# 21. V2-1 最终状态

| 项目 | 状态 |
|---|---|
| Metric-aligned Utility definition | **PASS** |
| Continuous magnitude target audit | **PASS** |
| Magnitude-aware utility learning | **PASS** |
| Sign auxiliary supervision | **PASS** |
| \(\lambda_{\mathrm{sign}}\) selection | **0.5** |
| Occupancy fusion-level validation | **PASS** |
| MSE loss improvement | **NO-GO** |
| Weighted-SmoothL1 improvement | **NO-GO** |
| Set-level IoU refinement | Marginal gain / not adopted |
| Final V2-1 model | **Frozen** |
| Test evaluation | **Not yet used for model selection** |
| Planning consistency | **Next stage** |

---

# 22. 最终一句话总结

> **STCV-Occ V2-1 将路侧 Occupancy 的合作可靠性从二值有益性判断扩展为与最终 Occupancy IoU 对齐的连续边际效用估计，并通过 magnitude regression 与 sign auxiliary supervision 学习候选区域的合作价值。在验证集上，该方法能够在不依赖推理阶段真实标注的情况下选择性接收路侧新增 Occupancy，使 100 m Occupancy IoU 从 Official OR 的 0.330526 提升至 0.360369，同时保留进一步向 Planning task consistency 与 uncertainty-aware decision 扩展的空间。**