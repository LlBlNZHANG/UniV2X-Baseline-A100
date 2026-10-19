# V2-2 Planning Task-Consistency Audit 总结

## 1. 阶段目标

V2-2 的核心目标不是继续提升 Occupancy 指标，而是验证：

> **STCV-Occ 在 Occupancy 层面的改进，是否能够进一步传递到下游 Planning，并减少车路协同信息对规划结果造成的负迁移。**

前期 V2-0 和 V2-1 已经证明，基于 metric-aligned utility 的选择性 Occupancy 融合能够改善 Occupancy IoU。但 Occupancy 指标提升并不必然带来 Planning 性能同步提升，因此需要进一步审计 UniV2X 中 Occupancy 与 Planning 的实际数据流，并通过受控反事实实验分析不同 Occupancy fusion strategy 对最终规划结果的影响。

V2-2 分为三个阶段：

1. **V2-2A：Planning Dataflow & Runtime Audit**
2. **V2-2B：Raw Planning Cache Construction**
3. **V2-2C：Controlled Counterfactual Planning Evaluation**

最终需要回答三个问题：

- Occupancy 在 UniV2X 中究竟如何影响 Planning；
- Official OR、V1、V2 对最终 Planning 的影响是否不同；
- Occupancy utility 与 Planning utility 是否保持一致。

---

# 2. V2-2A：Planning Dataflow & Runtime Audit

## 2.1 审计目的

首先对 UniV2X 的 Planning 数据流进行源代码级审计，明确 Occupancy 是否直接进入 Planning 网络，以及其具体作用阶段。

重点确认：

1. Occupancy 是否参与原始轨迹预测；
2. Occupancy 影响 Planning 的具体位置；
3. Occupancy horizon 与 Planning step 的对应关系；
4. Planning Metric 实际评估哪个轨迹；
5. 后续反事实实验应该在哪一层控制变量。

---

## 2.2 Occupancy 对 Planning 的真实作用位置

审计结果表明，Planning Head 首先输出原始规划轨迹：

\[
T_{\mathrm{raw}}
\]

即：

```text
sdc_traj_all
```

在推理阶段，当：

```text
use_col_optim = True
```

时，才依次执行：

```text
collision_optimization
        ↓
drivable_optimization
```

最终得到：

\[
T_{\mathrm{final}}
\]

因此整体 Planning 数据流为：

\[
T_{\mathrm{raw}}
\xrightarrow[\text{Occupancy}]
{\text{Collision Optimization}}
T_{\mathrm{col}}
\xrightarrow[\text{Drivable Map}]
{\text{Drivable Optimization}}
T_{\mathrm{final}}
\]

由此得到关键结论：

\[
\boxed{
\text{Occupancy 不直接参与原始 Planning 网络的轨迹预测}
}
\]

而是作为推理阶段的环境约束，作用于：

\[
\boxed{
\text{test-time trajectory post-processing}
}
\]

特别是 Collision Optimization。

因此，如果要严格研究不同 Occupancy fusion strategy 对 Planning 的影响，必须：

> 固定相同的原始轨迹 \(T_{\mathrm{raw}}\)，仅改变输入 collision optimizer 的 Occupancy。

这构成 V2-2C controlled counterfactual experiment 的理论基础。

---

## 2.3 Occupancy Horizon 与 Planning Horizon 对应关系

当前 Occupancy 共包含 5 个时间步：

\[
h=0,1,2,3,4
\]

而 Planning 预测：

\[
10
\]

个未来轨迹点。

官方 Collision Optimization 中采用：

\[
\text{cur\_t}=\min(t+1,\text{occ\_horizon}-1)
\]

当前：

\[
\text{occ\_horizon}=5
\]

因此 Planning step 与 Occupancy horizon 的实际映射为：

| Planning step | Occupancy horizon |
|---|---:|
| \(t_0\) | \(h_1\) |
| \(t_1\) | \(h_2\) |
| \(t_2\) | \(h_3\) |
| \(t_3\) | \(h_4\) |
| \(t_4\) | \(h_4\) |
| \(t_5\) | \(h_4\) |
| \(t_6\) | \(h_4\) |
| \(t_7\) | \(h_4\) |
| \(t_8\) | \(h_4\) |
| \(t_9\) | \(h_4\) |

即：

\[
[1,2,3,4,4,4,4,4,4,4]
\]

需要特别注意：

\[
\boxed{h=0\text{ 不进入当前官方 Planning collision optimization}}
\]

这意味着当前帧 Occupancy 即使得到改善，也不一定能够直接影响最终规划轨迹。

---

## 2.4 Collision Optimization 的局部敏感区域

官方 collision optimizer 的关键参数为：

```text
planning_steps = 10
occ_n_future_only_occ = 4
occ_filter_range = 5 m
dt = 0.5 s
sigma = 1
alpha_collision = 5
```

其中最关键的是：

\[
\boxed{\text{occ\_filter\_range}=5\text{ m}}
\]

也就是说，官方 Planning 并不会使用整个 Occupancy map，而只关注：

> **规划轨迹附近约 5 m 范围内的 Occupancy 信息。**

因此：

\[
\text{Global Occupancy IoU improvement}
\]

并不等价于：

\[
\text{Planning-sensitive Occupancy improvement}
\]

这也是后续出现“Occupancy 明显改善，但 Planning 改善有限”的重要机制原因。

---

## 2.5 Occupancy 与 Planning 网格尺度

当前 G0 Occupancy cache 对应：

```text
200 × 200
```

Occupancy nominal range 为：

\[
[-51.2,51.2]
\]

因此 nominal grid resolution 为：

\[
0.512\text{ m}
\]

但官方 Planning collision optimization 内部实际采用：

\[
0.5\text{ m}
\]

作为栅格到真实坐标的转换尺度。

V2-2 不修改这一实现，而严格复用官方 Planning 方法。

因此本阶段的实验结论应描述为：

> **在 UniV2X 官方 Planning post-processing 机制下，不同 Occupancy fusion strategy 对最终规划结果的影响。**

而不是重新定义一个新的 Planning evaluator。

---

# 3. V2-2B：Raw Planning Cache Construction

## 3.1 实验目的

为了严格执行 controlled counterfactual comparison，需要获得完全不经过 Occupancy collision optimization 的原始规划轨迹：

\[
T_{\mathrm{raw}}
\]

因此构建专门配置：

```text
projects/configs_e2e_univ2x/
univ2x_coop_e2e_stcv_planraw.py
```

该配置与原 STCV config 的关键区别只有：

```text
use_col_optim=True
```

修改为：

```text
use_col_optim=False
```

由于 Collision Optimization 和 Drivable Optimization 位于同一推理分支中，因此该设置可以直接获得未经后处理的：

\[
T_{\mathrm{raw}}
\]

---

## 3.2 Planning Cache 内容

最终成功导出完整：

\[
\boxed{675/675}
\]

个样本的 Raw Planning Cache。

每个样本保存：

```text
T_raw
drivable_pred
sdc_planning_eval
sdc_planning_mask_eval
gt_segmentation_eval
drivable_gt
command
sample_idx
scene_token
timestamp
planning_steps
use_col_optim
```

主要 shape：

```text
T_raw                  : [1, 10, 2]
sdc_planning_eval      : [1, 10, 2]
sdc_planning_mask_eval : [1, 10, 2]
gt_segmentation_eval   : [1, 10, 200, 200]
drivable_pred          : [200, 200]
drivable_gt            : [200, 200]
```

---

## 3.3 Cache 一致性检查

Planning Cache 与 G0 Occupancy Cache 对以下字段进行逐样本匹配：

```text
sample_idx
scene_token
timestamp
```

最终：

\[
\boxed{675/675\text{ PASS}}
\]

同时确认：

```text
planning_steps = 10
use_col_optim = 0
T_raw shape = [1,10,2]
```

因此可以在完全相同的：

\[
T_{\mathrm{raw}}
\]

基础上，分别输入 Ego、OfficialOR、V1、V2 Occupancy。

这样可以保证不同策略之间唯一变化因素为：

\[
\boxed{\text{Occupancy fusion result}}
\]

从而形成严格的反事实规划实验。

---

# 4. V2-2C：Controlled Counterfactual Planning Evaluation

## 4.1 实验设计

实验固定：

```text
Raw planning trajectory T_raw
Drivable prediction
Planning GT
Planning mask
Collision optimizer
Drivable optimizer
Optimizer hyperparameters
PlanningMetric
```

唯一改变：

\[
\boxed{\text{Occupancy input}}
\]

分别比较四种方法。

---

## 4.2 Ego Occupancy

定义：

\[
O_{\mathrm{Ego}}=O_v
\]

这里的 \(O_v\) 指 OccFusion 之前的 ego-side Occupancy prediction。

需要注意：

> \(O_v\) 并不应严格称为完全独立的 vehicle-only prediction，因为其上游 representation 可能已经受到 AgentFusion 的影响。

因此论文中更准确的表述应为：

> **pre-occupancy-fusion ego prediction**

而不是严格意义上的 vehicle-only baseline。

---

## 4.3 Official OR Fusion

UniV2X 当前 Occupancy fusion：

\[
O_{\mathrm{Official}}
=
O_v\lor O_i
\]

其中 \(O_i\) 为经过空间对齐后的 infrastructure occupancy。

这种方法将所有路侧新增 Occupancy 直接加入 Ego Occupancy。

---

## 4.4 STCV-Occ V1

V1 使用：

\[
O_{\mathrm{V1}}
=
O_v
\lor
(a_R^{V1}\land O_i)
\]

基于原始 binary utility predictor 判断 candidate patch 是否接受。

固定 threshold：

\[
\tau=0.70
\]

---

## 4.5 STCV-Occ V2

V2 使用 metric-aligned magnitude-aware utility：

\[
\hat U_R^{occ}
\]

当前冻结决策为：

\[
a_R^{V2}
=
\mathbf 1[
\hat U_R^{occ}>0
]
\]

从而：

\[
O_{\mathrm{V2}}
=
O_v
\lor
(a_R^{V2}\land O_i)
\]

---

# 5. V2-2C 数据划分

本阶段仅使用已经冻结的 V2-1 Val split。

规模：

```text
4 scenes
96 samples
18555 candidate patches
```

Val scene tokens：

```text
0014
0020
0063
0075
```

Candidate 数量：

\[
\boxed{18555}
\]

与原 V2-1 Val 数据完全一致。

---

# 6. Candidate Decision Reconstruction

## 6.1 V1

V1 接受：

\[
5691
\]

个 candidate。

接受率：

\[
\frac{5691}{18555}
=
30.670978\%
\]

---

## 6.2 V2

V2 接受：

\[
6156
\]

个 candidate。

接受率：

\[
\frac{6156}{18555}
=
33.177041\%
\]

最终 reconstruction：

\[
\boxed{
18555\text{ candidates}
}
\]

\[
\boxed{
V1=5691
}
\]

\[
\boxed{
V2=6156
}
\]

全部与冻结的历史结果一致。

---

# 7. V2 Numerical Execution Audit

在 V2-2C 中发现 CPU 与 CUDA 在极少数接近：

\[
\hat U_R^{occ}=0
\]

的 candidate 上存在 floating-point numerical difference。

审计结果：

```text
CPU sample-wise : 6160 accepted
CPU all         : 6160 accepted
CPU batch=2048  : 6160 accepted
CUDA batch=2048 : 6156 accepted
```

其中：

\[
4/18555
\]

个 candidate 出现 decision flip。

对应 candidate index：

```text
1050
5807
7077
16297
```

最大 CPU-CUDA magnitude difference：

\[
6.96\times10^{-4}
\]

平均差异：

\[
4.09\times10^{-5}
\]

因此最终冻结 V2 canonical inference path 为：

```text
CUDA
batch_size = 2048
decision = magnitude > 0
```

需要强调：

> 该差异仅涉及 4/18555 个 candidate，属于 hard-zero decision boundary 附近的数值敏感性，而不是模型、candidate definition、feature construction 或 checkpoint 不一致。

这一现象同时说明，在后续设计中引入 uncertainty / margin-aware decision 具有合理性。

---

# 8. Planning Evaluation Results

## 8.1 L2 Distance

| Method | Mean L2 | 1 s | 2 s | 3 s | 4 s | 5 s |
|---|---:|---:|---:|---:|---:|---:|
| Ego | **2.803602** | 1.282824 | 2.323013 | 2.986778 | 3.680300 | 4.976847 |
| OfficialOR | 2.845870 | 1.424733 | 2.328117 | 2.986786 | 3.680300 | **4.975702** |
| V1 | 2.804704 | **1.278777** | 2.323011 | 2.986778 | 3.680300 | 4.976847 |
| V2 | 2.804198 | 1.283341 | 2.323013 | 2.986778 | 3.680300 | 4.976847 |

整体来看：

\[
L2_{\mathrm{OfficialOR}}
>
L2_{\mathrm{V1}}
\approx
L2_{\mathrm{V2}}
\approx
L2_{\mathrm{Ego}}
\]

---

## 8.2 Mean L2 对比

### OfficialOR vs Ego

\[
2.845870-2.803602
=
+0.042268
\]

即：

\[
\boxed{
\text{OfficialOR 相比 Ego 的 Mean L2 恶化约 }1.51\%
}
\]

说明简单 OR 融合可能将有害路侧 Occupancy 传播到 Planning collision optimization，从而造成规划负迁移。

---

### V2 vs OfficialOR

\[
2.804198-2.845870
=
-0.041672
\]

即 V2 相比 OfficialOR：

\[
\boxed{
\text{Mean L2 降低约 }1.46\%
}
\]

说明选择性 Occupancy fusion 能够明显缓解 naïve OR cooperation 引入的 Planning degradation。

---

### V2 vs Ego

\[
2.804198-2.803602
=
+0.000596
\]

相对变化约：

\[
+0.021\%
\]

因此：

\[
\boxed{
\text{V2 并未在当前 Val 上实现 Planning L2 优于 Ego}
}
\]

两者结果非常接近，但不能表述为 V2 改善了 Ego Planning。

---

### V2 vs V1

\[
2.804198-2.804704
=
-0.000506
\]

V2 数值上略优于 V1，但差异非常小。

因此只能描述为：

> V2 在该 Val split 上获得了轻微的数值改善。

不能将该差异解释为统计显著提升。

---

# 9. Collision / Out-of-Drivable Metrics

## 9.1 Ego

```text
obj_col mean      = 0.002083
obj_box_col mean  = 0.005208
obj_out mean      = 0.002083
obj_box_out mean  = 0.004167
```

分时间点：

```text
obj_col
1s 0
2s 0
3s 0.010417
4s 0.010417
5s 0

obj_box_col
1s 0
2s 0
3s 0.020833
4s 0.010417
5s 0.020833

obj_out
1s 0
2s 0
3s 0
4s 0
5s 0.010417

obj_box_out
1s 0
2s 0
3s 0
4s 0.010417
5s 0.020833
```

---

## 9.2 OfficialOR

```text
obj_col mean      = 0.002083
obj_box_col mean  = 0.004167
obj_out mean      = 0.002083
obj_box_out mean  = 0.004167
```

其中：

```text
obj_box_col
1s 0
2s 0
3s 0.020833
4s 0.010417
5s 0.010417
```

因此需要注意：

\[
\boxed{
\text{OfficialOR 并不是在所有 Planning metric 上都差于 Ego/V2}
}
\]

OfficialOR 在 `obj_box_col` 上反而略低。

因此论文中不能写：

> Official OR 在所有 Planning 指标上均退化。

更准确的结论应为：

> Official OR 在 Mean L2 上产生明显负迁移，而 collision/out 指标由于事件稀疏，方法间差异较小且并不完全一致。

---

## 9.3 V1

V1 的：

```text
obj_col
obj_box_col
obj_out
obj_box_out
```

与 Ego 基本一致。

---

## 9.4 V2

V2 的：

```text
obj_col
obj_box_col
obj_out
obj_box_out
```

同样与 Ego 基本一致。

因此在当前 Val split 上，没有证据支持：

\[
\boxed{
\text{V2 显著改善 collision / out metrics}
}
\]

---

# 10. Planning Post-processing Trigger Statistics

进一步统计 Occupancy 对 Collision Optimization 的触发情况。

## Ego

```text
collision_changed : 32
drivable_changed  : 9
final_changed     : 34
```

## OfficialOR

```text
collision_changed : 44
drivable_changed  : 9
final_changed     : 46
```

## V1

```text
collision_changed : 39
drivable_changed  : 9
final_changed     : 41
```

## V2

```text
collision_changed : 36
drivable_changed  : 9
final_changed     : 38
```

可以观察到：

\[
32 < 36 < 39 < 44
\]

OfficialOR 最容易触发 Collision Optimization。

相比 OfficialOR：

\[
44\rightarrow36
\]

V2 减少了：

\[
8
\]

个 collision-optimization trajectory changes。

这进一步说明：

> 无选择地加入 infrastructure occupancy 会更频繁地扰动原始规划轨迹，而 STCV-Occ 能够过滤一部分产生无效或有害规划约束的路侧 Occupancy。

但：

\[
36>32
\]

说明 V2 相比 Ego 仍然引入了一部分额外的轨迹调整。

---

# 11. V2-2 核心实验结论

V2-2 最重要的结论不是：

> STCV-Occ 提升了 Planning。

当前结果并不支持这一表述。

正确结论是：

\[
\boxed{
\text{Naïve cooperative Occupancy fusion can introduce Planning negative transfer}
}
\]

OfficialOR 将全部路侧 Occupancy 直接加入 Ego：

\[
O_v\lor O_i
\]

导致 Mean L2：

\[
2.803602
\rightarrow
2.845870
\]

出现明显恶化。

而 STCV-Occ V2：

\[
2.804198
\]

基本恢复到 Ego 水平。

因此更加准确的结论为：

\[
\boxed{
\text{STCV-Occ 能够显著缓解 naïve OR cooperation 带来的 Planning negative transfer}
}
\]

而不是：

\[
\boxed{
\text{STCV-Occ 提升 Planning performance}
}
\]

---

# 12. Occupancy Improvement 与 Planning Improvement 不一致

V2-1 已经证明，STCV-Occ V2 在 Occupancy IoU 上具有明显优势。

但是 V2-2 显示：

\[
\text{Occupancy improvement}
\not\Rightarrow
\text{Planning improvement}
\]

原因主要包括以下三点。

---

## 12.1 Planning 仅使用局部 Occupancy

Collision Optimization 只使用轨迹附近：

\[
5\text{ m}
\]

范围内的 Occupancy。

因此大量能够改善全局 Occupancy IoU 的 candidate 根本不会进入 Planning。

---

## 12.2 当前 Occupancy Utility 是 perception-oriented

V2-1 的 utility：

\[
U_R^{occ}
\]

本质上衡量 candidate 对 Occupancy IoU 的贡献。

但 Planning 真正关心的是：

\[
U_R^{plan}
\]

即 candidate 是否能够改善最终轨迹。

因此：

\[
\boxed{
U_R^{occ}
\neq
U_R^{plan}
}
\]

二者存在 task-objective mismatch。

---

## 12.3 Planning Post-processing 具有非线性

Collision Optimization 是非线性优化过程。

因此：

\[
\text{Occupancy IoU 增益}
\]

不会线性映射为：

\[
\text{Planning L2 增益}
\]

即使一个 candidate 在 perception 层面是正确的，它对 collision optimizer 施加的空间约束也可能导致轨迹偏移，从而增加 L2。

---

# 13. V2-2 对 V2-3 的直接启示

V2-2 表明，仅依赖 Occupancy utility：

\[
\hat U_R^{occ}
\]

不足以保证 candidate 对 Planning 有益。

因此第三阶段不能简单继续调整：

```text
fixed threshold
```

而应该进一步考虑：

\[
\boxed{
\text{Planning relevance + Planning utility + uncertainty}
}
\]

即：

\[
a_R
=
g(
\hat U_R^{occ},
\hat U_R^{plan},
d_R^{plan},
\mathcal U_R
)
\]

其中：

- \(\hat U_R^{occ}\)：Occupancy utility；
- \(\hat U_R^{plan}\)：Planning utility；
- \(d_R^{plan}\)：candidate 与规划轨迹的空间关系；
- \(\mathcal U_R\)：预测不确定性或 decision margin。

---

# 14. 对第二个工作的整体意义

V2-2 将 STCV-Occ 从单纯的 Occupancy metric optimization 扩展到了：

> **task-consistent cooperative perception**

其核心科学问题由：

> 哪些 infrastructure Occupancy 能提高 Occupancy IoU？

进一步扩展为：

> 哪些 infrastructure Occupancy 不仅在 perception 层面有价值，而且不会给下游 Planning 带来负迁移？

因此第二个工作的逻辑可以逐渐形成：

\[
\text{V1}
\rightarrow
\text{Selective Cooperation}
\]

\[
\text{V2-0}
\rightarrow
\text{Metric-aligned Utility}
\]

\[
\text{V2-1}
\rightarrow
\text{Magnitude-aware Utility Learning}
\]

\[
\text{V2-2}
\rightarrow
\text{Planning Task-Consistency Diagnosis}
\]

\[
\text{V2-3}
\rightarrow
\text{Planning-aware Adaptive Cooperative Decision}
\]

从而形成：

\[
\boxed{
\text{Perception Utility}
\rightarrow
\text{Task Utility}
\rightarrow
\text{Reliable Cooperative Decision}
}
\]

---

# 15. 论文中可以使用的核心结论

## 15.1 中文正式版

> 进一步的任务一致性实验表明，Occupancy 层面的性能提升并不会直接转化为规划性能增益。UniV2X 中的 Occupancy 主要通过推理阶段的碰撞优化模块影响最终规划轨迹，而该模块仅对规划路径邻域内的占用信息敏感。直接采用 OR 融合全部路侧 Occupancy 会使平均规划 L2 由 2.8036 增加至 2.8459，表明未经筛选的协同信息可能向下游规划传播负迁移。相比之下，STCV-Occ V2 将平均 L2 降低至 2.8042，基本恢复至 Ego 基线水平，说明基于合作价值的选择性融合能够有效抑制由不可靠路侧占用信息引起的规划性能退化。然而，V2 尚未进一步优于 Ego，说明感知层面的 Occupancy utility 与规划任务实际需要的 Planning utility 并不完全一致，为后续引入任务相关的自适应协同决策提供了实验依据。

---

## 15.2 更凝练的论文版

> Planning task-consistency experiments reveal that improved Occupancy prediction does not necessarily translate into better downstream planning. Naïve OR fusion increases the mean planning L2 from 2.8036 to 2.8459, indicating that indiscriminate cooperative occupancy fusion may propagate harmful infrastructure information into trajectory post-processing. In contrast, STCV-Occ V2 reduces the mean L2 to 2.8042, largely recovering the Ego-level planning performance. These results demonstrate that selective cooperation effectively mitigates planning negative transfer, while the remaining gap between occupancy utility and planning utility motivates task-aware adaptive cooperative decision making.

---

# 16. 论文中不要使用的表述

当前 V2-2 结果不支持以下结论：

```text
STCV-Occ significantly improves downstream planning.
```

不要写：

```text
V2 significantly outperforms Ego in Planning.
```

不要写：

```text
STCV-Occ improves all planning metrics.
```

不要写：

```text
Better Occupancy directly leads to better Planning.
```

更准确的表达应该是：

```text
STCV-Occ mitigates the planning degradation caused by naïve cooperative occupancy fusion.
```

或者：

```text
Selective occupancy cooperation suppresses negative transfer from infrastructure perception to downstream planning.
```

---

# 17. 当前 V2-2 的实验边界

本阶段结果属于：

\[
\boxed{
\text{V2-1 Val subset task-consistency audit}
}
\]

当前使用：

```text
4 Val scenes
96 valid samples
```

因此它不是完整 675 个样本上的最终官方 Planning benchmark。

同时，V2-2C 使用固定：

\[
T_{\mathrm{raw}}
\]

进行 counterfactual replay。

这种设计的优势是因果关系清晰：

> 只比较 Occupancy 的变化如何影响 Planning post-processing。

但其结论应限定为：

> Occupancy fusion 对官方 test-time Planning post-processing 的影响。

不能扩展为：

> Occupancy fusion 改善了整个 Planning network 的学习能力。

---

# 18. V2-2 最终结论

V2-2 完成了从 Occupancy-level evaluation 到 Planning-level task-consistency evaluation 的关键过渡。

最终得到以下结论：

1. UniV2X 的 Occupancy 不参与原始 Planning trajectory generation，而主要作用于 test-time collision optimization。

2. Planning 只对规划轨迹附近约 5 m 范围内的 Occupancy 敏感，因此全局 Occupancy IoU 与 Planning performance 并不存在直接的一一对应关系。

3. Official OR fusion 将所有 infrastructure occupancy 无选择地加入 Ego，Mean L2 从：

\[
2.803602
\]

恶化至：

\[
2.845870
\]

证明 naïve cooperation 存在明显的 Planning negative transfer。

4. STCV-Occ V1 和 V2 均能够大幅缓解该负迁移，其中 V2 Mean L2 为：

\[
2.804198
\]

基本恢复至 Ego 水平。

5. V2 相比 OfficialOR 的 Mean L2 降低：

\[
0.041672
\]

约：

\[
1.46\%
\]

表明 selective cooperation 能够减少错误或无效路侧 Occupancy 对 Planning trajectory post-processing 的干扰。

6. V2 尚未进一步优于 Ego：

\[
2.804198
>
2.803602
\]

差异约：

\[
0.000596
\]

因此当前不能声称 STCV-Occ 改善了 Ego Planning。

7. Collision / out-of-drivable 等离散指标在当前 Val split 上事件较少，方法间差异有限，因此现阶段不能据此声称安全性得到显著提升。

8. Occupancy utility 与 Planning utility 存在明显的 task-objective mismatch：

\[
\boxed{
U_R^{occ}\neq U_R^{plan}
}
\]

9. 因此，仅基于 Occupancy utility 的固定决策仍然存在局限，需要进一步引入 Planning relevance、Planning utility 和 uncertainty-aware adaptive decision。

综上，V2-2 最重要的发现不是：

\[
\text{STCV-Occ improves Planning}
\]

而是：

\[
\boxed{
\text{Naïve cooperative Occupancy fusion can cause Planning negative transfer}
}
\]

以及：

\[
\boxed{
\text{STCV-Occ selective cooperation can substantially suppress this negative transfer}
}
\]

这一结果为 V2-3 进一步建立 Planning-aware adaptive cooperative decision 提供了直接的实验依据和任务层动机。