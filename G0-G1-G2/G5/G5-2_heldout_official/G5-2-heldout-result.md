# G5-2 阶段总结：STCV-Occ 官方评估与未见场景泛化验证

## 1. 阶段目标

本阶段主要验证 STCV-Occ 在 UniV2X 完整推理系统中的实际有效性，重点回答以下三个问题：

1. STCV-Occ 接入 UniV2X 后，在关闭 STCV 时能否保持原始模型行为不变；
2. STCV-Occ 在完整官方验证集上能否提升 Occupancy 预测性能；
3. STCV-Occ 在 predictor 从未参与训练和调参的未见场景上是否仍具有有效的泛化能力。

本阶段不对 UniV2X backbone 进行重新训练，仅加载已经训练完成的 STCV cooperation-utility predictor：

- STCV predictor：`g3_neighborhood_best.pth`
- Predictor 参数量：73,985
- STCV threshold：`τ = 0.70`
- UniV2X checkpoint：官方 `univ2x_coop_e2e_stg2.pth`
- Occupancy threshold：`0.1`

因此，本阶段属于 **Frozen Backbone / Frozen Predictor 的系统级推理评估**。

---

# 2. G5-2A：STCV-OFF Official Regression

## 2.1 实验目的

首先关闭 STCV-Occ，在修改后的 UniV2X 代码中重新执行完整官方验证集评估，用于确认 STCV 模块的接入没有破坏原始 UniV2X 推理逻辑。

实验设置：

- Validation frames：675
- Official-valid Occupancy frames：549
- `use_stcv_occ=False`
- 使用官方 UniV2X stage2 checkpoint
- 使用官方 Occupancy evaluator

## 2.2 实验结果

| Metric | 30×30 | 100×100 |
|---|---:|---:|
| IoU | 22.6 | 25.9 |
| PQ | 2.0 | 0.1 |
| SQ | 63.1 | 60.2 |
| RQ | 3.2 | 0.1 |

同时：

- Detection mAP = 0.0525
- NDS = 0.0520
- Tracking AMOTA = 0.118
- IDS = 119
- FRAG = 79

结果与原始 UniV2X baseline 保持一致。

## 2.3 结论

**G5-2A：PASS。**

STCV-Occ 模块在关闭状态下不会改变原始 UniV2X 的 Occupancy、Detection 和 Tracking 行为，说明模块集成具有良好的回归一致性。

该实验可以作为后续 STCV-ON 性能比较的可靠 baseline。

---

# 3. G5-2B：STCV-ON Full Official Validation

## 3.1 实验目的

在保持 UniV2X backbone 和官方 checkpoint 不变的情况下启用 STCV-Occ，对完整 675-frame 官方验证序列进行系统级评估。

实验设置：

- Validation frames：675
- Official-valid Occupancy frames：549
- `use_stcv_occ=True`
- STCV threshold：`τ=0.70`
- UniV2X backbone：Frozen
- STCV predictor：Frozen
- 不进行额外 end-to-end training

## 3.2 Occupancy 主结果

| Metric | Official OR | STCV-Occ | Absolute Δ |
|---|---:|---:|---:|
| IoU-30m | 22.6 | **28.5** | **+5.9** |
| IoU-100m | 25.9 | **29.1** | **+3.2** |
| PQ-30m | 2.0 | **3.7** | **+1.7** |
| PQ-100m | 0.1 | 0.1 | +0.0 |
| SQ-30m | **63.1** | 62.2 | -0.9 |
| SQ-100m | **60.2** | 60.0 | -0.2 |
| RQ-30m | 3.2 | **6.0** | **+2.8** |
| RQ-100m | 0.1 | **0.2** | **+0.1** |

IoU 相对提升约为：

\[
\frac{28.5-22.6}{22.6}\approx26.1\%
\]

\[
\frac{29.1-25.9}{25.9}\approx12.4\%
\]

因此，STCV-Occ 在完整官方验证序列上取得了明显的 Occupancy 性能提升。

---

# 4. Full Validation 下游规划指标

UniV2X 官方评估同时输出：

- `obj_col`
- `obj_box_col`
- `obj_out`
- `obj_box_out`
- `L2`

对 0.5 s–5.0 s 十个预测时刻进行简单平均后：

| Metric | Official OR | STCV-Occ | Relative Δ | 结果 |
|---|---:|---:|---:|---|
| mean obj_col | 0.000750 | **0.000600** | **-20.0%** | 改善 |
| mean obj_box_col | 0.004440 | **0.003850** | **-13.3%** | 改善 |
| mean obj_out | 0.006090 | 0.006080 | -0.16% | 基本持平 |
| mean obj_box_out | 0.026070 | **0.023560** | **-9.6%** | 改善 |
| mean L2 | 2.86461 | **2.79297** | **-2.5%** | 改善 |

其中短时预测阶段改善尤其明显。

### L2

| Horizon | Official OR | STCV-Occ |
|---|---:|---:|
| 0.5 s | 1.0838 | **0.7203** |
| 1.0 s | 1.4325 | **1.2135** |
| 1.5 s | 1.7195 | **1.6959** |
| 2.0 s | **2.1262** | 2.1273 |
| 2.5 s | 2.5556 | **2.5224** |
| 3.0 s | 2.9571 | **2.9361** |
| 3.5 s | 3.3509 | **3.3349** |
| 4.0 s | 3.8391 | **3.8352** |
| 4.5 s | 4.4741 | **4.4570** |
| 5.0 s | 5.1073 | **5.0871** |

0.5 s 和 1.0 s 时：

\[
1.0838\rightarrow0.7203
\]

\[
1.4325\rightarrow1.2135
\]

说明 STCV-Occ 对近期轨迹预测误差的改善较为明显。

### obj_box_out

前五个预测时刻均出现改善：

| Horizon | Official OR | STCV-Occ |
|---|---:|---:|
| 0.5 s | 0.0059 | **0.0030** |
| 1.0 s | 0.0119 | **0.0030** |
| 1.5 s | 0.0119 | **0.0074** |
| 2.0 s | 0.0148 | **0.0119** |
| 2.5 s | 0.0281 | **0.0222** |

因此 STCV-Occ 的收益并非仅体现在 Occupancy IoU 上，还表现出一定的下游规划收益。

需要注意：

> 上述 mean 指标为对 0.5–5.0 s 十个时刻进行的简单平均，并非 UniV2X 官方额外定义的单一 aggregate metric，论文中应明确说明计算方式。

---

# 5. Detection / Tracking 影响

STCV-Occ 主要作用于 Occupancy fusion，因此还需要检查其是否影响其他任务。

完整验证集结果表明：

| Metric | Official OR | STCV-Occ |
|---|---:|---:|
| Detection mAP | 0.0525 | 0.0525 |
| NDS | 0.0520 | 0.0520 |
| Tracking AMOTA | 0.118 | 0.118 |
| IDS | 119 | 119 |
| FRAG | 79 | 79 |

因此：

> STCV-Occ 在明显改善 Occupancy prediction 的同时，没有破坏原有 Detection 和 Tracking 分支性能。

这支持 STCV-Occ 属于相对独立的 Occupancy reliability intervention。

---

# 6. Full Validation 结果的局限

虽然完整 675-frame 官方验证结果表现较强：

\[
22.6/25.9
\rightarrow
28.5/29.1
\]

但当前 STCV-v1 predictor 最初是在这 21 个 validation scenes 内部进行：

- 13 scenes：Train
- 4 scenes：Validation
- 4 scenes：Test

因此完整 675-frame official evaluation 中包含部分 predictor 在训练或模型选择阶段接触过的 scene。

所以：

> Full-675 结果能够证明系统级性能提升，但不能单独作为“完全未见场景泛化”的最终证据。

为解决这一问题，本阶段进一步进行了 Held-out Scene Evaluation。

---

# 7. G5-2C：Held-out Scene Official-Metric Generalization Evaluation

## 7.1 实验目的

选取 G3 中完全未参与 predictor training 和 model selection 的 4 个 Test scenes：

- `0007`
- `0021`
- `0071`
- `0088`

在这些完全未见场景上重新采用 UniV2X 官方 Occupancy evaluator，比较：

\[
Official\ OR
\]

与

\[
STCV\text{-}v1
\]

从而验证 STCV-Occ 是否具有跨场景泛化能力。

---

# 8. Held-out 数据审计

4 个场景完整时序如下：

| Scene | Dataset Index | Total Frames | Official-valid Frames | G3 Candidate Frames |
|---|---:|---:|---:|---:|
| 0007 | 42–71 | 30 | 24 | 21 |
| 0021 | 202–239 | 38 | 32 | 29 |
| 0071 | 449–489 | 41 | 35 | 35 |
| 0088 | 582–588 | 7 | 1 | 1 |
| **Total** | — | **116** | **92** | **86** |

其中共有：

\[
92-86=6
\]

个 official-valid frame 没有 STCV candidate。

这些样本仍然被保留在正式评价中，没有因为“没有 candidate”而被人工过滤。

因此正式 Held-out evaluation 使用：

- Complete sequential forward frames：116
- Official-valid Occupancy frames：92

而不是只使用 G3 中的 86 candidate-containing samples。

这一设计避免了 candidate-dependent sample selection bias。

---

# 9. Held-out Official OR Baseline

在 4 个 held-out scenes 的相同 92 个 official-valid frames 上，Official OR 得到：

| Metric | 30×30 | 100×100 |
|---|---:|---:|
| IoU | 20.6 | 28.3 |
| PQ | 3.1 | 0.2 |
| SQ | 61.8 | 62.8 |
| RQ | 5.0 | 0.3 |

该结果仅作为 Held-out scenes 的同场景 baseline。

不能直接将：

\[
20.6/28.3
\]

与完整 549-frame validation 上的：

\[
22.6/25.9
\]

直接比较，因为二者评价数据范围不同。

---

# 10. Held-out STCV-v1 Results

在完全相同的：

- 4 scenes
- 116 sequential forward frames
- 92 official-valid frames
- Official evaluator

条件下启用 STCV-v1，得到：

| Metric | Official OR | STCV-v1 | Absolute Δ |
|---|---:|---:|---:|
| **IoU-30m** | 20.6 | **25.0** | **+4.4** |
| **IoU-100m** | 28.3 | **30.9** | **+2.6** |
| **PQ-30m** | 3.1 | **6.6** | **+3.5** |
| PQ-100m | 0.2 | 0.2 | 0.0 |
| SQ-30m | **61.8** | 59.8 | -2.0 |
| SQ-100m | 62.8 | 62.8 | 0.0 |
| **RQ-30m** | 5.0 | **11.0** | **+6.0** |
| RQ-100m | 0.3 | 0.3 | 0.0 |

IoU 相对提升：

\[
\frac{25.0-20.6}{20.6}
\approx21.4\%
\]

\[
\frac{30.9-28.3}{28.3}
\approx9.2\%
\]

因此，在 predictor 完全未参与训练和调参的场景中，STCV-v1 仍取得显著的 Occupancy IoU 提升。

---

# 11. Full Validation 与 Held-out Generalization 对照

目前可以形成两层互补证据。

## 11.1 系统级完整验证

\[
22.6/25.9
\rightarrow
28.5/29.1
\]

作用：

> 证明 STCV-Occ 接入完整 UniV2X 系统后，能够明显改善总体 Occupancy 性能。

## 11.2 未见场景泛化

\[
20.6/28.3
\rightarrow
25.0/30.9
\]

作用：

> 证明性能收益并不局限于 predictor 训练过程中接触过的场景，在完全 held-out scenes 上仍然能够保持明显提升。

两组结果方向一致，说明 STCV-Occ 的收益具有一定的跨场景泛化能力。

---

# 12. 对 PQ / SQ / RQ 的解释

Full Validation 和 Held-out Evaluation 均表现出类似趋势：

- IoU 明显提升；
- PQ 提升；
- RQ 提升较明显；
- SQ 基本保持或轻微下降。

Held-out 30 m 下：

\[
RQ:5.0\rightarrow11.0
\]

而：

\[
SQ:61.8\rightarrow59.8
\]

因此目前更合理的解释是：

> STCV-Occ 的主要收益来自对协同 Occupancy 信息的有效筛选，即减少错误路侧增量信息并保留有效合作信息，从而提高 Occupancy recognition / association quality；其主要作用并非进一步优化已经正确识别实例的空间形状质量。

论文中不要写成：

> “STCV-Occ 全面提升所有 Occupancy 指标。”

更准确应写成：

> “STCV-Occ 显著改善 IoU、PQ 和 RQ，同时 SQ 基本保持，仅在部分评价范围出现轻微下降，表明方法的主要收益来源于更可靠的协同信息选择。”

---

# 13. 本阶段最终结论

综合 G5-2A、G5-2B 和 G5-2C，可以得到以下结论。

### （1）集成正确性成立

关闭 STCV 时，修改后的 UniV2X 完整复现 Official OR baseline，证明 STCV 的接入没有改变原始模型行为。

### （2）系统级性能提升成立

在完整 675-frame official validation 上：

\[
IoU_{30m}:22.6\rightarrow28.5
\]

\[
IoU_{100m}:25.9\rightarrow29.1
\]

说明 STCV-Occ 能够显著改善 Occupancy prediction。

### （3）未见场景泛化成立

在完全 held-out 的 4 个 scenes、92 个 official-valid samples 上：

\[
IoU_{30m}:20.6\rightarrow25.0
\]

\[
IoU_{100m}:28.3\rightarrow30.9
\]

说明 STCV 的收益并非仅来源于 predictor 对训练场景的记忆。

### （4）收益具有机制一致性

STCV 同时提高 IoU、PQ 和 RQ，而 SQ 基本保持或略有下降，说明提升主要来源于：

> 对路侧新增 Occupancy 信息进行更有效的接受与拒绝，而非重新优化 Occupancy instance shape。

这与 STCV-Occ 的核心设计目标一致：

\[
\text{Estimate Cooperation Utility}
\rightarrow
\text{Selective Infrastructure Acceptance}
\]

### （5）没有破坏其他任务

Detection 和 Tracking 指标保持不变，说明 STCV-Occ 的影响主要局限于 Occupancy fusion 分支。

### （6）存在进一步优化空间

当前 predictor 仍未达到 Patch Oracle 上限。

此前 Test86 中：

\[
IoU_{STCV}=0.2732
\]

而：

\[
IoU_{PatchOracle}=0.3077
\]

同时：

\[
BA\ Retention\approx37.75\%
\]

\[
HA\ Admission\approx15.84\%
\]

说明当前 predictor 仍存在：

- 有益合作信息漏接；
- 有害合作信息误接。

因此继续开展 G5-3 Enhanced Utility Learning 具有充分依据。

---

# 14. 论文中可以直接使用的实验结果描述

## 14.1 完整验证集结果

> 为验证所提出 STCV-Occ 在完整车路协同 Occupancy 推理流程中的有效性，在保持 UniV2X 主干网络及官方 Stage-2 权重固定的条件下，将 STCV-Occ 集成至 Occupancy fusion 模块，并在官方验证序列上进行评估。实验结果表明，相较于原始 OR fusion，STCV-Occ 将 30 m 与 100 m 范围内的 Occupancy IoU 分别由 22.6 和 25.9 提升至 28.5 和 29.1，对应绝对提升 5.9 和 3.2 个百分点。同时，30 m 范围内 PQ 由 2.0 提升至 3.7，RQ 由 3.2 提升至 6.0，而 SQ 基本保持稳定。结果表明，基于合作价值估计的选择性融合能够有效缓解无条件接收路侧 Occupancy 所引起的负合作问题。

## 14.2 泛化结果

> 为进一步验证 STCV-Occ 的跨场景泛化能力，在 predictor 训练及模型选择过程中均未参与的 4 个 held-out scenes 上进行独立评价。完整保留各场景时序状态后，共包含 116 个推理帧，其中 92 个样本满足官方 Occupancy 评价条件。实验结果表明，STCV-Occ 将 30 m 范围内 IoU 由 20.6 提升至 25.0，将 100 m 范围内 IoU 由 28.3 提升至 30.9，绝对提升分别达到 4.4 和 2.6 个百分点。同时，30 m 范围内 PQ 由 3.1 提升至 6.6，RQ 由 5.0 提升至 11.0。该结果表明，STCV-Occ 的性能收益能够迁移至未见场景，并非仅依赖于训练场景特征。

## 14.3 下游结果

> 除 Occupancy prediction 外，STCV-Occ 对下游规划相关指标整体表现出正向影响。对 0.5–5.0 s 预测时域进行平均后，obj_col、obj_box_col 和 obj_box_out 分别下降约 20.0%、13.3% 和 9.6%，平均 L2 误差下降约 2.5%，其中 0.5–1.0 s 的短期轨迹误差改善更为明显。与此同时，Detection 与 Tracking 指标保持不变，说明 STCV-Occ 能够在不干扰其他感知任务的情况下改善 Occupancy fusion 质量。

---

# 15. 当前论文中可以形成的完整证据链

目前 STCV-Occ 已经形成：

**G1：Negative Cooperation Diagnosis**

\[
Official\ OR
\]

中存在大量 harmful additions，证明负合作问题真实存在。

↓

**G2：Patch Oracle Analysis**

证明如果能够正确选择基础设施新增 Occupancy，存在明显性能提升空间。

↓

**G3：Cooperation Utility Learning**

证明 patch cooperation utility 可以从 soft occupancy、horizon、spatial 和 neighborhood evidence 中学习。

↓

**G4：Held-out Offline Evaluation**

证明 learned selective fusion 在未见场景上优于 Official OR。

↓

**G5-1：Runtime Integration & Consistency**

证明离线 STCV 与 UniV2X 在线运行行为一致。

↓

**G5-2A：Official Regression**

证明关闭 STCV 后能够完整恢复 UniV2X baseline。

↓

**G5-2B：Full Official Validation**

\[
22.6/25.9
\rightarrow
28.5/29.1
\]

证明完整系统级性能提升。

↓

**G5-2C：Held-out Official-Metric Generalization**

\[
20.6/28.3
\rightarrow
25.0/30.9
\]

证明未见场景泛化能力。

因此目前已经形成：

\[
\boxed{
Problem\ Diagnosis
\rightarrow
Oracle\ Evidence
\rightarrow
Utility\ Learning
\rightarrow
Selective\ Fusion
\rightarrow
Runtime\ Integration
\rightarrow
Official\ Evaluation
\rightarrow
Held\text{-}out\ Generalization
}
\]

的完整研究链条。

---

# 16. 当前仍需注意的论文表述边界

目前可以写：

> STCV-Occ 在完整 UniV2X 验证序列上显著改善 Occupancy prediction，并在独立 held-out scenes 上保持一致的性能增益。

可以写：

> Held-out 结果为 STCV-Occ 的跨场景泛化能力提供了直接证据。

可以写：

> 方法无需重新训练 UniV2X backbone，仅利用轻量级 cooperation-utility predictor 即可改善 Occupancy fusion。

目前不宜写：

> “所有测试场景均取得提升。”

因为当前没有逐 scene 结果。

目前不宜写：

> “所有 Occupancy 指标均提高。”

因为 SQ 存在轻微下降。

目前不宜写：

> “已经充分证明广泛分布外泛化能力。”

因为当前 held-out evaluation 仅包含 4 个 scenes，并不等同于严格意义上的 OOD benchmark。

目前也不应将 Full-675 的：

\[
28.5/29.1
\]

描述成：

> “完全独立测试集上的最终结果。”

因为当前 STCV-v1 的 predictor training scenes 与完整 21-scene validation 存在重叠。

这一问题将在 G5-3 中通过新的训练协议解决。

---

# 17. 下一阶段：G5-3 Enhanced STCV Utility Learning

G5-3 的目标不再是证明 STCV 是否有效，因为当前阶段已经完成这一验证。

G5-3 的目标调整为：

> 在保持 UniV2X backbone frozen 的条件下，利用更加规范且规模更大的 Official Train 数据重新训练 STCV cooperation-utility predictor，从而进一步提高预测能力和跨场景泛化性能。

目标协议：

```text
Official Training Split
        ↓
Frozen UniV2X Forward
        ↓
生成 Pv / Pi_aligned / GT / warp mask
        ↓
构建 Cooperation Utility Labels
        ↓
Train STCV-v2 Predictor
        ↓
Independent Train-side Validation
选择 checkpoint / τ
        ↓
Official Validation 675
完全不参与 predictor training / tuning
        ↓
Official OR vs STCV-v2
```

这样最终完整 675-frame validation 将成为真正独立于 STCV predictor 训练过程的评价集合。

G5-3 后最终希望形成：

| Method | Training Protocol | Official Val IoU |
|---|---|---|
| Official OR | Official UniV2X | 22.6 / 25.9 |
| STCV-v1 | Current G3 predictor | 28.5 / 29.1 |
| STCV-v2 | Official Train only | 待实验 |

如果 STCV-v2 在完全独立的 Official Validation 上仍获得接近或高于当前结果的性能，则能够进一步显著增强论文主实验的说服力。

---

# 18. 本阶段状态

**G5-2A — STCV-OFF Official Regression：PASS**

**G5-2B — Full Official STCV Evaluation：PASS**

**G5-2C — Held-out Scene Official-Metric Generalization：PASS**

当前结论：

> STCV-Occ 已完成从模块级实现、在线一致性、系统级性能评估到未见场景泛化验证的完整闭环。现有结果已经证明合作价值驱动的选择性 Occupancy fusion 在 UniV2X 中具有实际有效性，并具备一定的跨场景泛化能力。下一阶段 G5-3 将重点从“证明有效”转向“进一步提升 predictor 性能并建立严格独立的 train/validation protocol”。