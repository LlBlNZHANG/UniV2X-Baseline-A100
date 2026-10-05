## 14. 当前阶段状态

当前已完成 G5-0A、G5-0B、G5-0C、G5-0D、G5-1A-1 和 G5-1A-2，共完成 6 个阶段。

当前完成度为：

\[
\frac{6}{8}=75\%
\]

当前阶段状态如下：

| Phase | Task | Status |
|---|---|---|
| G5-0A | Candidate Definition Audit | ✅ PASS |
| G5-0B | STCV Module Implementation | ✅ PASS |
| G5-0C | OccHead Minimal Integration | ✅ PASS |
| G5-0D | Runtime Config Audit | ✅ PASS |
| G5-1A-1 | Module-level Exact Audit | ✅ PASS |
| G5-1A-2 | Runtime Integration Audit | ✅ PASS |
| G5-1B | 86-sample Sequential Consistency Sweep | ⬜ NEXT |
| G5-2 | Frozen Official Evaluation | ⬜ PENDING |
| G5-3 | Frozen-backbone Fine-tuning | ⬜ CONDITIONAL |

---

## 15. 下一步：G5-1B 86-sample Sequential Consistency Sweep

下一阶段进入：

> **G5-1B：86-sample Sequential Consistency Sweep**

G5-1A 已经完成单个 held-out 样本从离线实现到真实 UniV2X Runtime 的完整一致性验证。G5-1B 将进一步把验证范围扩展至 G3 Test split 中的全部 86 个 held-out 样本。

由于 G5-1A-2 已经确认 UniV2X 推理过程存在显式时序状态依赖，因此 G5-1B 不能对 86 个目标样本分别进行独立 cold-start，而必须保持完整 validation sequence 的原始推理顺序。

正确的验证流程为：

```text
Validation Sequence
0 → 1 → 2 → ... → 674
        ↓
保持 UniV2X temporal runtime state
        ↓
遇到属于 G3 Test split 的目标样本
        ↓
执行 Runtime–Offline Consistency Audit
        ↓
继续顺序推理
```

其中，非 Test split 样本仍需正常执行 forward，用于维持：

- `prev_bev`
- `scene_token`
- `prev_pos`
- `prev_angle`
- tracking memory / memory bank

等 UniV2X 时序状态。

### 15.1 G5-1B 验证目标

G5-1B 需要对全部 86 个 held-out Test samples 验证以下内容。

#### （1）Sample-level 一致性

```text
Test samples             : 86 / 86
```

要求全部 86 个 Test 样本均能够在 Runtime 中正确定位，并完成一致性审计。

#### （2）Candidate Patch 一致性

G3 Test split 中共有：

```text
Candidate patches        : 14,750
```

需要验证：

```text
Offline candidates       : 14,750
Runtime candidates       : 14,750
Candidate indices        : exact
```

即全部 Candidate Patch 的：

- horizon；
- patch row；
- patch column；
- traversal order；

均与离线 G3/G4 数据完全一致。

#### （3）127-d Feature 一致性

对于全部 14,750 个 Candidate Patch，需要比较 Runtime 与 Offline 的 127 维输入特征。

判定标准：

```text
Feature max_abs_diff     < 1e-5
Feature mean_abs_diff    < 1e-6
```

127 维特征包括：

```text
Local Soft Feature       : 80 dims
Horizon One-hot          : 5 dims
Spatial Coordinate       : 2 dims
Neighborhood Feature     : 40 dims
--------------------------------
Total                    : 127 dims
```

#### （4）Predictor 输出一致性

需要逐 Candidate Patch 比较 Cooperation Utility Predictor 的：

- Logit；
- Score。

判定标准：

```text
Logit max_abs_diff       < 1e-5
Score max_abs_diff       < 1e-5
```

#### （5）Accept / Fallback 决策一致性

固定：

\[
\tau=0.70
\]

验证全部 Candidate Patch 的选择决策：

```text
Accept decisions         : 14,750 / 14,750 exact
Accept mismatches        : 0
```

即：

\[
a_R^{runtime}=a_R^{offline}
\]

对全部 14,750 个候选 Patch 成立。

#### （6）Learned Occupancy 一致性

需要逐样本、逐 horizon 比较 Runtime `Ofused` 与离线 Learned Occupancy。

目标：

```text
Learned occupancy        : 86 / 86 exact
Differing cells          : 0
```

即对全部 Test samples：

\[
O_{fused}^{runtime}
=
O_{learned}^{offline}
\]

#### （7）Candidate-domain IoU 一致性

在与 G4 完全相同的统计口径下，重新计算 86 个 Test samples 的 Learned Occupancy candidate-domain IoU。

参考结果为：

```text
Official OR IoU          ≈ 0.2526
Ego IoU                  ≈ 0.2671
Learned IoU              ≈ 0.2732
Patch Oracle IoU         ≈ 0.3077
```

G5-1B 的目标不是重新证明性能提升，而是验证 Runtime 输出能够复现 G4 的离线 Test 结果。

因此重点检查：

```text
Candidate-domain Learned IoU
≈ G4 Test reference 0.2732
```

其统计方式必须与 G4 保持一致。

---

## 16. G5-1B 最终 PASS 标准

G5-1B 的完整通过条件定义为：

```text
G5-1B 86-SAMPLE SEQUENTIAL CONSISTENCY SWEEP
================================================

Test samples
86 / 86                                        PASS

Candidate patches
Offline                14,750
Runtime                14,750
Candidate indices      exact                  PASS

127-d Features
max_abs_diff           < 1e-5                 PASS
mean_abs_diff          < 1e-6                 PASS

Predictor Logits
max_abs_diff           < 1e-5                 PASS

Predictor Scores
max_abs_diff           < 1e-5                 PASS

Accept Decisions
14,750 / 14,750 exact                          PASS
mismatches             0

Learned Occupancy
86 / 86 exact                                  PASS
differing cells        0

Candidate-domain IoU
reproduce G4 Test reference (~0.2732)          PASS

FINAL RESULT                                  PASS
================================================
```

只有满足上述全部条件后，才能正式判定：

> **G5-1 Offline–Online Consistency Validation：PASS**

届时可以确认，STCV-Occ 从 G3/G4 离线分析流程迁移至正式 UniV2X Runtime 后，在完整 held-out Test split 上不存在实现偏差。

---

## 17. G5-1B 完成后的下一阶段

G5-1B 完成并通过后，G5-1 阶段即可正式关闭。

随后进入：

> **G5-2：Frozen Official Evaluation**

G5-2 将使用：

```text
UniV2X checkpoint:
ckpts/univ2x_coop_e2e_stg2.pth

STCV config:
projects/configs_e2e_univ2x/univ2x_coop_e2e_stcv.py

STCV checkpoint:
G0-G1-G2/G3/checkpoints_soft/g3_neighborhood_best.pth

STCV threshold:
τ = 0.70
```

在保持 UniV2X 官方 sequential inference protocol 的条件下，执行正式官方 evaluator。

G5-2 的核心任务是回答：

> **在不重新训练 UniV2X Backbone、仅加载冻结的 STCV-Occ Predictor 的条件下，STCV-Occ 是否能够在官方 Occupancy Evaluation 中产生实际收益。**

需要特别注意：

> G4 中 `+8.2%` 为 held-out Test candidate-domain 下 Learned 相对 Official OR 的离线参考结果，不能直接作为 G5-2 官方 evaluator 的预期提升，也不能提前宣称其会迁移至官方 benchmark。

因此 G5-2 必须以实际官方 evaluator 输出为准。

---

## 18. 当前最终状态

```text
G5-0A   Candidate Definition Audit          ✅ PASS
G5-0B   STCV Module Implementation          ✅ PASS
G5-0C   OccHead Minimal Integration         ✅ PASS
G5-0D   Runtime Config Audit                ✅ PASS

G5-1A-1 Module-level Exact Audit            ✅ PASS
G5-1A-2 Runtime Integration Audit           ✅ PASS
G5-1B   86-sample Consistency Sweep         ⬜ NEXT

G5-2    Frozen Official Evaluation          ⬜ PENDING
G5-3    Frozen-backbone Fine-tuning         ⬜ CONDITIONAL
```

当前完成度：

\[
\boxed{\frac{6}{8}=75\%}
\]

当前下一步：

> **执行 G5-1B：86-sample Sequential Consistency Sweep。**