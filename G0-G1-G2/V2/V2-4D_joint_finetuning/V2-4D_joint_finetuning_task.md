# V2-4D 创新三：可学习自适应融合联合微调任务书

## 1. 任务目标

在官方 UniV2X Stage-2 模型基础上，实现并验证创新三：

\[
\boxed{\text{Utility-conditioned Learnable Adaptive Fusion}}
\]

完整逻辑：

\[
\hat U_R^{occ}\le0
\Rightarrow
g_R=0
\]

即创新二判断该区域不值得合作时，直接保持 Ego。

对于：

\[
\hat U_R^{occ}>0
\]

进入创新三，由可学习融合模块预测融合策略：

\[
\boxed{g_R\in\{0.1,1.0\}}
\]

其中：

- \(g=0.1\)：衰减融合；
- \(g=1.0\)：完整融合。

最终：

\[
\boxed{
P_{f,R}
=
P_{v,R}
+
g_R(P_{i,R}-P_{v,R})_+
}
\]

本阶段目标不是重新训练整个 UniV2X，而是：

> 训练创新三 Learnable Fusion Module，并联合微调少量 Occupancy late representation，使特征更适合判断“完整融合还是衰减融合”。

---

# 2. 固定基础

项目目录：

```text
/root/autodl-tmp/UniV2X
```

官方配置：

```text
projects/configs_e2e_univ2x/univ2x_coop_e2e.py
```

官方 Stage-2 checkpoint：

```text
ckpts/univ2x_coop_e2e_stg2.pth
```

创新二 Utility Predictor：

```text
G0-G1-G2/V2/V2-1_magnitude_learning/
checkpoints/main_lambda_0p50/best.pth
```

创新三当前最好离线模型：

```text
G0-G1-G2/V2/V2-4_learnable_adaptive_fusion/
checkpoints/V2-4C-3E1_binary_attenuation/best.pth
```

当前配置真实模型入口为：

```text
model_ego_agent
model_other_agent_inf
```

不得假设存在：

```text
cfg.model
```

本工作重点研究：

```text
model_ego_agent
└── occ_head
```

---

# 3. 当前固定结果

以下结果作为后续实验固定参考，不允许随意改变评价口径：

```text
Official OR 100m IoU       = 0.330526
V2 Hard 100m IoU           = 0.360369
E1 Frozen Fusion 100m IoU  = 0.363508
Binary Oracle 100m IoU     = 0.379668
```

创新三 Binary Oracle 已证明：

\[
0.360369
\rightarrow
0.379668
\]

存在约：

\[
+0.019299
\]

的 100m IoU 理论空间。

当前问题是：

> 固定 Occupancy representation 下的 Learnable Fusion 只能达到 0.363508，尚未充分利用 Oracle Headroom。

---

# 4. V2-4D-0：代码与参数审计

## 4.1 目标

先完整阅读当前 UniV2X 代码，不直接训练。

必须确认：

1. `model_ego_agent` 的实际构建入口；
2. 官方 Stage-2 checkpoint 的加载方式；
3. `occ_head` 的完整模块结构；
4. Occupancy 最后 1～2 个 late blocks 的真实模块名；
5. `Pv` 的产生位置；
6. `Pi_aligned` 的产生位置；
7. Occupancy fusion 的实际执行位置；
8. 最终 occupancy probability 的产生位置；
9. 哪个中间 feature 最适合供创新三使用；
10. 哪些参数可以安全解冻而不影响 Detection / Tracking / Planning。

重点阅读：

```text
projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py
```

同时追踪其调用链。

输出：

```text
- model_ego_agent 实际构建入口
- occ_head 模块树
- occ_head named_parameters
- 推荐冻结模块
- 推荐解冻模块
- 每个候选模块参数量
- 推荐作为创新三输入的 feature
```

本阶段：

```text
Training : NONE
Test used: NO
```

---

# 5. V2-4D-1：组装完整方法链

必须形成以下前向结构：

```text
Official UniV2X Stage2
        │
        ▼
 Pv / Pi / Occupancy Features
        │
        ▼
  P4 Candidate Features
        │
        ▼
 Innovation 2
 Frozen Utility Predictor
        │
        ▼
      U_hat
      /   \
     /     \
U_hat<=0   U_hat>0
   │          │
   ▼          ▼
  g=0    Innovation 3
         Learnable Fusion
              │
              ▼
          g=0.1 / 1
              │
              ▼
       Final Occupancy
```

创新二必须：

```text
正确加载 checkpoint
参与 forward
参数冻结
不加入 optimizer
```

创新三必须：

```text
参与 forward
参与 backward
加入 optimizer
```

---

# 6. 创新三定义

创新三名称暂定：

```text
Utility-conditioned Learnable Adaptive Fusion
```

中文：

```text
合作价值条件约束下的可学习自适应融合
```

创新二负责：

```text
是否合作
```

创新三负责：

```text
合作以后融合多强
```

第一版动作空间固定：

\[
g_R\in\{0.1,1.0\}
\]

不得在第一轮同时改成连续 gate 或多档 gate。

训练标签：

\[
\boxed{
y_R^{att}
=
\mathbf1[
U_R(0.1)>U_R(1.0)
]
}
\]

即：

```text
y=1 → g=0.1
y=0 → g=1.0
```

GT 只用于训练监督。

推理阶段禁止使用 GT。

---

# 7. Smoke Test

正式训练前必须先完成少量 batch smoke test。

检查：

```text
1. Stage-2 checkpoint 正确加载
2. V2 Utility Predictor checkpoint 正确加载
3. 创新三模块正确接入
4. V2 reject 必须严格 g=0
5. V2 accept 才进入创新三
6. candidate 顺序与原 V2 完全一致
7. forward 无 NaN / Inf
8. backward 正常
9. 只有预期模块存在梯度
10. 冻结模块 grad=None
```

同时打印：

```text
V2 candidate count
V2 accepted count
attenuation count
g=0 count
g=0.1 count
g=1 count
trainable parameter count
frozen parameter count
```

只有 Smoke Test PASS 后才能训练。

---

# 8. V2-4D-2：局部 Joint Fine-tuning

## 8.1 冻结模块

第一轮原则上冻结：

```text
Image Backbone
Image Neck
BEV Encoder
AgentFusion
Detection Head
Tracking相关模块
Motion Head
Planning Head
Infrastructure Model
Innovation-2 Utility Predictor
Occupancy early / middle layers
```

## 8.2 可训练模块

仅训练：

```text
Innovation-3 Learnable Fusion Module

+

model_ego_agent.occ_head 中
最靠近最终 Occupancy 输出的 1～2 个 late blocks
```

具体模块名称必须根据 V2-4D-0 代码审计结果确定。

禁止凭名称猜测。

---

# 9. Optimizer

不要 resume 官方 Stage-2 optimizer。

使用：

```text
Stage-2 checkpoint initialization
+
new optimizer
```

建议分组学习率：

```python
optimizer = AdamW(
    [
        {
            "params": innovation3_params,
            "lr": 3e-4,
        },
        {
            "params": occ_late_params,
            "lr": 1e-5,
        },
    ],
    weight_decay=1e-4,
)
```

原则：

\[
LR_{innovation3}
>
LR_{occ\_late}
\]

Occupancy Stage-2 representation 只能小幅调整。

---

# 10. 数据划分

保持现有 scene-level split：

```text
Train : 13 scenes
Val   : 4 scenes
Test  : 4 scenes
```

已有 candidate：

```text
Train candidates : 86143
Val candidates   : 18555
Test candidates  : 14750
```

创新三当前 V2 accepted：

```text
Train accepted : 19878
Val accepted   : 6156
```

要求：

```text
Test 完全封存
```

禁止：

```text
用 Test 选 epoch
用 Test 调 threshold
用 Test 调网络结构
用 Test 决定是否继续方法
```

---

# 11. 第一轮训练规模

不要直接训练完整 20 epochs。

第一阶段建议：

```text
600～900 iterations
```

或者根据项目实际 dataloader / epoch 长度设置一个等价短训练。

建议定期保存并进行 Val evaluation。

例如：

```text
每 100 iterations
```

记录一次核心指标。

如果出现明显 representation collapse 或性能下降，立即停止。

---

# 12. 主评价指标

最终成败首先看官方 Occupancy evaluator：

\[
\boxed{100m\ Occupancy\ IoU}
\]

同时必须报告：

\[
\boxed{30m\ Occupancy\ IoU}
\]

核心比较：

| Method | 30m IoU | 100m IoU |
|---|---:|---:|
| Official OR | 0.229482 | 0.330526 |
| V2 Hard | 0.309799 | 0.360369 |
| E1 Frozen Fusion | 0.313465 | 0.363508 |
| Joint Learnable Fusion | 待测 | 待测 |
| Binary Oracle | 0.314732 | 0.379668 |

必须额外计算：

\[
\Delta IoU_{100}^{V2}
=
IoU_{Joint}
-
0.360369
\]

以及：

\[
\Delta IoU_{100}^{E1}
=
IoU_{Joint}
-
0.363508
\]

---

# 13. 创新三诊断指标

除 IoU 外，同时报告：

```text
Attenuation Rate
Harmful Attenuation Rate
Mean True Gain
Precision
Recall
F1
V2 Acceptance Rate
```

重点观察：

\[
\boxed{Mean\ True\ Gain>0}
\]

以及：

\[
\boxed{Harmful\ Attenuation\ Rate}
\]

是否明显下降。

分类指标不能替代最终 IoU。

---

# 14. Oracle Recovery

计算：

\[
\boxed{
Recovery
=
\frac{
IoU_{Joint}-0.360369
}{
0.379668-0.360369
}
}
\]

用于衡量创新三实际利用了多少 Binary Oracle Headroom。

---

# 15. GO / NO-GO

## GO

若：

```text
100m IoU >= 0.366
```

同时：

```text
30m IoU 不明显下降
V2 行为没有严重破坏
```

则：

\[
\boxed{GO}
\]

## STRONG GO

若：

```text
100m IoU >= 0.370
```

则：

\[
\boxed{STRONG\ GO}
\]

创新三可作为论文核心贡献。

## NO-GO

若经过合理的局部 joint fine-tuning 后仍然：

```text
100m IoU ≈ 0.363～0.364
```

且没有稳定改善，则：

\[
\boxed{NO\text{-}GO}
\]

此时停止：

```text
继续扩大 MLP
继续增加 gate 数量
无限调 loss
全模型解冻
使用 Test 调参
```

创新三降级为分析 / enhancement。

---

# 16. 实验约束

必须遵守：

```text
1. 不修改创新1 Utility定义
2. 不重新训练创新2
3. 创新2第一阶段必须冻结
4. 不修改 P4 candidate 定义
5. 不改变 V2 accept 规则
6. 不使用 Test
7. 不全模型解冻
8. 不修改 SQSA 工作
9. 不覆盖旧 checkpoint
10. 所有新实验放入 V2-4D 目录
```

建议目录：

```text
G0-G1-G2/V2/V2-4_learnable_adaptive_fusion/
└── V2-4D_joint_finetuning/
    ├── configs/
    ├── scripts/
    ├── checkpoints/
    ├── results/
    └── logs/
```

---

# 17. 必须交付的结果

完成后提交：

```text
1. 当前 UniV2X 的真实模型构建入口

2. ego occ_head 完整关键模块结构

3. 最终选择解冻的模块名

4. Trainable / Frozen 参数数量

5. 创新二 checkpoint 加载验证

6. 创新三模块结构

7. Smoke Test 结果

8. Joint Fine-tuning 设置
   - optimizer
   - learning rate
   - iterations
   - batch size
   - loss

9. 最佳 Val 结果
   - 30m IoU
   - 100m IoU
   - Δ100m vs V2
   - Δ100m vs E1
   - Oracle Recovery

10. 创新三诊断指标
    - attenuation rate
    - harmful attenuation rate
    - mean true gain
    - precision
    - recall
    - F1

11. 最佳 checkpoint 路径

12. 明确输出：
    Test used = NO
```

---

# 18. 最终原则

本阶段不是重新训练 UniV2X。

核心目标是：

\[
\boxed{
\text{Frozen Utility Selection}
\rightarrow
\text{Learnable Adaptive Fusion}
}
\]

通过局部 joint fine-tuning，让：

\[
\boxed{
\text{Occupancy late representation}
+
\text{Innovation-3 Fusion Module}
}
\]

共同适应融合强度决策。

必须：

```text
先阅读代码
→ 确认真实模块
→ Smoke Test
→ 短程 Joint Fine-tuning
→ Val 判定
```

禁止：

```text
凭模块名猜解冻层
直接全模型训练
直接打开 Test
为了得到更高结果无限修改方法
```