# V2-4D-0 代码审计报告

Training : NONE
Test used : NO

---

## 1. model_ego_agent 实际构建入口

文件：`projects/configs_e2e_univ2x/univ2x_coop_e2e.py`，第 521 行起。

```
model_ego_agent = dict(
    type="UniV2X",
    is_cooperation=True,      # line 519-520 赋值
    is_ego_agent=True,        # line 519-520 赋值
    ...
)
```

注意：配置文件顶部（第 105 行）`is_ego_agent = False`，第 519-520 行被覆盖为 `True`。
真实模型入口是 `model_ego_agent`（不是 `cfg.model`，后者不存在）。

对应 Python 类：`UniV2X`（继承链 UniV2X → UniV2XTrack → ...）
注册在：`projects/mmdet3d_plugin/univ2x/detectors/univ2x_e2e.py`

---

## 2. 官方 Stage-2 Checkpoint 加载方式

配置文件第 1215 行：
```
load_from = "ckpts/univ2x_coop_e2e_stg1.pth"
```
Stage-2 checkpoint（任务书指定）：
```
ckpts/univ2x_coop_e2e_stg2.pth
```
加载方式：mmcv Runner 通过 `load_from` 字段在训练启动时 `load_checkpoint(model, load_from, strict=False)` 加载。
联合微调时不 resume optimizer，只 init 权重。

---

## 3. occ_head 完整模块结构与参数量

```
model_ego_agent.occ_head                          总计 14,456,581
│
├── bev_sampler              (BevFeatureSlicer)          0          [无可训练参数]
├── bev_light_proj           (SimpleConv2d 4层)   1,836,800
├── base_downscale           (2×Bottleneck /4)      560,128
│   ├── [0] Bottleneck                             280,064
│   └── [1] Bottleneck                             280,064
├── transformer_decoder      (5层 DetrDecoder)    7,894,272
│   ├── layers (5× DetrTransformerDecoderLayer)  7,893,760
│   └── post_norm (LayerNorm 256)                      512
├── temporal_mlps            (5克隆 MLP)            657,920        [每个 131,584]
├── downscale_convs          (5克隆 Bottleneck)   1,400,320        [每个 280,064]
├── upsample_adds            (5克隆 UpsamplingAdd)  330,240        [每个  66,048]
├── dense_decoder            (CVT_Decoder 2块)      788,480
├── mode_fuser               (Linear+LN+ReLU)        66,304
├── multi_query_fuser        (Linear*2+LN+ReLU)     526,080
├── query_to_occ_feat        (3层 MLP)              197,376
├── temporal_mlp_for_mask    (3层 MLP, deepcopy)    197,376
├── pred_prob                (5× Conv2d 256→1 1×1)    1,285        [每个 257]
├── loss_dice                                             0
└── loss_mask                                             0
```

---

## 4. occ_head 前向数据流

### 4.1 训练路径（forward_train）

```
bev_feat [hw, b, 256]
    │
    ▼ forward(bev_feat, ins_query)
    ├── bev_sampler         → [b, 256, 200, 200]
    ├── bev_light_proj      → [b, 256, 200, 200]   (bev_proj_dim=256)
    ├── base_downscale      → [b, 256, 50, 50]     (两次 /2 → /4)
    │
    └── for i in range(5):   (n_future_blocks = n_future+1 = 5)
        ├── downscale_convs[i]    → [b, 256, 25, 25]     (/8)
        ├── temporal_mlps[i]      → cur_ins_query [b,q,256]
        ├── get_attn_mask(...)    → attn_mask
        ├── transformer_decoder.layers[i*1]  (5层，每块1层)
        └── upsample_adds[i]      → [b, 256, 50, 50]    (back to /4)
            → future_states [b, 5, 256, 50, 50]
    │
    ├── dense_decoder       → future_states [b, 5, 256, 50, 50]  (2×upsample)
    └── pred_prob[ti]       → ins_occ_logits [b, 1, 5, 200, 200]  (Conv 256→1)
```

### 4.2 测试路径（forward_test）

```
ins_query = merge_queries(outs_motion, ...)
_, pred_ins_logits = self(bev_feat, ins_query)    # [b, q, 5, 200, 200]
pred_ins_sigmoid = pred_ins_logits.sigmoid()       # [b, q, 5, 200, 200]
pred_seg_scores = pred_ins_sigmoid.max(dim=1)[0]   # [b, 5, 200, 200] = Pv

if is_ego_agent and is_cooperation:
    new_pred_seg_scores, _, fusion_aux = occ_prob_fusion(pred_seg_scores, inf_occ, veh2inf_rt)
    # fusion_aux 包含 Pv, Pi_aligned, Ov, Oi, Oofficial, Ofused, warp_valid_mask
    pred_seg_scores = new_pred_seg_scores

seg_out = (pred_seg_scores > test_seg_thresh).long()   # 最终二值 occupancy
```

---

## 5. 关键变量产生位置

| 变量 | 产生位置 | 形状 |
|---|---|---|
| `Pv` | `occ_head.forward_test` L534：`pred_ins_sigmoid.max(1)[0]` | [b, 5, 200, 200] |
| `Pi_aligned` | `occ_prob_fusion` L737：`F.grid_sample(inf_occ, bev_grid)` | [b, 5, 200, 200] |
| Occupancy fusion | `occ_prob_fusion` L813-825 | `torch.maximum(Ov, Oi)` 或 STCVOcc |
| 最终 occ prob | `pred_seg_scores`（融合后） | [b, 5, 200, 200] |
| 最终二值输出 | `(pred_seg_scores > 0.1).long()` | [b, 5, 1, 200, 200] |

---

## 6. 创新三推荐输入 Feature

**推荐：dense_decoder 输出的 current frame BEV feature**

理由：
- `future_states[:, 0, :, :, :]` 在 `dense_decoder` 之后形状为 `[b, 256, 50, 50]`
- 包含了 transformer_decoder 注入的 instance query 信息，是 occ_head 最高语义层特征
- 通过 RoI pooling 或 grid_sample 可以针对每个 candidate 区域 R 提取局部特征
- 比直接用 Pv/Pi 的概率值（127维离线特征）包含更丰富的结构信息
- 解冻 dense_decoder 可以让该特征适应融合决策任务

备选（更轻量）：直接使用 Pv[R] 和 Pi_aligned[R] 的统计量（均值、方差、差值），与离线 V2-4C 方案一致，实现更简单，但 representation 适应性弱。

**建议第一轮使用：dense_decoder 输出 → 区域均值池化 → Innovation-3 MLP**

---

## 7. 推荐冻结 / 解冻方案

### 冻结（不加入 optimizer）

```
model_other_agent_inf                   （整个 infrastructure model）
model_ego_agent.img_backbone            （freeze_img_backbone=True 已有）
model_ego_agent.img_neck                （freeze_img_neck=True 已有）
model_ego_agent.pts_bbox_head           （BEV encoder + track head）
model_ego_agent.seg_head                （map head）
model_ego_agent.motion_head
model_ego_agent.planning_head
Innovation-2 Utility Predictor          （V2UtilityPredictor，单独加载）

model_ego_agent.occ_head 内部冻结：
  ├── bev_sampler                        (0 params)
  ├── bev_light_proj                     (1,836,800)
  ├── base_downscale                     (560,128)
  ├── transformer_decoder                (7,894,272)
  ├── temporal_mlps                      (657,920)
  ├── downscale_convs                    (1,400,320)
  ├── upsample_adds                      (330,240)     ← 可选解冻，见下
  ├── mode_fuser                         (66,304)
  ├── multi_query_fuser                  (526,080)
  ├── query_to_occ_feat                  (197,376)
  └── temporal_mlp_for_mask              (197,376)

冻结 occ_head 合计：11,936,256
```

### 解冻（加入 optimizer，lr=1e-5）

```
model_ego_agent.occ_head.dense_decoder        788,480   ← 第一个解冻候选
model_ego_agent.occ_head.pred_prob              1,285   ← 最靠近输出的卷积
（可选）model_ego_agent.occ_head.upsample_adds 330,240   ← 第二轮再考虑
```

### Innovation-3 Learnable Fusion Module（新建，lr=3e-4）

参数量待新模块确定（预估 ~100K-500K）

---

## 8. 参数量汇总

| 模块 | 参数量 | 状态 |
|---|---:|---|
| occ_head.transformer_decoder | 7,894,272 | 冻结 |
| occ_head.downscale_convs | 1,400,320 | 冻结 |
| occ_head.bev_light_proj | 1,836,800 | 冻结 |
| occ_head.temporal_mlps | 657,920 | 冻结 |
| occ_head.multi_query_fuser | 526,080 | 冻结 |
| occ_head.base_downscale | 560,128 | 冻结 |
| occ_head.upsample_adds | 330,240 | 冻结（第一轮） |
| occ_head.query_to_occ_feat | 197,376 | 冻结 |
| occ_head.temporal_mlp_for_mask | 197,376 | 冻结 |
| occ_head.mode_fuser | 66,304 | 冻结 |
| **occ_head.dense_decoder** | **788,480** | **解冻，lr=1e-5** |
| **occ_head.pred_prob** | **1,285** | **解冻，lr=1e-5** |
| **Innovation-3 Fusion Module** | **~TBD** | **解冻，lr=3e-4** |

---

## 9. V2-1 Utility Predictor（创新二）结构

```python
class V2UtilityPredictor(nn.Module):
    # input: 127维候选特征
    input_proj:    Linear(127→512) + LayerNorm + ReLU + Dropout
    residual_block: Linear(512→512) + LayerNorm + ReLU + Dropout
    down_256:      Linear(512→256) + LayerNorm + ReLU
    down_128:      Linear(256→128) + LayerNorm + ReLU
    magnitude_head: Linear(128→1)
    sign_head:      Linear(128→1)
```

checkpoint：`G0-G1-G2/V2/V2-1_magnitude_learning/checkpoints/main_lambda_0p50/best.pth`

---

## 10. V2-4C-3E1 BinaryAttenuationPredictor（当前最好离线模型）结构

```python
class BinaryAttenuationPredictor(nn.Module):
    # input: 129维（127 + 2个额外特征）
    input_proj:     Linear(129→512) + LayerNorm + ReLU + Dropout
    residual:       Linear(512→512) + LayerNorm + ReLU + Dropout
    down_256:       Linear(512→256) + LayerNorm + ReLU
    down_128:       Linear(256→128) + LayerNorm + ReLU
    head:           Linear(128→1)
```

checkpoint：`G0-G1-G2/V2/V2-4_learnable_adaptive_fusion/checkpoints/V2-4C-3E1_binary_attenuation/best.pth`

---

## 11. 待 V2-4D-1 确认的问题

在开始写代码之前，需要确认以下 **2 个问题**，请用户回答：

**Q1. Innovation-3 的输入 feature 来源**

方案 A（推荐）：从 `occ_head.forward` 内部 hook `dense_decoder` 的输出，针对每个 candidate 区域 R 做 RoI mean pooling，得到 [N_cand, 256] 特征，送入新的 MLP。

方案 B（简单）：沿用离线方案，用 Pv[R] 和 Pi_aligned[R] 的统计量（均值等，~127维），不解冻 occ 内部特征。

你倾向于哪个？（任务书 § 4.9 说"哪个中间 feature 最适合供创新三使用"，未强制规定。）

**Q2. Joint fine-tuning 的 loss 设计**

创新三的训练标签是 `y = 1[U(0.1) > U(1.0)]`（已在离线数据中预计算）。  
联合微调时 occ_head 也需要 occupancy loss（dice + mask）。两个 loss 如何加权？

- 选项 A：只训练创新三 binary cross-entropy loss，occ 那部分 loss 权重=0（彻底不更新 occ late layers via occ loss，只通过 fusion 决策 loss 更新）
- 选项 B：occ loss（dice+mask）+ BCE for 创新三，分别带权，occ loss 起 regularizer 作用

任务书未明确，建议 B，但需你确认。
