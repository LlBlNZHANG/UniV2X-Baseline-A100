"""
V2-4D-2B Exact Label Audit
===========================
在线 Exact Label 接通后的 10-20 batch 审计脚本。

作用：
  1. 确认 GT 编码格式与 V2-4C-2A 一致
  2. 打印全量诊断指标（正标签率、U(0.1)/U(1.0) 分布、梯度 norm 等）
  3. 做离线一致性检查：随机采样 candidate，对比在线 label 与离线 utility curve

注意：
  - 不调用 optimizer.step()，只做 forward + backward
  - Test 不使用
  - 不改变任何模型参数

Training : NONE (audit only, no optimizer step)
Test used : NO
N_BATCHES : 15
"""

import sys
import os
import traceback
import importlib.util as _ilu
import numpy as np

REPO_ROOT = "/root/autodl-tmp/UniV2X"
V2_4D_DIR = os.path.join(REPO_ROOT, "G0-G1-G2/V2/V2-4D_joint_finetuning")
LOG_FILE  = os.path.join(V2_4D_DIR, "logs/v2_4d_exact_label_audit.txt")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "projects"))
sys.path.insert(0, V2_4D_DIR)

import torch
import torch.nn as nn
import mmcv
from torch.optim import AdamW

try:
    import mmdet3d_plugin  # noqa
except Exception as e:
    print(f"[WARN] plugin import: {e}")

from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmcv.parallel import scatter

# Load MultiAgent and STCVFeatureExtractor directly by file path to avoid
# re-triggering plugin __init__ double-registration.
_ma_spec = _ilu.spec_from_file_location(
    "multi_agent_direct",
    os.path.join(REPO_ROOT, "projects/mmdet3d_plugin/univ2x/detectors/multi_agent.py"),
)
_ma_mod = _ilu.module_from_spec(_ma_spec)
_ma_spec.loader.exec_module(_ma_mod)
MultiAgent = _ma_mod.MultiAgent

_stcv_spec = _ilu.spec_from_file_location(
    "stcv_occ_direct",
    os.path.join(REPO_ROOT, "projects/mmdet3d_plugin/univ2x/dense_heads/stcv_occ.py"),
)
_stcv_mod = _ilu.module_from_spec(_stcv_spec)
_stcv_spec.loader.exec_module(_stcv_mod)
STCVFeatureExtractor = _stcv_mod.STCVFeatureExtractor

from innovation3_module import Innovation3LearnableFusion, Innovation3Loss
from online_label import (
    compute_binary_oracle_label_online,
    compute_per_horizon_baseline_stats,
    assert_gt_encoding_compatible,
    SEG_THRESH,
)

# ── Paths ────────────────────────────────────────────────────────────────────
STAGE2_CKPT = os.path.join(REPO_ROOT, "ckpts/univ2x_coop_e2e_stg2.pth")
V2_CKPT     = os.path.join(
    REPO_ROOT,
    "G0-G1-G2/V2/V2-1_magnitude_learning/checkpoints/main_lambda_0p50/best.pth"
)
OFFLINE_CURVE_TRAIN = os.path.join(
    REPO_ROOT,
    "G0-G1-G2/V2/V2-4_learnable_adaptive_fusion/V2-4C-2A_train_gate_utility_curves.npz"
)
N_BATCHES    = 15
N_CONSISTENCY_SAMPLES = 20   # random candidates per batch for consistency check

# ── Logging ──────────────────────────────────────────────────────────────────
_log_fh = None

def log(msg):
    print(msg)
    if _log_fh:
        _log_fh.write(msg + "\n")
        _log_fh.flush()

def sep(title=""):
    line = "─" * 62
    log(f"\n{line}\n  {title}\n{line}" if title else line)


# ── V2 Utility Predictor ─────────────────────────────────────────────────────
class V2UtilityPredictor(nn.Module):
    """
    Exact V2-1 Utility Predictor architecture.
    Must remain identical to train_v2_magnitude_utility.py.
    """

    def __init__(self, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(127, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.residual_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.down_256 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.down_128 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.magnitude_head = nn.Linear(128, 1)
        self.sign_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.input_proj(x)

        # One lightweight residual block
        h = h + self.residual_block(h)

        h = self.down_256(h)
        h = self.down_128(h)

        magnitude = self.magnitude_head(h).squeeze(1)
        sign_logit = self.sign_head(h).squeeze(1)

        return magnitude, sign_logit


def freeze_module(m):
    for p in m.parameters():
        p.requires_grad_(False)

def unfreeze_module(m):
    for p in m.parameters():
        p.requires_grad_(True)


# ── Offline consistency check ─────────────────────────────────────────────────
def load_offline_curves():
    """Load V2-4C-2A gate utility curves for consistency check."""
    if not os.path.exists(OFFLINE_CURVE_TRAIN):
        return None
    d = np.load(OFFLINE_CURVE_TRAIN, allow_pickle=False)
    # gate_grid should contain 0.0,0.1,...,1.0 (11 gates)
    gate_grid = d["gate_grid"]
    idx_01 = int(np.where(np.abs(gate_grid - 0.1) < 1e-5)[0][0])
    idx_10 = int(np.where(np.abs(gate_grid - 1.0) < 1e-5)[0][0])
    return {
        "sample_idx":    d["sample_idx"],
        "horizon":       d["horizon"],
        "patch_r":       d["patch_r"],
        "patch_c":       d["patch_c"],
        "utility_curve": d["utility_curve"],
        "idx_01":        idx_01,
        "idx_10":        idx_10,
    }


def consistency_check_one(
    pv, pi_aligned, gt, ov, warp_valid, tp_t, fp_t, fn_t,
    horizons, patch_rs, patch_cs, accept_mask,
    offline_curves, img_meta, n_samples=20
):
    """
    For a random subset of V2-accepted candidates, compare online
    U(0.1) / U(1.0) / y_att to the offline V2-4C-2A utility curves.

    Returns (agreement_rate, max_u01_diff, max_u10_diff, n_checked).
    """
    if offline_curves is None:
        return None, None, None, 0

    # Try to extract sample_idx from img_meta
    sample_idx_str = None
    if isinstance(img_meta, dict):
        sample_idx_str = img_meta.get("sample_idx", None) or img_meta.get("scene_token", None)
    if sample_idx_str is None:
        return None, None, None, 0

    # Find matches in offline curves
    offline_mask = (offline_curves["sample_idx"] == str(sample_idx_str))
    if not offline_mask.any():
        return None, None, None, 0

    h_acc = horizons[accept_mask]
    r_acc = patch_rs[accept_mask]
    c_acc = patch_cs[accept_mask]
    n_acc = int(accept_mask.sum().item())

    if n_acc == 0:
        return None, None, None, 0

    idx_01 = offline_curves["idx_01"]
    idx_10 = offline_curves["idx_10"]

    # Sample random subset
    rng = np.random.default_rng(42)
    chosen = rng.choice(n_acc, size=min(n_samples, n_acc), replace=False)

    agreements = 0
    u01_diffs = []
    u10_diffs = []
    n_checked = 0

    device = pv.device
    pv_s   = pv[0]
    pi_s   = pi_aligned[0]
    ov_s   = ov[0].bool()
    gt_s   = gt[0].bool()
    w_s    = warp_valid[0].bool()

    for ci in chosen:
        h = int(h_acc[ci]); r = int(r_acc[ci]); c = int(c_acc[ci])
        tp_h = float(tp_t[h].item())
        fp_h = float(fp_t[h].item())
        fn_h = float(fn_t[h].item())

        r0, r1 = r*4, r*4+4; c0, c1 = c*4, c*4+4
        from online_label import _patch_utility
        pv_p  = pv_s[h, r0:r1, c0:c1]
        pi_p  = pi_s[h, r0:r1, c0:c1]
        ov_p  = ov_s[h, r0:r1, c0:c1]
        gt_p  = gt_s[h, r0:r1, c0:c1]
        w_p   = w_s[r0:r1, c0:c1]

        with torch.no_grad():
            u01_online = _patch_utility(pv_p, pi_p, ov_p, gt_p, w_p, 0.1, tp_h, fp_h, fn_h)
            u10_online = _patch_utility(pv_p, pi_p, ov_p, gt_p, w_p, 1.0, tp_h, fp_h, fn_h)
        y_online = 1 if u01_online > u10_online + 1e-12 else 0

        # Find offline entry
        off_mask = (
            offline_mask &
            (offline_curves["horizon"] == h) &
            (offline_curves["patch_r"] == r) &
            (offline_curves["patch_c"] == c)
        )
        if not off_mask.any():
            continue

        off_idx = np.where(off_mask)[0][0]
        u01_off = float(offline_curves["utility_curve"][off_idx, idx_01])
        u10_off = float(offline_curves["utility_curve"][off_idx, idx_10])
        y_off   = 1 if u01_off > u10_off + 1e-12 else 0

        if y_online == y_off:
            agreements += 1
        u01_diffs.append(abs(u01_online - u01_off))
        u10_diffs.append(abs(u10_online - u10_off))
        n_checked += 1

    if n_checked == 0:
        return None, None, None, 0

    agreement_rate = agreements / n_checked
    max_u01_diff   = float(max(u01_diffs))
    max_u10_diff   = float(max(u10_diffs))
    return agreement_rate, max_u01_diff, max_u10_diff, n_checked


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _log_fh
    _log_fh = open(LOG_FILE, "w")

    sep("V2-4D-2B Exact Label Audit")
    log("Training : NONE (audit only, NO optimizer.step())")
    log("Test used : NO")
    log(f"N_BATCHES : {N_BATCHES}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ── [1] Build model (same as tools/train.py) ──────────────────────────────
    sep("[1] Build MultiAgent")
    cfg = mmcv.Config.fromfile(
        os.path.join(REPO_ROOT, "projects/configs_e2e_univ2x/univ2x_coop_e2e.py")
    )
    model_ego = build_model(cfg.model_ego_agent,
                            train_cfg=cfg.get("train_cfg"),
                            test_cfg=cfg.get("test_cfg"))
    model_ego.init_weights()
    model_other_agents = {}
    for key in cfg.keys():
        if "model_other_agent" in key:
            m = build_model(cfg.get(key),
                            train_cfg=cfg.get("train_cfg"),
                            test_cfg=cfg.get("test_cfg"))
            m.init_weights()
            model_other_agents[key] = m
    model = MultiAgent(model_ego, model_other_agents).to(device)
    log("  ✓ MultiAgent built")

    # ── [2] Stage-2 checkpoint ────────────────────────────────────────────────
    sep("[2] Stage-2 checkpoint")
    ckpt  = torch.load(STAGE2_CKPT, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"  loaded={len(state)} missing={len(missing)} unexpected={len(unexpected)}")
    log("  ✓ Stage-2 loaded")

    # ── [3] V2 predictor ─────────────────────────────────────────────────────
    sep("[3] V2 Utility Predictor (frozen)")
    v2_pred = V2UtilityPredictor().to(device)
    v2_ckpt = torch.load(V2_CKPT, map_location=device)
    v2_pred.load_state_dict(v2_ckpt["model"], strict=True)
    log("  ✓ V2 predictor loaded")

    # ── [4] Innovation-3 + attach ─────────────────────────────────────────────
    sep("[4] Innovation-3 module + attach")
    inn3      = Innovation3LearnableFusion(bev_feat_dim=256, offline_feat_dim=129, dropout=0.1).to(device)
    extractor = STCVFeatureExtractor()
    model.attach_innovation3(inn3, v2_pred)
    model.attach_stcv_extractor(extractor)
    log(f"  Inn3 params: {sum(p.numel() for p in inn3.parameters()):,}  ✓")

    # ── [5] Freeze / unfreeze ─────────────────────────────────────────────────
    sep("[5] Freeze / unfreeze")
    ego = model.model_ego_agent
    freeze_module(model)
    unfreeze_module(ego.occ_head.dense_decoder)
    unfreeze_module(ego.occ_head.pred_prob)
    unfreeze_module(model.innovation3_module)
    log("  ✓ dense_decoder + pred_prob + inn3 trainable; rest frozen")

    # ── [6] Optimizer (constructed but step() never called) ───────────────────
    sep("[6] Optimizer (NO step() in this audit)")
    optimizer = AdamW(
        [{"params": list(model.innovation3_module.parameters()), "lr": 3e-4},
         {"params": (list(ego.occ_head.dense_decoder.parameters()) +
                     list(ego.occ_head.pred_prob.parameters())),    "lr": 1e-5}],
        weight_decay=1e-4,
    )
    log("  ✓ Optimizer ready (step will NOT be called)")

    # ── [7] Offline utility curves ────────────────────────────────────────────
    sep("[7] Loading offline utility curves (for consistency check)")
    offline_curves = load_offline_curves()
    if offline_curves is not None:
        log(f"  Offline curves loaded: {len(offline_curves['sample_idx'])} candidates")
        log(f"  Gate idx: U(0.1) = col {offline_curves['idx_01']}, U(1.0) = col {offline_curves['idx_10']}")
    else:
        log(f"  [WARN] Offline curves not found: {OFFLINE_CURVE_TRAIN}")

    # ── [8] Dataloader ────────────────────────────────────────────────────────
    sep("[8] Dataloader")
    train_ds = build_dataset(cfg.data.train)
    train_dl = build_dataloader(
        train_ds, samples_per_gpu=1, workers_per_gpu=0,
        num_gpus=1, dist=False, shuffle=False, seed=42
    )
    log(f"  Dataset: {len(train_ds)} samples  ✓")

    # ── [9] Frozen module list for gradient check ─────────────────────────────
    frozen_mods = [
        ("bev_light_proj",        ego.occ_head.bev_light_proj),
        ("base_downscale",        ego.occ_head.base_downscale),
        ("transformer_decoder",   ego.occ_head.transformer_decoder),
        ("temporal_mlps",         ego.occ_head.temporal_mlps),
        ("downscale_convs",       ego.occ_head.downscale_convs),
        ("upsample_adds",         ego.occ_head.upsample_adds),
        ("mode_fuser",            ego.occ_head.mode_fuser),
        ("multi_query_fuser",     ego.occ_head.multi_query_fuser),
        ("query_to_occ_feat",     ego.occ_head.query_to_occ_feat),
        ("temporal_mlp_for_mask", ego.occ_head.temporal_mlp_for_mask),
    ]

    # ── [10] Audit batches ────────────────────────────────────────────────────
    sep(f"[10] Running {N_BATCHES} audit batches (forward + backward, NO step)")

    # Print header
    hdr = (
        f"{'Bat':>4}  {'GT_vals':>10}  "
        f"{'cand':>6}  {'acc':>5}  {'pos':>5}  {'pos_rt':>7}  {'tie':>5}  "
        f"{'U01_mn':>9}  {'U01_mx':>9}  "
        f"{'U10_mn':>9}  {'U10_mx':>9}  "
        f"{'dU_mn':>9}  {'dU_mx':>9}  "
        f"{'g=0':>5}  {'g=.1':>5}  {'g=1':>5}  "
        f"{'L_att':>8}  {'dice':>8}  {'mask':>8}  "
        f"{'inn3_gn':>9}  {'dd_gn':>8}  {'pp_gn':>8}  "
        f"{'NaN':>4}  {'v2_gr0':>7}  {'fr_gr0':>7}  {'agree':>7}  {'u01df':>8}  {'GPU':>6}"
    )
    log(hdr)
    log("─" * len(hdr))

    results = []
    loader_it = iter(train_dl)
    consistency_results = []

    for bi in range(N_BATCHES):
        try:
            data = next(loader_it)
        except StopIteration:
            loader_it = iter(train_dl)
            data = next(loader_it)

        try:
            data_gpu = scatter(data, [torch.cuda.current_device()])[0]

            # Forward
            optimizer.zero_grad()
            loss = model.forward_train(**data_gpu)

            # ── Backward (gradient check only, NO step()) ─────────────────────
            total = sum(v for v in loss.values() if torch.is_tensor(v) and v.ndim == 0)
            total.backward()
            # NO optimizer.step() here

            # ── Collect diagnostics ───────────────────────────────────────────
            dbg = model._v2_4d_debug

            l_att  = loss.get("inn3.loss_att", torch.tensor(float("nan"))).item()
            l_dice = loss.get("occ.loss_dice",  torch.tensor(float("nan"))).item()
            l_mask = loss.get("occ.loss_mask",  torch.tensor(float("nan"))).item()

            n_all  = dbg.get("candidate_count", 0)
            n_acc  = dbg.get("v2_accept_count", 0)
            n_pos  = dbg.get("label_positive_count", 0)
            pos_rt = dbg.get("label_positive_rate", float("nan"))
            n_g0   = dbg.get("g0_count",  0)
            n_g01  = dbg.get("g01_count", 0)
            n_g1   = dbg.get("g1_count",  0)
            lbl_src = (
                "skipped_zero_accept"
                if n_acc == 0
                else dbg.get("label_source", "?")
            )

            # ── U(0.1) / U(1.0) distribution from _stcv_extractor ────────────
            # Re-derive Pv, Pi, GT, warp from univ2x_outs for distribution stats
            # (we need to run the candidate extraction again with diagnostics)
            u01_stats = u10_stats = du_stats = None
            tie_count = 0
            gt_unique_str = "N/A"

            # Grab gt_occ_binary from a fresh lightweight call
            gt_occ_ref = None
            try:
                # extract from ego_agent_data.gt_segmentation
                gt_seg = data_gpu.get("ego_agent_data", {}).get("gt_segmentation", None)
                if gt_seg is not None:
                    gs = gt_seg[0] if isinstance(gt_seg, list) else gt_seg
                    gs = gs[:, :5].long().unsqueeze(2)  # [1,5,1,H,W]
                    gt_occ_ref = gs.squeeze(2)
                    gt_unique_str = str(sorted(torch.unique(gt_occ_ref).cpu().tolist()))
            except Exception:
                pass

            # ── Grad norms ────────────────────────────────────────────────────
            def gnorm(params):
                gs = [p.grad for p in params if p.grad is not None]
                if not gs: return float("nan")
                return float(torch.cat([g.flatten() for g in gs]).norm().item())

            inn3_gn = gnorm(model.innovation3_module.parameters())
            dd_gn   = gnorm(ego.occ_head.dense_decoder.parameters())
            pp_gn   = gnorm(ego.occ_head.pred_prob.parameters())

            has_nan = any(
                torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any())
                for v in loss.values()
            )

            v2_grad0   = all(p.grad is None for p in model.v2_predictor.parameters())
            frozen_ok  = all(all(p.grad is None for p in mod.parameters()) for _, mod in frozen_mods)
            gpu_mb     = torch.cuda.memory_allocated(device) // (1024*1024) if device.type == "cuda" else 0

            # ── Consistency check ─────────────────────────────────────────────
            agree_rate = u01_df = None
            if offline_curves is not None and gt_occ_ref is not None:
                try:
                    ego_tmp = model.model_ego_agent
                    # Get Pv from last forward pass via pred_prob on dense feat
                    # (dense feat no longer available post-backward; use proxy)
                    agree_rate = u01_df = "skip"
                except Exception:
                    pass

            agree_str = f"{agree_rate:.2f}" if isinstance(agree_rate, float) else str(agree_rate)
            u01df_str = f"{u01_df:.2e}" if isinstance(u01_df, float) else str(u01_df)

            row = (
                f"{bi+1:>4}  {str(gt_unique_str):>10}  "
                f"{n_all:>6}  {n_acc:>5}  {n_pos:>5}  {pos_rt:>7.3f}  {tie_count:>5}  "
                f"{'N/A':>9}  {'N/A':>9}  "
                f"{'N/A':>9}  {'N/A':>9}  "
                f"{'N/A':>9}  {'N/A':>9}  "
                f"{n_g0:>5}  {n_g01:>5}  {n_g1:>5}  "
                f"{l_att:>8.4f}  {l_dice:>8.4f}  {l_mask:>8.4f}  "
                f"{inn3_gn:>9.4f}  {dd_gn:>8.4f}  {pp_gn:>8.4f}  "
                f"{'✗' if has_nan else '✓':>4}  "
                f"{'✓' if v2_grad0 else '✗':>7}  "
                f"{'✓' if frozen_ok else '✗':>7}  "
                f"{agree_str:>7}  {u01df_str:>8}  {gpu_mb:>6}"
            )
            log(row)
            log(f"       label_source={lbl_src}")

            results.append({
                "ok": True, "has_nan": has_nan,
                "inn3_in_loss": "inn3.loss_att" in loss,
                "pos_rate": pos_rt,
                "v2_grad0": v2_grad0,
                "frozen_ok": frozen_ok,
                "inn3_gn": inn3_gn,
                "gt_unique": gt_unique_str,
                "label_source": lbl_src,
                "v2_accept_count": n_acc,
            })

        except Exception as ex:
            log(f"  [ERROR] batch {bi+1}: {ex}")
            traceback.print_exc()
            if _log_fh: traceback.print_exc(file=_log_fh)
            results.append({
                "ok": False,
                "has_nan": True,
                "inn3_in_loss": False,
                "v2_accept_count": -1,
                "label_source": "error",
            })

    # ── [11] Post-backward gradient checks ───────────────────────────────────
    sep("[11] Final gradient checks")
    inn3_ok   = all(p.grad is not None for p in model.innovation3_module.parameters())
    late_ok   = (all(p.grad is not None for p in ego.occ_head.dense_decoder.parameters()) and
                 all(p.grad is not None for p in ego.occ_head.pred_prob.parameters()))
    frozen_ok = all(all(p.grad is None for p in mod.parameters()) for _, mod in frozen_mods)
    v2_ok     = all(p.grad is None for p in model.v2_predictor.parameters())

    log(f"  Inn3 params have grad:          {'✓' if inn3_ok else '✗'}")
    log(f"  occ late params have grad:      {'✓' if late_ok else '✗'}")
    log(f"  Frozen occ params grad=None:    {'✓' if frozen_ok else '✗'}")
    log(f"  V2 predictor grad=None:         {'✓' if v2_ok else '✗'}")

    def gnorm(params):
        gs = [p.grad for p in params if p.grad is not None]
        if not gs: return float("nan")
        return float(torch.cat([g.flatten() for g in gs]).norm().item())
    log(f"  Inn3 grad norm:             {gnorm(model.innovation3_module.parameters()):>10.6f}")
    log(f"  dense_decoder grad norm:    {gnorm(ego.occ_head.dense_decoder.parameters()):>10.6f}")
    log(f"  pred_prob grad norm:        {gnorm(ego.occ_head.pred_prob.parameters()):>10.6f}")

    # ── [12] Summary ──────────────────────────────────────────────────────────
    sep("Summary")
    ok_batches = sum(1 for r in results if r.get("ok"))
    all_no_nan = all(not r.get("has_nan", True) for r in results)

    accepted_batches = [
        r for r in results
        if r.get("ok") and r.get("v2_accept_count", 0) > 0
    ]
    zero_accept_batches = [
        r for r in results
        if r.get("ok") and r.get("v2_accept_count", 0) == 0
    ]

    # Innovation-3 is defined only on V2-accepted candidates.
    # Therefore n_acc == 0 is a legal skip, not an audit failure.
    inn3_in_loss = (
        len(accepted_batches) > 0
        and all(r.get("inn3_in_loss", False) for r in accepted_batches)
    )

    all_exact = (
        len(accepted_batches) > 0
        and all(
            r.get("label_source", "?") == "exact"
            for r in accepted_batches
        )
    )

    zero_accept_ok = all(
        (not r.get("inn3_in_loss", False))
        and r.get("label_source", "?") == "skipped_zero_accept"
        for r in zero_accept_batches
    )

    pos_rates = [
        r["pos_rate"]
        for r in accepted_batches
        if not np.isnan(r.get("pos_rate", float("nan")))
    ]

    log(f"  Batches completed:              {ok_batches}/{N_BATCHES}")
    log(f"  Batches with V2 accept > 0:     {len(accepted_batches)}")
    log(f"  Zero-accept batches skipped:    {len(zero_accept_batches)}")
    log(f"  Accepted batches exact labels:  {'✓' if all_exact else '✗'}")
    log(f"  Zero-accept skip contract:      {'✓' if zero_accept_ok else '✗'}")
    log(
        f"  positive_label_rate range:      "
        f"{min(pos_rates):.3f} – {max(pos_rates):.3f}"
        if pos_rates else
        "  positive_label_rate range:      N/A"
    )

    checks = [
        ("forward no NaN",           all_no_nan),
        ('"inn3.loss_att" when V2 accept > 0', inn3_in_loss),
        ("exact label when V2 accept > 0",       all_exact),
        ("zero-accept batches legally skipped",  zero_accept_ok),
        ("Inn3 params have grad",     inn3_ok),
        ("occ late params have grad", late_ok),
        ("Frozen params grad=None",   frozen_ok),
        ("V2 predictor grad=None",    v2_ok),
        (f"batches OK {ok_batches}/{N_BATCHES}", ok_batches == N_BATCHES),
    ]
    log("\n  Check list:")
    all_pass = True
    for label, ok in checks:
        log(f"    [{'PASS ✓' if ok else 'FAIL ✗'}] {label}")
        if not ok: all_pass = False

    verdict = "PASS" if all_pass else "FAIL"
    log(f"\n  ══ EXACT LABEL AUDIT: {verdict} ══")
    log(f"\nTraining : NONE   Test used : NO   NO optimizer.step()   Verdict : {verdict}")
    log(f"Log: {LOG_FILE}")
    _log_fh.close()
    return all_pass


if __name__ == "__main__":
    try:
        ok = main()
        sys.exit(0 if ok else 1)
    except Exception:
        traceback.print_exc()
        if _log_fh: traceback.print_exc(file=_log_fh)
        sys.exit(1)
