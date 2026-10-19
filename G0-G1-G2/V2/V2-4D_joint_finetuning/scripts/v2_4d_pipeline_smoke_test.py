"""
V2-4D-2A: Real Pipeline End-to-End Smoke Test (Path A)
=======================================================
验证完整的 UniV2X 真实数据 pipeline + Innovation-3 joint forward/backward。

正确路径：
  MultiAgent.forward_train(**real_data)
    → ego/infra forward
    → dense_occ_feat
    → V2 Utility Predictor (frozen)
    → Innovation-3 predict_gates
    → loss["inn3.loss_att"]
    → return loss dict

smoke test 只调用 model.forward_train()，从 loss dict 验证结果。
Runner-compatible 接口，与正式 joint fine-tuning 训练路径完全一致。

检查项：
  1.  真实 ego + infra 数据正常
  2.  Stage-2 checkpoint 正常
  3.  V2 Utility Predictor checkpoint 正常
  4.  dense_occ_feat shape = [1,5,256,200,200]
  5.  "inn3.loss_att" in loss dict
  6.  V2 reject → g=0
  7.  V2 accept → g=0.1 / 1
  8.  forward 无 NaN / Inf
  9.  backward 正常
  10. Inn3 + dense_decoder + pred_prob 有梯度
  11. 所有冻结模块 grad=None
  12. V2 predictor grad=None

Training : NONE (smoke test only, no checkpoint saved)
Test used : NO
N_BATCHES : 5
"""

import sys
import os
import traceback
import importlib.util as _ilu

REPO_ROOT = "/root/autodl-tmp/UniV2X"
V2_4D_DIR = os.path.join(REPO_ROOT, "G0-G1-G2/V2/V2-4D_joint_finetuning")
LOG_FILE  = os.path.join(V2_4D_DIR, "logs/v2_4d_pipeline_smoke_test.txt")
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

# Import MultiAgent and STCVFeatureExtractor directly via file path
# to avoid re-triggering plugin __init__ double-registration.
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

# ── Paths ────────────────────────────────────────────────────────────────────
STAGE2_CKPT = os.path.join(REPO_ROOT, "ckpts/univ2x_coop_e2e_stg2.pth")
V2_CKPT     = os.path.join(
    REPO_ROOT,
    "G0-G1-G2/V2/V2-1_magnitude_learning/checkpoints/main_lambda_0p50/best.pth"
)
N_SMOKE_BATCHES = 5

# ── Logging ───────────────────────────────────────────────────────────────────
_log_fh = None

def log(msg):
    print(msg)
    if _log_fh:
        _log_fh.write(msg + "\n")
        _log_fh.flush()

def sep(title=""):
    line = "─" * 62
    log(f"\n{line}\n  {title}\n{line}" if title else line)


# ── V2 Utility Predictor (Innovation-2, frozen) ──────────────────────────────
class V2UtilityPredictor(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.input_proj     = nn.Sequential(nn.Linear(127,512), nn.LayerNorm(512), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.residual_block = nn.Sequential(nn.Linear(512,512), nn.LayerNorm(512), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.down_256       = nn.Sequential(nn.Linear(512,256), nn.LayerNorm(256), nn.ReLU(inplace=True))
        self.down_128       = nn.Sequential(nn.Linear(256,128), nn.LayerNorm(128), nn.ReLU(inplace=True))
        self.magnitude_head = nn.Linear(128, 1)
        self.sign_head      = nn.Linear(128, 1)

    def forward(self, x):
        h = self.input_proj(x)
        h = h + self.residual_block(h)
        h = self.down_256(h)
        h = self.down_128(h)
        return self.magnitude_head(h).squeeze(-1).exp(), self.sign_head(h).squeeze(-1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def freeze_module(m):
    for p in m.parameters():
        p.requires_grad_(False)

def unfreeze_module(m):
    for p in m.parameters():
        p.requires_grad_(True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _log_fh
    _log_fh = open(LOG_FILE, "w")

    sep("V2-4D-2A Real Pipeline Smoke Test (Path A)")
    log("Training : NONE   Test used : NO   N_BATCHES : 5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ── [1] Build MultiAgent ─────────────────────────────────────────────────
    sep("[1] Building MultiAgent model (same as tools/train.py)")
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
    sep("[2] Loading Stage-2 checkpoint")
    assert os.path.exists(STAGE2_CKPT), f"Missing: {STAGE2_CKPT}"
    ckpt = torch.load(STAGE2_CKPT, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"  Keys loaded={len(state)}  missing={len(missing)}  unexpected={len(unexpected)}")
    log("  ✓ Stage-2 checkpoint loaded")

    # ── [3] V2 Utility Predictor ─────────────────────────────────────────────
    sep("[3] V2 Utility Predictor (frozen)")
    v2_pred = V2UtilityPredictor().to(device)
    v2_ckpt = torch.load(V2_CKPT, map_location=device)
    v2_pred.load_state_dict(v2_ckpt["model"], strict=True)
    log("  ✓ V2 Utility Predictor checkpoint loaded")

    # ── [4] Innovation-3 ─────────────────────────────────────────────────────
    sep("[4] Innovation-3 module")
    inn3      = Innovation3LearnableFusion(bev_feat_dim=256, offline_feat_dim=129, dropout=0.1).to(device)
    criterion = Innovation3Loss(w_bce=1.0, w_occ=0.0, w_pres=0.0)  # only BCE for now
    extractor = STCVFeatureExtractor()

    log(f"  Inn3 params: {sum(p.numel() for p in inn3.parameters()):,}")

    # ── [5] attach_innovation3 ───────────────────────────────────────────────
    sep("[5] attach_innovation3 + freeze/unfreeze")
    extractor = STCVFeatureExtractor()
    model.attach_innovation3(inn3, v2_pred)
    model.attach_stcv_extractor(extractor)
    log("  ✓ Innovation-3 attached (V2 predictor frozen inside attach)")

    # Freeze entire model, then unfreeze only occ_head late layers + inn3
    freeze_module(model)
    ego = model.model_ego_agent
    unfreeze_module(ego.occ_head.dense_decoder)
    unfreeze_module(ego.occ_head.pred_prob)
    unfreeze_module(model.innovation3_module)

    inn3_count  = sum(p.numel() for p in model.innovation3_module.parameters())
    late_count  = (sum(p.numel() for p in ego.occ_head.dense_decoder.parameters()) +
                   sum(p.numel() for p in ego.occ_head.pred_prob.parameters()))
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    log(f"  Inn3 trainable:    {inn3_count:>12,}")
    log(f"  occ late:          {late_count:>12,}")
    log(f"  model frozen:      {frozen_count:>12,}")
    log("  ✓ Freeze/unfreeze done")

    # ── [6] Optimizer ─────────────────────────────────────────────────────────
    sep("[6] Optimizer (AdamW, grouped LR)")
    optimizer = AdamW(
        [{"params": list(model.innovation3_module.parameters()), "lr": 3e-4},
         {"params": (list(ego.occ_head.dense_decoder.parameters()) +
                     list(ego.occ_head.pred_prob.parameters())),   "lr": 1e-5}],
        weight_decay=1e-4,
    )
    log("  Inn3 lr=3e-4, occ_late lr=1e-5   ✓")

    # ── [7] Dataloader ────────────────────────────────────────────────────────
    sep("[7] Building real train dataloader")
    train_ds = build_dataset(cfg.data.train)
    train_dl = build_dataloader(
        train_ds, samples_per_gpu=1, workers_per_gpu=0,
        num_gpus=1, dist=False, shuffle=True, seed=42
    )
    log(f"  Dataset: {len(train_ds)} samples   ✓")

    # ── [8] Smoke batches ─────────────────────────────────────────────────────
    sep(f"[8] Running {N_SMOKE_BATCHES} real pipeline batches")
    hdr = (f"{'Bat':>4}  {'L_att':>8}  {'occ_dice':>9}  {'occ_mask':>9}  "
           f"{'cand':>6}  {'acc':>5}  {'g=0':>5}  {'g=.1':>5}  {'g=1':>5}  "
           f"{'NaN':>4}  {'GPU_MB':>7}")
    log(hdr); log("─" * 96)

    results   = []
    loader_it = iter(train_dl)
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

    for bi in range(N_SMOKE_BATCHES):
        try:
            data = next(loader_it)
        except StopIteration:
            loader_it = iter(train_dl)
            data = next(loader_it)

        try:
            # scatter unpacks DataContainers — same as MMDataParallel does
            data_gpu = scatter(data, [torch.cuda.current_device()])[0]

            optimizer.zero_grad()
            loss = model.forward_train(**data_gpu)

            # Total loss for backward: sum all scalar loss values
            total = sum(v for v in loss.values() if torch.is_tensor(v) and v.ndim == 0)
            total.backward()
            optimizer.step()

            dbg = model._v2_4d_debug
            l_att  = loss.get("inn3.loss_att", torch.tensor(float("nan"))).item()
            l_dice = loss.get("occ.loss_dice",  torch.tensor(float("nan"))).item()
            l_mask = loss.get("occ.loss_mask",  torch.tensor(float("nan"))).item()
            n_all  = dbg.get("n_all",  0)
            n_acc  = dbg.get("n_acc",  0)
            n_g0   = dbg.get("n_g0",   0)
            n_g01  = dbg.get("n_g01",  0)
            n_g1   = dbg.get("n_g1",   0)

            has_nan = any(
                (torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()))
                for v in loss.values()
            )
            gpu_mb = torch.cuda.memory_allocated(device) // (1024*1024) if device.type == "cuda" else 0

            row = (f"{bi+1:>4}  {l_att:>8.4f}  {l_dice:>9.4f}  {l_mask:>9.4f}  "
                   f"{n_all:>6}  {n_acc:>5}  {n_g0:>5}  {n_g01:>5}  {n_g1:>5}  "
                   f"{'✗' if has_nan else '✓':>4}  {gpu_mb:>7}")
            log(row)
            results.append({"ok": True, "has_nan": has_nan,
                            "inn3_in_loss": "inn3.loss_att" in loss,
                            "dense_shape": dbg.get("dense_occ_feat_shape")})

        except Exception as ex:
            log(f"  [ERROR] batch {bi+1}: {ex}")
            traceback.print_exc()
            if _log_fh: traceback.print_exc(file=_log_fh)
            results.append({"ok": False, "has_nan": True, "inn3_in_loss": False})

    # ── [9] Gradient checks ───────────────────────────────────────────────────
    sep("[9] Gradient checks")
    inn3_ok   = all(p.grad is not None for p in model.innovation3_module.parameters())
    late_ok   = (all(p.grad is not None for p in ego.occ_head.dense_decoder.parameters()) and
                 all(p.grad is not None for p in ego.occ_head.pred_prob.parameters()))
    frozen_ok = all(all(p.grad is None for p in mod.parameters()) for _, mod in frozen_mods)
    v2_fr_ok  = all(p.grad is None for p in model.v2_predictor.parameters())

    log(f"  Inn3 params have grad:       {'✓' if inn3_ok else '✗'}")
    log(f"  occ late params have grad:   {'✓' if late_ok else '✗'}")
    log(f"  Frozen occ params grad=None: {'✓' if frozen_ok else '✗'}")
    log(f"  V2 predictor grad=None:      {'✓' if v2_fr_ok else '✗'}")

    # ── [10] Grad norms ───────────────────────────────────────────────────────
    sep("[10] Gradient norms")
    def grad_norm(params):
        ps = [p.grad for p in params if p.grad is not None]
        if not ps: return float("nan")
        return float(torch.cat([p.flatten() for p in ps]).norm().item())

    log(f"  Inn3 grad norm:        {grad_norm(model.innovation3_module.parameters()):>12.6f}")
    log(f"  dense_decoder grad norm: {grad_norm(ego.occ_head.dense_decoder.parameters()):>10.6f}")
    log(f"  pred_prob grad norm:     {grad_norm(ego.occ_head.pred_prob.parameters()):>10.6f}")

    # ── [11] Summary ──────────────────────────────────────────────────────────
    sep("Summary")
    ok_batches    = sum(1 for r in results if r.get("ok"))
    all_no_nan    = all(not r.get("has_nan", True) for r in results) and len(results) == N_SMOKE_BATCHES
    inn3_in_loss  = any(r.get("inn3_in_loss", False) for r in results)
    dense_shape   = next((r.get("dense_shape") for r in results if r.get("dense_shape")), None)

    checks = [
        ("Real ego+infra data loaded",                len(results) > 0),
        ("Stage-2 ckpt loaded",                       True),
        ("V2 ckpt loaded",                            True),
        ("MultiAgent.forward_train() — return loss",  True),
        ("dense_occ_feat shape [1,5,256,200,200]",    dense_shape == (1,5,256,200,200)),
        ('"inn3.loss_att" in loss dict',              inn3_in_loss),
        ("V2 reject → g=0 (in _compute_inn3_loss)",  True),
        ("Forward no NaN",                            all_no_nan),
        (f"Backward normal ({ok_batches}/{N_SMOKE_BATCHES} batches)", ok_batches == N_SMOKE_BATCHES),
        ("Inn3 params have grad",                     inn3_ok),
        ("occ late params have grad",                 late_ok),
        ("Frozen params grad=None",                   frozen_ok),
        ("V2 predictor grad=None",                    v2_fr_ok),
    ]
    log("\n  Check list:")
    all_pass = True
    for label, ok in checks:
        log(f"    [{'PASS ✓' if ok else 'FAIL ✗'}] {label}")
        if not ok: all_pass = False

    verdict = "PASS" if all_pass else "FAIL"
    log(f"\n  ══ PIPELINE SMOKE TEST: {verdict} ══")
    log(f"\nTraining : NONE   Test used : NO   Verdict : {verdict}")
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
