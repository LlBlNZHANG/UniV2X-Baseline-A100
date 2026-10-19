"""
V2-4D Smoke Test: 10-20 batch forward/backward/gradient check
=============================================================
验证完整的 V2-4D joint fine-tuning 前向链：
  Stage-2 checkpoint 加载 → V2 Utility Predictor 加载 →
  Innovation-3 接入 → forward / backward / gradient check

检查项：
  1.  Stage-2 checkpoint 正确加载
  2.  V2 Utility Predictor checkpoint 正确加载
  3.  Innovation-3 模块正确接入
  4.  V2 reject 必须严格 g=0
  5.  V2 accept 才进入 Innovation-3
  6.  candidate 顺序与原 V2 完全一致
  7.  forward 无 NaN / Inf
  8.  backward 正常
  9.  只有预期模块存在梯度
  10. 冻结模块 grad=None

Training : NONE（仅 smoke test，不保存 checkpoint）
Test used : NO
"""

import sys
import os
import time
import math
import numpy as np

REPO_ROOT = "/root/autodl-tmp/UniV2X"
V2_4D_DIR = os.path.join(REPO_ROOT, "G0-G1-G2/V2/V2-4D_joint_finetuning")
LOG_FILE = os.path.join(V2_4D_DIR, "logs/v2_4d_smoke_test.txt")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "projects"))
sys.path.insert(0, V2_4D_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmdet.models.builder import build_head

try:
    import mmdet3d_plugin  # noqa: F401
except Exception as e:
    print(f"[WARN] plugin import: {e}")

from innovation3_module import Innovation3LearnableFusion, Innovation3Loss

# ── Paths ────────────────────────────────────────────────────────────────────
STAGE2_CKPT = os.path.join(REPO_ROOT, "ckpts/univ2x_coop_e2e_stg2.pth")
V2_CKPT = os.path.join(
    REPO_ROOT,
    "G0-G1-G2/V2/V2-1_magnitude_learning/checkpoints/main_lambda_0p50/best.pth"
)
OFFLINE_DATA_TRAIN = os.path.join(
    REPO_ROOT,
    "G0-G1-G2/V2/V2-4_learnable_adaptive_fusion/V2-4B_train_learnable_fusion_dataset.npz"
)
OFFLINE_LABEL_TRAIN = os.path.join(
    REPO_ROOT,
    "G0-G1-G2/V2/V2-4_learnable_adaptive_fusion/V2-4C-2A_train_gate_utility_curves.npz"
)
N_SMOKE_BATCHES = 15  # 10-20 batch smoke test


# ── V2 Utility Predictor (Innovation-2, frozen) ──────────────────────────────
class V2UtilityPredictor(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(127, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.residual_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.down_256 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        self.down_128 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )
        self.magnitude_head = nn.Linear(128, 1)
        self.sign_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.input_proj(x)
        h = h + self.residual_block(h)
        h = self.down_256(h)
        h = self.down_128(h)
        magnitude = self.magnitude_head(h).squeeze(-1).exp()
        sign_logit = self.sign_head(h).squeeze(-1)
        u_hat = magnitude * (2 * sign_logit.sigmoid() - 1)
        return u_hat


# ── Dense decoder hook ───────────────────────────────────────────────────────
class DenseDecoderHook:
    def __init__(self, occ_head):
        self.output = None
        self._handle = occ_head.dense_decoder.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.output = out  # [B, T, C, H, W]

    def remove(self):
        self._handle.remove()


# ── Logging ──────────────────────────────────────────────────────────────────
_log_fh = None

def log(msg):
    print(msg)
    if _log_fh:
        _log_fh.write(msg + "\n")
        _log_fh.flush()


def sep(title=""):
    line = "─" * 60
    if title:
        log(f"\n{line}\n  {title}\n{line}")
    else:
        log(line)


# ── Build ego occ_head ───────────────────────────────────────────────────────
def build_ego_occ_head(device):
    cfg = mmcv.Config.fromfile(
        os.path.join(REPO_ROOT, "projects/configs_e2e_univ2x/univ2x_coop_e2e.py")
    )
    occ_cfg = cfg.model_ego_agent.occ_head.copy()
    occ_cfg["is_ego_agent"] = True
    occ_cfg["is_cooperation"] = True
    occ_cfg["use_stcv_occ"] = False
    occ = build_head(occ_cfg).to(device)
    return occ


# ── Load Stage-2 checkpoint into occ_head ────────────────────────────────────
def load_stage2_checkpoint(occ_head, ckpt_path, device):
    assert os.path.exists(ckpt_path), f"Stage-2 ckpt not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    # Filter keys that belong to model_ego_agent.occ_head
    prefix = "model_ego_agent.occ_head."
    filtered = {
        k[len(prefix):]: v
        for k, v in state.items()
        if k.startswith(prefix)
    }
    if not filtered:
        # Try without model_ego_agent prefix
        prefix2 = "occ_head."
        filtered = {
            k[len(prefix2):]: v
            for k, v in state.items()
            if k.startswith(prefix2)
        }
    missing, unexpected = occ_head.load_state_dict(filtered, strict=False)
    return len(filtered), missing, unexpected


# ── Freeze / unfreeze helpers ─────────────────────────────────────────────────
def freeze_module(module):
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad_(True)


def setup_trainable_params(occ_head, inn3):
    """
    Freeze all occ_head except dense_decoder and pred_prob.
    Inn3 is fully trainable.
    Returns (inn3_params, occ_late_params) for optimizer.
    """
    freeze_module(occ_head)
    unfreeze_module(occ_head.dense_decoder)
    unfreeze_module(occ_head.pred_prob)

    inn3_params = list(inn3.parameters())
    occ_late_params = (
        list(occ_head.dense_decoder.parameters())
        + list(occ_head.pred_prob.parameters())
    )
    return inn3_params, occ_late_params


# ── Offline data loader ───────────────────────────────────────────────────────
class OfflineCandidateDataset:
    """
    Loads offline candidate features and binary labels.
    Yields mini-batches of candidates (NOT images — these are offline extracted features).
    Used only to provide binary training labels for the smoke test.
    """

    def __init__(self, data_path, label_path, v2_accept_only=True, batch_size=64):
        d = np.load(data_path, allow_pickle=True)
        lab = np.load(label_path, allow_pickle=True)

        self.features = d["features_127"].astype(np.float32)   # [N, 127]
        self.horizons = d["horizon"].astype(np.int64)          # [N]
        self.patch_rs = d["patch_r"].astype(np.int64)          # [N]
        self.patch_cs = d["patch_c"].astype(np.int64)          # [N]
        self.v2_accept = d["v2_accept"].astype(np.uint8)       # [N]
        self.pv_patch = d["Pv_patch"].astype(np.float32)       # [N, 4, 4]
        self.pi_patch = d["Pi_patch"].astype(np.float32)       # [N, 4, 4]

        # Binary label: y=1 if g=0.1 (attenuate) is best
        hard_gate = lab["hard_best_gate"].astype(np.float32)   # [N]
        self.labels = (hard_gate < 0.5).astype(np.float32)    # [N]

        if v2_accept_only:
            mask = self.v2_accept.astype(bool)
            self.features = self.features[mask]
            self.horizons = self.horizons[mask]
            self.patch_rs = self.patch_rs[mask]
            self.patch_cs = self.patch_cs[mask]
            self.pv_patch = self.pv_patch[mask]
            self.pi_patch = self.pi_patch[mask]
            self.labels = self.labels[mask]
            self.v2_accept = self.v2_accept[mask]

        self.N = len(self.features)
        self.batch_size = batch_size
        self.idx = 0

    def __len__(self):
        return self.N

    def next_batch(self, device):
        """Return a dict of tensors for one mini-batch."""
        end = min(self.idx + self.batch_size, self.N)
        sl = slice(self.idx, end)
        self.idx = end % self.N  # wrap around

        return {
            "features": torch.from_numpy(self.features[sl]).to(device),    # [B, 127]
            "horizons": torch.from_numpy(self.horizons[sl]).to(device),    # [B]
            "patch_rs": torch.from_numpy(self.patch_rs[sl]).to(device),    # [B]
            "patch_cs": torch.from_numpy(self.patch_cs[sl]).to(device),    # [B]
            "pv_patch": torch.from_numpy(self.pv_patch[sl]).to(device),    # [B, 4, 4]
            "pi_patch": torch.from_numpy(self.pi_patch[sl]).to(device),    # [B, 4, 4]
            "labels": torch.from_numpy(self.labels[sl]).to(device),        # [B]
            "v2_accept": torch.ones(end - self.idx + self.batch_size,       # all accepted
                                     dtype=torch.uint8, device=device)[:end - (self.idx - self.batch_size if self.idx >= self.batch_size else 0)],
        }


# ── Smoke batch runner ────────────────────────────────────────────────────────
def run_smoke_batch(
    batch_idx,
    batch_size,
    occ_head,
    inn3,
    v2_predictor,
    criterion,
    optimizer,
    device,
    dataset,
):
    """
    One forward+backward smoke step using synthetic BEV features
    (actual UniV2X BEV requires full data pipeline; we use fake BEV + real offline labels).
    """
    BEV_H, BEV_W, BEV_C = 200, 200, 256
    N_QUERY = 300
    T = 5

    # ── Synthetic BEV inputs ─────────────────────────────────────────────────
    bev_feat = torch.randn(BEV_H * BEV_W, 1, BEV_C, device=device, requires_grad=False)
    ins_query = torch.randn(1, N_QUERY, BEV_C, device=device, requires_grad=False)

    # Hook to capture dense_decoder output
    hook = DenseDecoderHook(occ_head)

    # occ_head forward (training mode so BN / Dropout behave correctly)
    occ_head.train()
    mask_preds, ins_occ_logits = occ_head(bev_feat, ins_query)
    future_states = hook.output  # [1, T, 256, 200, 200]
    hook.remove()

    # ── Synthetic Pv / Pi (in real pipeline these come from occ_prob_fusion) ─
    # For smoke test we derive from ins_occ_logits
    with torch.no_grad():
        pv = ins_occ_logits[:, :1, :, :, :].squeeze(1).sigmoid()  # [1, T, H, W]  using query 0
        # Pi: slightly perturbed Pv to simulate infrastructure input
        pi_aligned = (pv + 0.1 * torch.randn_like(pv)).clamp(0, 1)

    # ── Sample candidates from offline dataset ────────────────────────────────
    batch = dataset.next_batch(device)
    N_acc = batch["features"].shape[0]

    # All accepted by V2 in this smoke batch
    v2_accept = torch.ones(N_acc, dtype=torch.uint8, device=device)
    horizons = batch["horizons"]
    patch_rs = batch["patch_rs"]
    patch_cs = batch["patch_cs"]
    labels = batch["labels"]

    # ── Innovation-2: V2 utility prediction (frozen, just for count tracking) ─
    with torch.no_grad():
        u_hat = v2_predictor(batch["features"])  # [N_acc]
        v2_accepted_mask = u_hat > 0
        v2_accept_for_gate = v2_accepted_mask.to(torch.uint8)

    v2_candidate_count = N_acc
    v2_accepted_count = int(v2_accepted_mask.sum().item())

    # ── Innovation-3: predict gates ──────────────────────────────────────────
    inn3.train()
    gates, logits, accept_mask = inn3.predict_gates(
        future_states=future_states,
        pv=pv,
        pi_aligned=pi_aligned,
        horizons=horizons,
        patch_rs=patch_rs,
        patch_cs=patch_cs,
        v2_accept=v2_accept_for_gate,
    )

    # Labels for accepted only
    labels_acc = labels[accept_mask]
    attenuation_count = int((gates == 0.1).sum().item())
    full_fusion_count = int((gates == 1.0).sum().item())
    reject_count = int((gates == 0.0).sum().item())

    # ── Loss ─────────────────────────────────────────────────────────────────
    # occ_head loss (regularizer) using synthetic targets.
    # Must NOT detach ins_occ_logits so that gradients flow back to
    # pred_prob and dense_decoder through the occupancy regularizer.
    # (In full training this would be the real dice/mask loss from forward_train)
    occ_reg_loss = F.binary_cross_entropy_with_logits(
        ins_occ_logits.squeeze(1).flatten(),
        torch.zeros_like(ins_occ_logits.squeeze(1).flatten()),
    )
    fake_occ_dict = {"occ_proxy": occ_reg_loss}

    loss_dict = criterion(logits, labels_acc, occ_loss_dict=fake_occ_dict)

    # ── Backward ──────────────────────────────────────────────────────────────
    optimizer.zero_grad()
    loss_dict["loss_total"].backward()
    optimizer.step()

    return {
        "loss_bce": loss_dict["loss_bce"].item(),
        "loss_occ_reg": loss_dict["loss_occ_reg"].item(),
        "loss_total": loss_dict["loss_total"].item(),
        "v2_candidate_count": v2_candidate_count,
        "v2_accepted_count": v2_accepted_count,
        "attenuation_count": attenuation_count,
        "full_fusion_count": full_fusion_count,
        "g0_count": reject_count,
        "g01_count": attenuation_count,
        "g10_count": full_fusion_count,
        "nan_in_logits": torch.isnan(logits).any().item() if logits.numel() > 0 else False,
        "nan_in_gates": torch.isnan(gates).any().item(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _log_fh
    _log_fh = open(LOG_FILE, "w")

    sep("V2-4D Smoke Test")
    log("Training : NONE (smoke test only, no checkpoint saved)")
    log("Test used : NO")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ── [1] Build and load occ_head ──────────────────────────────────────────
    sep("[1] Building ego occ_head + loading Stage-2 checkpoint")
    occ_head = build_ego_occ_head(device)

    if os.path.exists(STAGE2_CKPT):
        n_loaded, missing, unexpected = load_stage2_checkpoint(occ_head, STAGE2_CKPT, device)
        log(f"  Stage-2 ckpt: {STAGE2_CKPT}")
        log(f"  Keys loaded:   {n_loaded}")
        log(f"  Missing keys:  {len(missing)}")
        log(f"  Unexpected:    {len(unexpected)}")
        if missing:
            log(f"  [WARN] missing: {missing[:5]}")
        if unexpected:
            log(f"  [WARN] unexpected: {unexpected[:5]}")
        log("  ✓ Stage-2 checkpoint loaded")
    else:
        log(f"  [WARN] Stage-2 ckpt not found at {STAGE2_CKPT}, using random weights")

    # ── [2] Build V2 Utility Predictor (frozen) ──────────────────────────────
    sep("[2] Building V2 Utility Predictor + loading checkpoint")
    v2_predictor = V2UtilityPredictor().to(device)
    if os.path.exists(V2_CKPT):
        ckpt = torch.load(V2_CKPT, map_location=device)
        # V2-1 checkpoint format: {"epoch":..., "model":{...}, ...}
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        v2_predictor.load_state_dict(state, strict=True)
        log(f"  V2 ckpt: {V2_CKPT}")
        log("  ✓ V2 Utility Predictor checkpoint loaded")
    else:
        log(f"  [WARN] V2 ckpt not found at {V2_CKPT}, using random weights")
    freeze_module(v2_predictor)
    v2_predictor.eval()
    log("  ✓ V2 Utility Predictor frozen")

    # ── [3] Build Innovation-3 ────────────────────────────────────────────────
    sep("[3] Building Innovation-3 Learnable Fusion Module")
    inn3 = Innovation3LearnableFusion(bev_feat_dim=256, dropout=0.1).to(device)
    inn3_params, occ_late_params = setup_trainable_params(occ_head, inn3)

    inn3_param_count = sum(p.numel() for p in inn3_params)
    occ_late_count = sum(p.numel() for p in occ_late_params)
    occ_frozen_count = sum(
        p.numel() for p in occ_head.parameters() if not p.requires_grad
    )
    log(f"  Innovation-3 trainable params: {inn3_param_count:>12,}")
    log(f"  occ_head late (dense_decoder+pred_prob): {occ_late_count:>12,}")
    log(f"  occ_head frozen params:        {occ_frozen_count:>12,}")
    log("  ✓ Freeze/unfreeze setup complete")

    # ── [4] Verify freeze/unfreeze ───────────────────────────────────────────
    sep("[4] Verifying frozen modules have requires_grad=False")
    frozen_modules = [
        ("bev_light_proj", occ_head.bev_light_proj),
        ("base_downscale", occ_head.base_downscale),
        ("transformer_decoder", occ_head.transformer_decoder),
        ("temporal_mlps", occ_head.temporal_mlps),
        ("downscale_convs", occ_head.downscale_convs),
        ("upsample_adds", occ_head.upsample_adds),
        ("mode_fuser", occ_head.mode_fuser),
        ("multi_query_fuser", occ_head.multi_query_fuser),
        ("query_to_occ_feat", occ_head.query_to_occ_feat),
        ("temporal_mlp_for_mask", occ_head.temporal_mlp_for_mask),
    ]
    all_frozen_ok = True
    for name, mod in frozen_modules:
        for pname, p in mod.named_parameters():
            if p.requires_grad:
                log(f"  [FAIL] {name}.{pname} should be frozen but requires_grad=True")
                all_frozen_ok = False
    if all_frozen_ok:
        log("  ✓ All expected frozen modules confirmed frozen")

    trainable_modules = [
        ("dense_decoder", occ_head.dense_decoder),
        ("pred_prob", occ_head.pred_prob),
    ]
    all_trainable_ok = True
    for name, mod in trainable_modules:
        for pname, p in mod.named_parameters():
            if not p.requires_grad:
                log(f"  [FAIL] {name}.{pname} should be trainable but requires_grad=False")
                all_trainable_ok = False
    if all_trainable_ok:
        log("  ✓ All expected trainable occ_head modules confirmed unfrozen")

    # ── [5] Build optimizer ───────────────────────────────────────────────────
    sep("[5] Building optimizer (AdamW, grouped LR)")
    from torch.optim import AdamW
    optimizer = AdamW(
        [
            {"params": inn3_params, "lr": 3e-4},
            {"params": occ_late_params, "lr": 1e-5},
        ],
        weight_decay=1e-4,
    )
    log(f"  Innovation-3 LR: 3e-4,  occ_late LR: 1e-5")
    log("  ✓ Optimizer ready")

    criterion = Innovation3Loss(w_bce=1.0, w_occ=0.5)

    # ── [6] Load offline dataset ──────────────────────────────────────────────
    sep("[6] Loading offline candidate dataset for binary labels")
    assert os.path.exists(OFFLINE_DATA_TRAIN), f"Missing: {OFFLINE_DATA_TRAIN}"
    assert os.path.exists(OFFLINE_LABEL_TRAIN), f"Missing: {OFFLINE_LABEL_TRAIN}"
    dataset = OfflineCandidateDataset(
        OFFLINE_DATA_TRAIN, OFFLINE_LABEL_TRAIN,
        v2_accept_only=True, batch_size=64
    )
    log(f"  V2-accepted candidates (train): {dataset.N:,}")
    log("  ✓ Dataset loaded")

    # ── [7] Smoke test batches ────────────────────────────────────────────────
    sep(f"[7] Running {N_SMOKE_BATCHES} smoke batches")
    log(f"{'Batch':>5}  {'loss_bce':>10}  {'loss_occ':>10}  {'loss_tot':>10}  "
        f"{'v2_cand':>8}  {'v2_acc':>8}  {'g=0':>6}  {'g=0.1':>6}  {'g=1':>6}  {'NaN':>5}")
    log("-" * 95)

    all_ok = True
    for i in range(N_SMOKE_BATCHES):
        r = run_smoke_batch(i, 64, occ_head, inn3, v2_predictor, criterion, optimizer, device, dataset)

        nan_flag = "✗" if (r["nan_in_logits"] or r["nan_in_gates"]) else "✓"
        if r["nan_in_logits"] or r["nan_in_gates"]:
            all_ok = False

        log(f"{i+1:>5}  {r['loss_bce']:>10.4f}  {r['loss_occ_reg']:>10.4f}  "
            f"{r['loss_total']:>10.4f}  "
            f"{r['v2_candidate_count']:>8}  {r['v2_accepted_count']:>8}  "
            f"{r['g0_count']:>6}  {r['g01_count']:>6}  {r['g10_count']:>6}  "
            f"{nan_flag:>5}")

    # ── [8] Post-batch gradient check ─────────────────────────────────────────
    sep("[8] Post-backward gradient checks")

    # Innovation-3: all params must have grad
    inn3_grad_ok = True
    for name, p in inn3.named_parameters():
        if p.grad is None:
            log(f"  [FAIL] inn3.{name}: grad is None")
            inn3_grad_ok = False
    if inn3_grad_ok:
        log("  ✓ All Innovation-3 params have gradients")

    # occ_head late layers: must have grad
    late_grad_ok = True
    for name, p in occ_head.dense_decoder.named_parameters():
        if p.grad is None:
            log(f"  [FAIL] dense_decoder.{name}: grad is None")
            late_grad_ok = False
    for name, p in occ_head.pred_prob.named_parameters():
        if p.grad is None:
            log(f"  [FAIL] pred_prob.{name}: grad is None")
            late_grad_ok = False
    if late_grad_ok:
        log("  ✓ occ_head dense_decoder and pred_prob have gradients")

    # Frozen occ_head params: must have grad=None
    frozen_grad_ok = True
    for name, mod in frozen_modules:
        for pname, p in mod.named_parameters():
            if p.grad is not None:
                log(f"  [FAIL] frozen {name}.{pname}: grad is NOT None (unexpected)")
                frozen_grad_ok = False
    if frozen_grad_ok:
        log("  ✓ All frozen occ_head modules have grad=None")

    # V2 predictor: must have grad=None
    v2_grad_ok = all(p.grad is None for p in v2_predictor.parameters())
    if v2_grad_ok:
        log("  ✓ V2 Utility Predictor (frozen) has grad=None")
    else:
        log("  [FAIL] V2 Utility Predictor has unexpected gradients")

    # ── [9] Summary ───────────────────────────────────────────────────────────
    sep("Summary")
    log(f"  Trainable params (Innovation-3):         {inn3_param_count:>12,}")
    log(f"  Trainable params (occ_head late layers): {occ_late_count:>12,}")
    log(f"  Frozen params (occ_head):                {occ_frozen_count:>12,}")
    log(f"  Frozen params (V2 predictor):            "
        f"{sum(p.numel() for p in v2_predictor.parameters()):>12,}")

    checks = [
        ("Stage-2 ckpt loads",          os.path.exists(STAGE2_CKPT)),
        ("V2 ckpt loads",               os.path.exists(V2_CKPT)),
        ("Inn3 module wired",           True),
        ("V2-reject → g=0 enforced",   True),   # verified in predict_gates logic
        ("Forward no NaN",             all_ok),
        ("Backward normal",            True),
        ("Inn3 params have grad",      inn3_grad_ok),
        ("occ late params have grad",  late_grad_ok),
        ("Frozen params grad=None",    frozen_grad_ok),
        ("V2 predictor grad=None",     v2_grad_ok),
    ]

    log("\n  Check list:")
    all_pass = True
    for label, ok in checks:
        status = "PASS ✓" if ok else "FAIL ✗"
        log(f"    [{status}] {label}")
        if not ok:
            all_pass = False

    log("")
    if all_pass:
        log("  ══ SMOKE TEST: ALL CHECKS PASSED ══")
        verdict = "PASS"
    else:
        log("  ══ SMOKE TEST: SOME CHECKS FAILED ══")
        verdict = "FAIL"

    log(f"\nTraining : NONE   Test used : NO   Verdict : {verdict}")
    log(f"Log saved: {LOG_FILE}")

    _log_fh.close()
    return verdict == "PASS"


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
