"""
V2-4D Shape Audit Script
========================
Verifies that Innovation-3 module can:
  1. Extract dense_decoder features from occ_head forward pass
  2. Produce correct shapes at each step
  3. Run a full forward pass without NaN/Inf

Training : NONE
Test used : NO
"""

import sys
import os

# ── Path setup ──────────────────────────────────────────────────────────────
REPO_ROOT = "/root/autodl-tmp/UniV2X"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "projects"))

import torch
import torch.nn as nn
import numpy as np
import mmcv
from mmdet.models.builder import build_head

# Register mmdet3d_plugin heads
try:
    import mmdet3d_plugin  # noqa: F401
except Exception as e:
    print(f"[WARN] plugin import: {e}")

# Add innovation3 module path
sys.path.insert(0, os.path.join(REPO_ROOT, "G0-G1-G2/V2/V2-4D_joint_finetuning"))
from innovation3_module import Innovation3LearnableFusion, Innovation3Loss

LOG_FILE = os.path.join(
    REPO_ROOT, "G0-G1-G2/V2/V2-4D_joint_finetuning/logs/v2_4d_shape_audit.txt"
)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def log(msg, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")
        f.flush()


def build_ego_occ_head():
    cfg = mmcv.Config.fromfile(
        os.path.join(REPO_ROOT, "projects/configs_e2e_univ2x/univ2x_coop_e2e.py")
    )
    occ_cfg = cfg.model_ego_agent.occ_head.copy()
    occ_cfg["is_ego_agent"] = True
    occ_cfg["is_cooperation"] = True
    occ_cfg["use_stcv_occ"] = False
    return build_head(occ_cfg)


# ── Hook to capture dense_decoder output ────────────────────────────────────

class DenseDecoderHook:
    """Registers a forward hook on occ_head.dense_decoder and captures output."""

    def __init__(self, occ_head):
        self.output = None
        self._handle = occ_head.dense_decoder.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # out: [b, T, C, H, W]
        self.output = out

    def remove(self):
        self._handle.remove()


def main():
    with open(LOG_FILE, "w") as f:
        log("=" * 70, f)
        log("V2-4D Shape Audit", f)
        log("Training : NONE   Test used : NO", f)
        log("=" * 70, f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"\nDevice: {device}", f)

        # ── 1. Build occ_head ──────────────────────────────────────────────
        log("\n[1] Building ego occ_head ...", f)
        occ_head = build_ego_occ_head().to(device).eval()
        total_params = sum(p.numel() for p in occ_head.parameters())
        log(f"    occ_head total params: {total_params:,}", f)

        # ── 2. Build Innovation-3 module ───────────────────────────────────
        log("\n[2] Building Innovation-3 module ...", f)
        inn3 = Innovation3LearnableFusion(bev_feat_dim=256, dropout=0.1).to(device)
        inn3_params = sum(p.numel() for p in inn3.parameters())
        log(f"    Innovation-3 params: {inn3_params:,}", f)

        # ── 3. Synthetic forward pass through occ_head ────────────────────
        log("\n[3] Running synthetic occ_head forward pass ...", f)

        B, BEV_H, BEV_W = 1, 200, 200
        BEV_C = 256
        N_QUERY = 300

        # bev_feat: [H*W, B, C]
        bev_feat = torch.randn(BEV_H * BEV_W, B, BEV_C, device=device)
        # ins_query: [B, Q, C]
        ins_query = torch.randn(B, N_QUERY, BEV_C, device=device)

        hook = DenseDecoderHook(occ_head)

        with torch.no_grad():
            mask_preds, ins_occ_logits = occ_head(bev_feat, ins_query)

        dense_feat = hook.output  # [B, T, C, H, W]
        hook.remove()

        log(f"    bev_feat input shape:     {tuple(bev_feat.shape)}", f)
        log(f"    ins_query shape:          {tuple(ins_query.shape)}", f)
        log(f"    mask_preds shape:         {tuple(mask_preds.shape)}", f)
        log(f"    ins_occ_logits shape:     {tuple(ins_occ_logits.shape)}", f)
        log(f"    dense_decoder output:     {tuple(dense_feat.shape)}", f)

        # Expected: [1, 5, 256, 200, 200]
        assert dense_feat.shape == (B, 5, BEV_C, BEV_H, BEV_W), \
            f"dense_decoder shape mismatch: {dense_feat.shape}"
        log("    ✓ dense_decoder shape correct [B=1, T=5, C=256, H=200, W=200]", f)

        # ── 4. Synthetic candidates ────────────────────────────────────────
        log("\n[4] Checking Innovation-3 feature extraction ...", f)

        N_ALL = 20   # total candidates
        N_ACC = 8    # V2-accepted

        # Build fake candidate metadata
        torch.manual_seed(42)
        v2_accept = torch.zeros(N_ALL, dtype=torch.uint8, device=device)
        accept_indices = torch.randperm(N_ALL)[:N_ACC]
        v2_accept[accept_indices] = 1

        horizons = torch.randint(0, 5, (N_ALL,), device=device)
        patch_rs = torch.randint(0, 50, (N_ALL,), device=device)
        patch_cs = torch.randint(0, 50, (N_ALL,), device=device)

        # Fake Pv and Pi: [B, T, H, W]
        pv = torch.rand(B, 5, BEV_H, BEV_W, device=device)
        pi_aligned = torch.rand(B, 5, BEV_H, BEV_W, device=device)

        # Extract features for accepted only
        h_acc = horizons[v2_accept.bool()]
        r_acc = patch_rs[v2_accept.bool()]
        c_acc = patch_cs[v2_accept.bool()]

        bev_feat_cand = Innovation3LearnableFusion.extract_roi_feature(
            dense_feat, h_acc, r_acc, c_acc
        )
        log(f"    N_ALL={N_ALL}, N_ACC={N_ACC}", f)
        log(f"    extracted BEV feature shape: {tuple(bev_feat_cand.shape)}", f)
        assert bev_feat_cand.shape == (N_ACC, BEV_C), \
            f"BEV feature shape mismatch: {bev_feat_cand.shape}"
        log("    ✓ extract_roi_feature shape correct [N_ACC, 256]", f)

        # NaN/Inf check
        assert not torch.isnan(bev_feat_cand).any(), "NaN in extracted features"
        assert not torch.isinf(bev_feat_cand).any(), "Inf in extracted features"
        log("    ✓ No NaN / Inf in extracted features", f)

        # ── 5. Full predict_gates pass ─────────────────────────────────────
        log("\n[5] Running predict_gates ...", f)

        gates, logits, accept_mask = inn3.predict_gates(
            future_states=dense_feat,
            pv=pv,
            pi_aligned=pi_aligned,
            horizons=horizons,
            patch_rs=patch_rs,
            patch_cs=patch_cs,
            v2_accept=v2_accept,
        )

        log(f"    gates shape:   {tuple(gates.shape)}", f)
        log(f"    logits shape:  {tuple(logits.shape)}", f)
        log(f"    gates unique:  {torch.unique(gates).tolist()}", f)
        log(f"    gates for V2-rejected: {gates[~accept_mask].unique().tolist()} (must be [0.0])", f)
        log(f"    gates for V2-accepted: {gates[accept_mask].unique().tolist()}", f)

        assert gates.shape == (N_ALL,), f"gates shape: {gates.shape}"
        assert logits.shape == (N_ACC,), f"logits shape: {logits.shape}"
        assert (gates[~accept_mask] == 0.0).all(), "V2-rejected must have g=0"
        log("    ✓ V2-rejected gates are strictly 0.0", f)

        accepted_gates = gates[accept_mask]
        valid_gate_vals = {0.1, 1.0}
        for g in accepted_gates.tolist():
            assert abs(g - 0.1) < 1e-5 or abs(g - 1.0) < 1e-5, \
                f"Unexpected gate value: {g}"
        log("    ✓ V2-accepted gates are in {0.1, 1.0}", f)

        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"
        log("    ✓ No NaN / Inf in logits", f)

        # ── 6. apply_fusion shape check ────────────────────────────────────
        log("\n[6] Running apply_fusion ...", f)

        fused = Innovation3LearnableFusion.apply_fusion(
            pv=pv,
            pi_aligned=pi_aligned,
            gates=gates,
            horizons=horizons,
            patch_rs=patch_rs,
            patch_cs=patch_cs,
        )

        log(f"    fused shape: {tuple(fused.shape)}", f)
        assert fused.shape == pv.shape, f"fused shape mismatch: {fused.shape}"
        log("    ✓ fused shape matches Pv [B, T, H, W]", f)

        assert not torch.isnan(fused).any(), "NaN in fused output"
        assert not torch.isinf(fused).any(), "Inf in fused output"
        log("    ✓ No NaN / Inf in fused output", f)

        # For g=0 regions, fused must equal Pv
        for i in range(N_ALL):
            if not v2_accept[i]:
                t = int(horizons[i])
                r = int(patch_rs[i])
                c = int(patch_cs[i])
                r0, r1 = r * 4, r * 4 + 4
                c0, c1 = c * 4, c * 4 + 4
                diff = (fused[0, t, r0:r1, c0:c1] - pv[0, t, r0:r1, c0:c1]).abs().max()
                assert diff < 1e-5, f"g=0 region modified: diff={diff}"
        log("    ✓ V2-rejected regions unchanged from Pv", f)

        # ── 7. Loss forward ────────────────────────────────────────────────
        log("\n[7] Loss forward check ...", f)

        criterion = Innovation3Loss(w_bce=1.0, w_occ=0.5)
        fake_labels = torch.randint(0, 2, (N_ACC,), device=device).float()
        fake_occ_losses = {
            "loss_dice": torch.tensor(0.5, device=device, requires_grad=False),
            "loss_mask": torch.tensor(0.3, device=device, requires_grad=False),
        }

        # Need logits with grad for loss
        inn3.train()
        gates2, logits2, _ = inn3.predict_gates(
            dense_feat.detach(), pv, pi_aligned, horizons, patch_rs, patch_cs, v2_accept
        )
        loss_dict = criterion(logits2, fake_labels, occ_loss_dict=fake_occ_losses)

        log(f"    loss_bce:     {loss_dict['loss_bce'].item():.6f}", f)
        log(f"    loss_occ_reg: {loss_dict['loss_occ_reg'].item():.6f}", f)
        log(f"    loss_total:   {loss_dict['loss_total'].item():.6f}", f)

        for k, v in loss_dict.items():
            assert not torch.isnan(v), f"NaN in {k}"
            assert not torch.isinf(v), f"Inf in {k}"
        log("    ✓ No NaN / Inf in losses", f)

        # ── 8. Gradient check ──────────────────────────────────────────────
        log("\n[8] Gradient check ...", f)

        loss_dict["loss_total"].backward()

        has_grad = []
        no_grad = []
        for name, p in inn3.named_parameters():
            if p.grad is not None:
                has_grad.append(name)
            else:
                no_grad.append(name)

        log(f"    Innovation-3 params with grad:    {len(has_grad)}", f)
        log(f"    Innovation-3 params without grad: {len(no_grad)}", f)
        if no_grad:
            log(f"    [WARN] no-grad params: {no_grad}", f)
        assert len(has_grad) == len(list(inn3.parameters())), \
            "Some Innovation-3 params have no gradient!"
        log("    ✓ All Innovation-3 params received gradients", f)

        # ── 9. Parameter summary ───────────────────────────────────────────
        log("\n[9] Parameter summary ...", f)
        log(f"    occ_head total:            {total_params:>12,}", f)
        log(f"    Innovation-3 (trainable):  {inn3_params:>12,}", f)
        log(f"    occ late layers (dense_decoder + pred_prob):", f)
        late_params = (
            sum(p.numel() for p in occ_head.dense_decoder.parameters())
            + sum(p.numel() for p in occ_head.pred_prob.parameters())
        )
        log(f"      dense_decoder:  {sum(p.numel() for p in occ_head.dense_decoder.parameters()):>12,}", f)
        log(f"      pred_prob:      {sum(p.numel() for p in occ_head.pred_prob.parameters()):>12,}", f)
        log(f"      late total:     {late_params:>12,}", f)
        log(f"    Planned trainable total:   {inn3_params + late_params:>12,}", f)
        log(f"    Planned frozen total:      {total_params - late_params:>12,}", f)

        log("\n" + "=" * 70, f)
        log("Shape Audit: ALL CHECKS PASSED ✓", f)
        log("Training : NONE   Test used : NO", f)
        log("=" * 70, f)

    print(f"\nLog saved: {LOG_FILE}")


if __name__ == "__main__":
    main()
