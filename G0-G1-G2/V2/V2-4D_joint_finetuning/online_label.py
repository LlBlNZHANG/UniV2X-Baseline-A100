"""
V2-4D Online Exact Binary Label
================================
Computes y_att = 1[U_R(0.1) > U_R(1.0) + EPS] for each V2-accepted candidate.

Implementation is a strict Python/PyTorch translation of the NumPy logic in:
  V2-4C-2A_build_gate_utility_curves.py  (compute_curves)
  V2-4B_build_learnable_fusion_dataset.py (binary_counts, gt_np construction)

Six audited details from the source:

1. Patch boundary:  r0=r*4, r1=r0+4, c0=c*4, c1=c0+4  (y→r, x→c)
2. Fusion formula:  Pf(g) = Pv + g * max(Pi-Pv, 0) * warp_valid_mask
                    (warp is applied to positive_delta BEFORE multiplying gate)
3. Binarisation:    fused = Pf > 0.1   (strict >)
4. ADD definition:  ADD = (~Ov) & fused & warp_valid_mask
5. Baseline tp/fp/fn: computed over full 200×200, from Ov and gt.astype(bool)
   — GT ignore_index=255 is treated as occupied (True) because
     binary_counts does gt.astype(bool), making 255 → True
   — warp_valid_mask does NOT filter the baseline TP/FP/FN
6. Tie rule:  U(0.1) > U(1.0) + EPS  with EPS=1e-12;
              tie (U(0.1) == U(1.0)) → y=0 (full fusion stays default)

Training : N/A (label utility only)
Test used : NO
"""

import torch

EPS       = 1e-12
SEG_THRESH = 0.1
PATCH     = 4


# ---------------------------------------------------------------------------
# Runtime GT encoding check (call once per session before training)
# ---------------------------------------------------------------------------

def assert_gt_encoding_compatible(gt_occ: torch.Tensor, label: str = "train"):
    """
    Verify that the live dataloader GT is binary-compatible with the offline
    V2-4C-2A data, where gt_np = d["GT"].astype(bool) maps {0→False, 1→True,
    255→True}.

    Requirement: GT must contain only values in {0, 1, 255} (long/int tensor).
    If other values appear, the label computation is undefined.

    Args:
        gt_occ: integer tensor from the dataloader, any shape
        label:  human-readable split name for the error message
    """
    vals = torch.unique(gt_occ)
    allowed = {0, 1, 255}
    bad = [int(v) for v in vals.tolist() if int(v) not in allowed]
    if bad:
        raise ValueError(
            f"[GT encoding check — {label}] Unexpected GT values: {bad}. "
            f"Expected subset of {{0, 1, 255}}. "
            f"Online label computation requires GT identical to V2-4C-2A offline encoding."
        )


# ---------------------------------------------------------------------------
# Per-horizon baseline TP / FP / FN over the full 200×200 grid
# ---------------------------------------------------------------------------


def compute_valid_gt_mask(
    gt: torch.Tensor,
    frame_valid: torch.Tensor,
) -> torch.Tensor:
    """Valid cells: GT is not 255 and the horizon is valid."""
    assert gt.ndim == 4, "GT must have shape [B,T,H,W]"
    assert frame_valid.shape == gt.shape[:2], (
        "frame_valid must have shape [B,T]"
    )
    return (gt != 255) & frame_valid.bool()[:, :, None, None]


def compute_per_horizon_baseline_stats(
    ov:  torch.Tensor,  # [1, T, 200, 200]  ego binary occupancy  (bool or 0/1 int)
    gt:  torch.Tensor,  # [1, T, 200, 200]  GT occupancy (int, values in {0,1,255})
) -> tuple:
    """
    Replicate V2-4B binary_counts(ov_np[h], gt_np[h]) for each horizon h.

    gt_np is built as d["GT"].astype(bool), so 255 → True.
    warp_valid_mask is NOT used here — baseline is over the full 200×200 frame.

    Returns:
        tp_t, fp_t, fn_t : [T] float32 tensors on the same device as ov
    """
    with torch.no_grad():
        # Ensure both tensors are on the same device
        device = ov.device
        ov_bool = ov[0].bool().to(device)    # [T, 200, 200]
        gt_bool = (gt[0] == 1).to(device)
        valid = (gt[0] != 255).to(device)

        tp_t = (ov_bool & gt_bool & valid).sum(dim=(-1, -2)).float()
        fp_t = (ov_bool & ~gt_bool & valid).sum(dim=(-1, -2)).float()
        fn_t = (~ov_bool & gt_bool & valid).sum(dim=(-1, -2)).float()

    return tp_t, fp_t, fn_t


# ---------------------------------------------------------------------------
# Per-patch utility under a single gate value
# ---------------------------------------------------------------------------

def _patch_utility(
    pv_patch:  torch.Tensor,  # [4, 4]  float
    pi_patch:  torch.Tensor,  # [4, 4]  float
    ov_patch:  torch.Tensor,  # [4, 4]  bool
    gt_patch:  torch.Tensor,  # [4, 4]  bool  (255 already converted to True before calling)
    w_patch:   torch.Tensor,  # [4, 4]  bool
    gate:      float,
    tp:        float,
    fp:        float,
    fn:        float,
) -> float:
    """
    U_R(gate) = (TP + BA) / (D + HA) - TP / D,    D = TP+FP+FN > 0
    U_R(gate) = 0                                   when D = 0

    Exact translation of V2-4C-2A compute_curves inner loop.
    """
    # Pf(g) = Pv + g * max(Pi - Pv, 0) * warp    [warp applied before gate]
    pos_delta = (pi_patch - pv_patch).clamp(min=0) * w_patch.float()
    pf        = pv_patch + gate * pos_delta

    fused = pf > SEG_THRESH                  # strict >

    add   = (~ov_patch) & fused & w_patch    # ADD = (~Ov) & pred & warp
    ba    = int((add &  gt_patch).sum().item())
    ha    = int((add & ~gt_patch).sum().item())

    D = tp + fp + fn
    if D <= 0:
        return 0.0
    return float((tp + ba) / (D + ha) - tp / D)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_binary_oracle_label_online(
    pv:          torch.Tensor,  # [1, T, 200, 200]  soft prob,   detached
    pi_aligned:  torch.Tensor,  # [1, T, 200, 200]  soft prob,   detached
    ov:          torch.Tensor,  # [1, T, 200, 200]  ego binary occupancy (bool/int)
    gt:          torch.Tensor,  # [1, T, 200, 200]  GT (int, {0,1,255})
    warp_valid:  torch.Tensor,  # [1,    200, 200]  warp valid mask (bool)
    horizons:    torch.Tensor,  # [N_acc]  long
    patch_rs:    torch.Tensor,  # [N_acc]  long  (0..49)
    patch_cs:    torch.Tensor,  # [N_acc]  long  (0..49)
    tp_t:        torch.Tensor,  # [T]  float  — from compute_per_horizon_baseline_stats
    fp_t:        torch.Tensor,
    fn_t:        torch.Tensor,
) -> torch.Tensor:
    """
    Compute y_att = 1[U_R(0.1) > U_R(1.0) + EPS] for each accepted candidate.

    Entire function runs inside torch.no_grad() — labels are never part of the
    computation graph.

    GT encoding assumption (verified by assert_gt_encoding_compatible):
        GT values in {0, 1, 255};  gt.bool() maps 255 → True (same as V2-4C-2A).

    Returns:
        labels: [N_acc]  float32  values in {0.0, 1.0}
    """
    with torch.no_grad():
        N   = int(horizons.shape[0])
        dev = pv.device

        labels = torch.zeros(N, device=dev, dtype=torch.float32)

        pv_s  = pv[0]          # [T, 200, 200]
        pi_s  = pi_aligned[0]
        ov_s  = ov[0].bool()   # [T, 200, 200]
        gt_s  = gt[0].bool()   # [T, 200, 200]  255 → True
        w_s   = warp_valid[0].bool()  # [200, 200]

        for i in range(N):
            h  = int(horizons[i])
            r  = int(patch_rs[i])
            c  = int(patch_cs[i])
            r0, r1 = r * PATCH, r * PATCH + PATCH
            c0, c1 = c * PATCH, c * PATCH + PATCH

            pv_p  = pv_s[h, r0:r1, c0:c1]
            pi_p  = pi_s[h, r0:r1, c0:c1]
            ov_p  = ov_s[h, r0:r1, c0:c1]
            gt_p  = gt_s[h, r0:r1, c0:c1]
            w_p   = w_s[r0:r1, c0:c1]

            tp_h = float(tp_t[h].item())
            fp_h = float(fp_t[h].item())
            fn_h = float(fn_t[h].item())

            U_01 = _patch_utility(pv_p, pi_p, ov_p, gt_p, w_p, 0.1, tp_h, fp_h, fn_h)
            U_10 = _patch_utility(pv_p, pi_p, ov_p, gt_p, w_p, 1.0, tp_h, fp_h, fn_h)

            # y = 1 only when U(0.1) strictly exceeds U(1.0) by more than EPS
            labels[i] = 1.0 if U_01 > U_10 + EPS else 0.0

    return labels


# ---------------------------------------------------------------------------
# V2-4D Valid-GT Oracle (new supervision definition)
# ---------------------------------------------------------------------------

def compute_valid_binary_oracle_label_online(
    pv: torch.Tensor,
    pi_aligned: torch.Tensor,
    ov: torch.Tensor,
    gt: torch.Tensor,
    warp_valid: torch.Tensor,
    horizons: torch.Tensor,
    patch_rs: torch.Tensor,
    patch_cs: torch.Tensor,
    frame_valid: torch.Tensor,
) -> tuple:
    """
    Returns:
        labels: [N] float32, 1 means g=0.1 is preferable to g=1.
        supervise_mask: [N] bool, valid candidates for BCE.

    Invalid GT cells (255) and invalid horizons are excluded.
    Baseline statistics use all valid cells in each horizon.
    """
    with torch.no_grad():
        assert pv.ndim == 4 and pv.shape[0] == 1
        assert pi_aligned.shape == pv.shape
        assert ov.shape == pv.shape
        assert gt.shape == pv.shape
        assert frame_valid.shape == gt.shape[:2]
        assert warp_valid.shape == (1, *gt.shape[-2:])

        assert_gt_encoding_compatible(gt, label="valid_oracle")

        n = int(horizons.numel())
        assert patch_rs.numel() == n and patch_cs.numel() == n

        device = pv.device
        labels = torch.zeros(n, device=device, dtype=torch.float32)
        supervise_mask = torch.zeros(n, device=device, dtype=torch.bool)

        valid = compute_valid_gt_mask(gt, frame_valid)
        gt_positive = gt == 1
        ov_positive = ov.bool()

        tp = (ov_positive & gt_positive & valid).sum(
            dim=(-1, -2)
        ).float()[0]
        fp = (ov_positive & ~gt_positive & valid).sum(
            dim=(-1, -2)
        ).float()[0]
        fn = (~ov_positive & gt_positive & valid).sum(
            dim=(-1, -2)
        ).float()[0]

        T, H, W = gt.shape[1:]
        assert H % PATCH == 0 and W % PATCH == 0

        for i in range(n):
            h = int(horizons[i].item())
            r = int(patch_rs[i].item())
            c = int(patch_cs[i].item())

            assert 0 <= h < T
            assert 0 <= r < H // PATCH
            assert 0 <= c < W // PATCH

            if not bool(frame_valid[0, h]):
                continue

            rs = slice(r * PATCH, (r + 1) * PATCH)
            cs = slice(c * PATCH, (c + 1) * PATCH)

            cell_valid = valid[0, h, rs, cs]
            warp_patch = warp_valid[0, rs, cs].bool()
            effective_valid = cell_valid & warp_patch

            if not bool(effective_valid.any()):
                continue

            supervise_mask[i] = True

            pv_p = pv[0, h, rs, cs]
            pi_p = pi_aligned[0, h, rs, cs]

            pos_delta = (
                (pi_p - pv_p).clamp(min=0)
                * warp_patch.float()
            )

            add_base = ~ov_positive[0, h, rs, cs]

            D = float((tp[h] + fp[h] + fn[h]).item())

            # Undefined baseline foreground IoU: do not supervise.
            if D <= 0:
                supervise_mask[i] = False
                continue

            def utility(gate):
                pf = pv_p + gate * pos_delta
                fused = pf > SEG_THRESH
                add = add_base & fused & effective_valid

                ba = int((add & gt_positive[0, h, rs, cs]).sum().item())
                ha = int((add & ~gt_positive[0, h, rs, cs]).sum().item())

                if D <= 0:
                    return 0.0

                return (
                    (float(tp[h].item()) + ba) / (D + ha)
                    - float(tp[h].item()) / D
                )

            labels[i] = float(
                utility(0.1) > utility(1.0) + EPS
            )

    return labels, supervise_mask
