"""
Innovation-3: Utility-conditioned Learnable Adaptive Fusion Module
==================================================================
V2-4D joint fine-tuning — 385-D input version.

Input composition per accepted candidate R:
  dense_occ_feat[R]   256-D  (from occ_head.dense_decoder, horizon-specific)
  features_127        127-D  (V2 offline candidate features)
  v2_mag                1-D  (V2 utility magnitude)
  v2_sign_prob          1-D  (V2 utility sign probability)
  ─────────────────────────
  total               385-D

Gate space: {0.0, 0.1, 1.0}
  g=0:   V2-rejected (forced, no MLP call)
  g=0.1: attenuate   (MLP predicts y=1)
  g=1.0: full fusion (MLP predicts y=0)

Fusion formula:
  P_f,R = Pv_R + g_R * (Pi_R - Pv_R).clamp(min=0)

Training : NONE  (this file only defines the module)
Test used : NO
"""

import torch
import torch.nn as nn


class LearnableFusionMLP(nn.Module):
    """
    MLP predicting binary attenuation gate per V2-accepted candidate.

    Input:  [N, 385]
    Output: [N, 1]  scalar logit  (sigmoid > 0.5 → y=1 → g=0.1 attenuate)
    """

    def __init__(self, in_dim: int = 385, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.residual = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.down = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, in_dim]  →  logit: [N, 1]"""
        h = self.proj(x)
        h = h + self.residual(h)
        h = self.down(h)
        return self.head(h)


class Innovation3LearnableFusion(nn.Module):
    """
    Full Innovation-3 module.

    Compared with the earlier 259-D prototype, the MLP now also receives
    the offline V2 utility features (features_127, v2_mag, v2_sign_prob) so
    that joint fine-tuning can leverage both the live dense_decoder BEV
    representation and the pre-computed candidate statistics.

    The V2 utility predictor (Innovation-2) is frozen and called externally;
    only its scalar outputs (v2_mag, v2_sign_prob) are passed in here.
    """

    GATE_ATTENUATE = 0.1
    GATE_FULL = 1.0

    def __init__(self, bev_feat_dim: int = 256, offline_feat_dim: int = 129,
                 dropout: float = 0.1):
        """
        Args:
            bev_feat_dim:     dim of dense_decoder output per candidate (256)
            offline_feat_dim: dim of offline V2 features
                              (features_127=127 + v2_mag=1 + v2_sign_prob=1 = 129)
            dropout:          dropout probability for MLP
        """
        super().__init__()
        in_dim = bev_feat_dim + offline_feat_dim  # 256 + 129 = 385
        self.bev_feat_dim = bev_feat_dim
        self.offline_feat_dim = offline_feat_dim
        self.mlp = LearnableFusionMLP(in_dim=in_dim, dropout=dropout)

    # ------------------------------------------------------------------
    # Feature extraction from dense_decoder output
    # ------------------------------------------------------------------

    @staticmethod
    def extract_roi_feature(
        future_states: torch.Tensor,
        horizons: torch.Tensor,
        patch_rs: torch.Tensor,
        patch_cs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract per-candidate horizon-specific BEV feature.

        Args:
            future_states: [b, T, C, H, W]  dense_decoder output, b=1, T=5, C=256, H=W=200
            horizons:      [N]  int, values in {0..4}
            patch_rs:      [N]  int, values in {0..49}  (row index in 50×50 grid)
            patch_cs:      [N]  int, values in {0..49}  (col index in 50×50 grid)

        Returns:
            feat: [N, C]  mean-pooled over the 4×4 BEV cell
                  (patch_r/c index a 50×50 grid on 200×200 BEV →
                   pixel window = [r*4 : r*4+4, c*4 : c*4+4])
        """
        assert future_states.shape[0] == 1, "batch_size must be 1"
        bev = future_states[0]  # [T, C, H, W]
        N = horizons.shape[0]
        C = bev.shape[1]
        device = bev.device

        if N == 0:
            return torch.zeros((0, C), device=device, dtype=bev.dtype)

        feats = []
        for i in range(N):
            t = int(horizons[i].item())
            r = int(patch_rs[i].item())
            c = int(patch_cs[i].item())
            r0, r1 = r * 4, r * 4 + 4
            c0, c1 = c * 4, c * 4 + 4
            feats.append(bev[t, :, r0:r1, c0:c1].mean(dim=[-1, -2]))  # [C]

        return torch.stack(feats, dim=0)  # [N, C]

    # ------------------------------------------------------------------
    # Gate prediction
    # ------------------------------------------------------------------

    def predict_gates(
        self,
        future_states: torch.Tensor,
        horizons: torch.Tensor,
        patch_rs: torch.Tensor,
        patch_cs: torch.Tensor,
        v2_accept: torch.Tensor,
        features_127: torch.Tensor,
        v2_mag: torch.Tensor,
        v2_sign_prob: torch.Tensor,
    ) -> tuple:
        """
        Predict gates for ALL candidates; force g=0 for V2-rejected.

        Args:
            future_states:  [b, T, C, H, W]
            horizons:       [N_all]  int
            patch_rs:       [N_all]  int
            patch_cs:       [N_all]  int
            v2_accept:      [N_all]  bool/uint8  (1=accepted by V2, 0=rejected)
            features_127:   [N_all, 127]  offline V2 candidate features
            v2_mag:         [N_all]  V2 utility magnitude (scalar)
            v2_sign_prob:   [N_all]  V2 utility sign probability (scalar)

        Returns:
            gates:       [N_all]  float, values in {0.0, 0.1, 1.0}
            logits:      [N_acc]  float, raw MLP logits for accepted candidates
            accept_mask: [N_all]  bool
        """
        accept_mask = v2_accept.bool()
        N_all = horizons.shape[0]
        device = future_states.device
        dtype = future_states.dtype

        gates = torch.zeros(N_all, device=device, dtype=dtype)

        if accept_mask.sum() == 0:
            return gates, torch.zeros(0, device=device, dtype=dtype), accept_mask

        h_acc = horizons[accept_mask]
        r_acc = patch_rs[accept_mask]
        c_acc = patch_cs[accept_mask]

        # BEV feature: [N_acc, 256]
        bev_feat = self.extract_roi_feature(future_states, h_acc, r_acc, c_acc)

        # Offline features for accepted: [N_acc, 127+1+1=129]
        f127_acc = features_127[accept_mask]                                  # [N_acc, 127]
        mag_acc = v2_mag[accept_mask].unsqueeze(1)                             # [N_acc, 1]
        sign_acc = v2_sign_prob[accept_mask].unsqueeze(1)                      # [N_acc, 1]
        offline_feat = torch.cat([f127_acc, mag_acc, sign_acc], dim=1)        # [N_acc, 129]

        # Concat: [N_acc, 385]
        x = torch.cat([bev_feat, offline_feat], dim=1)

        logits = self.mlp(x).squeeze(1)  # [N_acc]
        probs = logits.sigmoid()

        # Hard gate: g=0.1 if prob>0.5, else g=1.0
        hard_gate = torch.where(
            probs > 0.5,
            torch.full_like(probs, self.GATE_ATTENUATE),
            torch.full_like(probs, self.GATE_FULL),
        )
        gates[accept_mask] = hard_gate

        return gates, logits, accept_mask

    # ------------------------------------------------------------------
    # Pixel-level fusion
    # ------------------------------------------------------------------

    @staticmethod
    def apply_fusion(
        pv: torch.Tensor,
        pi_aligned: torch.Tensor,
        gates: torch.Tensor,
        horizons: torch.Tensor,
        patch_rs: torch.Tensor,
        patch_cs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply per-candidate gates to produce fused probability map.

        Formula: P_f,R = Pv_R + g_R * (Pi_R - Pv_R).clamp(min=0)

        g=0   (V2-rejected): P_f,R = Pv_R
        g=0.1 (attenuate):   slight pull toward Pi
        g=1.0 (full fusion): P_f,R = Pv_R + (Pi_R - Pv_R)+ = max(Pv, Pi) via soft path

        Args:
            pv:          [b, T, H, W]
            pi_aligned:  [b, T, H, W]
            gates:       [N_all]
            horizons:    [N_all]
            patch_rs:    [N_all]
            patch_cs:    [N_all]

        Returns:
            fused:       [b, T, H, W]
        """
        fused = pv.clone()
        N = gates.shape[0]

        for i in range(N):
            g = gates[i].item()
            if g == 0.0:
                continue
            t = int(horizons[i].item())
            r = int(patch_rs[i].item())
            c = int(patch_cs[i].item())
            r0, r1 = r * 4, r * 4 + 4
            c0, c1 = c * 4, c * 4 + 4
            pv_p = pv[:, t, r0:r1, c0:c1]
            pi_p = pi_aligned[:, t, r0:r1, c0:c1]
            fused[:, t, r0:r1, c0:c1] = pv_p + g * (pi_p - pv_p).clamp(min=0)

        return fused


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------

class Innovation3Loss(nn.Module):
    """
    Combined loss for V2-4D joint fine-tuning:

      L = w_bce * L_att  +  w_occ * L_occ  +  w_pres * L_pres

    L_att:  BCE(logits, labels)              — Innovation-3 main supervision
    L_occ:  dice + mask from occ_head        — representation regularizer
    L_pres: |Pv_current - Pv_stage2|.mean() — behavior preservation
            (optional; zero if pv_stage2 not provided)
    """

    def __init__(self, w_bce: float = 1.0, w_occ: float = 0.5, w_pres: float = 0.1):
        super().__init__()
        self.w_bce = w_bce
        self.w_occ = w_occ
        self.w_pres = w_pres
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        occ_loss_dict: dict = None,
        pv_current: torch.Tensor = None,
        pv_stage2: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            logits:       [N_acc]  raw logits for V2-accepted candidates
            labels:       [N_acc]  float  0 or 1  (1 = attenuate)
            occ_loss_dict: dict from occ_head.forward_train (loss_dice, loss_mask, ...)
            pv_current:   [b, T, H, W]  current Pv for preservation monitoring
            pv_stage2:    [b, T, H, W]  Stage-2 reference Pv (detached)

        Returns:
            loss_dict with keys: loss_bce, loss_occ_reg, loss_pres, loss_total
        """
        result = {}

        # Innovation-3 BCE loss
        if logits.numel() > 0:
            loss_bce = self.bce(logits, labels.float())
        else:
            loss_bce = logits.new_zeros(1)[0]
        result['loss_bce'] = loss_bce

        # Occupancy regularizer
        loss_occ = loss_bce.new_zeros(1)[0]
        if occ_loss_dict is not None and self.w_occ > 0:
            for v in occ_loss_dict.values():
                loss_occ = loss_occ + v
        result['loss_occ_reg'] = loss_occ

        # Behavior preservation: keep Pv close to Stage-2 reference
        loss_pres = loss_bce.new_zeros(1)[0]
        if pv_current is not None and pv_stage2 is not None and self.w_pres > 0:
            loss_pres = (pv_current - pv_stage2.detach()).abs().mean()
        result['loss_pres'] = loss_pres

        loss_total = (
            self.w_bce * loss_bce
            + self.w_occ * loss_occ
            + self.w_pres * loss_pres
        )
        result['loss_total'] = loss_total

        return result
