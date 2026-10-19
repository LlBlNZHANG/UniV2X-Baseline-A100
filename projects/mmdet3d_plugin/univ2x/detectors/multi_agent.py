#----------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                 #
# Copyright (c) DAIR-V2X. All rights reserved.                   #
# Contact: yuhaibao94@gmail.com                                   #
#----------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os as _os
_V2_4D_DIR = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "../../../../G0-G1-G2/V2/V2-4D_joint_finetuning"
)
if _V2_4D_DIR not in sys.path:
    sys.path.insert(0, _V2_4D_DIR)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


class MultiAgent(MVXTwoStageDetector):
    def __init__(self, model_ego_agent, model_other_agents={}):
        super(MultiAgent, self).__init__()
        self.model_ego_agent = model_ego_agent

        self.other_agent_names = []
        self.pc_range_dict = {}
        for name_other_agent, model_other_agent in model_other_agents.items():
            setattr(self, name_other_agent, model_other_agent)
            self.other_agent_names.append(name_other_agent)
            if hasattr(model_other_agent, 'pc_range'):
                self.pc_range_dict[name_other_agent] = model_other_agent.pc_range

        # V2-4D: registered as nn.Module so params appear in state_dict / optimizer.
        # Set by attach_innovation3() BEFORE optimizer and DDP construction.
        self.innovation3_module = None  # Innovation3LearnableFusion (trainable)
        self.v2_predictor       = None  # V2UtilityPredictor         (frozen)

        # STCVFeatureExtractor is stateless (no parameters); stored as plain attr.
        self._stcv_extractor = None

        # Lightweight diagnostics — only scalars/shapes, no live tensors.
        self._v2_4d_debug = {}
        self.v2_4d_strict_exact_label = False

    # ------------------------------------------------------------------
    # V2-4D: attach Innovation-3 components
    # ------------------------------------------------------------------

    def attach_innovation3(self, innovation3_module, v2_predictor):
        """
        Register Innovation-3 modules before optimizer / DDP construction.

        Args:
            innovation3_module: Innovation3LearnableFusion  (trainable)
            v2_predictor:       V2UtilityPredictor          (frozen)
        """
        self.innovation3_module = innovation3_module
        self.v2_predictor = v2_predictor

        for p in self.v2_predictor.parameters():
            p.requires_grad = False
        self.v2_predictor.eval()

        for p in self.innovation3_module.parameters():
            p.requires_grad = True

    def attach_stcv_extractor(self, extractor):
        """Attach STCVFeatureExtractor (stateless, no nn.Module registration needed)."""
        self._stcv_extractor = extractor

    # ------------------------------------------------------------------
    # P4 candidate extraction + V2 utility (all detached from graph)
    # ------------------------------------------------------------------

    def _extract_candidates(self, pv, pi_aligned, warp_valid, device):
        """
        Enumerate all 12500 = 5*50*50 P4 candidates and compute:
          - 127-D spatial features via STCVFeatureExtractor
          - V2 utility magnitude and sign probability (frozen predictor)
          - V2 accept mask (u_hat > 0)

        All returned tensors are detached — gradient flows only through
        dense_occ_feat in the Innovation-3 path.

        Args:
            pv:         [1, 5, 200, 200]  ego soft probability  (detach before calling)
            pi_aligned: [1, 5, 200, 200]  aligned infra prob    (detach before calling)
            warp_valid: [1, 200, 200]     warp valid mask

        Returns:
            dict of detached tensors
        """
        ext   = self._stcv_extractor
        pv_s  = pv[0]           # [5, 200, 200]
        pi_s  = pi_aligned[0]
        w_s   = warp_valid[0]

        feats, hs, rs, cs = [], [], [], []

        # Canonical V2 candidate definition:
        # A P4 patch is a candidate only when infrastructure adds
        # at least one occupied cell not already covered by Ego.
        ov_s = pv_s > 0.1      # [5, 200, 200]  ego binary occupancy
        oi_s = pi_s > 0.1      # [5, 200, 200]  infra binary occupancy
        warp_bool = w_s.bool()
        add_s = (~ov_s) & oi_s & warp_bool.unsqueeze(0)  # [5, 200, 200]

        for h in range(5):
            for r in range(50):
                y0 = 4 * r
                y1 = y0 + 4
                for c in range(50):
                    x0 = 4 * c
                    x1 = x0 + 4
                    if not add_s[h, y0:y1, x0:x1].any():
                        continue
                    feats.append(
                        ext.extract_127_features(
                            pv_s, pi_s, w_s, h, r, c,
                        )
                    )
                    hs.append(h)
                    rs.append(r)
                    cs.append(c)

        if len(feats) == 0:
            return {
                "horizons":     torch.empty(0, dtype=torch.long,    device=device),
                "patch_rs":     torch.empty(0, dtype=torch.long,    device=device),
                "patch_cs":     torch.empty(0, dtype=torch.long,    device=device),
                "features_127": torch.empty(0, 127, dtype=pv.dtype, device=device),
                "v2_mag":       torch.empty(0, dtype=pv.dtype,      device=device),
                "v2_sign_prob": torch.empty(0, dtype=pv.dtype,      device=device),
                "v2_accept":    torch.empty(0, dtype=torch.uint8,   device=device),
            }

        features_127 = torch.stack(feats).to(device)                   # [12500, 127]
        horizons     = torch.tensor(hs, dtype=torch.long, device=device)
        patch_rs     = torch.tensor(rs, dtype=torch.long, device=device)
        patch_cs     = torch.tensor(cs, dtype=torch.long, device=device)

        # V2 utility — frozen, no gradient
        self.v2_predictor.eval()
        with torch.no_grad():
            mag, sign_logit = self.v2_predictor(features_127.detach())
            mag        = mag.squeeze(-1) if mag.dim() > 1 else mag
            sign_prob  = sign_logit.sigmoid().squeeze(-1) if sign_logit.dim() > 1 else sign_logit.sigmoid()
            # Canonical V2 decision: accept when predicted utility magnitude > 0
            v2_accept  = (mag > 0).to(torch.uint8)

        return {
            "horizons":     horizons,
            "patch_rs":     patch_rs,
            "patch_cs":     patch_cs,
            "features_127": features_127.detach(),
            "v2_mag":       mag.detach(),
            "v2_sign_prob": sign_prob.detach(),
            "v2_accept":    v2_accept,
        }

    # ------------------------------------------------------------------
    # Innovation-3 loss (computed inside forward_train)
    # ------------------------------------------------------------------

    def _compute_inn3_loss(self, dense_occ_feat, pv_detached,
                           other_agent_results, device, gt_occ_binary=None,
                           gt_occ_frame_valid=None):
        """
        Full Innovation-3 forward pass with online exact binary label.

        Gradient path:
            L_att → Inn3 MLP → roi_feat (gradient ON) → dense_occ_feat
            features_127 / v2_mag / v2_sign_prob / labels — all detached.

        Args:
            dense_occ_feat:   [1, 5, 256, 200, 200]  with gradient
            pv_detached:      [1, 5, 200, 200]        detached
            other_agent_results: dict from infra forward
            device:           torch.device
            gt_occ_binary:    [1, 5, 200, 200] int tensor {0,1,255} or None
                              When None, falls back to proxy label (smoke test only).
        """
        pi_aligned = None
        warp_valid = None

        ego_occ = self.model_ego_agent.occ_head
        for oout in other_agent_results.values():
            occ_data = oout if isinstance(oout, dict) else {}
            inf_prob_raw = occ_data.get("occ", {})
            if isinstance(inf_prob_raw, dict):
                inf_prob_raw = inf_prob_raw.get("univ2x_occ_prob_data", None)
            if inf_prob_raw is not None:
                veh2inf = occ_data.get("ego2other_rt")
                if veh2inf is not None:
                    with torch.no_grad():
                        _, _, fusion_aux = ego_occ.occ_prob_fusion(
                            pv_detached, inf_prob_raw.unsqueeze(0), veh2inf
                        )
                        pi_aligned = fusion_aux["Pi_aligned"]
                        warp_valid = fusion_aux["warp_valid_mask"]
            break

        if pi_aligned is None or warp_valid is None:
            raise RuntimeError(
                "[V2-4D] Real infrastructure soft occupancy (univ2x_occ_prob_data) "
                "was not found in training other_agent_results. "
                "Pi must come from the infrastructure forward pass; "
                "do NOT fall back to Pv."
            )

        # ── Extract P4 candidates + V2 utility (all detached) ─────────────
        cand  = self._extract_candidates(
            pv_detached, pi_aligned.detach(), warp_valid, device
        )
        n_all = int(cand["horizons"].shape[0])
        n_acc = int(cand["v2_accept"].sum().item())

        self._v2_4d_debug = {
            "dense_occ_feat_shape": tuple(dense_occ_feat.shape),
            "candidate_count":      n_all,
            "v2_accept_count":      n_acc,
            "g0_count":  0,
            "g01_count": 0,
            "g1_count":  0,
        }

        if n_acc == 0:
            return {}

        # ── Innovation-3 predict gates ─────────────────────────────────────
        # roi_feat inside predict_gates retains gradient through dense_occ_feat.
        # All offline features are detached (enforced inside predict_gates).
        gates, logits, accept_mask = self.innovation3_module.predict_gates(
            future_states=dense_occ_feat,
            horizons=cand["horizons"],
            patch_rs=cand["patch_rs"],
            patch_cs=cand["patch_cs"],
            v2_accept=cand["v2_accept"],
            features_127=cand["features_127"],
            v2_mag=cand["v2_mag"],
            v2_sign_prob=cand["v2_sign_prob"],
        )

        self._v2_4d_debug.update({
            "g0_count":  int((gates == 0.0).sum().item()),
            "g01_count": int((gates == 0.1).sum().item()),
            "g1_count":  int((gates == 1.0).sum().item()),
        })

        # ── Online Exact Binary Label  y_att = 1[U_R(0.1) > U_R(1.0) + EPS] ─
        # Strict translation of V2-4C-2A compute_curves; all under no_grad.
        # Falls back to proxy (Pi_mean > Pv_mean) when GT is not available.
        h_acc = cand["horizons"][accept_mask]
        r_acc = cand["patch_rs"][accept_mask]
        c_acc = cand["patch_cs"][accept_mask]

        if gt_occ_binary is None or gt_occ_frame_valid is None:
            if self.v2_4d_strict_exact_label:
                raise RuntimeError(
                    "V2-4D Valid-GT requires GT and frame-valid mask"
                )
            # Non-strict smoke-test path: no GT supervision.
            labels = logits.new_zeros(logits.shape)
            supervise_mask = torch.zeros_like(labels, dtype=torch.bool)
            self._v2_4d_debug["label_source"] = "none"
        else:
            from online_label import (
                assert_gt_encoding_compatible,
                compute_valid_binary_oracle_label_online,
            )

            assert_gt_encoding_compatible(gt_occ_binary, label="train")
            ov = (pv_detached > 0.1).long()

            labels, supervise_mask = compute_valid_binary_oracle_label_online(
                pv=pv_detached,
                pi_aligned=pi_aligned.detach(),
                ov=ov,
                gt=gt_occ_binary,
                warp_valid=warp_valid,
                horizons=h_acc,
                patch_rs=r_acc,
                patch_cs=c_acc,
                frame_valid=gt_occ_frame_valid,
            )

            self._v2_4d_debug.update({
                "label_source": "valid_gt_exact",
                "label_valid_count": int(supervise_mask.sum().item()),
                "label_positive_count": int(
                    labels[supervise_mask].sum().item()
                ),
                "label_positive_rate": float(
                    labels[supervise_mask].float().mean().item()
                ) if supervise_mask.any() else 0.0,
            })

        if logits.numel() > 0 and supervise_mask.any():
            loss_att = F.binary_cross_entropy_with_logits(
                logits[supervise_mask],
                labels[supervise_mask],
            )
        else:
            # Connected zero: backward remains safe with no valid labels.
            loss_att = logits.sum() * 0.0

        self._v2_4d_debug["loss_att"] = float(loss_att.item())

        return {"inn3.loss_att": loss_att}

    # ------------------------------------------------------------------
    # Standard forward entry points — public interface unchanged
    # ------------------------------------------------------------------

    def forward(self, ego_agent_data=None, other_agent_data_dict={},
                return_loss=True, w_label=True, **kwargs):
        if return_loss:
            return self.forward_train(
                ego_agent_data=ego_agent_data,
                other_agent_data_dict=other_agent_data_dict,
                return_loss=return_loss,
                **kwargs,
            )
        else:
            return self.forward_test(
                ego_agent_data=ego_agent_data,
                other_agent_data_dict=other_agent_data_dict,
                return_loss=return_loss,
                w_label=w_label,
                **kwargs,
            )

    def forward_train(self, ego_agent_data=None, other_agent_data_dict={},
                      return_loss=True, **kwargs):
        # UniV2X TODO: hardcode with 'img_metas'
        kwargs.pop('img_metas')

        # Infrastructure forward — frozen, no gradient
        with torch.no_grad():
            other_agent_results = {}
            for name_other_agent in self.other_agent_names:
                loss_other, univ2x_outs_other = getattr(self, name_other_agent)(
                    return_loss=return_loss,
                    **(other_agent_data_dict[name_other_agent]),
                    **kwargs,
                )
                univ2x_outs_other['ego2other_rt'] = \
                    other_agent_data_dict[name_other_agent]['veh2inf_rt']
                univ2x_outs_other['pc_range'] = self.pc_range_dict[name_other_agent]
                other_agent_results[name_other_agent] = univ2x_outs_other

        # Ego forward — returns (loss_dict, univ2x_outs)
        # V2-4D: occ_head.forward_train sets return_occ_feature=True for ego+coop,
        # placing dense_occ_feat [1,5,256,200,200] in univ2x_outs["dense_occ_feat"].
        loss, univ2x_outs = self.model_ego_agent(
            return_loss=return_loss,
            other_agent_results=other_agent_results,
            **ego_agent_data,
            **kwargs,
        )

        # V2-4D Innovation-3: compute and add BCE loss when module is attached
        if (
            self.innovation3_module is not None
            and self.v2_predictor is not None
            and self._stcv_extractor is not None
        ):
            dense_occ_feat = univ2x_outs.get("dense_occ_feat", None)
            if dense_occ_feat is not None:
                device = dense_occ_feat.device

                # Derive Pv (detached) from pred_prob applied to dense_occ_feat
                ego_occ = self.model_ego_agent.occ_head
                with torch.no_grad():
                    probs = [ego_occ.pred_prob[ti](dense_occ_feat.detach()[:, ti])
                             for ti in range(5)]
                    pv_detached = (torch.stack(probs, dim=1)
                                   .permute(0, 2, 1, 3, 4)
                                   .squeeze(1)
                                   .sigmoid())

                inn3_loss = self._compute_inn3_loss(
                    dense_occ_feat, pv_detached, other_agent_results, device,
                    gt_occ_binary=univ2x_outs.get("gt_occ_binary", None),
                    gt_occ_frame_valid=univ2x_outs.get(
                        "gt_occ_frame_valid", None
                    ),
                )
                # Only scalars enter loss dict — no diagnostics here
                loss.update(inn3_loss)

        # Runner-compatible: always return loss dict only
        return loss


    @torch.no_grad()
    def _v2_4d_infer_fusion(
        self, dense_occ_feat, pv, pi_aligned, warp_valid
    ):
        """V2-4D inference: frozen V2 selection + learned gate refinement."""
        if self.innovation3_module is None or self.v2_predictor is None:
            raise RuntimeError("V2-4D inference modules not attached")
        if self._stcv_extractor is None:
            raise RuntimeError("V2-4D feature extractor not attached")
        if dense_occ_feat is None:
            raise RuntimeError("V2-4D dense occupancy feature missing")

        self.v2_predictor.eval()
        self.innovation3_module.eval()

        cand = self._extract_candidates(
            pv, pi_aligned, warp_valid, pv.device
        )

        gates, _, _ = self.innovation3_module.predict_gates(
            future_states=dense_occ_feat,
            horizons=cand["horizons"],
            patch_rs=cand["patch_rs"],
            patch_cs=cand["patch_cs"],
            v2_accept=cand["v2_accept"],
            features_127=cand["features_127"],
            v2_mag=cand["v2_mag"],
            v2_sign_prob=cand["v2_sign_prob"],
        )

        pf = self.innovation3_module.apply_fusion(
            pv=pv,
            pi_aligned=pi_aligned,
            gates=gates,
            horizons=cand["horizons"],
            patch_rs=cand["patch_rs"],
            patch_cs=cand["patch_cs"],
        )

        self._v2_4d_debug = {
            "candidate_count": int(gates.numel()),
            "v2_accept_count": int(cand["v2_accept"].sum().item()),
            "g0_count": int((gates == 0).sum().item()),
            "g01_count": int((gates == 0.1).sum().item()),
            "g1_count": int((gates == 1.0).sum().item()),
            "mode": "inference",
        }

        return pf

    def forward_test(self, ego_agent_data=None, other_agent_data_dict={},
                     w_label=True, return_loss=False, **kwargs):
        other_agent_results = {}
        for name_other_agent in self.other_agent_names:
            other_agent_result = getattr(self, name_other_agent)(
                return_loss=return_loss,
                w_label=w_label,
                **(other_agent_data_dict[name_other_agent]),
                **kwargs,
            )
            other_agent_result[0]['ego2other_rt'] = \
                other_agent_data_dict[name_other_agent]['veh2inf_rt']
            other_agent_result[0]['pc_range'] = self.pc_range_dict[name_other_agent]
            other_agent_results[name_other_agent] = other_agent_result

        # V2-4D: enable only when all inference components are attached.
        v2_4d_fn = None
        if (
            self.innovation3_module is not None
            and self.v2_predictor is not None
            and self._stcv_extractor is not None
        ):
            v2_4d_fn = self._v2_4d_infer_fusion

        result = self.model_ego_agent(
            return_loss=return_loss,
            w_label=w_label,
            other_agent_results=other_agent_results,
            v2_4d_fusion_fn=v2_4d_fn,
            **ego_agent_data,
            **kwargs,
        )
        return result
