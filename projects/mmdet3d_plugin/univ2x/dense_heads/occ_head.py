import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule
from einops import rearrange
from mmdet.core import reduce_mean
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
import copy
import os
import numpy as np
from .occ_head_plugin import MLP, BevFeatureSlicer, SimpleConv2d, CVT_Decoder, Bottleneck, UpsamplingAdd, \
                             predict_instance_segmentation_and_trajectories
from .stcv_occ import STCVOcc

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

@HEADS.register_module()
class OccHead(BaseModule):
    def __init__(self, 
                 # General
                 receptive_field=3,
                 n_future=4,
                 spatial_extent=(50, 50),
                 ignore_index=255,

                 # BEV
                 bevformer_bev_conf = {
                    'xbound': [-51.2, 51.2, 0.512],
                    'ybound': [-51.2, 51.2, 0.512],
                    'zbound': [-10.0, 10.0, 20.0],
                    },
                 grid_conf = None,

                 bev_size=(200, 200),
                 bev_emb_dim=256,
                 bev_proj_dim=64,
                 bev_proj_nlayers=1,

                 # Query
                 query_dim=256,
                 query_mlp_layers=3,
                 detach_query_pos=True,
                 temporal_mlp_layer=2,

                 # Transformer
                 transformer_decoder=None,

                 attn_mask_thresh=0.5,
                 # Loss
                 sample_ignore_mode='all_valid',
                 aux_loss_weight=1.,

                 loss_mask=None,
                 loss_dice=None,

                 # Cfgs
                 init_cfg=None,

                 # Eval
                 pan_eval=False,
                 test_seg_thresh:float=0.5,
                 test_with_track_score=False,

                 # coop
                 is_old_mode=False,
                 is_cooperation=False,
                 is_ego_agent=False,
                 return_occ_data=True,
                 bev_h=200,
                 bev_w=200,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 inf_pc_range=[0, -51.2, -5.0, 102.4, 51.2, 3.0],
                 # STCV-Occ
                 use_stcv_occ=False,
                 stcv_checkpoint_path=None,
                 stcv_threshold=0.70,
                 ):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg)
        self.receptive_field = receptive_field  # NOTE: Used by prepare_future_labels in E2EPredTransformer
        self.n_future = n_future
        self.spatial_extent = spatial_extent
        self.ignore_index  = ignore_index

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.inf_pc_range = inf_pc_range
        self.is_cooperation = is_cooperation
        self.is_ego_agent = is_ego_agent
        self.return_occ_data = return_occ_data
        self.is_old_mode = is_old_mode

        # STCV-Occ initialization
        self.use_stcv_occ = use_stcv_occ
        self.stcv_checkpoint_path = stcv_checkpoint_path
        self.stcv_threshold = stcv_threshold
        self.stcv_occ = None

        if self.use_stcv_occ:
            if not self.is_ego_agent:
                raise ValueError(
                    "STCV-Occ should only be enabled for the ego OccHead."
                )
            if not self.is_cooperation:
                raise ValueError(
                    "STCV-Occ requires is_cooperation=True."
                )
            if self.stcv_checkpoint_path is None:
                raise ValueError(
                    "stcv_checkpoint_path must be provided when use_stcv_occ=True."
                )
            self.stcv_occ = STCVOcc(
                checkpoint_path=self.stcv_checkpoint_path,
                threshold=self.stcv_threshold,
            )
            print(f"[OccHead] STCV-Occ enabled with τ={self.stcv_threshold}")

        #bevformer_bev_conf = {
        #    'xbound': [-51.2, 51.2, 0.512],
        #    'ybound': [-51.2, 51.2, 0.512],
        #    'zbound': [-10.0, 10.0, 20.0],
        #}
        self.bev_sampler =  BevFeatureSlicer(bevformer_bev_conf, grid_conf)
        
        self.bev_size = bev_size
        self.bev_proj_dim = bev_proj_dim

        if bev_proj_nlayers == 0:
            self.bev_light_proj = nn.Sequential()
        else:
            self.bev_light_proj = SimpleConv2d(
                in_channels=bev_emb_dim,
                conv_channels=bev_emb_dim,
                out_channels=bev_proj_dim,
                num_conv=bev_proj_nlayers,
            )

        # Downscale bev_feat -> /4
        self.base_downscale = nn.Sequential(
            Bottleneck(in_channels=bev_proj_dim, downsample=True),
            Bottleneck(in_channels=bev_proj_dim, downsample=True)
        )

        # Future blocks with transformer
        self.n_future_blocks = self.n_future + 1

        # - transformer
        self.attn_mask_thresh = attn_mask_thresh
        
        self.num_trans_layers = transformer_decoder.num_layers
        assert self.num_trans_layers % self.n_future_blocks == 0

        self.num_heads = transformer_decoder.transformerlayers.\
            attn_cfgs.num_heads
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)

        # - temporal-mlps
        # query_out_dim = bev_proj_dim

        temporal_mlp = MLP(query_dim, query_dim, bev_proj_dim, num_layers=temporal_mlp_layer)
        self.temporal_mlps = _get_clones(temporal_mlp, self.n_future_blocks)
            
        # - downscale-convs
        downscale_conv = Bottleneck(in_channels=bev_proj_dim, downsample=True)
        self.downscale_convs = _get_clones(downscale_conv, self.n_future_blocks)
        
        # - upsampleAdds
        upsample_add = UpsamplingAdd(in_channels=bev_proj_dim, out_channels=bev_proj_dim)
        self.upsample_adds = _get_clones(upsample_add, self.n_future_blocks)

        # Decoder
        self.dense_decoder = CVT_Decoder(
            dim=bev_proj_dim,
            blocks=[bev_proj_dim, bev_proj_dim],
        )

        # Query
        self.mode_fuser = nn.Sequential(
                nn.Linear(query_dim, bev_proj_dim),
                nn.LayerNorm(bev_proj_dim),
                nn.ReLU(inplace=True)
            )
        self.multi_query_fuser =  nn.Sequential(
                nn.Linear(query_dim * 3, query_dim * 2),
                nn.LayerNorm(query_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(query_dim * 2, bev_proj_dim),
            )

        self.detach_query_pos = detach_query_pos

        self.query_to_occ_feat = MLP(
            query_dim, query_dim, bev_proj_dim, num_layers=query_mlp_layers
        )
        self.temporal_mlp_for_mask = copy.deepcopy(self.query_to_occ_feat)
        
        if not self.is_old_mode:
            conv1x1 = nn.Conv2d(bev_emb_dim, 1, kernel_size=1)
            self.pred_prob = _get_clones(conv1x1, self.n_future_blocks)
        # Loss
        # For matching
        self.sample_ignore_mode = sample_ignore_mode
        assert self.sample_ignore_mode in ['all_valid', 'past_valid', 'none']

        self.aux_loss_weight = aux_loss_weight

        self.loss_dice = build_loss(loss_dice)
        self.loss_mask = build_loss(loss_mask)

        self.pan_eval = pan_eval
        self.test_seg_thresh = test_seg_thresh
        self.test_with_track_score = test_with_track_score

        # [DEBUG] one-shot sanity print flags
        self._debug_ft_once = False
        self._debug_fusion_once = False

        # [DEBUG] alignment statistics for first N cooperative samples
        self._align_diag_count = 0
        self._align_diag_limit = 50

        self._align_valid_ratio_sum = 0.0
        self._align_zero_valid_samples = 0
        self._align_zero_active_samples = 0
        self._align_inf_active_sum = 0
        self._align_aligned_active_sum = 0

        # ===== G0/G1/G2 offline export =====
        self._g012_export_dir = os.environ.get(
            "UNIV2X_G012_EXPORT_DIR", "")
        self._g012_export_limit = int(
            os.environ.get("UNIV2X_G012_EXPORT_LIMIT", "-1"))
        self._g012_export_idx = 0
        if self._g012_export_dir and self.is_ego_agent:
            os.makedirs(self._g012_export_dir, exist_ok=True)

        self.init_weights()

    def init_weights(self):
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def get_attn_mask(self, state, ins_query):
        # state: b, c, h, w
        # ins_query: b, q, c
        ins_embed = self.temporal_mlp_for_mask(
            ins_query 
        )
        mask_pred = torch.einsum("bqc,bchw->bqhw", ins_embed, state)
        attn_mask = mask_pred.sigmoid() < self.attn_mask_thresh
        attn_mask = rearrange(attn_mask, 'b q h w -> b (h w) q').unsqueeze(1).repeat(
            1, self.num_heads, 1, 1).flatten(0, 1)
        attn_mask = attn_mask.detach()
        
        # if a mask is all True(all background), then set it all False.
        attn_mask[torch.where(
            attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        upsampled_mask_pred = F.interpolate(
            mask_pred,
            self.bev_size,
            mode='bilinear',
            align_corners=False
        )  # Supervised by gt

        return attn_mask, upsampled_mask_pred, ins_embed

    def forward(self, x, ins_query, return_occ_feature=False):
        """
        Args:
            x:                  [(h*w), b, d]  BEV feature
            ins_query:          [b, q, d]      instance queries
            return_occ_feature: bool (default False).
                When True, additionally returns dense_occ_feat [b, T, C, H, W]
                from the dense_decoder output, for use by V2-4D Innovation-3.
                Default False preserves the original two-tuple return.

        Returns (return_occ_feature=False):
            mask_preds, ins_occ_logits
        Returns (return_occ_feature=True):
            mask_preds, ins_occ_logits, dense_occ_feat
        """
        base_state = rearrange(x, '(h w) b d -> b d h w', h=self.bev_size[0])

        base_state = self.bev_sampler(base_state)
        base_state = self.bev_light_proj(base_state)
        base_state = self.base_downscale(base_state)
        base_ins_query = ins_query

        last_state = base_state
        last_ins_query = base_ins_query
        future_states = []
        mask_preds = []
        temporal_query = []
        temporal_embed_for_mask_attn = []
        n_trans_layer_each_block = self.num_trans_layers // self.n_future_blocks
        assert n_trans_layer_each_block >= 1

        for i in range(self.n_future_blocks):
            # Downscale
            cur_state = self.downscale_convs[i](last_state)  # /4 -> /8

            # Attention
            # temporal_aware ins_query
            cur_ins_query = self.temporal_mlps[i](last_ins_query)  # [b, q, d]
            temporal_query.append(cur_ins_query)

            # Generate attn mask
            attn_mask, mask_pred, cur_ins_emb_for_mask_attn = self.get_attn_mask(cur_state, cur_ins_query)
            attn_masks = [None, attn_mask]

            mask_preds.append(mask_pred)  # /1
            temporal_embed_for_mask_attn.append(cur_ins_emb_for_mask_attn)

            cur_state = rearrange(cur_state, 'b c h w -> (h w) b c')
            cur_ins_query = rearrange(cur_ins_query, 'b q c -> q b c')

            for j in range(n_trans_layer_each_block):
                trans_layer_ind = i * n_trans_layer_each_block + j
                trans_layer = self.transformer_decoder.layers[trans_layer_ind]
                cur_state = trans_layer(
                    query=cur_state,  # [h'*w', b, c]
                    key=cur_ins_query,  # [nq, b, c]
                    value=cur_ins_query,  # [nq, b, c]
                    query_pos=None,
                    key_pos=None,
                    attn_masks=attn_masks,
                    query_key_padding_mask=None,
                    key_padding_mask=None
                )  # out size: [h'*w', b, c]

            cur_state = rearrange(cur_state, '(h w) b c -> b c h w', h=self.bev_size[0]//8)

            # Upscale to /4
            cur_state = self.upsample_adds[i](cur_state, last_state)

            # Out
            future_states.append(cur_state)  # [b, d, h/4, w/4]
            last_state = cur_state

        future_states = torch.stack(future_states, dim=1)  # [b, t, d, h/4, w/4]
        temporal_query = torch.stack(temporal_query, dim=1)  # [b, t, q, d]
        mask_preds = torch.stack(mask_preds, dim=2)  # [b, q, t, h, w]
        ins_query = torch.stack(temporal_embed_for_mask_attn, dim=1)  # [b, t, q, d]

        # Decode future states to larger resolution
        future_states = self.dense_decoder(future_states)  # [b, t, c, H, W]
        ins_occ_query = self.query_to_occ_feat(ins_query)    # [b, t, q, query_out_dim]

        # old_mode: same as UniAD
        if self.is_old_mode:
            # Generate final outputs
            ins_occ_logits = torch.einsum("btqc,btchw->bqthw", ins_occ_query, future_states)
        else:
            t = future_states.size(1)
            probs = []
            for ti in range(t):
                p = self.pred_prob[ti](future_states[:, ti, :, :, :])
                probs.append(p)
            ins_occ_logits = torch.stack(probs, dim=1).permute(0, 2, 1, 3, 4) # [b, 1, t, h, w]

        if return_occ_feature:
            # future_states: [b, T, C, H, W] — dense_decoder output, used by Innovation-3
            return mask_preds, ins_occ_logits, future_states
        return mask_preds, ins_occ_logits

    def merge_queries(self, outs_dict, detach_query_pos=True):
        ins_query = outs_dict.get('traj_query', None)       # [n_dec, b, nq, n_modes, dim]
        track_query = outs_dict['track_query']              # [b, nq, d]
        track_query_pos = outs_dict['track_query_pos']      # [b, nq, d]

        if detach_query_pos:
            track_query_pos = track_query_pos.detach()

        ins_query = ins_query[-1]
        ins_query = self.mode_fuser(ins_query).max(2)[0]
        ins_query = self.multi_query_fuser(torch.cat([ins_query, track_query, track_query_pos], dim=-1))
        
        return ins_query

    # With matched queries [a small part of all queries] and matched_gt results
    def forward_train(
                    self,
                    bev_feat,
                    outs_dict,
                    gt_inds_list=None,
                    gt_segmentation=None,
                    gt_instance=None,
                    gt_img_is_valid=None,
                    return_occ_feature=False,
                ):
        # Generate warpped gt and related inputs
        gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(gt_segmentation, gt_instance, gt_img_is_valid)

        all_matched_gt_ids = outs_dict['all_matched_idxes']  # list of tensor, length bs

        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)

        # Forward the occ-flow model
        # V2-4D: when return_occ_feature=True, also return dense_decoder output
        # for use by Innovation-3 Learnable Fusion Module.
        if return_occ_feature:
            mask_preds_batch, ins_seg_preds_batch, dense_occ_feat = self(
                bev_feat, ins_query=ins_query, return_occ_feature=True)
        else:
            mask_preds_batch, ins_seg_preds_batch = self(bev_feat, ins_query=ins_query)
        
        # Get pred and gt
        ins_seg_targets_batch  = gt_instance # [1, 5, 200, 200] [b, t, h, w] # ins targets of a batch
        
        # img_valid flag, for filtering out invalid samples in sequence when calculating loss
        img_is_valid = gt_img_is_valid  # [1, 7]
        assert img_is_valid.size(1) == self.receptive_field + self.n_future,  \
                f"Img_is_valid can only be 7 as for loss calculation and evaluation!!! Don't change it"
        frame_valid_mask = img_is_valid.bool()
        past_valid_mask  = frame_valid_mask[:, :self.receptive_field]
        future_frame_mask = frame_valid_mask[:, (self.receptive_field-1):]  # [1, 5]  including current frame

        # only supervise when all 3 past frames are valid
        past_valid = past_valid_mask.all(dim=1)
        future_frame_mask[~past_valid] = False
        
        # Calculate loss in the batch
        loss_dict = dict()
        loss_dice = ins_seg_preds_batch.new_zeros(1)[0].float()
        loss_mask = ins_seg_preds_batch.new_zeros(1)[0].float()
        loss_aux_dice = ins_seg_preds_batch.new_zeros(1)[0].float()
        loss_aux_mask = ins_seg_preds_batch.new_zeros(1)[0].float()

        bs = ins_query.size(0)
        assert bs == 1
        for ind in range(bs):
            # Each gt_bboxes contains 3 frames, we only use the last one
            cur_gt_inds   = gt_inds_list[ind][-1]

            cur_matched_gt = all_matched_gt_ids[ind]  # [n_gt]
            
            # Re-order gt according to matched_gt_inds
            cur_gt_inds   = cur_gt_inds[cur_matched_gt]
            
            # Deal matched_gt: -1, its actually background(unmatched)
            cur_gt_inds[cur_matched_gt == -1] = -1  # Bugfixed
            cur_gt_inds[cur_matched_gt == -2] = -2  

            frame_mask = future_frame_mask[ind]  # [t]

            # Prediction
            ins_seg_preds = ins_seg_preds_batch[ind]   # [q(n_gt for matched), t, h, w]
            ins_seg_targets = ins_seg_targets_batch[ind]  # [t, h, w]
            mask_preds = mask_preds_batch[ind]
            
            # Assigned-gt
            ins_seg_targets_ordered = []
            for ins_id in cur_gt_inds:
                # -1 for unmatched query
                # If ins_seg_targets is all 255, ignore (directly append occ-and-flow gt to list)
                # 255 for special object --> change to -20 (same as in occ_label.py)
                # -2 for no_query situation
                if (ins_seg_targets == self.ignore_index).all().item() is True:
                    ins_tgt = ins_seg_targets.long()
                elif ins_id.item() in [-1, -2] :  # false positive query (unmatched)
                    ins_tgt = torch.ones_like(ins_seg_targets).long() * self.ignore_index
                else:
                    SPECIAL_INDEX = -20
                    if ins_id.item() == self.ignore_index:
                        ins_id = torch.ones_like(ins_id) * SPECIAL_INDEX
                    ins_tgt = (ins_seg_targets == ins_id).long()  # [t, h, w], 0 or 1
                
                ins_seg_targets_ordered.append(ins_tgt)
            
            ins_seg_targets_ordered_for_mask = torch.stack(ins_seg_targets_ordered, dim=0)  # [n_gt, t, h, w]
            ins_seg_targets_ordered = ins_seg_targets_ordered_for_mask.max(0)[0].unsqueeze(0)
            
            # Sanity check
            t, h, w = ins_seg_preds.shape[-3:]
            assert t == 1+self.n_future, f"{ins_seg_preds.size()}"
            assert ins_seg_preds.size() == ins_seg_targets_ordered.size(),   \
                            f"{ins_seg_preds.size()}, {ins_seg_targets_ordered.size()}"
            
            num_total_pos = ins_seg_preds.size(0)  # Check this line 

            # loss for a sample in batch
            num_total_pos = ins_seg_preds.new_tensor([num_total_pos])
            num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
            
            cur_dice_loss = self.loss_dice(
                ins_seg_preds, ins_seg_targets_ordered, avg_factor=num_total_pos, frame_mask=frame_mask)

            cur_mask_loss = self.loss_mask(
                ins_seg_preds, ins_seg_targets_ordered, frame_mask=frame_mask
            )

            cur_aux_dice_loss = self.loss_dice(
                mask_preds, ins_seg_targets_ordered_for_mask, avg_factor=num_total_pos, frame_mask=frame_mask
            )
            cur_aux_mask_loss = self.loss_mask(
                mask_preds, ins_seg_targets_ordered_for_mask, frame_mask=frame_mask
            )

            loss_dice += cur_dice_loss
            loss_mask += cur_mask_loss
            loss_aux_dice += cur_aux_dice_loss * self.aux_loss_weight
            loss_aux_mask += cur_aux_mask_loss * self.aux_loss_weight

        loss_dict['loss_dice'] = loss_dice / bs
        loss_dict['loss_mask'] = loss_mask / bs
        loss_dict['loss_aux_dice'] = loss_aux_dice / bs
        loss_dict['loss_aux_mask'] = loss_aux_mask / bs

        if return_occ_feature:
            # V2-4D: return dense_occ_feat [b, T, C, H, W] alongside losses
            return loss_dict, dense_occ_feat
        return loss_dict

    def forward_test(
                    self,
                    bev_feat,
                    outs_dict,
                    no_query=False,
                    gt_segmentation=None,
                    gt_instance=None,
                    gt_img_is_valid=None,
                    w_label=True,
                    other_agent_results=None,
                    g012_meta=None,
                    v2_4d_fusion_fn=None,
                ):
        
        out_dict = dict()

        if w_label:
            gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(gt_segmentation, gt_instance, gt_img_is_valid)
            out_dict['seg_gt']  = gt_segmentation[:, :1+self.n_future]  # [1, 5, 1, 200, 200]
            out_dict['ins_seg_gt'] = self.get_ins_seg_gt(gt_instance[:, :1+self.n_future])  # [1, 5, 200, 200]

        dense_occ_feat = None  # V2-4D: will be populated when return_occ_feature=True
        if no_query:
            b = outs_dict['track_scores'].shape[0]
            q = outs_dict['track_scores'].shape[1]
            pred_ins_logits = torch.zeros([b,q,5,200,200]).to(bev_feat) # [b, q, t, h, w] hard code
        else:
            ins_query = self.merge_queries(outs_dict, self.detach_query_pos)
            # V2-4D: request dense_occ_feat for Innovation-3 (ego cooperative agent only)
            if self.is_ego_agent and self.is_cooperation:
                _, pred_ins_logits, dense_occ_feat = self(
                    bev_feat, ins_query=ins_query, return_occ_feature=True)
            else:
                _, pred_ins_logits = self(bev_feat, ins_query=ins_query)

        out_dict['pred_ins_logits'] = pred_ins_logits

        pred_ins_logits = pred_ins_logits[:,:,:1+self.n_future]  # [b, q, t, h, w]
        pred_ins_sigmoid = pred_ins_logits.sigmoid()  # [b, q, t, h, w]

        # if self.test_with_track_score:
        #     track_scores = outs_dict['track_scores'].to(pred_ins_sigmoid)  # [b, q]
        #     track_scores = track_scores[:, :, None, None, None]
        #     pred_ins_sigmoid = pred_ins_sigmoid * track_scores  # [b, q, t, h, w]

        out_dict['pred_ins_sigmoid'] = pred_ins_sigmoid
        if pred_ins_sigmoid.shape[1] != 0:
            pred_seg_scores = pred_ins_sigmoid.max(1)[0] #[b, t, h, w]
        else:
            pred_seg_scores = torch.zeros([b,5,200,200]).to(bev_feat) # [b, t, h, w] hard code

        # [DEBUG] one-shot sanity check: dtype / range / q dim (pre-fusion)
        if not self._debug_ft_once:
            tag = "EGO" if getattr(self, 'is_ego_agent', False) else "INF"
            print(f"\n===== [DEBUG forward_test | {tag}] =====")
            print("no_query:", no_query)
            print("is_old_mode:", self.is_old_mode)
            if pred_ins_logits.numel() > 0:
                print("pred_ins_logits:", tuple(pred_ins_logits.shape),
                      pred_ins_logits.dtype,
                      pred_ins_logits.min().item(), pred_ins_logits.max().item())
            else:
                print("pred_ins_logits:", tuple(pred_ins_logits.shape), "(empty, q=0)")
            if pred_ins_sigmoid.numel() > 0:
                print("pred_ins_sigmoid:", tuple(pred_ins_sigmoid.shape),
                      pred_ins_sigmoid.dtype,
                      pred_ins_sigmoid.min().item(), pred_ins_sigmoid.max().item())
            else:
                print("pred_ins_sigmoid:", tuple(pred_ins_sigmoid.shape), "(empty, q=0)")
            print("pred_seg_scores (pre-fusion):", tuple(pred_seg_scores.shape),
                  pred_seg_scores.dtype,
                  pred_seg_scores.min().item(), pred_seg_scores.max().item())
            print("=====================================\n")
            self._debug_ft_once = True

        fusion_aux = None
        if self.is_ego_agent and self.is_cooperation and other_agent_results:
            for other_agent_name, other_agent_result in other_agent_results.items():
                if 'univ2x_occ_prob_data' not in other_agent_results[other_agent_name][0]['occ']:
                    import pdb;pdb.set_trace()
                other_agent_occ_data = other_agent_results[other_agent_name][0]['occ']['univ2x_occ_prob_data']
                other_agent_occ_data = torch.stack([other_agent_occ_data], dim=0)
                new_pred_seg_scores, new_inf_occ, fusion_aux = self.occ_prob_fusion(pred_seg_scores, other_agent_occ_data, other_agent_results[other_agent_name][0]['ego2other_rt'])   # [1, t, h, w]
                pred_seg_scores = new_pred_seg_scores
        
        # V2-4D: independent inference fusion before seg_out.
        if v2_4d_fusion_fn is not None:
            if not (self.is_ego_agent and self.is_cooperation):
                raise RuntimeError("V2-4D requires cooperative Ego OccHead")
            if self.use_stcv_occ:
                raise RuntimeError(
                    "V2-4D and historical STCV cannot run together"
                )
            if dense_occ_feat is None or fusion_aux is None:
                raise RuntimeError(
                    "V2-4D requires dense_occ_feat and real fusion_aux"
                )

            pred_seg_scores = v2_4d_fusion_fn(
                dense_occ_feat=dense_occ_feat,
                pv=fusion_aux["Pv"],
                pi_aligned=fusion_aux["Pi_aligned"],
                warp_valid=fusion_aux["warp_valid_mask"],
            )

        seg_out = (pred_seg_scores > self.test_seg_thresh).long().unsqueeze(2)  # [b, t, 1, h, w]
        out_dict['seg_out'] = seg_out
        if self.pan_eval:
            if pred_ins_sigmoid.shape[1] != 0:
                # ins_pred
                pred_consistent_instance_seg =  \
                    predict_instance_segmentation_and_trajectories(seg_out, pred_ins_sigmoid)  # bg is 0, fg starts with 1, consecutive
                
                out_dict['ins_seg_out'] = pred_consistent_instance_seg  # [1, 5, 200, 200]
            else:
                device = pred_ins_logits.device
                out_dict['ins_seg_out'] = torch.zeros([1,5,200,200]).long().to(device)

        if not self.is_ego_agent and self.return_occ_data:
            out_dict['univ2x_occ_prob_data'] = pred_seg_scores[0]

        # ============================================================
        # G0/G1/G2 offline export
        # Only export from ego cooperative OccHead with GT
        # ============================================================
        if (
            self.is_ego_agent
            and fusion_aux is not None
            and w_label
            and self._g012_export_dir
        ):
            do_export = (
                self._g012_export_limit < 0
                or self._g012_export_idx < self._g012_export_limit
            )
            if do_export:
                export_idx = self._g012_export_idx

                # GT: [1, 5, 1, 200, 200] -> [1, 5, 200, 200]
                gt_occ = out_dict["seg_gt"].squeeze(2)

                # Same temporal validity rule as training code
                frame_valid_mask = gt_img_is_valid.bool()
                past_valid_mask = frame_valid_mask[:, :self.receptive_field]
                future_valid_mask = frame_valid_mask[
                    :, (self.receptive_field - 1):
                ].clone()
                past_valid = past_valid_mask.all(dim=1)
                future_valid_mask[~past_valid] = False

                # Preserve ignore_index information
                gt_cell_valid_mask = (gt_occ != self.ignore_index)

                # Move all to CPU and numpy
                if g012_meta is None:
                    g012_meta = {}

                scene_token = g012_meta.get("scene_token", "")
                sample_idx = g012_meta.get("sample_idx", "")
                sample_timestamp = g012_meta.get("timestamp", "")
                export_data = {
                    "Pv": fusion_aux["Pv"][0].cpu().numpy().astype(np.float32),
                    "Pi_aligned": fusion_aux["Pi_aligned"][0].cpu().numpy().astype(np.float32),

                    "Ov": fusion_aux["Ov"][0].cpu().numpy().astype(np.uint8),
                    "Oi": fusion_aux["Oi"][0].cpu().numpy().astype(np.uint8),
                    "Oofficial": fusion_aux["Oofficial"][0].cpu().numpy().astype(np.uint8),
                    "Ofused": fusion_aux["Ofused"][0].cpu().numpy().astype(np.uint8),

                    "GT": gt_occ[0].cpu().numpy().astype(np.int16),

                    "gt_cell_valid_mask":
                        gt_cell_valid_mask[0].cpu().numpy().astype(np.uint8),

                    "future_valid_mask":
                        future_valid_mask[0].cpu().numpy().astype(np.uint8),

                    "warp_valid_mask":
                        fusion_aux["warp_valid_mask"][0]
                        .cpu().numpy().astype(np.uint8),

                    "test_seg_thresh": self.test_seg_thresh,
                    "export_idx": export_idx,

                    # metadata
                    "scene_token": np.asarray(str(scene_token)),
                    "sample_idx": np.asarray(str(sample_idx)),
                    "timestamp": np.asarray(
                                            sample_timestamp,
                                            dtype=np.float64
                                        ),
                }

                export_path = os.path.join(
                    self._g012_export_dir,
                    f"sample_{export_idx:05d}.npz"
                )
                np.savez_compressed(export_path, **export_data)

                self._g012_export_idx += 1

                if export_idx == 0 or (export_idx + 1) % 10 == 0:
                    print(f"[G0 Export] Saved {export_idx + 1} samples to {self._g012_export_dir}")

        return out_dict

    def occ_prob_fusion(self, veh_occ, inf_occ, veh2inf_rt):
        # veh_occ: 1,5,200,200
        # inf_occ: 1,5,200,200
        bs = veh_occ.shape[0]
                    
        xs = torch.linspace(0.5, self.bev_w - 0.5, self.bev_w, dtype=veh_occ.dtype,
                            device=veh_occ.device).view(1, self.bev_w).expand(self.bev_h, self.bev_w) / self.bev_w
        ys = torch.linspace(0.5, self.bev_h - 0.5, self.bev_h, dtype=veh_occ.dtype,
                            device=veh_occ.device).view(self.bev_h, 1).expand(self.bev_h, self.bev_w) / self.bev_h
        bev_grid = torch.stack((xs, ys), -1)
        # [bs, bev_h, bev_2, 2]
        bev_grid = bev_grid[None].repeat(bs, 1, 1, 1) 

        # compute x coordinate in veh frame
        bev_grid[..., 0:1] = bev_grid[..., 0:1] * \
            (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        # compute y coordinate in veh frame
        bev_grid[..., 1:2] = bev_grid[..., 1:2] * \
            (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        
        # [bs, 3, 3]
        veh2inf_rt = veh2inf_rt[:, [0, 1, 3]][:, :, [0, 1, 3]]
        # compute coordinates in inf frame
        add_ones = torch.ones((bs, self.bev_h, self.bev_w, 1), dtype=bev_grid.dtype, device=bev_grid.device)
        bev_grid = torch.cat((bev_grid, add_ones), dim=-1)
        bev_grid = bev_grid @ veh2inf_rt

        # [DEBUG] 保存归一化前的路侧物理坐标，便于检查
        bev_grid_metric = bev_grid[..., :-1].clone()

        # compute sample location in inf bev feature
        bev_grid = bev_grid[..., :-1]
        bev_grid[..., 0:1] = (2 * bev_grid[..., 0:1] - (self.inf_pc_range[3] - self.inf_pc_range[0])) / (self.inf_pc_range[3] - self.inf_pc_range[0]) # [0, 100]
        bev_grid[..., 1:2] = (2 * bev_grid[..., 1:2]) / (self.inf_pc_range[4] - self.inf_pc_range[1]) # [-50, 50]

        # ===== 统一计算 warp_valid_mask =====
        grid_x = bev_grid[..., 0]
        grid_y = bev_grid[..., 1]
        warp_valid_mask = (
            (grid_x >= -1.0) & (grid_x <= 1.0) &
            (grid_y >= -1.0) & (grid_y <= 1.0)
        )  # [bs, h, w]

        # [DEBUG] 检查 alignment sampling grid（与末尾 fusion DEBUG 共用 flag，仅此块不置位）
        if not self._debug_fusion_once:
            print("\n===== [DEBUG alignment grid] =====")
            print("transformed metric x range:",
                  bev_grid_metric[..., 0].min().item(),
                  bev_grid_metric[..., 0].max().item())
            print("transformed metric y range:",
                  bev_grid_metric[..., 1].min().item(),
                  bev_grid_metric[..., 1].max().item())
            print("bev_grid normalized x range:",
                  grid_x.min().item(), grid_x.max().item())
            print("bev_grid normalized y range:",
                  grid_y.min().item(), grid_y.max().item())
            print("valid grid ratio:", warp_valid_mask.float().mean().item())
            print("valid grid count:", warp_valid_mask.sum().item(),
                  "/", warp_valid_mask.numel())
            print("veh2inf_rt:")
            print(veh2inf_rt)
            print("inf_occ > threshold count:",
                  (inf_occ > self.test_seg_thresh).sum().item())
            print("==================================\n")
        new_inf_occ = F.grid_sample(inf_occ, bev_grid, align_corners=True)

        # ===== 前 50 个样本 alignment 统计 =====
        if self._align_diag_count < self._align_diag_limit:
            self._align_diag_count += 1
            idx = self._align_diag_count

            valid_grid_ratio = warp_valid_mask.float().mean().item()
            valid_grid_count = warp_valid_mask.sum().item()
            total_grid_count = warp_valid_mask.numel()

            inf_active_count = (
                inf_occ > self.test_seg_thresh
            ).sum().item()

            aligned_active_count = (
                new_inf_occ > self.test_seg_thresh
            ).sum().item()

            aligned_nonzero_count = (
                new_inf_occ != 0
            ).sum().item()

            self._align_valid_ratio_sum += valid_grid_ratio
            self._align_inf_active_sum += inf_active_count
            self._align_aligned_active_sum += aligned_active_count

            if valid_grid_count == 0:
                self._align_zero_valid_samples += 1

            if aligned_active_count == 0:
                self._align_zero_active_samples += 1

            print(
                f"[ALIGN {idx:02d}/{self._align_diag_limit}] "
                f"valid_ratio={valid_grid_ratio:.4f} | "
                f"valid_grid={valid_grid_count}/{total_grid_count} | "
                f"inf>thr={inf_active_count} | "
                f"aligned_nonzero={aligned_nonzero_count} | "
                f"aligned>thr={aligned_active_count}"
            )

            if idx == self._align_diag_limit:
                print("\n========== [ALIGNMENT SUMMARY] ==========")
                print("samples:", self._align_diag_count)
                print(
                    "mean valid_grid_ratio:",
                    self._align_valid_ratio_sum / self._align_diag_count
                )
                print(
                    "zero-valid-grid samples:",
                    self._align_zero_valid_samples,
                    "/",
                    self._align_diag_count
                )
                print(
                    "zero-aligned-active samples:",
                    self._align_zero_active_samples,
                    "/",
                    self._align_diag_count
                )
                print(
                    "total original inf > threshold:",
                    self._align_inf_active_sum
                )
                print(
                    "total aligned inf > threshold:",
                    self._align_aligned_active_sum
                )
                print("=========================================\n")
        # ========================================

        veh_occ_log = (veh_occ > self.test_seg_thresh).long()
        inf_occ_log = (new_inf_occ > self.test_seg_thresh).long()

        # Official UniV2X OR fusion
        official_occ = torch.maximum(veh_occ_log, inf_occ_log)

        # STCV-Occ selective fusion
        if self.use_stcv_occ:
            fused_occ = self.stcv_occ(
                pv=veh_occ,
                pi=new_inf_occ,
                ov=veh_occ_log,
                oi=inf_occ_log,
                warp=warp_valid_mask,
            )
        else:
            fused_occ = official_occ

        # Official OR invariant check (corrected: checks official_occ, not fused_occ)
        official_check = torch.maximum(veh_occ_log, inf_occ_log)
        if not torch.equal(official_occ.long(), official_check.long()):
            raise RuntimeError("Official OccFusion != Ov OR Oi")

        # [DEBUG] one-shot sanity check inside fusion
        if not self._debug_fusion_once:
            print("\n===== [DEBUG occ_prob_fusion] =====")
            print("veh_occ:", tuple(veh_occ.shape), veh_occ.dtype,
                  veh_occ.min().item(), veh_occ.max().item())
            print("inf_occ (transmitted):", tuple(inf_occ.shape), inf_occ.dtype,
                  inf_occ.min().item(), inf_occ.max().item())
            print("new_inf_occ (aligned):", tuple(new_inf_occ.shape), new_inf_occ.dtype,
                  new_inf_occ.min().item(), new_inf_occ.max().item())
            print("test_seg_thresh:", self.test_seg_thresh)
            print("veh_occ_log unique:", torch.unique(veh_occ_log))
            print("inf_occ_log unique:", torch.unique(inf_occ_log))
            print("official_occ unique:", torch.unique(official_occ))
            print("fused_occ:", tuple(fused_occ.shape), fused_occ.dtype,
                  "unique:", torch.unique(fused_occ))
            print("use_stcv_occ:", self.use_stcv_occ)
            print("===================================\n")
            self._debug_fusion_once = True

        fusion_aux = {
            # Soft probability
            "Pv": veh_occ.detach(),
            "Pi_aligned": new_inf_occ.detach(),
            # Binary occupancy
            "Ov": veh_occ_log.detach(),
            "Oi": inf_occ_log.detach(),
            # Official binary OR (always computed)
            "Oofficial": official_occ.detach(),
            # Fused occupancy (Official OR or STCV Learned)
            "Ofused": fused_occ.detach(),
            # Spatial validity of infrastructure warp
            "warp_valid_mask": warp_valid_mask.detach(),
        }

        return fused_occ, inf_occ_log, fusion_aux
    
    def get_ins_seg_gt(self, gt_instance):
        ins_gt_old = gt_instance  # Not consecutive, 0 for bg, otherwise ins_ind(start from 1)
        ins_gt_new = torch.zeros_like(ins_gt_old).to(ins_gt_old)  # Make it consecutive
        ins_inds_unique = torch.unique(ins_gt_old)
        new_id = 1
        for uni_id in ins_inds_unique:
            if uni_id.item() in [0, self.ignore_index]:  # ignore background_id
                continue
            ins_gt_new[ins_gt_old == uni_id] = new_id
            new_id += 1
        return ins_gt_new  # Consecutive

    def get_occ_labels(self, gt_segmentation, gt_instance, gt_img_is_valid):
        if not self.training:
            gt_segmentation = gt_segmentation[0]
            gt_instance = gt_instance[0]
            gt_img_is_valid = gt_img_is_valid[0]

        gt_segmentation = gt_segmentation[:, :self.n_future+1].long().unsqueeze(2)
        gt_instance = gt_instance[:, :self.n_future+1].long()
        gt_img_is_valid = gt_img_is_valid[:, :self.receptive_field + self.n_future]
        return gt_segmentation, gt_instance, gt_img_is_valid
