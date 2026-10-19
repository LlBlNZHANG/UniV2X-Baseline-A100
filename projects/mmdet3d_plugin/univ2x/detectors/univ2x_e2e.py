#----------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                 #
# Copyright (c) DAIR-V2X. All rights reserved.                   #
# Modified from UniAD (https://github.com/OpenDriveLab/UniAD)    #
#----------------------------------------------------------------#

import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
import copy
import os 
from ..dense_heads.seg_head_plugin import IOU
from .univ2x_track import UniV2XTrack
from mmdet.models.builder import build_head
import mmcv

@DETECTORS.register_module()
class UniV2X(UniV2XTrack):
    """
    UniV2X: End-to-End Autonomous Driving through V2X Cooperation
    """
    def __init__(
        self,
        seg_head=None,
        motion_head=None,
        occ_head=None,
        planning_head=None,
        task_loss_weight=dict(
            track=1.0,
            map=1.0,
            motion=1.0,
            occ=1.0,
            planning=1.0
        ),
        is_cooperation = False,
        is_ego_agent = False,
        save_track_query=False,
        save_track_query_file_root = '',
        read_track_query_file_root = '',
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        inf_pc_range= [0, -51.2, -5.0, 102.4, 51.2, 3.0],
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        load_from='',
        **kwargs,
    ):
        super(UniV2X, self).__init__(is_cooperation=is_cooperation,
                                                                    is_ego_agent=is_ego_agent,
                                                                    pc_range=pc_range,
                                                                    inf_pc_range=inf_pc_range,
                                                                    post_center_range=post_center_range,
                                                                    **kwargs)
        if seg_head:
            self.seg_head = build_head(seg_head)
            self.seg_head.is_ego_agent = is_ego_agent
        if occ_head:
            self.occ_head = build_head(occ_head)
            self.occ_head.is_ego_agent = is_ego_agent
        if motion_head:
            self.motion_head = build_head(motion_head)
        if planning_head:
            self.planning_head = build_head(planning_head)
        
        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == \
               {'track', 'occ', 'motion', 'map', 'planning'}
    
        self.is_cooperation = is_cooperation
        self.save_track_query = save_track_query
        self.save_track_query_file_root = save_track_query_file_root
        self.read_track_query_file_root = read_track_query_file_root

        self.is_ego_agent = is_ego_agent


    @property
    def with_planning_head(self):
        return hasattr(self, 'planning_head') and self.planning_head is not None
    
    @property
    def with_occ_head(self):
        return hasattr(self, 'occ_head') and self.occ_head is not None

    @property
    def with_motion_head(self):
        return hasattr(self, 'motion_head') and self.motion_head is not None

    @property
    def with_seg_head(self):
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
        

    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_past_traj=None,
                      gt_past_traj_mask=None,
                      gt_sdc_bbox=None,
                      gt_sdc_label=None,
                      gt_sdc_fut_traj=None,
                      gt_sdc_fut_traj_mask=None,
                      
                      # Occ_gt
                      gt_segmentation=None,
                      gt_instance=None, 
                      gt_occ_img_is_valid=None,
                      
                      #planning
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      
                      # fut gt for planning
                      gt_future_boxes=None,

                      #for coop
                     other_agent_results=None,

                      **kwargs,  # [1, 9]
                      ):
        """Forward training function for the model that includes multiple tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning.

            Args:
            img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
            img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
            gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
            gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
            l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
            l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
            timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): List of tensors containing ground truth 2D bounding boxes in images to be ignored. Defaults to None.
            gt_lane_labels (list[torch.Tensor], optional): List of tensors containing ground truth lane labels. Defaults to None.
            gt_lane_bboxes (list[torch.Tensor], optional): List of tensors containing ground truth lane bounding boxes. Defaults to None.
            gt_lane_masks (list[torch.Tensor], optional): List of tensors containing ground truth lane masks. Defaults to None.
            gt_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth future trajectories. Defaults to None.
            gt_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth future trajectory masks. Defaults to None.
            gt_past_traj (list[torch.Tensor], optional): List of tensors containing ground truth past trajectories. Defaults to None.
            gt_past_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth past trajectory masks. Defaults to None.
            gt_sdc_bbox (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car bounding boxes. Defaults to None.
            gt_sdc_label (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car labels. Defaults to None.
            gt_sdc_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectories. Defaults to None.
            gt_sdc_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectory masks. Defaults to None.
            gt_segmentation (list[torch.Tensor], optional): List of tensors containing ground truth segmentation masks. Defaults to
            gt_instance (list[torch.Tensor], optional): List of tensors containing ground truth instance segmentation masks. Defaults to None.
            gt_occ_img_is_valid (list[torch.Tensor], optional): List of tensors containing binary flags indicating whether an image is valid for occupancy prediction. Defaults to None.
            sdc_planning (list[torch.Tensor], optional): List of tensors containing self-driving car planning information. Defaults to None.
            sdc_planning_mask (list[torch.Tensor], optional): List of tensors containing self-driving car planning masks. Defaults to None.
            command (list[torch.Tensor], optional): List of tensors containing high-level command information for planning. Defaults to None.
            gt_future_boxes (list[torch.Tensor], optional): List of tensors containing ground truth future bounding boxes for planning. Defaults to None.
            gt_future_labels (list[torch.Tensor], optional): List of tensors containing ground truth future labels for planning. Defaults to None.
            
            Returns:
                dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary 
                    is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
        """
        losses = dict()
        len_queue = img.size(1)

        losses_track, outs_track = self.forward_track_train(img, gt_bboxes_3d, gt_labels_3d, gt_past_traj, gt_past_traj_mask, gt_inds, gt_sdc_bbox, gt_sdc_label,
                                                        l2g_t, l2g_r_mat, img_metas, timestamp, other_agent_results=other_agent_results)
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
        losses.update(losses_track)
        
        # Upsample bev for tiny version
        outs_track = self.upsample_bev_if_tiny(outs_track)

        bev_embed = outs_track["bev_embed"]
        bev_pos  = outs_track["bev_pos"]

        img_metas = [each[len_queue-1] for each in img_metas]

        outs_seg = dict()
        if self.with_seg_head:          
            losses_seg, outs_seg = self.seg_head.forward_train(bev_embed, img_metas,
                                                          gt_lane_labels, gt_lane_bboxes, gt_lane_masks, other_agent_results=other_agent_results)
            
            losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix='map')
            losses.update(losses_seg)

        outs_motion = dict()
        # Forward Motion Head
        if self.with_motion_head:
            ret_dict_motion = self.motion_head.forward_train(bev_embed,
                                                        gt_bboxes_3d, gt_labels_3d, 
                                                        gt_fut_traj, gt_fut_traj_mask, 
                                                        gt_sdc_fut_traj, gt_sdc_fut_traj_mask, 
                                                        outs_track=outs_track, outs_seg=outs_seg
                                                    )
            losses_motion = ret_dict_motion["losses"]
            outs_motion = ret_dict_motion["outs_motion"]
            outs_motion['bev_pos'] = bev_pos
            losses_motion = self.loss_weighted_and_prefixed(losses_motion, prefix='motion')
            losses.update(losses_motion)

        # Forward Occ Head
        if self.with_occ_head:
            if outs_motion['track_query'].shape[1] == 0:
                # TODO: rm hard code
                outs_motion['track_query'] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion['track_query_pos'] = torch.zeros((1,1, 256)).to(bev_embed)
                outs_motion['traj_query'] = torch.zeros((3, 1, 1, 6, 256)).to(bev_embed)
                outs_motion['all_matched_idxes'] = [[-1]]

            # V2-4D: ego cooperative agent requests dense_occ_feat for Innovation-3
            need_occ_feat = (
                (self.is_ego_agent and self.is_cooperation)
                or (
                    (not self.is_ego_agent)
                    and getattr(self.occ_head, "return_occ_data", False)
                )
            )
            if need_occ_feat:
                losses_occ, dense_occ_feat = self.occ_head.forward_train(
                                bev_embed,
                                outs_motion,
                                gt_inds_list=gt_inds,
                                gt_segmentation=gt_segmentation,
                                gt_instance=gt_instance,
                                gt_img_is_valid=gt_occ_img_is_valid,
                                return_occ_feature=True,
                            )
            else:
                losses_occ = self.occ_head.forward_train(
                                bev_embed,
                                outs_motion,
                                gt_inds_list=gt_inds,
                                gt_segmentation=gt_segmentation,
                                gt_instance=gt_instance,
                                gt_img_is_valid=gt_occ_img_is_valid,
                            )
                dense_occ_feat = None
            losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix='occ')
            losses.update(losses_occ)
        
        # Forward Plan Head
        if self.with_planning_head:
            outs_planning = self.planning_head.forward_train(bev_embed, outs_motion, sdc_planning, sdc_planning_mask, command, gt_future_boxes)
            losses_planning = outs_planning['losses']
            losses_planning = self.loss_weighted_and_prefixed(losses_planning, prefix='planning')
            losses.update(losses_planning)
        
        for k,v in losses.items():
            losses[k] = torch.nan_to_num(v)

        # used for ego-agent fusion
        univ2x_outs = {}
        if not self.is_ego_agent:
            univ2x_outs['univ2x_track_instances_list'] = outs_track['univ2x_track_instances_list']
            univ2x_outs['univ2x_lane_query_data'] = outs_seg['univ2x_lane_query_data']
            # V2-4D: expose infrastructure soft occupancy during training.
            # Reproduces the same representation used in forward_test().
            if self.with_occ_head and dense_occ_feat is not None:
                with torch.no_grad():
                    _occ_logits = []
                    for ti in range(1 + self.occ_head.n_future):
                        _occ_logits.append(
                            self.occ_head.pred_prob[ti](
                                dense_occ_feat[:, ti, :, :, :]
                            )
                        )
                    _occ_logits = (
                        torch.stack(_occ_logits, dim=1)
                        .permute(0, 2, 1, 3, 4)
                    )
                    _occ_prob = _occ_logits.squeeze(1).sigmoid()
                univ2x_outs["occ"] = {
                    "univ2x_occ_prob_data": _occ_prob[0]
                }

        # V2-4D: pass dense_occ_feat and GT to caller for Innovation-3 joint fine-tuning
        if self.is_ego_agent and self.is_cooperation:
            univ2x_outs['dense_occ_feat'] = dense_occ_feat if self.with_occ_head else None
            # GT binary occupancy for online label computation: [1, T, 200, 200] int
            # Process gt_segmentation same way occ_head.get_occ_labels does
            if gt_segmentation is not None and self.with_occ_head:
                try:
                    import torch as _t
                    # gt_segmentation arrives as list-of-list after scatter; unwrap
                    gs = gt_segmentation[0] if isinstance(gt_segmentation, list) else gt_segmentation
                    gs = gs[:, :self.occ_head.n_future + 1].long()  # [1, T, H, W]
                    # V2-4D Valid-GT: preserve 0/1/255 encoding.
                    univ2x_outs['gt_occ_binary'] = gs.detach()

                    # Match the official OccHead temporal validity rule.
                    fv = (
                        gt_occ_img_is_valid[0]
                        if isinstance(gt_occ_img_is_valid, list)
                        else gt_occ_img_is_valid
                    )
                    assert fv is not None, "Valid-GT requires frame-valid flags"
                    fv = fv[:, :self.occ_head.receptive_field + self.occ_head.n_future].bool()
                    assert fv.shape[1] == self.occ_head.receptive_field + self.occ_head.n_future
                    past_valid = fv[:, :self.occ_head.receptive_field].all(dim=1)
                    future_valid = fv[:, self.occ_head.receptive_field - 1:].clone()
                    future_valid[~past_valid] = False
                    assert future_valid.shape == gs.shape[:2]
                    univ2x_outs['gt_occ_frame_valid'] = future_valid.detach()
                except Exception:
                    univ2x_outs['gt_occ_binary'] = None
            else:
                univ2x_outs['gt_occ_binary'] = None

        return losses, univ2x_outs
    
    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}" : v*loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def forward_test(self,
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     rescale=False,
                     #  planning gt(for evaluation only)
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
                     w_label=True,
                     # Occ_gt (for evaluation only)
                     gt_segmentation=None,
                     gt_instance=None, 
                     gt_occ_img_is_valid=None,

                     #for coop
                     other_agent_results=None,
                     v2_4d_fusion_fn=None,
                     **kwargs
                    ):
        """Test function
        """
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # first frame
        if self.prev_frame_info['scene_token'] is None:
            img_metas[0][0]['can_bus'][:3] = 0
            img_metas[0][0]['can_bus'][-1] = 0
        # following frames
        else:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp,
                                                                                    other_agent_results=other_agent_results)

        # Upsample bev for tiny model        
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])
        
        bev_embed = result_track[0]["bev_embed"]

        if self.with_seg_head:
            result_seg, drivable_gt, drivable_pred =  self.seg_head.forward_test(
                bev_embed, 
                gt_lane_labels, 
                gt_lane_masks=gt_lane_masks, 
                img_metas=img_metas, 
                rescale=rescale, 
                w_label=w_label, 
                other_agent_results=other_agent_results
            )

        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(bev_embed, outs_track=result_track[0], outs_seg=result_seg[0])
            outs_motion['bev_pos'] = result_track[0]['bev_pos']

        outs_occ = dict()
        if self.with_occ_head:
            occ_no_query = outs_motion['track_query'].shape[1] == 0

            # ============================================================
            # G0/G1/G2 metadata
            # img_metas is already flattened above:
            # img_metas[0] -> current sample metadata dict
            # ============================================================
            if timestamp is None:
                g012_timestamp = float("nan")
            elif torch.is_tensor(timestamp):
                g012_timestamp = float(
                    timestamp.detach().cpu().reshape(-1)[0].item()
                )
            else:
                g012_timestamp = float(timestamp)

            g012_meta = {
                "scene_token": img_metas[0].get("scene_token", ""),
                "sample_idx": img_metas[0].get("sample_idx", ""),
                "timestamp": g012_timestamp,
            }

            outs_occ = self.occ_head.forward_test(
                bev_embed,
                outs_motion,
                no_query=occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
                w_label=w_label,
                other_agent_results=other_agent_results,
                g012_meta=g012_meta,
                v2_4d_fusion_fn=v2_4d_fusion_fn,
            )

            result[0]['occ'] = outs_occ



        # outs_occ = dict()
        # if self.with_occ_head:
        #     occ_no_query = outs_motion['track_query'].shape[1] == 0
        #     outs_occ = self.occ_head.forward_test(
        #         bev_embed,
        #         outs_motion,
        #         no_query = occ_no_query,
        #         gt_segmentation=gt_segmentation,
        #         gt_instance=gt_instance,
        #         gt_img_is_valid=gt_occ_img_is_valid,
        #         w_label=w_label,
        #         other_agent_results=other_agent_results
        #     )
        #     result[0]['occ'] = outs_occ
        
        if self.with_planning_head:
            result_planning = self.planning_head.forward_test(bev_embed, outs_motion, outs_occ, command, drivable_pred)
            if w_label:
                planning_gt=dict(
                    segmentation=gt_segmentation,
                    sdc_planning=sdc_planning,
                    sdc_planning_mask=sdc_planning_mask,
                    command=command,
                    drivable_gt=drivable_gt, # for out of boundary evaluation
                )
                result[0]['planning'] = dict(
                    planning_gt=planning_gt,
                    result_planning=result_planning,
                )
            else:
                result[0]['planning'] = dict(
                    result_planning=result_planning,
                )

            # ============================================================
            # V2-2B Planning Counterfactual Cache Export
            # Enabled only when STCV_PLAN_CACHE_DIR is set.
            #
            # IMPORTANT:
            #   Run with use_col_optim=False.
            #   Then result_planning['sdc_traj'] is T_raw, i.e. the
            #   trajectory before collision/drivable post-processing.
            # ============================================================
            import os as _os
            import numpy as _np

            _plan_cache_dir = _os.environ.get("STCV_PLAN_CACHE_DIR", "")

            if _plan_cache_dir and w_label:
                _os.makedirs(_plan_cache_dir, exist_ok=True)

                if not hasattr(self, "_stcv_plan_export_idx"):
                    self._stcv_plan_export_idx = 0

                _export_idx = int(self._stcv_plan_export_idx)
                self._stcv_plan_export_idx += 1

                def _to_numpy_v22b(x):
                    # Unwrap common singleton wrappers used by the
                    # UniV2X/MMCV data pipeline.
                    while isinstance(x, (list, tuple)) and len(x) == 1:
                        x = x[0]

                    if torch.is_tensor(x):
                        return x.detach().cpu().numpy()

                    return _np.asarray(x)

                _scene_token = img_metas[0].get("scene_token", "")
                _sample_idx = img_metas[0].get("sample_idx", "")

                if timestamp is None:
                    _timestamp = _np.nan
                elif torch.is_tensor(timestamp):
                    _timestamp = float(
                        timestamp.detach().cpu().reshape(-1)[0].item()
                    )
                else:
                    _timestamp = float(timestamp)

                _out_path = _os.path.join(
                    _plan_cache_dir,
                    f"sample_{_export_idx:05d}.npz"
                )

                _np.savez_compressed(
                    _out_path,

                    # -------------------------
                    # Fixed raw planning output
                    # -------------------------
                    T_raw=_to_numpy_v22b(
                        result_planning["sdc_traj"]
                    ),

                    # -------------------------
                    # Fixed drivable prediction
                    # -------------------------
                    drivable_pred=_to_numpy_v22b(
                        drivable_pred
                    ),

                    # -------------------------
                    # Official planning GT
                    # Keep raw structures here;
                    # offline evaluator will reproduce
                    # the exact slicing in apis/test.py.
                    # -------------------------
                    # Exactly reproduce the slicing used in
                    # projects/mmdet3d_plugin/univ2x/apis/test.py
                    sdc_planning_eval=_to_numpy_v22b(
                        sdc_planning[0][
                            0, :, :self.planning_head.planning_steps, :2
                        ]
                    ),
                    sdc_planning_mask_eval=_to_numpy_v22b(
                        sdc_planning_mask[0][
                            0, :, :self.planning_head.planning_steps, :2
                        ]
                    ),
                    gt_segmentation_eval=_to_numpy_v22b(
                        gt_segmentation[0][
                            :, 1:self.planning_head.planning_steps + 1
                        ]
                    ),
                    drivable_gt=_to_numpy_v22b(
                        drivable_gt
                    ),
                    command=_to_numpy_v22b(
                        command
                    ),

                    # -------------------------
                    # Alignment metadata
                    # -------------------------
                    export_idx=_np.asarray(
                        _export_idx,
                        dtype=_np.int64
                    ),
                    scene_token=_np.asarray(
                        str(_scene_token)
                    ),
                    sample_idx=_np.asarray(
                        str(_sample_idx)
                    ),
                    timestamp=_np.asarray(
                        _timestamp,
                        dtype=_np.float64
                    ),

                    # -------------------------
                    # Runtime audit metadata
                    # -------------------------
                    planning_steps=_np.asarray(
                        int(self.planning_head.planning_steps),
                        dtype=_np.int64
                    ),
                    use_col_optim=_np.asarray(
                        int(self.planning_head.use_col_optim),
                        dtype=_np.int64
                    ),
                )

                if _export_idx < 3:
                    print(
                        "[V2-2B CACHE]",
                        _out_path,
                        "T_raw",
                        tuple(result_planning["sdc_traj"].shape),
                        "use_col_optim",
                        self.planning_head.use_col_optim,
                    )

        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=['pts_bbox', 'args_tuple'])
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]['occ'] = pop_elem_in_result(result[0]['occ'],  \
                pop_list=['seg_out_mask', 'flow_out', 'future_states_occ', 'pred_ins_masks', 'pred_raw_occ', 'pred_ins_logits', 'pred_ins_sigmoid'])

        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])

        return result
    
    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (dict): Input points and the information of the sample.
            result (dict): Prediction results.
            out_dir (str): Output directory of visualization result.
        """

        from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
        import mmcv
        from os import path as osp
        from mmcv.parallel import DataContainer as DC
        
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            # inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            # pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]
            inds = result[batch_id]['scores_3d'].numpy() > 0.1
            pred_bboxes = result[batch_id]['boxes_3d'][inds]

            if len(pred_bboxes) <= 0:
                continue

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for convertion!')

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name)


def pop_elem_in_result(task_result:dict, pop_list:list=None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result
