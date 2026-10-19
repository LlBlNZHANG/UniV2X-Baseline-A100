#!/usr/bin/env python3

import argparse
import csv
import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Repository path
# =============================================================================

ROOT = Path("/root/autodl-tmp/UniV2X")

# Make "projects.*" importable even when this script is launched directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from projects.mmdet3d_plugin.univ2x.dense_heads.planning_head import (
    PlanningHeadSingleMode,
)
from projects.mmdet3d_plugin.univ2x.dense_heads.planning_head_plugin import (
    PlanningMetric,
)


# =============================================================================
# Paths
# =============================================================================

V22C_SCRIPT = (
    ROOT
    / "G0-G1-G2/V2/V2-2_planning_task_consistency/"
      "V2-2C_counterfactual_planning.py"
)

OUT_DIR = (
    ROOT
    / "G0-G1-G2/V2/V2-3_adaptive_decision"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_OUT = (
    OUT_DIR
    / "V2-3C_train_planning_utility_labels.csv"
)

NPZ_OUT = (
    OUT_DIR
    / "V2-3C_train_planning_utility_dataset.npz"
)

JSON_OUT = (
    OUT_DIR
    / "V2-3C_train_planning_utility_summary.json"
)


# =============================================================================
# Reuse already audited V2-2C implementation
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "v22c",
    V22C_SCRIPT,
)

m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


# =============================================================================
# Constants
# =============================================================================

PLANNING_STEPS_BY_OCC_HORIZON = {
    0: [],
    1: [0],
    2: [1],
    3: [2],
    4: [3, 4, 5, 6, 7, 8, 9],
}

GRID_CENTER = 100

# IMPORTANT:
# Reuse the official planning implementation convention,
# not nominal Occupancy 0.512 m resolution.
PLANNING_GRID_RESOLUTION = 0.5
PLANNING_CELL_OFFSET = 0.25

OFFICIAL_OCC_FILTER_RANGE = 5.0

UTILITY_EPS = 1e-6


# =============================================================================
# Helpers
# =============================================================================

def norm(x, ndim):

    x = np.asarray(x)

    while (
        x.ndim > ndim
        and x.shape[0] == 1
    ):
        x = x[0]

    assert x.ndim == ndim, (
        f"Expected ndim={ndim}, "
        f"got shape={x.shape}"
    )

    return x


def grid_cells_to_planning_xy(
    global_rows,
    global_cols,
):

    global_rows = np.asarray(
        global_rows,
        dtype=np.float32,
    )

    global_cols = np.asarray(
        global_cols,
        dtype=np.float32,
    )

    x = (
        (
            global_cols
            - GRID_CENTER
        )
        * PLANNING_GRID_RESOLUTION
        + PLANNING_CELL_OFFSET
    )

    y = (
        (
            global_rows
            - GRID_CENTER
        )
        * PLANNING_GRID_RESOLUTION
        + PLANNING_CELL_OFFSET
    )

    return np.stack(
        [x, y],
        axis=1,
    )


def candidate_planning_relevance(
    h,
    r,
    c,
    ov,
    oi,
    warp,
    raw_traj,
):
    """
    Reproduce V2-3A planning relevance definition.

    Candidate patch:
        4x4.

    ADD:
        (~Ov) & Oi & warp_valid

    Returns:
        n_add_cells
        min_plan_distance_m
        n_add_cells_within_5m
        planning_used
    """

    y0 = 4 * int(r)
    y1 = y0 + 4

    x0 = 4 * int(c)
    x1 = x0 + 4

    add_patch = (
        (~ov[
            int(h),
            y0:y1,
            x0:x1,
        ])
        &
        oi[
            int(h),
            y0:y1,
            x0:x1,
        ]
        &
        warp[
            y0:y1,
            x0:x1,
        ]
    )

    local_rc = np.argwhere(
        add_patch
    )

    n_add_cells = int(
        len(local_rc)
    )

    assert n_add_cells > 0

    planning_steps = (
        PLANNING_STEPS_BY_OCC_HORIZON[
            int(h)
        ]
    )

    if len(planning_steps) == 0:

        return (
            n_add_cells,
            float("inf"),
            0,
            False,
        )

    global_rows = (
        local_rc[:, 0] + y0
    )

    global_cols = (
        local_rc[:, 1] + x0
    )

    cell_xy = (
        grid_cells_to_planning_xy(
            global_rows,
            global_cols,
        )
    )

    traj_xy = raw_traj[
        planning_steps,
        :2,
    ].astype(
        np.float32
    )

    delta = (
        cell_xy[:, None, :]
        -
        traj_xy[None, :, :]
    )

    distance = np.sqrt(
        np.sum(
            delta ** 2,
            axis=-1,
        )
    )

    min_dist = float(
        np.min(distance)
    )

    cell_min_dist = np.min(
        distance,
        axis=1,
    )

    n_within_5m = int(
        (
            cell_min_dist
            <
            OFFICIAL_OCC_FILTER_RANGE
        ).sum()
    )

    return (
        n_add_cells,
        min_dist,
        n_within_5m,
        True,
    )


# =============================================================================
# Official planner replay
# =============================================================================

def build_planner():

    planner = PlanningHeadSingleMode.__new__(
        PlanningHeadSingleMode
    )

    nn.Module.__init__(planner)

    planner.planning_steps = 10
    planner.occ_n_future_only_occ = 4

    planner.bev_h = 200
    planner.bev_w = 200

    planner.occ_filter_range = 5.0
    planner.sigma = 1.0
    planner.alpha_collision = 5.0

    planner.eval()

    return planner


@torch.no_grad()
def postprocess(
    planner,
    raw_traj,
    occ,
    drivable_pred,
):

    occ_input = (
        occ.bool()
        .unsqueeze(0)
    )

    collision_traj = (
        planner.collision_optimization(
            raw_traj.clone(),
            occ_input,
        )
    )

    final_traj = (
        planner.drivable_optimization(
            collision_traj.clone(),
            drivable_pred.clone(),
        )
    )

    return (
        collision_traj,
        final_traj,
    )


@torch.no_grad()
def planning_metric_one(
    pred_traj,
    gt_traj,
    gt_mask,
    gt_seg,
    drivable_gt,
):

    metric = PlanningMetric(
        n_future=10
    ).cpu()

    metric.update(
        pred_traj.clone(),
        gt_traj.clone(),
        gt_mask.clone(),
        gt_seg.clone(),
        drivable_gt.clone(),
    )

    out = metric.compute()

    result = {}

    for key in [
        "L2",
        "obj_col",
        "obj_box_col",
        "obj_out",
        "obj_box_out",
    ]:

        arr = (
            out[key]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        assert arr.shape == (10,)

        result[key] = arr

        result[
            key + "_mean"
        ] = float(
            arr.mean()
        )

        result[
            key + "_sum"
        ] = float(
            arr.sum()
        )

    return result


def add_single_candidate(
    ov,
    oi,
    h,
    r,
    c,
):

    out = ov.clone()

    y0 = 4 * int(r)
    y1 = y0 + 4

    x0 = 4 * int(c)
    x1 = x0 + 4

    out[
        int(h),
        y0:y1,
        x0:x1,
    ] = (
        out[
            int(h),
            y0:y1,
            x0:x1,
        ]
        |
        oi[
            int(h),
            y0:y1,
            x0:x1,
        ]
    )

    return out


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Smoke-test first N planning-sensitive "
            "Train candidates."
        ),
    )

    args = parser.parse_args()

    # =========================================================================
    # Exact split
    # =========================================================================

    (
        valid_sample_ids,
        train_scenes,
        val_scenes,
        test_scenes,
    ) = m.build_exact_val_split()

    plan_files_all = sorted(
        m.PLAN_CACHE.glob(
            "sample_*.npz"
        )
    )

    train_files = []

    for p in plan_files_all:

        with np.load(
            p,
            allow_pickle=False,
        ) as d:

            sample_idx = str(
                d["sample_idx"].item()
            )

            scene_token = str(
                d["scene_token"].item()
            )

        if (
            sample_idx in valid_sample_ids
            and scene_token in train_scenes
        ):
            train_files.append(p)

    # Frozen split audit.
    assert len(train_files) == 361, (
        f"Expected 361 Train samples, "
        f"got {len(train_files)}"
    )

    print("=" * 100)
    print("V2-3C TRAIN PLANNING-UTILITY LABEL GENERATION")
    print("=" * 100)

    print(
        "Valid samples      :",
        len(valid_sample_ids),
    )

    print(
        "Train scenes       :",
        len(train_scenes),
    )

    print(
        "Val scenes         :",
        len(val_scenes),
    )

    print(
        "Test scenes        :",
        len(test_scenes),
    )

    print(
        "Train samples      :",
        len(train_files),
    )

    print(
        "Val/Test used      : NO"
    )

    print(
        "Planning filter    : < 5.0 m"
    )

    print(
        "Planning label     : U_plan = -Delta L2"
    )

    # =========================================================================
    # Frozen V1 candidate extractor
    # =========================================================================

    v1 = m.STCVOcc(
        checkpoint_path=str(
            m.V1_CKPT
        ),
        threshold=0.70,
    )

    v1.eval()

    # =========================================================================
    # PASS 1
    #
    # Reconstruct ALL Train candidates exactly, but keep only candidates
    # that can actually enter official planning (< 5m).
    #
    # Also save the corresponding exact 127-D features.
    # =========================================================================

    manifest = []
    feature_chunks = []

    all_candidate_count = 0

    print()
    print("=" * 100)
    print("PASS 1: TRAIN CANDIDATE + PLANNING-RELEVANCE RECONSTRUCTION")
    print("=" * 100)

    for sample_order, p_plan in enumerate(
        train_files
    ):

        p_g0 = (
            m.G0_CACHE
            / p_plan.name
        )

        assert p_g0.exists()

        with np.load(
            p_plan,
            allow_pickle=False,
        ) as plan:

            raw_traj = norm(
                plan["T_raw"],
                3,
            )[0, :, :2].astype(
                np.float32
            )

            plan_sample_idx = str(
                plan["sample_idx"].item()
            )

            scene_token = str(
                plan["scene_token"].item()
            )

        with np.load(
            p_g0,
            allow_pickle=False,
        ) as g0:

            g0_sample_idx = str(
                g0["sample_idx"].item()
            )

            assert (
                plan_sample_idx
                ==
                g0_sample_idx
            )

            pv = torch.from_numpy(
                norm(
                    g0["Pv"],
                    3,
                )
            ).float()

            pi = torch.from_numpy(
                norm(
                    g0["Pi_aligned"],
                    3,
                )
            ).float()

            ov = torch.from_numpy(
                norm(
                    g0["Ov"],
                    3,
                )
            ).bool()

            oi = torch.from_numpy(
                norm(
                    g0["Oi"],
                    3,
                )
            ).bool()

            warp = torch.from_numpy(
                norm(
                    g0[
                        "warp_valid_mask"
                    ],
                    2,
                )
            ).bool()

        (
            candidate_indices,
            features,
        ) = (
            v1.extract_candidates_and_features(
                pv,
                pi,
                ov,
                oi,
                warp,
            )
        )

        n_candidates = len(
            candidate_indices
        )

        assert features.shape == (
            n_candidates,
            127,
        )

        ov_np = ov.numpy()
        oi_np = oi.numpy()
        warp_np = warp.numpy()

        for local_idx, (
            h,
            r,
            c,
        ) in enumerate(
            candidate_indices
        ):

            global_candidate_idx = (
                all_candidate_count
                +
                local_idx
            )

            (
                n_add_cells,
                min_plan_dist,
                n_cells_within_5m,
                planning_used,
            ) = (
                candidate_planning_relevance(
                    int(h),
                    int(r),
                    int(c),
                    ov_np,
                    oi_np,
                    warp_np,
                    raw_traj,
                )
            )

            near5 = (
                planning_used
                and
                min_plan_dist
                <
                OFFICIAL_OCC_FILTER_RANGE
            )

            if not near5:
                continue

            manifest.append({
                "train_candidate_idx":
                    int(
                        global_candidate_idx
                    ),

                "sample_order":
                    int(
                        sample_order
                    ),

                "cache_file":
                    p_plan.name,

                "sample_idx":
                    plan_sample_idx,

                "scene_token":
                    scene_token,

                "horizon":
                    int(h),

                "patch_r":
                    int(r),

                "patch_c":
                    int(c),

                "n_add_cells":
                    int(
                        n_add_cells
                    ),

                "add_cells_within_5m":
                    int(
                        n_cells_within_5m
                    ),

                "min_plan_dist_m":
                    float(
                        min_plan_dist
                    ),
            })

            feature_chunks.append(
                features[
                    local_idx
                ].cpu()
            )

        all_candidate_count += (
            n_candidates
        )

        if (
            sample_order < 3
            or (sample_order + 1) % 25 == 0
            or sample_order + 1
            == len(train_files)
        ):

            print(
                f"[{sample_order+1:03d}/"
                f"{len(train_files):03d}] "
                f"{p_plan.name} "
                f"sample={plan_sample_idx} "
                f"cand={n_candidates} "
                f"near5_total={len(manifest)}"
            )

    # Exact historical Train candidate count from V2-1.
    assert all_candidate_count == 86143, (
        f"Train candidate mismatch: "
        f"{all_candidate_count} != 86143"
    )

    assert len(manifest) > 0

    features_127 = torch.stack(
        feature_chunks,
        dim=0,
    )

    assert features_127.shape == (
        len(manifest),
        127,
    )

    print()
    print(
        "Train all candidates:",
        all_candidate_count,
    )

    print(
        "Train near5 candidates:",
        len(manifest),
    )

    print(
        "Train near5 rate      :",
        f"{len(manifest)/all_candidate_count:.6%}",
    )

    # =========================================================================
    # Frozen V2 predictions
    #
    # Exact canonical numerical path:
    # CUDA + batch=2048.
    # =========================================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "Exact V2 execution requires CUDA."
        )

    device = torch.device(
        "cuda"
    )

    v2 = m.V2UtilityPredictor(
        dropout=0.1
    )

    ckpt = torch.load(
        m.V2_CKPT,
        map_location=device,
    )

    v2.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    v2 = v2.to(
        device
    )

    v2.eval()

    mag_chunks = []
    sign_chunks = []

    with torch.no_grad():

        for i in range(
            0,
            len(features_127),
            2048,
        ):

            x = features_127[
                i:i + 2048
            ].to(
                device,
                non_blocking=True,
            )

            (
                mag_pred,
                sign_logit,
            ) = v2(x)

            mag_chunks.append(
                mag_pred.detach().cpu()
            )

            sign_chunks.append(
                sign_logit.detach().cpu()
            )

    v2_mag = torch.cat(
        mag_chunks
    ).numpy()

    v2_sign_logit = torch.cat(
        sign_chunks
    ).numpy()

    v2_sign_prob = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -v2_sign_logit
            )
        )
    )

    v2_accept = (
        v2_mag > 0.0
    )

    for i in range(
        len(manifest)
    ):

        manifest[i][
            "v2_mag"
        ] = float(
            v2_mag[i]
        )

        manifest[i][
            "v2_sign_logit"
        ] = float(
            v2_sign_logit[i]
        )

        manifest[i][
            "v2_sign_prob"
        ] = float(
            v2_sign_prob[i]
        )

        manifest[i][
            "v2_sign_margin"
        ] = float(
            abs(
                v2_sign_prob[i]
                - 0.5
            )
        )

        manifest[i][
            "v2_accept"
        ] = int(
            v2_accept[i]
        )

    print(
        "Train near5 V2 accepted:",
        int(
            v2_accept.sum()
        ),
        "/",
        len(v2_accept),
    )

    # =========================================================================
    # Limit for smoke
    # =========================================================================

    eval_indices = np.arange(
        len(manifest),
        dtype=np.int64,
    )

    if args.limit is not None:

        eval_indices = eval_indices[
            :args.limit
        ]

        print(
            "SMOKE LIMIT         :",
            args.limit,
        )

    # =========================================================================
    # PASS 2
    #
    # Exact single-candidate planning counterfactual.
    # =========================================================================

    planner = build_planner()

    sample_cache = {}

    records = []

    print()
    print("=" * 100)
    print("PASS 2: COUNTERFACTUAL TRAIN LABEL GENERATION")
    print("=" * 100)

    for eval_order, manifest_idx in enumerate(
        eval_indices,
        start=1,
    ):

        meta = manifest[
            int(manifest_idx)
        ]

        sorder = int(
            meta[
                "sample_order"
            ]
        )

        p_plan = train_files[
            sorder
        ]

        p_g0 = (
            m.G0_CACHE
            / p_plan.name
        )

        # ---------------------------------------------------------------------
        # Cache one sample's Ego planning baseline
        # ---------------------------------------------------------------------

        if sorder not in sample_cache:

            with np.load(
                p_plan,
                allow_pickle=False,
            ) as plan:

                raw_np = norm(
                    plan["T_raw"],
                    3,
                )

                gt_traj_np = norm(
                    plan[
                        "sdc_planning_eval"
                    ],
                    3,
                )

                gt_mask_np = norm(
                    plan[
                        "sdc_planning_mask_eval"
                    ],
                    3,
                )

                gt_seg_np = norm(
                    plan[
                        "gt_segmentation_eval"
                    ],
                    4,
                )

                drivable_pred_np = norm(
                    plan[
                        "drivable_pred"
                    ],
                    2,
                )

                drivable_gt_np = norm(
                    plan[
                        "drivable_gt"
                    ],
                    2,
                )

                sample_idx = str(
                    plan[
                        "sample_idx"
                    ].item()
                )

                scene_token = str(
                    plan[
                        "scene_token"
                    ].item()
                )

            with np.load(
                p_g0,
                allow_pickle=False,
            ) as g0:

                ov_np = norm(
                    g0["Ov"],
                    3,
                )

                oi_np = norm(
                    g0["Oi"],
                    3,
                )

                g0_sample_idx = str(
                    g0[
                        "sample_idx"
                    ].item()
                )

            assert (
                sample_idx
                ==
                g0_sample_idx
            )

            raw = torch.from_numpy(
                raw_np
            ).float()

            gt_traj = torch.from_numpy(
                gt_traj_np
            ).float()

            gt_mask = torch.from_numpy(
                gt_mask_np
            ).float()

            gt_seg = torch.from_numpy(
                gt_seg_np
            )

            drivable_pred = (
                torch.from_numpy(
                    drivable_pred_np
                )
            )

            drivable_gt = (
                torch.from_numpy(
                    drivable_gt_np
                )
            )

            ov = torch.from_numpy(
                ov_np
            ).bool()

            oi = torch.from_numpy(
                oi_np
            ).bool()

            assert raw.shape == (
                1,
                10,
                2,
            )

            assert ov.shape == (
                5,
                200,
                200,
            )

            (
                ego_collision,
                ego_final,
            ) = postprocess(
                planner,
                raw,
                ov,
                drivable_pred,
            )

            ego_metric = (
                planning_metric_one(
                    ego_final,
                    gt_traj,
                    gt_mask,
                    gt_seg,
                    drivable_gt,
                )
            )

            sample_cache[
                sorder
            ] = {
                "sample_idx":
                    sample_idx,

                "scene_token":
                    scene_token,

                "raw":
                    raw,

                "gt_traj":
                    gt_traj,

                "gt_mask":
                    gt_mask,

                "gt_seg":
                    gt_seg,

                "drivable_pred":
                    drivable_pred,

                "drivable_gt":
                    drivable_gt,

                "ov":
                    ov,

                "oi":
                    oi,

                "ego_collision":
                    ego_collision,

                "ego_final":
                    ego_final,

                "ego_metric":
                    ego_metric,
            }

        d = sample_cache[
            sorder
        ]

        h = int(
            meta["horizon"]
        )

        r = int(
            meta["patch_r"]
        )

        c = int(
            meta["patch_c"]
        )

        assert h > 0
        assert (
            meta[
                "min_plan_dist_m"
            ]
            <
            5.0
        )

        # ---------------------------------------------------------------------
        # Counterfactual:
        # Ego + exactly ONE candidate
        # ---------------------------------------------------------------------

        occ_cf = add_single_candidate(
            d["ov"],
            d["oi"],
            h,
            r,
            c,
        )

        (
            cf_collision,
            cf_final,
        ) = postprocess(
            planner,
            d["raw"],
            occ_cf,
            d["drivable_pred"],
        )

        cf_metric = (
            planning_metric_one(
                cf_final,
                d["gt_traj"],
                d["gt_mask"],
                d["gt_seg"],
                d["drivable_gt"],
            )
        )

        collision_max_abs = float(
            (
                cf_collision
                -
                d["ego_collision"]
            )
            .abs()
            .max()
            .item()
        )

        final_max_abs = float(
            (
                cf_final
                -
                d["ego_final"]
            )
            .abs()
            .max()
            .item()
        )

        delta_L2 = (
            cf_metric[
                "L2_mean"
            ]
            -
            d[
                "ego_metric"
            ][
                "L2_mean"
            ]
        )

        plan_utility = (
            -delta_L2
        )

        if plan_utility > UTILITY_EPS:

            plan_label = 1

        elif plan_utility < -UTILITY_EPS:

            plan_label = -1

        else:

            plan_label = 0

        delta_obj_col = (
            cf_metric[
                "obj_col_sum"
            ]
            -
            d[
                "ego_metric"
            ][
                "obj_col_sum"
            ]
        )

        delta_obj_box_col = (
            cf_metric[
                "obj_box_col_sum"
            ]
            -
            d[
                "ego_metric"
            ][
                "obj_box_col_sum"
            ]
        )

        delta_obj_out = (
            cf_metric[
                "obj_out_sum"
            ]
            -
            d[
                "ego_metric"
            ][
                "obj_out_sum"
            ]
        )

        delta_obj_box_out = (
            cf_metric[
                "obj_box_out_sum"
            ]
            -
            d[
                "ego_metric"
            ][
                "obj_box_out_sum"
            ]
        )

        delta_safety_sum = (
            delta_obj_col
            +
            delta_obj_box_col
            +
            delta_obj_out
            +
            delta_obj_box_out
        )

        rec = dict(
            meta
        )

        rec.update({
            "collision_changed":
                int(
                    collision_max_abs
                    > 1e-6
                ),

            "final_changed":
                int(
                    final_max_abs
                    > 1e-6
                ),

            "collision_max_abs":
                collision_max_abs,

            "final_max_abs":
                final_max_abs,

            "ego_L2_mean":
                float(
                    d[
                        "ego_metric"
                    ][
                        "L2_mean"
                    ]
                ),

            "cf_L2_mean":
                float(
                    cf_metric[
                        "L2_mean"
                    ]
                ),

            "delta_L2_mean":
                float(
                    delta_L2
                ),

            "plan_utility":
                float(
                    plan_utility
                ),

            # +1 beneficial
            #  0 neutral
            # -1 harmful
            "plan_label":
                int(
                    plan_label
                ),

            "delta_obj_col_sum":
                float(
                    delta_obj_col
                ),

            "delta_obj_box_col_sum":
                float(
                    delta_obj_box_col
                ),

            "delta_obj_out_sum":
                float(
                    delta_obj_out
                ),

            "delta_obj_box_out_sum":
                float(
                    delta_obj_box_out
                ),

            "delta_safety_sum":
                float(
                    delta_safety_sum
                ),
        })

        records.append(
            rec
        )

        if (
            eval_order <= 5
            or eval_order % 25 == 0
            or eval_order
            == len(eval_indices)
        ):

            print(
                f"[{eval_order:04d}/"
                f"{len(eval_indices):04d}] "
                f"cand={meta['train_candidate_idx']:05d} "
                f"sample={meta['sample_idx']} "
                f"h={h} "
                f"d={meta['min_plan_dist_m']:.3f}m "
                f"V2={meta['v2_accept']} "
                f"traj={rec['final_changed']} "
                f"dL2={delta_L2:+.8f} "
                f"Uplan={plan_utility:+.8f} "
                f"label={plan_label:+d}"
            )

    # =========================================================================
    # Summary
    # =========================================================================

    plan_utility_arr = np.asarray(
        [
            x[
                "plan_utility"
            ]
            for x in records
        ],
        dtype=np.float64,
    )

    plan_label_arr = np.asarray(
        [
            x[
                "plan_label"
            ]
            for x in records
        ],
        dtype=np.int8,
    )

    traj_changed_arr = np.asarray(
        [
            x[
                "final_changed"
            ]
            for x in records
        ],
        dtype=bool,
    )

    v2_selected_arr = np.asarray(
        [
            x[
                "v2_accept"
            ]
            for x in records
        ],
        dtype=bool,
    )

    beneficial = (
        plan_label_arr == 1
    )

    harmful = (
        plan_label_arr == -1
    )

    neutral = (
        plan_label_arr == 0
    )

    summary = {
        "protocol": {
            "split":
                "Train only",

            "train_scenes":
                13,

            "train_samples":
                361,

            "all_train_candidates":
                int(
                    all_candidate_count
                ),

            "near5_train_candidates":
                int(
                    len(manifest)
                ),

            "evaluated_candidates":
                int(
                    len(records)
                ),

            "planning_utility":
                "-delta_L2_mean",

            "positive_definition":
                "plan_utility > 1e-6",

            "negative_definition":
                "plan_utility < -1e-6",

            "counterfactual":
                "Ego vs Ego + one candidate",

            "test_used":
                False,

            "val_used_for_labels":
                False,
        },

        "labels": {
            "beneficial":
                int(
                    beneficial.sum()
                ),

            "harmful":
                int(
                    harmful.sum()
                ),

            "neutral":
                int(
                    neutral.sum()
                ),

            "beneficial_rate":
                float(
                    beneficial.mean()
                ),

            "harmful_rate":
                float(
                    harmful.mean()
                ),

            "neutral_rate":
                float(
                    neutral.mean()
                ),
        },

        "trajectory": {
            "changed":
                int(
                    traj_changed_arr.sum()
                ),

            "changed_rate":
                float(
                    traj_changed_arr.mean()
                ),
        },

        "utility": {
            "mean":
                float(
                    plan_utility_arr.mean()
                ),

            "median":
                float(
                    np.median(
                        plan_utility_arr
                    )
                ),

            "p10":
                float(
                    np.percentile(
                        plan_utility_arr,
                        10,
                    )
                ),

            "p90":
                float(
                    np.percentile(
                        plan_utility_arr,
                        90,
                    )
                ),

            "min":
                float(
                    plan_utility_arr.min()
                ),

            "max":
                float(
                    plan_utility_arr.max()
                ),
        },

        "v2_alignment": {
            "v2_accept":
                int(
                    v2_selected_arr.sum()
                ),

            "v2_accept_beneficial":
                int(
                    (
                        v2_selected_arr
                        &
                        beneficial
                    ).sum()
                ),

            "v2_accept_harmful":
                int(
                    (
                        v2_selected_arr
                        &
                        harmful
                    ).sum()
                ),

            "v2_reject_beneficial":
                int(
                    (
                        (~v2_selected_arr)
                        &
                        beneficial
                    ).sum()
                ),

            "v2_reject_harmful":
                int(
                    (
                        (~v2_selected_arr)
                        &
                        harmful
                    ).sum()
                ),
        },
    }

    print()
    print("=" * 100)
    print("TRAIN PLANNING-UTILITY LABEL SUMMARY")
    print("=" * 100)

    print(
        "All Train candidates :",
        all_candidate_count,
    )

    print(
        "Near5 candidates     :",
        len(manifest),
    )

    print(
        "Evaluated            :",
        len(records),
    )

    print()

    print(
        "Beneficial U>0       :",
        int(
            beneficial.sum()
        ),
        f"({beneficial.mean():.6%})",
    )

    print(
        "Harmful U<0          :",
        int(
            harmful.sum()
        ),
        f"({harmful.mean():.6%})",
    )

    print(
        "Neutral              :",
        int(
            neutral.sum()
        ),
        f"({neutral.mean():.6%})",
    )

    print()

    print(
        "Trajectory changed   :",
        int(
            traj_changed_arr.sum()
        ),
        f"({traj_changed_arr.mean():.6%})",
    )

    print()

    print(
        "V2 accepted          :",
        int(
            v2_selected_arr.sum()
        ),
    )

    print(
        "V2 accept beneficial :",
        summary[
            "v2_alignment"
        ][
            "v2_accept_beneficial"
        ],
    )

    print(
        "V2 accept harmful    :",
        summary[
            "v2_alignment"
        ][
            "v2_accept_harmful"
        ],
    )

    print(
        "V2 reject beneficial :",
        summary[
            "v2_alignment"
        ][
            "v2_reject_beneficial"
        ],
    )

    print(
        "V2 reject harmful    :",
        summary[
            "v2_alignment"
        ][
            "v2_reject_harmful"
        ],
    )

    print()
    print(
        "Uplan mean           :",
        f"{plan_utility_arr.mean():+.10f}",
    )

    print(
        "Uplan median         :",
        f"{np.median(plan_utility_arr):+.10f}",
    )

    print(
        "Uplan min            :",
        f"{plan_utility_arr.min():+.10f}",
    )

    print(
        "Uplan max            :",
        f"{plan_utility_arr.max():+.10f}",
    )

    # =========================================================================
    # Save smoke separately.
    # Full run writes canonical training dataset.
    # =========================================================================

    if args.limit is not None:

        suffix = (
            f"_limit_{args.limit}"
        )

        csv_out = (
            OUT_DIR
            / (
                "V2-3C_train_planning_utility_labels"
                f"{suffix}.csv"
            )
        )

        json_out = (
            OUT_DIR
            / (
                "V2-3C_train_planning_utility_summary"
                f"{suffix}.json"
            )
        )

    else:

        csv_out = CSV_OUT
        json_out = JSON_OUT

    with open(
        csv_out,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                records[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            records
        )

    with open(
        json_out,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # Full run additionally writes exact arrays for V2-3D training.
    if args.limit is None:

        evaluated_manifest_indices = (
            np.arange(
                len(manifest),
                dtype=np.int64,
            )
        )

        assert len(records) == len(
            manifest
        )

        np.savez_compressed(
            NPZ_OUT,

            features_127=(
                features_127
                .numpy()
                .astype(
                    np.float32
                )
            ),

            train_candidate_idx=np.asarray(
                [
                    x[
                        "train_candidate_idx"
                    ]
                    for x in records
                ],
                dtype=np.int64,
            ),

            sample_order=np.asarray(
                [
                    x[
                        "sample_order"
                    ]
                    for x in records
                ],
                dtype=np.int64,
            ),

            horizon=np.asarray(
                [
                    x[
                        "horizon"
                    ]
                    for x in records
                ],
                dtype=np.int64,
            ),

            patch_r=np.asarray(
                [
                    x[
                        "patch_r"
                    ]
                    for x in records
                ],
                dtype=np.int64,
            ),

            patch_c=np.asarray(
                [
                    x[
                        "patch_c"
                    ]
                    for x in records
                ],
                dtype=np.int64,
            ),

            n_add_cells=np.asarray(
                [
                    x[
                        "n_add_cells"
                    ]
                    for x in records
                ],
                dtype=np.int64,
            ),

            add_cells_within_5m=np.asarray(
                [
                    x[
                        "add_cells_within_5m"
                    ]
                    for x in records
                ],
                dtype=np.int64,
            ),

            min_plan_dist_m=np.asarray(
                [
                    x[
                        "min_plan_dist_m"
                    ]
                    for x in records
                ],
                dtype=np.float32,
            ),

            v2_mag=np.asarray(
                [
                    x[
                        "v2_mag"
                    ]
                    for x in records
                ],
                dtype=np.float32,
            ),

            v2_sign_logit=np.asarray(
                [
                    x[
                        "v2_sign_logit"
                    ]
                    for x in records
                ],
                dtype=np.float32,
            ),

            v2_sign_prob=np.asarray(
                [
                    x[
                        "v2_sign_prob"
                    ]
                    for x in records
                ],
                dtype=np.float32,
            ),

            v2_sign_margin=np.asarray(
                [
                    x[
                        "v2_sign_margin"
                    ]
                    for x in records
                ],
                dtype=np.float32,
            ),

            v2_accept=np.asarray(
                [
                    x[
                        "v2_accept"
                    ]
                    for x in records
                ],
                dtype=np.uint8,
            ),

            delta_L2_mean=np.asarray(
                [
                    x[
                        "delta_L2_mean"
                    ]
                    for x in records
                ],
                dtype=np.float32,
            ),

            plan_utility=np.asarray(
                [
                    x[
                        "plan_utility"
                    ]
                    for x in records
                ],
                dtype=np.float32,
            ),

            plan_label=np.asarray(
                [
                    x[
                        "plan_label"
                    ]
                    for x in records
                ],
                dtype=np.int8,
            ),

            delta_safety_sum=np.asarray(
                [
                    x[
                        "delta_safety_sum"
                    ]
                    for x in records
                ],
                dtype=np.float32,
            ),
        )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)

    print(
        "CSV :",
        csv_out,
    )

    print(
        "JSON:",
        json_out,
    )

    if args.limit is None:

        print(
            "NPZ :",
            NPZ_OUT,
        )

        print()
        print(
            "[V2-3C TRAIN LABEL GENERATION] COMPLETE"
        )

    else:

        print()
        print(
            "[V2-3C SMOKE] COMPLETE"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
