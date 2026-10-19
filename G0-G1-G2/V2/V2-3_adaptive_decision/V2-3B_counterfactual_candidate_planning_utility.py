#!/usr/bin/env python3

import argparse
import csv
import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from projects.mmdet3d_plugin.univ2x.dense_heads.planning_head import (
    PlanningHeadSingleMode,
)
from projects.mmdet3d_plugin.univ2x.dense_heads.planning_head_plugin import (
    PlanningMetric,
)


# =============================================================================
# Paths
# =============================================================================

ROOT = Path("/root/autodl-tmp/UniV2X")

V22C_SCRIPT = (
    ROOT
    / "G0-G1-G2/V2/V2-2_planning_task_consistency/"
      "V2-2C_counterfactual_planning.py"
)

V23A_NPZ = (
    ROOT
    / "G0-G1-G2/V2/V2-3_adaptive_decision/"
      "V2-3A_candidate_decision_risk_arrays.npz"
)

OUT_DIR = (
    ROOT
    / "G0-G1-G2/V2/V2-3_adaptive_decision"
)


# =============================================================================
# Reuse audited V2-2C helpers / paths
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "v22c",
    V22C_SCRIPT,
)

m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


# =============================================================================
# Minimal official planning post-process object
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


# =============================================================================
# Normalize cached arrays
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


# =============================================================================
# Official planning post-processing
# =============================================================================

@torch.no_grad()
def postprocess(
    planner,
    raw_traj,
    occ,
    drivable_pred,
):

    # raw_traj: [1,10,2]
    # occ:      [5,200,200]

    raw = raw_traj.clone()

    occ_input = (
        occ.bool()
        .unsqueeze(0)
    )

    collision_traj = (
        planner.collision_optimization(
            raw.clone(),
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


# =============================================================================
# One-sample official PlanningMetric
# =============================================================================

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

    # IMPORTANT:
    # PlanningMetric modifies x coordinates internally.
    # Always pass clones.
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
        result[key + "_mean"] = float(
            arr.mean()
        )

        result[key + "_sum"] = float(
            arr.sum()
        )

    return result


# =============================================================================
# Candidate insertion
# =============================================================================

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
            "Smoke-test only the first N near-5m candidates. "
            "Default: evaluate all 306."
        ),
    )

    args = parser.parse_args()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Exact Val split
    # -------------------------------------------------------------------------

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

    val_files = []

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
            and scene_token in val_scenes
        ):
            val_files.append(p)

    assert len(val_files) == 96

    # -------------------------------------------------------------------------
    # V2-3A candidate arrays
    # -------------------------------------------------------------------------

    with np.load(
        V23A_NPZ,
        allow_pickle=False,
    ) as d:

        candidate_idx = (
            d["candidate_idx"]
            .astype(np.int64)
        )

        sample_order = (
            d["sample_order"]
            .astype(np.int64)
        )

        horizon = (
            d["horizon"]
            .astype(np.int64)
        )

        patch_r = (
            d["patch_r"]
            .astype(np.int64)
        )

        patch_c = (
            d["patch_c"]
            .astype(np.int64)
        )

        near_5m = (
            d["near_5m"]
            .astype(bool)
        )

        min_plan_dist = (
            d["min_plan_dist_m"]
            .astype(np.float64)
        )

        v2_mag = (
            d["v2_mag"]
            .astype(np.float64)
        )

        v2_accept = (
            d["v2_accept"]
            .astype(bool)
        )

        sign_prob = (
            d["sign_prob"]
            .astype(np.float64)
        )

        sign_margin = (
            d["sign_margin"]
            .astype(np.float64)
        )

    assert len(candidate_idx) == 18555
    assert int(near_5m.sum()) == 306
    assert int(v2_accept.sum()) == 6156

    near_ids = np.where(
        near_5m
    )[0]

    assert len(near_ids) == 306

    if args.limit is not None:

        near_ids = near_ids[
            :args.limit
        ]

    # -------------------------------------------------------------------------
    # Planner
    # -------------------------------------------------------------------------

    planner = build_planner()

    # -------------------------------------------------------------------------
    # Cache per-sample data and Ego result.
    #
    # Only 24/96 samples contain near5 candidates.
    # -------------------------------------------------------------------------

    sample_cache = {}

    records = []

    print("=" * 100)
    print("V2-3B COUNTERFACTUAL CANDIDATE PLANNING UTILITY AUDIT")
    print("=" * 100)

    print(
        "Val samples         :",
        96,
    )

    print(
        "All candidates      :",
        18555,
    )

    print(
        "Near-5m candidates  :",
        306,
    )

    print(
        "Evaluated candidates:",
        len(near_ids),
    )

    if args.limit is not None:

        print(
            "SMOKE LIMIT         :",
            args.limit,
        )

    print(
        "Counterfactual       : Ego vs Ego + ONE candidate"
    )

    print(
        "Planning postprocess : official collision + drivable optimization"
    )

    print()

    # =========================================================================
    # Candidate loop
    # =========================================================================

    for order, global_idx in enumerate(
        near_ids,
        start=1,
    ):

        sorder = int(
            sample_order[
                global_idx
            ]
        )

        assert 0 <= sorder < 96

        p_plan = val_files[
            sorder
        ]

        p_g0 = (
            m.G0_CACHE
            / p_plan.name
        )

        # ---------------------------------------------------------------------
        # Load sample once
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
                    plan["drivable_pred"],
                    2,
                )

                drivable_gt_np = norm(
                    plan["drivable_gt"],
                    2,
                )

                plan_sample_idx = str(
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
                plan_sample_idx
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

            # -------------------------------------------------------------
            # Ego baseline post-processing once per sample
            # -------------------------------------------------------------

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
                    plan_sample_idx,

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
            horizon[
                global_idx
            ]
        )

        r = int(
            patch_r[
                global_idx
            ]
        )

        c = int(
            patch_c[
                global_idx
            ]
        )

        assert h > 0
        assert bool(
            near_5m[
                global_idx
            ]
        )

        # ---------------------------------------------------------------------
        # Ego + exactly this ONE candidate
        # ---------------------------------------------------------------------

        occ_cf = (
            add_single_candidate(
                d["ov"],
                d["oi"],
                h,
                r,
                c,
            )
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

        # ---------------------------------------------------------------------
        # Trajectory effects
        # ---------------------------------------------------------------------

        collision_delta = (
            cf_collision
            -
            d["ego_collision"]
        )

        final_delta = (
            cf_final
            -
            d["ego_final"]
        )

        collision_max_abs = float(
            collision_delta
            .abs()
            .max()
            .item()
        )

        final_max_abs = float(
            final_delta
            .abs()
            .max()
            .item()
        )

        collision_changed = (
            collision_max_abs
            > 1e-6
        )

        final_changed = (
            final_max_abs
            > 1e-6
        )

        # ---------------------------------------------------------------------
        # Metric deltas: CF - Ego
        #
        # Negative = improvement.
        # Positive = deterioration.
        # ---------------------------------------------------------------------

        delta = {}

        for key in [
            "L2",
            "obj_col",
            "obj_box_col",
            "obj_out",
            "obj_box_out",
        ]:

            delta[
                key + "_mean"
            ] = (
                cf_metric[
                    key + "_mean"
                ]
                -
                d["ego_metric"][
                    key + "_mean"
                ]
            )

            delta[
                key + "_sum"
            ] = (
                cf_metric[
                    key + "_sum"
                ]
                -
                d["ego_metric"][
                    key + "_sum"
                ]
            )

        # Do NOT use this as a final training target yet.
        # It is only a compact diagnostic of event metrics.
        delta_safety_sum = (
            delta["obj_col_sum"]
            +
            delta["obj_box_col_sum"]
            +
            delta["obj_out_sum"]
            +
            delta["obj_box_out_sum"]
        )

        rec = {
            "candidate_idx":
                int(global_idx),

            "sample_order":
                sorder,

            "cache_file":
                p_plan.name,

            "sample_idx":
                d["sample_idx"],

            "scene_token":
                d["scene_token"],

            "horizon":
                h,

            "patch_r":
                r,

            "patch_c":
                c,

            "min_plan_dist_m":
                float(
                    min_plan_dist[
                        global_idx
                    ]
                ),

            "v2_mag":
                float(
                    v2_mag[
                        global_idx
                    ]
                ),

            "v2_accept":
                int(
                    v2_accept[
                        global_idx
                    ]
                ),

            "sign_prob":
                float(
                    sign_prob[
                        global_idx
                    ]
                ),

            "sign_margin":
                float(
                    sign_margin[
                        global_idx
                    ]
                ),

            "collision_changed":
                int(
                    collision_changed
                ),

            "final_changed":
                int(
                    final_changed
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
                    delta[
                        "L2_mean"
                    ]
                ),

            "delta_obj_col_mean":
                float(
                    delta[
                        "obj_col_mean"
                    ]
                ),

            "delta_obj_box_col_mean":
                float(
                    delta[
                        "obj_box_col_mean"
                    ]
                ),

            "delta_obj_out_mean":
                float(
                    delta[
                        "obj_out_mean"
                    ]
                ),

            "delta_obj_box_out_mean":
                float(
                    delta[
                        "obj_box_out_mean"
                    ]
                ),

            "delta_obj_col_sum":
                float(
                    delta[
                        "obj_col_sum"
                    ]
                ),

            "delta_obj_box_col_sum":
                float(
                    delta[
                        "obj_box_col_sum"
                    ]
                ),

            "delta_obj_out_sum":
                float(
                    delta[
                        "obj_out_sum"
                    ]
                ),

            "delta_obj_box_out_sum":
                float(
                    delta[
                        "obj_box_out_sum"
                    ]
                ),

            "delta_safety_sum":
                float(
                    delta_safety_sum
                ),
        }

        records.append(
            rec
        )

        if (
            order <= 5
            or order % 25 == 0
            or order == len(near_ids)
        ):

            print(
                f"[{order:03d}/"
                f"{len(near_ids):03d}] "
                f"cand={global_idx:05d} "
                f"sample={d['sample_idx']} "
                f"h={h} "
                f"d={rec['min_plan_dist_m']:.3f}m "
                f"V2={rec['v2_accept']} "
                f"traj={rec['final_changed']} "
                f"dL2={rec['delta_L2_mean']:+.8f} "
                f"dSafety={rec['delta_safety_sum']:+.1f}"
            )

    # =========================================================================
    # Summary
    # =========================================================================

    eps = 1e-6

    dL2 = np.asarray(
        [
            x["delta_L2_mean"]
            for x in records
        ],
        dtype=np.float64,
    )

    dSafety = np.asarray(
        [
            x["delta_safety_sum"]
            for x in records
        ],
        dtype=np.float64,
    )

    changed = np.asarray(
        [
            x["final_changed"]
            for x in records
        ],
        dtype=bool,
    )

    selected = np.asarray(
        [
            x["v2_accept"]
            for x in records
        ],
        dtype=bool,
    )

    l2_better = (
        dL2 < -eps
    )

    l2_worse = (
        dL2 > eps
    )

    l2_same = (
        np.abs(dL2) <= eps
    )

    safety_better = (
        dSafety < -eps
    )

    safety_worse = (
        dSafety > eps
    )

    safety_same = (
        np.abs(dSafety) <= eps
    )

    # -------------------------------------------------------------------------
    # Descriptive categories only.
    # No scalar planning-utility label is frozen here.
    # -------------------------------------------------------------------------

    summary = {
        "protocol": {
            "split":
                "Val only",

            "all_candidates":
                18555,

            "near5_candidates_total":
                306,

            "evaluated_candidates":
                int(
                    len(records)
                ),

            "counterfactual":
                "Ego vs Ego + one near5 candidate",

            "planning_postprocess":
                "official collision_optimization + drivable_optimization",

            "metric_delta_definition":
                "counterfactual - Ego; negative is better",

            "test_used":
                False,
        },

        "effect": {
            "trajectory_changed":
                int(
                    changed.sum()
                ),

            "trajectory_changed_rate":
                float(
                    changed.mean()
                ),

            "trajectory_unchanged":
                int(
                    (~changed).sum()
                ),

            "l2_improved":
                int(
                    l2_better.sum()
                ),

            "l2_worsened":
                int(
                    l2_worse.sum()
                ),

            "l2_unchanged":
                int(
                    l2_same.sum()
                ),

            "safety_improved":
                int(
                    safety_better.sum()
                ),

            "safety_worsened":
                int(
                    safety_worse.sum()
                ),

            "safety_unchanged":
                int(
                    safety_same.sum()
                ),
        },

        "v2_current_decision": {
            "near5_accepted":
                int(
                    selected.sum()
                ),

            "near5_rejected":
                int(
                    (~selected).sum()
                ),

            "accepted_and_traj_changed":
                int(
                    (
                        selected
                        & changed
                    ).sum()
                ),

            "accepted_and_l2_improved":
                int(
                    (
                        selected
                        & l2_better
                    ).sum()
                ),

            "accepted_and_l2_worsened":
                int(
                    (
                        selected
                        & l2_worse
                    ).sum()
                ),

            "accepted_and_safety_improved":
                int(
                    (
                        selected
                        & safety_better
                    ).sum()
                ),

            "accepted_and_safety_worsened":
                int(
                    (
                        selected
                        & safety_worse
                    ).sum()
                ),

            "rejected_but_l2_improved":
                int(
                    (
                        (~selected)
                        & l2_better
                    ).sum()
                ),

            "rejected_but_safety_improved":
                int(
                    (
                        (~selected)
                        & safety_better
                    ).sum()
                ),
        },

        "delta_L2": {
            "mean":
                float(
                    dL2.mean()
                ),

            "median":
                float(
                    np.median(
                        dL2
                    )
                ),

            "min_best":
                float(
                    dL2.min()
                ),

            "max_worst":
                float(
                    dL2.max()
                ),

            "p10":
                float(
                    np.percentile(
                        dL2,
                        10,
                    )
                ),

            "p90":
                float(
                    np.percentile(
                        dL2,
                        90,
                    )
                ),
        },

        "delta_safety_sum": {
            "mean":
                float(
                    dSafety.mean()
                ),

            "min_best":
                float(
                    dSafety.min()
                ),

            "max_worst":
                float(
                    dSafety.max()
                ),
        },
    }

    # =========================================================================
    # Print
    # =========================================================================

    print()
    print("=" * 100)
    print("COUNTERFACTUAL PLANNING EFFECT")
    print("=" * 100)

    print(
        "Evaluated candidates :",
        len(records),
    )

    print(
        "Trajectory changed   :",
        int(
            changed.sum()
        ),
        f"({changed.mean():.6%})",
    )

    print()
    print(
        "L2 improved          :",
        int(
            l2_better.sum()
        ),
    )

    print(
        "L2 worsened          :",
        int(
            l2_worse.sum()
        ),
    )

    print(
        "L2 unchanged         :",
        int(
            l2_same.sum()
        ),
    )

    print()
    print(
        "Safety improved      :",
        int(
            safety_better.sum()
        ),
    )

    print(
        "Safety worsened      :",
        int(
            safety_worse.sum()
        ),
    )

    print(
        "Safety unchanged     :",
        int(
            safety_same.sum()
        ),
    )

    print()
    print("=" * 100)
    print("CURRENT V2 DECISION VS PLANNING UTILITY")
    print("=" * 100)

    cur = summary[
        "v2_current_decision"
    ]

    print(
        "Near5 V2 accepted        :",
        cur[
            "near5_accepted"
        ],
    )

    print(
        "Near5 V2 rejected        :",
        cur[
            "near5_rejected"
        ],
    )

    print(
        "Accepted & traj changed  :",
        cur[
            "accepted_and_traj_changed"
        ],
    )

    print(
        "Accepted & L2 improved   :",
        cur[
            "accepted_and_l2_improved"
        ],
    )

    print(
        "Accepted & L2 worsened   :",
        cur[
            "accepted_and_l2_worsened"
        ],
    )

    print(
        "Accepted & safety improve:",
        cur[
            "accepted_and_safety_improved"
        ],
    )

    print(
        "Accepted & safety worsen :",
        cur[
            "accepted_and_safety_worsened"
        ],
    )

    print(
        "Rejected but L2 improved :",
        cur[
            "rejected_but_l2_improved"
        ],
    )

    print(
        "Rejected but safety impr.:",
        cur[
            "rejected_but_safety_improved"
        ],
    )

    print()
    print("=" * 100)
    print("DELTA L2 DISTRIBUTION")
    print("=" * 100)

    for key, value in summary[
        "delta_L2"
    ].items():

        print(
            f"{key:12s}: "
            f"{value:+.10f}"
        )

    # =========================================================================
    # Save
    # =========================================================================

    suffix = (
        f"_limit_{args.limit}"
        if args.limit is not None
        else ""
    )

    csv_path = (
        OUT_DIR
        / (
            "V2-3B_counterfactual_candidate_planning_utility"
            f"{suffix}.csv"
        )
    )

    json_path = (
        OUT_DIR
        / (
            "V2-3B_counterfactual_candidate_planning_utility"
            f"{suffix}.json"
        )
    )

    with open(
        csv_path,
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
        json_path,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)

    print(
        "CSV :",
        csv_path,
    )

    print(
        "JSON:",
        json_path,
    )

    if args.limit is None:

        assert len(records) == 306

        print()
        print(
            "[V2-3B FULL AUDIT] COMPLETE"
        )

    else:

        print()
        print(
            "[V2-3B SMOKE] COMPLETE"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
