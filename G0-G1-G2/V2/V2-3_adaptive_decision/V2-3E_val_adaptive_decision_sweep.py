#!/usr/bin/env python3

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path("/root/autodl-tmp/UniV2X")

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

BASE = (
    ROOT
    / "G0-G1-G2/V2/V2-3_adaptive_decision"
)

V22C_SCRIPT = (
    ROOT
    / "G0-G1-G2/V2/V2-2_planning_task_consistency/"
      "V2-2C_counterfactual_planning.py"
)

UNCERTAINTY_FILE = (
    BASE
    / "results/V2-3D-2_uncertainty/"
      "V2-3D-2_val_mc_dropout_predictions.npz"
)

VAL_DATASET_FILE = (
    BASE
    / "V2-3D_val_planning_utility_dataset.npz"
)

OUT_JSON = (
    BASE
    / "V2-3E_val_adaptive_decision_sweep.json"
)

OUT_NPZ = (
    BASE
    / "V2-3E_val_adaptive_decision_sweep.npz"
)


# =============================================================================
# Reuse exact frozen V2-2C definitions
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

BETAS = [
    0.0,
    0.25,
    0.5,
    1.0,
    1.5,
    2.0,
]

MODES = [
    "GateOnly",
    "RecoveryOnly",
    "Adaptive",
]

# Official UniV2X evaluator ranges.
OCC_RANGES = {
    "30m": (70, 130),
    "100m": (0, 200),
}


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
        f"Expected ndim={ndim}, got {x.shape}"
    )

    return x


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

        result[key + "_mean"] = float(
            arr.mean()
        )

    return result


def apply_candidates(
    ov,
    oi,
    candidate_indices,
    accept,
):

    out = ov.clone()

    assert len(candidate_indices) == len(
        accept
    )

    for i, (h, r, c) in enumerate(
        candidate_indices
    ):

        if not bool(
            accept[i]
        ):
            continue

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


def update_occ_counts(
    acc,
    pred,
    gt,
):

    pred = pred.bool().cpu().numpy()
    gt = gt.bool().cpu().numpy()

    assert pred.shape == (
        5,
        200,
        200,
    )

    assert gt.shape == pred.shape

    for name, (
        start,
        end,
    ) in OCC_RANGES.items():

        p = pred[
            :,
            start:end,
            start:end,
        ]

        g = gt[
            :,
            start:end,
            start:end,
        ]

        acc[name]["tp"] += int(
            np.logical_and(
                p,
                g,
            ).sum()
        )

        acc[name]["fp"] += int(
            np.logical_and(
                p,
                np.logical_not(g),
            ).sum()
        )

        acc[name]["fn"] += int(
            np.logical_and(
                np.logical_not(p),
                g,
            ).sum()
        )


def occ_iou_from_counts(
    d,
):

    denom = (
        d["tp"]
        +
        d["fp"]
        +
        d["fn"]
    )

    if denom == 0:
        return 0.0

    return (
        d["tp"]
        /
        denom
    )


def make_occ_accumulator():

    return {
        name: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for name in OCC_RANGES
    }


# =============================================================================
# Exact split
# =============================================================================

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
        and
        scene_token in val_scenes
    ):

        val_files.append(p)


assert len(val_files) == 96, (
    f"Expected 96 Val samples, "
    f"got {len(val_files)}"
)


print("=" * 110)
print("V2-3E VAL ADAPTIVE DECISION SWEEP")
print("=" * 110)

print(
    "Val scenes              :",
    len(val_scenes),
)

print(
    "Val samples             :",
    len(val_files),
)

print(
    "Test used               : NO"
)

print(
    "Betas                   :",
    BETAS,
)

print(
    "Decision modes          :",
    MODES,
)


# =============================================================================
# Load frozen near5 Planning predictions
# =============================================================================

with np.load(
    VAL_DATASET_FILE,
    allow_pickle=False,
) as d:

    near_candidate_idx = d[
        "candidate_idx"
    ].astype(
        np.int64
    )

    near_dataset_v2_accept = d[
        "v2_accept"
    ].astype(
        bool
    )

    near_min_dist = d[
        "min_plan_dist_m"
    ].astype(
        np.float32
    )


with np.load(
    UNCERTAINTY_FILE,
    allow_pickle=False,
) as d:

    utility_mean = d[
        "utility_mean"
    ].astype(
        np.float64
    )

    utility_std = d[
        "utility_std"
    ].astype(
        np.float64
    )

    label_true = d[
        "label_true"
    ].astype(
        np.int8
    )


assert len(
    near_candidate_idx
) == 306

assert utility_mean.shape == (
    306,
)

assert utility_std.shape == (
    306,
)

assert np.all(
    near_min_dist < 5.0
)

assert len(
    np.unique(
        near_candidate_idx
    )
) == 306


# =============================================================================
# Reconstruct all Val candidates + 127-D features
# =============================================================================

v1 = m.STCVOcc(
    checkpoint_path=str(
        m.V1_CKPT
    ),
    threshold=0.70,
)

v1.eval()


all_features = []

sample_records = []

candidate_count = 0


print()
print("=" * 110)
print("PASS 1: RECONSTRUCT EXACT VAL CANDIDATE ORDERING")
print("=" * 110)


for sample_order, p_plan in enumerate(
    val_files
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

        sample_idx = str(
            plan["sample_idx"].item()
        )

        scene_token = str(
            plan["scene_token"].item()
        )

    with np.load(
        p_g0,
        allow_pickle=False,
    ) as g0:

        assert str(
            g0["sample_idx"].item()
        ) == sample_idx

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

    n_cand = len(
        candidate_indices
    )

    assert features.shape == (
        n_cand,
        127,
    )

    start = candidate_count
    end = start + n_cand

    sample_records.append({
        "sample_order":
            sample_order,

        "file":
            p_plan,

        "sample_idx":
            sample_idx,

        "scene_token":
            scene_token,

        "candidate_indices":
            candidate_indices,

        "global_start":
            start,

        "global_end":
            end,
    })

    if n_cand > 0:

        all_features.append(
            features.cpu()
        )

    candidate_count = end

    if (
        sample_order < 3
        or (sample_order + 1) % 20 == 0
        or sample_order + 1
        == len(val_files)
    ):

        print(
            f"[{sample_order+1:03d}/"
            f"{len(val_files):03d}] "
            f"{p_plan.name} "
            f"sample={sample_idx} "
            f"cand={n_cand} "
            f"running={candidate_count}"
        )


assert candidate_count == 18555, (
    f"Candidate mismatch: "
    f"{candidate_count} != 18555"
)


all_features = torch.cat(
    all_features,
    dim=0,
)


assert all_features.shape == (
    18555,
    127,
)


print()
print(
    "Val candidate count     :",
    candidate_count,
)

print(
    "Candidate ordering      : PASS"
)


# =============================================================================
# Frozen V2 predictor: exact canonical CUDA batch=2048
# =============================================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "Exact V2 path requires CUDA."
    )


device = torch.device(
    "cuda"
)


v2 = m.V2UtilityPredictor(
    dropout=0.1
)


v2_ckpt = torch.load(
    m.V2_CKPT,
    map_location=device,
)


v2.load_state_dict(
    v2_ckpt["model"],
    strict=True,
)


v2 = v2.to(
    device
)

v2.eval()


mag_chunks = []


with torch.no_grad():

    for start in range(
        0,
        len(all_features),
        2048,
    ):

        x = all_features[
            start:
            start + 2048
        ].to(
            device,
            non_blocking=True,
        )

        (
            mag,
            _,
        ) = v2(x)

        mag_chunks.append(
            mag.detach().cpu()
        )


v2_mag = torch.cat(
    mag_chunks
).numpy()


v2_accept = (
    v2_mag > 0.0
)


assert int(
    v2_accept.sum()
) == 6156, (
    f"Frozen V2 accept mismatch: "
    f"{int(v2_accept.sum())} != 6156"
)


assert np.array_equal(
    v2_accept[
        near_candidate_idx
    ],
    near_dataset_v2_accept,
), (
    "Near5 V2 decision mismatch."
)


assert int(
    near_dataset_v2_accept.sum()
) == 11


print()
print(
    "Frozen V2 accepted      :",
    int(
        v2_accept.sum()
    ),
    "/18555",
)

print(
    "Near5 candidates        :",
    len(
        near_candidate_idx
    ),
)

print(
    "Near5 V2 accepted       :",
    int(
        near_dataset_v2_accept.sum()
    ),
)

print(
    "V2 reconstruction       : PASS"
)


# =============================================================================
# Build strategy acceptance arrays
# =============================================================================

strategy_accept = {
    "V2": v2_accept.copy(),
}


strategy_decision_stats = {}


for beta in BETAS:

    lcb = (
        utility_mean
        -
        beta
        *
        utility_std
    )

    plan_accept = (
        lcb > 0.0
    )

    for mode in MODES:

        accept = (
            v2_accept.copy()
        )

        old = (
            accept[
                near_candidate_idx
            ].copy()
        )

        if mode == "GateOnly":

            new = (
                old
                &
                plan_accept
            )

        elif mode == "RecoveryOnly":

            new = (
                old
                |
                plan_accept
            )

        elif mode == "Adaptive":

            new = (
                plan_accept.copy()
            )

        else:

            raise ValueError(
                mode
            )

        accept[
            near_candidate_idx
        ] = new

        name = (
            f"{mode}_b{beta:.2f}"
        )

        strategy_accept[
            name
        ] = accept

        strategy_decision_stats[
            name
        ] = {
            "beta":
                float(beta),

            "mode":
                mode,

            "total_accept":
                int(
                    accept.sum()
                ),

            "total_accept_rate":
                float(
                    accept.mean()
                ),

            "near5_accept":
                int(
                    new.sum()
                ),

            "near5_accept_rate":
                float(
                    new.mean()
                ),

            "v2_to_reject":
                int(
                    np.logical_and(
                        old,
                        np.logical_not(
                            new
                        ),
                    ).sum()
                ),

            "v2_reject_to_accept":
                int(
                    np.logical_and(
                        np.logical_not(
                            old
                        ),
                        new,
                    ).sum()
                ),
        }


# =============================================================================
# Reference strategies
# =============================================================================

strategy_names = [
    "Ego",
    "OfficialOR",
    "V2",
]

for beta in BETAS:

    for mode in MODES:

        strategy_names.append(
            f"{mode}_b{beta:.2f}"
        )


# =============================================================================
# Accumulators
# =============================================================================

occ_acc = {
    name: make_occ_accumulator()
    for name in strategy_names
}


planning_acc = {
    name: {
        "L2": [],
        "obj_col": [],
        "obj_box_col": [],
        "obj_out": [],
        "obj_box_out": [],
        "collision_changed": 0,
        "final_changed": 0,
    }
    for name in strategy_names
}


planner = build_planner()


# =============================================================================
# PASS 2: Full 96-sample fused occupancy + official Planning replay
# =============================================================================

print()
print("=" * 110)
print("PASS 2: FULL VAL OCCUPANCY + OFFICIAL PLANNING REPLAY")
print("=" * 110)


for i, rec in enumerate(
    sample_records
):

    p_plan = rec[
        "file"
    ]

    p_g0 = (
        m.G0_CACHE
        / p_plan.name
    )

    with np.load(
        p_plan,
        allow_pickle=False,
    ) as plan:

        raw = torch.from_numpy(
            norm(
                plan["T_raw"],
                3,
            )
        ).float()

        gt_traj = torch.from_numpy(
            norm(
                plan[
                    "sdc_planning_eval"
                ],
                3,
            )
        ).float()

        gt_mask = torch.from_numpy(
            norm(
                plan[
                    "sdc_planning_mask_eval"
                ],
                3,
            )
        ).float()

        gt_seg = torch.from_numpy(
            norm(
                plan[
                    "gt_segmentation_eval"
                ],
                4,
            )
        )

        drivable_pred = torch.from_numpy(
            norm(
                plan[
                    "drivable_pred"
                ],
                2,
            )
        )

        drivable_gt = torch.from_numpy(
            norm(
                plan[
                    "drivable_gt"
                ],
                2,
            )
        )

    with np.load(
        p_g0,
        allow_pickle=False,
    ) as g0:

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

        official = torch.from_numpy(
            norm(
                g0["Oofficial"],
                3,
            )
        ).bool()

        gt_occ = torch.from_numpy(
            norm(
                g0["GT"],
                3,
            )
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

    start = rec[
        "global_start"
    ]

    end = rec[
        "global_end"
    ]

    cand = rec[
        "candidate_indices"
    ]

    # -------------------------------------------------------------------------
    # Build occupancy for all strategies
    # -------------------------------------------------------------------------

    occ_by_strategy = {
        "Ego":
            ov,

        "OfficialOR":
            official,
    }

    for name in strategy_names:

        if name in [
            "Ego",
            "OfficialOR",
        ]:
            continue

        local_accept = (
            strategy_accept[
                name
            ][
                start:end
            ]
        )

        occ_by_strategy[
            name
        ] = apply_candidates(
            ov,
            oi,
            cand,
            local_accept,
        )

    # -------------------------------------------------------------------------
    # Ego baseline postprocess once
    # -------------------------------------------------------------------------

    (
        ego_collision,
        ego_final,
    ) = postprocess(
        planner,
        raw,
        ov,
        drivable_pred,
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    for name in strategy_names:

        occ = occ_by_strategy[
            name
        ]

        update_occ_counts(
            occ_acc[
                name
            ],
            occ,
            gt_occ,
        )

        if name == "Ego":

            collision = ego_collision
            final = ego_final

        else:

            (
                collision,
                final,
            ) = postprocess(
                planner,
                raw,
                occ,
                drivable_pred,
            )

        pm = planning_metric_one(
            final,
            gt_traj,
            gt_mask,
            gt_seg,
            drivable_gt,
        )

        for key in [
            "L2",
            "obj_col",
            "obj_box_col",
            "obj_out",
            "obj_box_out",
        ]:

            planning_acc[
                name
            ][key].append(
                pm[key]
            )

        planning_acc[
            name
        ][
            "collision_changed"
        ] += int(
            (
                collision
                -
                raw
            )
            .abs()
            .max()
            .item()
            > 1e-6
        )

        planning_acc[
            name
        ][
            "final_changed"
        ] += int(
            (
                final
                -
                raw
            )
            .abs()
            .max()
            .item()
            > 1e-6
        )

    if (
        i < 3
        or (i + 1) % 10 == 0
        or i + 1
        == len(sample_records)
    ):

        print(
            f"[{i+1:03d}/"
            f"{len(sample_records):03d}] "
            f"sample={rec['sample_idx']} "
            f"cand={len(cand)}"
        )


# =============================================================================
# Finalize metrics
# =============================================================================

results = {}


for name in strategy_names:

    planning = {}

    for key in [
        "L2",
        "obj_col",
        "obj_box_col",
        "obj_out",
        "obj_box_out",
    ]:

        arr = np.stack(
            planning_acc[
                name
            ][key],
            axis=0,
        )

        assert arr.shape == (
            96,
            10,
        )

        planning[
            key + "_mean"
        ] = float(
            arr.mean()
        )

        # V2-2 presentation uses 1s..5s at planning indices
        # [1,3,5,7,9]?  To avoid silently redefining,
        # save the full 10-step mean vector as canonical output.
        planning[
            key + "_per_step"
        ] = (
            arr.mean(
                axis=0
            )
            .astype(
                np.float64
            )
            .tolist()
        )

    planning[
        "collision_changed"
    ] = int(
        planning_acc[
            name
        ][
            "collision_changed"
        ]
    )

    planning[
        "final_changed"
    ] = int(
        planning_acc[
            name
        ][
            "final_changed"
        ]
    )

    occ = {}

    for range_name in OCC_RANGES:

        occ[
            range_name + "_iou"
        ] = float(
            occ_iou_from_counts(
                occ_acc[
                    name
                ][
                    range_name
                ]
            )
        )

    result = {
        "occupancy":
            occ,

        "planning":
            planning,
    }

    if name == "V2":

        result[
            "decision"
        ] = {
            "total_accept":
                int(
                    v2_accept.sum()
                ),

            "total_accept_rate":
                float(
                    v2_accept.mean()
                ),

            "near5_accept":
                int(
                    near_dataset_v2_accept.sum()
                ),
        }

    elif name in strategy_decision_stats:

        result[
            "decision"
        ] = (
            strategy_decision_stats[
                name
            ]
        )

    results[
        name
    ] = result


# =============================================================================
# Frozen-reference audit
# =============================================================================

print()
print("=" * 110)
print("REFERENCE REPRODUCTION AUDIT")
print("=" * 110)


expected_planning_l2 = {
    "Ego": 2.803602,
    "OfficialOR": 2.845870,
    "V2": 2.804198,
}


for name, expected in (
    expected_planning_l2.items()
):

    actual = (
        results[
            name
        ][
            "planning"
        ][
            "L2_mean"
        ]
    )

    diff = actual - expected

    print(
        f"{name:12s} "
        f"L2={actual:.6f} "
        f"expected={expected:.6f} "
        f"diff={diff:+.8f}"
    )


print()
print("Occupancy reference:")

for name in [
    "Ego",
    "OfficialOR",
    "V2",
]:

    d = results[
        name
    ][
        "occupancy"
    ]

    print(
        f"{name:12s} "
        f"30m={d['30m_iou']:.6f} "
        f"100m={d['100m_iou']:.6f}"
    )


# =============================================================================
# Main sweep table
# =============================================================================

print()
print("=" * 150)
print("V2-3E ADAPTIVE DECISION RESULTS")
print("=" * 150)

print(
    f"{'Strategy':24s} "
    f"{'Accept':>7s} "
    f"{'Near5':>7s} "
    f"{'V2->R':>7s} "
    f"{'R->A':>7s} "
    f"{'IoU30':>10s} "
    f"{'IoU100':>10s} "
    f"{'L2':>10s} "
    f"{'obj_col':>10s} "
    f"{'box_col':>10s} "
    f"{'obj_out':>10s} "
    f"{'box_out':>10s} "
    f"{'ColChg':>7s}"
)

print("-" * 150)


for name in strategy_names:

    d = results[
        name
    ]

    occ = d[
        "occupancy"
    ]

    plan = d[
        "planning"
    ]

    decision = d.get(
        "decision",
        {},
    )

    print(
        f"{name:24s} "
        f"{decision.get('total_accept', 0):7d} "
        f"{decision.get('near5_accept', 0):7d} "
        f"{decision.get('v2_to_reject', 0):7d} "
        f"{decision.get('v2_reject_to_accept', 0):7d} "
        f"{occ['30m_iou']:10.6f} "
        f"{occ['100m_iou']:10.6f} "
        f"{plan['L2_mean']:10.6f} "
        f"{plan['obj_col_mean']:10.6f} "
        f"{plan['obj_box_col_mean']:10.6f} "
        f"{plan['obj_out_mean']:10.6f} "
        f"{plan['obj_box_out_mean']:10.6f} "
        f"{plan['collision_changed']:7d}"
    )


# =============================================================================
# Relative-to-V2 table
# =============================================================================

v2_l2 = results[
    "V2"
][
    "planning"
][
    "L2_mean"
]

v2_iou30 = results[
    "V2"
][
    "occupancy"
][
    "30m_iou"
]

v2_iou100 = results[
    "V2"
][
    "occupancy"
][
    "100m_iou"
]


print()
print("=" * 110)
print("DELTA RELATIVE TO FROZEN V2")
print("=" * 110)

print(
    f"{'Strategy':24s} "
    f"{'dL2':>12s} "
    f"{'dIoU30':>12s} "
    f"{'dIoU100':>12s}"
)

print("-" * 65)


for name in strategy_names:

    if name in [
        "Ego",
        "OfficialOR",
        "V2",
    ]:
        continue

    d = results[
        name
    ]

    dl2 = (
        d[
            "planning"
        ][
            "L2_mean"
        ]
        -
        v2_l2
    )

    diou30 = (
        d[
            "occupancy"
        ][
            "30m_iou"
        ]
        -
        v2_iou30
    )

    diou100 = (
        d[
            "occupancy"
        ][
            "100m_iou"
        ]
        -
        v2_iou100
    )

    print(
        f"{name:24s} "
        f"{dl2:+12.8f} "
        f"{diou30:+12.8f} "
        f"{diou100:+12.8f}"
    )


# =============================================================================
# Diagnostic best-by-L2
# =============================================================================

adaptive_names = [
    x
    for x in strategy_names
    if x not in [
        "Ego",
        "OfficialOR",
        "V2",
    ]
]


best_l2_name = min(
    adaptive_names,
    key=lambda x:
        results[x][
            "planning"
        ][
            "L2_mean"
        ],
)


print()
print("=" * 110)
print("VAL DIAGNOSTIC ONLY — DO NOT FREEZE TEST YET")
print("=" * 110)

print(
    "Lowest-L2 adaptive strategy:",
    best_l2_name,
)

print(
    "L2:",
    f"{results[best_l2_name]['planning']['L2_mean']:.8f}",
)

print(
    "30m IoU:",
    f"{results[best_l2_name]['occupancy']['30m_iou']:.8f}",
)

print(
    "100m IoU:",
    f"{results[best_l2_name]['occupancy']['100m_iou']:.8f}",
)

print(
    "Decision:",
    results[
        best_l2_name
    ].get(
        "decision",
        {},
    ),
)


# =============================================================================
# Save
# =============================================================================

summary = {
    "stage":
        "V2-3E validation adaptive decision sweep",

    "protocol": {
        "val_scenes":
            4,

        "val_samples":
            96,

        "all_candidates":
            18555,

        "planning_sensitive_candidates":
            306,

        "test_used":
            False,

        "betas":
            BETAS,

        "modes":
            MODES,

        "outside_near5":
            "Frozen V2 decision",

        "near5_GateOnly":
            "V2_accept AND (mu_plan - beta*sigma_plan > 0)",

        "near5_RecoveryOnly":
            "V2_accept OR (mu_plan - beta*sigma_plan > 0)",

        "near5_Adaptive":
            "(mu_plan - beta*sigma_plan > 0)",
    },

    "results":
        results,

    "best_l2_diagnostic":
        best_l2_name,
}


with open(
    OUT_JSON,
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


np.savez_compressed(
    OUT_NPZ,

    near_candidate_idx=(
        near_candidate_idx
    ),

    utility_mean=(
        utility_mean.astype(
            np.float32
        )
    ),

    utility_std=(
        utility_std.astype(
            np.float32
        )
    ),

    v2_accept=(
        v2_accept.astype(
            np.uint8
        )
    ),

    v2_mag=(
        v2_mag.astype(
            np.float32
        )
    ),
)


print()
print("=" * 110)
print("SAVED")
print("=" * 110)

print(
    "JSON:",
    OUT_JSON,
)

print(
    "NPZ :",
    OUT_NPZ,
)

print()
print(
    "[V2-3E VAL ADAPTIVE DECISION SWEEP] COMPLETE"
)

print("=" * 110)
