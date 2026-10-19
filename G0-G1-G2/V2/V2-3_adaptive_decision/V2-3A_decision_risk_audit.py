#!/usr/bin/env python3

import csv
import json
import math
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


# =============================================================================
# Paths
# =============================================================================

ROOT = Path("/root/autodl-tmp/UniV2X")

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
    / "V2-3A_candidate_decision_risk_audit.csv"
)

NPZ_OUT = (
    OUT_DIR
    / "V2-3A_candidate_decision_risk_arrays.npz"
)

JSON_OUT = (
    OUT_DIR
    / "V2-3A_decision_risk_summary.json"
)


# =============================================================================
# Reuse the already audited V2-2C implementation
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "v22c",
    V22C_SCRIPT,
)

m = importlib.util.module_from_spec(spec)

spec.loader.exec_module(m)


# =============================================================================
# Helper
# =============================================================================

def safe_float(x):

    if x is None:
        return None

    x = float(x)

    if not math.isfinite(x):
        return None

    return x


def percentile_stats(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:

        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }

    return {
        "count": int(len(x)),
        "mean": float(
            np.mean(x)
        ),
        "median": float(
            np.median(x)
        ),
        "p10": float(
            np.percentile(x, 10)
        ),
        "p25": float(
            np.percentile(x, 25)
        ),
        "p75": float(
            np.percentile(x, 75)
        ),
        "p90": float(
            np.percentile(x, 90)
        ),
    }


def rate(
    numerator,
    denominator,
):

    if denominator == 0:
        return None

    return float(
        numerator / denominator
    )


# =============================================================================
# Official planning temporal relation
# =============================================================================
#
# planning_head.py:
#
#   cur_t = min(
#       t + 1,
#       occ_horizon - 1
#   )
#
# with:
#   planning_steps = 10
#   occ_horizon    = 5
#
# Therefore:
#
#   planning step 0 -> occ h=1
#   planning step 1 -> occ h=2
#   planning step 2 -> occ h=3
#   planning step 3 -> occ h=4
#   planning step 4 -> occ h=4
#   ...
#   planning step 9 -> occ h=4
#
# h=0 is NOT used by collision optimization.
#
# =============================================================================

PLANNING_STEPS_BY_OCC_HORIZON = {
    0: [],
    1: [0],
    2: [1],
    3: [2],
    4: [3, 4, 5, 6, 7, 8, 9],
}


# =============================================================================
# EXACT official collision-optimizer coordinate convention
# =============================================================================
#
# planning_head.py converts occupied-grid coordinates using:
#
#   pos_xy = torch.nonzero(...)
#   pos_xy = pos_xy[:, [1, 0]]
#   pos_xy = (
#       pos_xy - bev_h // 2
#   ) * 0.5 + 0.25
#
# IMPORTANT:
#
# This audit intentionally uses the planning implementation's 0.5 m
# convention, NOT the occupancy model's nominal 0.512 m BEV resolution.
#
# This is a task-consistency audit.
#
# =============================================================================

GRID_CENTER = 100
PLANNING_GRID_RESOLUTION = 0.5
PLANNING_CELL_OFFSET = 0.25
OFFICIAL_OCC_FILTER_RANGE = 5.0


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


def candidate_planning_distance(
    h,
    r,
    c,
    ov,
    oi,
    warp,
    raw_traj,
):
    """
    Compute exact candidate-level planning relevance proxy.

    Candidate region:
        4x4 patch.

    Only ADD cells are considered:
        (~Ov) & Oi & warp_valid.

    Relevance:
        Minimum Euclidean distance between any ADD cell in this patch
        and the raw planning point(s) that consume this occupancy horizon.

    Returns:
        n_add_cells
        min_plan_distance_m
        n_add_cells_within_5m
        planning_used
    """

    y0 = 4 * r
    y1 = y0 + 4

    x0 = 4 * c
    x1 = x0 + 4

    patch_ov = ov[
        h,
        y0:y1,
        x0:x1,
    ]

    patch_oi = oi[
        h,
        y0:y1,
        x0:x1,
    ]

    patch_warp = warp[
        y0:y1,
        x0:x1,
    ]

    add_patch = (
        (~patch_ov)
        & patch_oi
        & patch_warp
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

    # h=0 does not enter official collision optimization.
    if len(planning_steps) == 0:

        return (
            n_add_cells,
            float("inf"),
            0,
            False,
        )

    global_rows = (
        local_rc[:, 0]
        + y0
    )

    global_cols = (
        local_rc[:, 1]
        + x0
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

    # [N_cells, N_steps, 2]
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

    # For each ADD cell:
    # does it enter the official <=5m trajectory filter
    # for at least one associated planning timestep?
    cell_min_dist = np.min(
        distance,
        axis=1,
    )

    n_cells_within_5m = int(
        (
            cell_min_dist
            <
            OFFICIAL_OCC_FILTER_RANGE
        ).sum()
    )

    return (
        n_add_cells,
        min_dist,
        n_cells_within_5m,
        True,
    )


# =============================================================================
# Exact Val split
# =============================================================================

(
    valid_sample_ids,
    train_scenes,
    val_scenes,
    test_scenes,
) = m.build_exact_val_split()


plan_files = sorted(
    m.PLAN_CACHE.glob(
        "sample_*.npz"
    )
)

val_files = []

for p in plan_files:

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
        sample_idx
        in valid_sample_ids
        and scene_token
        in val_scenes
    ):
        val_files.append(p)


assert len(val_files) == 96, (
    f"Expected 96 Val samples, "
    f"got {len(val_files)}"
)


# =============================================================================
# Load frozen V1
# =============================================================================

v1 = m.STCVOcc(
    checkpoint_path=str(
        m.V1_CKPT
    ),
    threshold=0.70,
)

v1.eval()


# =============================================================================
# First pass:
#   exact candidate reconstruction
#   exact 127-D features
#   V1 decision
#   raw planning relevance
# =============================================================================

records = []

feature_chunks = []

candidate_global_idx = 0


print("=" * 100)
print("V2-3A DECISION-RISK AUDIT")
print("=" * 100)

print(
    "Seed               :",
    2026,
)

print(
    "Valid sample IDs   :",
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
    "Val scene tokens   :",
    sorted(val_scenes),
)

print(
    "Val samples        :",
    len(val_files),
)

print(
    "Planning relation  :",
    "h0=unused, h1->t0, h2->t1, "
    "h3->t2, h4->t3...t9",
)

print(
    "Planning distance  :",
    "official 0.5m grid convention",
)

print(
    "Planning filter    :",
    "< 5.0 m",
)


for sample_order, p_plan in enumerate(
    val_files
):

    p_g0 = (
        m.G0_CACHE
        / p_plan.name
    )

    assert p_g0.exists()

    # -------------------------------------------------------------------------
    # Planning raw trajectory
    # -------------------------------------------------------------------------

    with np.load(
        p_plan,
        allow_pickle=False,
    ) as plan:

        raw_traj = m.normalize_array(
            plan["T_raw"],
            3,
        )[0, :, :2].astype(
            np.float32
        )

        assert raw_traj.shape == (
            10,
            2,
        )

        plan_sample_idx = str(
            plan["sample_idx"].item()
        )

        plan_scene_token = str(
            plan["scene_token"].item()
        )

    # -------------------------------------------------------------------------
    # Occupancy
    # -------------------------------------------------------------------------

    with np.load(
        p_g0,
        allow_pickle=False,
    ) as g0:

        g0_sample_idx = str(
            g0["sample_idx"].item()
        )

        g0_scene_token = str(
            g0["scene_token"].item()
        )

        assert (
            plan_sample_idx
            ==
            g0_sample_idx
        )

        assert (
            plan_scene_token
            ==
            g0_scene_token
        )

        pv = torch.from_numpy(
            m.normalize_array(
                g0["Pv"],
                3,
            )
        ).float()

        pi = torch.from_numpy(
            m.normalize_array(
                g0["Pi_aligned"],
                3,
            )
        ).float()

        ov = torch.from_numpy(
            m.normalize_array(
                g0["Ov"],
                3,
            )
        ).bool()

        oi = torch.from_numpy(
            m.normalize_array(
                g0["Oi"],
                3,
            )
        ).bool()

        warp = torch.from_numpy(
            m.normalize_array(
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

    if n_candidates > 0:

        feature_chunks.append(
            features.cpu()
        )

        with torch.no_grad():

            v1_logits = (
                v1.predictor(
                    features
                )
            )

            v1_scores = torch.sigmoid(
                v1_logits
            ).cpu().numpy()

    else:

        v1_scores = np.empty(
            0,
            dtype=np.float32,
        )

    ov_np = ov.cpu().numpy()
    oi_np = oi.cpu().numpy()
    warp_np = warp.cpu().numpy()

    # -------------------------------------------------------------------------
    # Candidate-level audit
    # -------------------------------------------------------------------------

    for local_idx, (
        h,
        r,
        c,
    ) in enumerate(
        candidate_indices
    ):

        (
            n_add_cells,
            min_plan_dist,
            n_add_cells_within_5m,
            planning_used,
        ) = candidate_planning_distance(
            int(h),
            int(r),
            int(c),
            ov_np,
            oi_np,
            warp_np,
            raw_traj,
        )

        near_1m = (
            planning_used
            and min_plan_dist < 1.0
        )

        near_2m = (
            planning_used
            and min_plan_dist < 2.0
        )

        near_3m = (
            planning_used
            and min_plan_dist < 3.0
        )

        near_5m = (
            planning_used
            and min_plan_dist
            <
            OFFICIAL_OCC_FILTER_RANGE
        )

        records.append({
            "candidate_idx":
                candidate_global_idx,

            "sample_order":
                sample_order,

            "cache_file":
                p_plan.name,

            "sample_idx":
                plan_sample_idx,

            "scene_token":
                plan_scene_token,

            "horizon":
                int(h),

            "patch_r":
                int(r),

            "patch_c":
                int(c),

            "n_add_cells":
                int(n_add_cells),

            "planning_used":
                int(planning_used),

            "min_plan_dist_m":
                float(min_plan_dist),

            "add_cells_within_5m":
                int(
                    n_add_cells_within_5m
                ),

            "near_1m":
                int(near_1m),

            "near_2m":
                int(near_2m),

            "near_3m":
                int(near_3m),

            "near_5m":
                int(near_5m),

            "v1_score":
                float(
                    v1_scores[
                        local_idx
                    ]
                ),

            "v1_accept":
                int(
                    v1_scores[
                        local_idx
                    ]
                    >
                    0.70
                ),
        })

        candidate_global_idx += 1

    if (
        sample_order < 3
        or (sample_order + 1) % 10 == 0
        or sample_order + 1
        == len(val_files)
    ):

        print(
            f"[{sample_order+1:03d}/"
            f"{len(val_files):03d}] "
            f"{p_plan.name} "
            f"sample={plan_sample_idx} "
            f"cand={n_candidates}"
        )


# =============================================================================
# Exact candidate audit
# =============================================================================

assert candidate_global_idx == 18555, (
    f"Expected 18555 candidates, "
    f"got {candidate_global_idx}"
)

all_features = torch.cat(
    feature_chunks,
    dim=0,
)

assert all_features.shape == (
    18555,
    127,
)


v1_accept_count = int(
    sum(
        x["v1_accept"]
        for x in records
    )
)

assert v1_accept_count == 5691, (
    f"Frozen V1 mismatch: "
    f"{v1_accept_count} != 5691"
)


# =============================================================================
# Frozen V2 inference
#
# EXACT V2-1C numerical path:
#
#   CUDA
#   batch_size = 2048
#   accept iff magnitude > 0
#
# =============================================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "Exact frozen V2-1C audit "
        "requires CUDA."
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
        len(all_features),
        2048,
    ):

        x = all_features[
            i:i + 2048
        ].to(
            device,
            non_blocking=True,
        )

        (
            mag,
            sign_logit,
        ) = v2(x)

        mag_chunks.append(
            mag.detach().cpu()
        )

        sign_chunks.append(
            sign_logit.detach().cpu()
        )


mag = torch.cat(
    mag_chunks
).numpy()

sign_logit = torch.cat(
    sign_chunks
).numpy()

sign_prob = (
    1.0
    /
    (
        1.0
        +
        np.exp(
            -sign_logit
        )
    )
)


assert len(mag) == 18555


v2_accept = (
    mag > 0.0
)

v2_accept_count = int(
    v2_accept.sum()
)

assert v2_accept_count == 6156, (
    f"Frozen V2 mismatch: "
    f"{v2_accept_count} != 6156"
)


# =============================================================================
# Add V2 uncertainty-related raw quantities
# =============================================================================
#
# Do NOT freeze a final uncertainty formula here.
#
# We retain:
#
#   1. abs_mag
#      distance from current hard utility boundary mag=0
#
#   2. sign_prob
#
#   3. sign_margin
#      |p(sign positive) - 0.5|
#
#   4. sign_entropy
#
# These are diagnostic quantities only.
#
# =============================================================================

eps = 1e-12

abs_mag = np.abs(
    mag
)

sign_margin = np.abs(
    sign_prob
    - 0.5
)

sign_entropy = -(
    sign_prob
    * np.log(
        sign_prob + eps
    )
    +
    (
        1.0
        - sign_prob
    )
    * np.log(
        1.0
        - sign_prob
        + eps
    )
)


for i, rec in enumerate(
    records
):

    rec[
        "v2_mag"
    ] = float(
        mag[i]
    )

    rec[
        "v2_accept"
    ] = int(
        v2_accept[i]
    )

    rec[
        "abs_mag"
    ] = float(
        abs_mag[i]
    )

    rec[
        "sign_logit"
    ] = float(
        sign_logit[i]
    )

    rec[
        "sign_prob"
    ] = float(
        sign_prob[i]
    )

    rec[
        "sign_margin"
    ] = float(
        sign_margin[i]
    )

    rec[
        "sign_entropy"
    ] = float(
        sign_entropy[i]
    )


# =============================================================================
# Numpy arrays
# =============================================================================

horizon = np.asarray(
    [
        x["horizon"]
        for x in records
    ],
    dtype=np.int64,
)

planning_used = np.asarray(
    [
        x["planning_used"]
        for x in records
    ],
    dtype=bool,
)

min_plan_dist = np.asarray(
    [
        x["min_plan_dist_m"]
        for x in records
    ],
    dtype=np.float64,
)

near_1m = np.asarray(
    [
        x["near_1m"]
        for x in records
    ],
    dtype=bool,
)

near_2m = np.asarray(
    [
        x["near_2m"]
        for x in records
    ],
    dtype=bool,
)

near_3m = np.asarray(
    [
        x["near_3m"]
        for x in records
    ],
    dtype=bool,
)

near_5m = np.asarray(
    [
        x["near_5m"]
        for x in records
    ],
    dtype=bool,
)

v1_accept = np.asarray(
    [
        x["v1_accept"]
        for x in records
    ],
    dtype=bool,
)

sample_order = np.asarray(
    [
        x["sample_order"]
        for x in records
    ],
    dtype=np.int64,
)

patch_r = np.asarray(
    [
        x["patch_r"]
        for x in records
    ],
    dtype=np.int64,
)

patch_c = np.asarray(
    [
        x["patch_c"]
        for x in records
    ],
    dtype=np.int64,
)

n_add_cells = np.asarray(
    [
        x["n_add_cells"]
        for x in records
    ],
    dtype=np.int64,
)

add_cells_within_5m = np.asarray(
    [
        x[
            "add_cells_within_5m"
        ]
        for x in records
    ],
    dtype=np.int64,
)


# =============================================================================
# Summary
# =============================================================================

summary = {
    "protocol": {
        "seed": 2026,
        "split": "Val only",
        "val_samples": 96,
        "candidate_count": 18555,
        "v1_accept_expected": 5691,
        "v2_accept_expected": 6156,
        "v2_execution":
            "CUDA batch=2048",
        "v2_decision":
            "magnitude_pred > 0",
        "planning_steps": 10,
        "occupancy_horizons": 5,
        "planning_horizon_mapping": {
            "0": [],
            "1": [0],
            "2": [1],
            "3": [2],
            "4": [
                3, 4, 5, 6,
                7, 8, 9
            ],
        },
        "planning_grid_resolution_m":
            0.5,
        "planning_cell_offset_m":
            0.25,
        "official_occ_filter_range_m":
            5.0,
    },

    "global": {
        "candidates":
            int(len(records)),

        "v1_accepted":
            int(
                v1_accept.sum()
            ),

        "v1_accept_rate":
            float(
                v1_accept.mean()
            ),

        "v2_accepted":
            int(
                v2_accept.sum()
            ),

        "v2_accept_rate":
            float(
                v2_accept.mean()
            ),

        "planning_used_candidates":
            int(
                planning_used.sum()
            ),

        "planning_used_rate":
            float(
                planning_used.mean()
            ),

        "near_5m_candidates":
            int(
                near_5m.sum()
            ),

        "near_5m_rate":
            float(
                near_5m.mean()
            ),

        "v2_accepted_near_5m":
            int(
                (
                    v2_accept
                    & near_5m
                ).sum()
            ),

        "v2_accepted_near_5m_rate_among_all_accepts":
            rate(
                int(
                    (
                        v2_accept
                        & near_5m
                    ).sum()
                ),
                int(
                    v2_accept.sum()
                ),
            ),

        "v2_accepted_outside_5m_or_h0":
            int(
                (
                    v2_accept
                    & (~near_5m)
                ).sum()
            ),

        "v2_accepted_outside_5m_or_h0_rate_among_all_accepts":
            rate(
                int(
                    (
                        v2_accept
                        & (~near_5m)
                    ).sum()
                ),
                int(
                    v2_accept.sum()
                ),
            ),

        "samples_with_any_near_5m_candidate":
            int(
                len(
                    set(
                        sample_order[
                            near_5m
                        ].tolist()
                    )
                )
            ),

        "samples_with_v2_accepted_near_5m":
            int(
                len(
                    set(
                        sample_order[
                            v2_accept
                            & near_5m
                        ].tolist()
                    )
                )
            ),
    },

    "distance_stats_m": (
        percentile_stats(
            min_plan_dist[
                planning_used
            ]
        )
    ),

    "accepted_distance_stats_m": (
        percentile_stats(
            min_plan_dist[
                planning_used
                & v2_accept
            ]
        )
    ),

    "rejected_distance_stats_m": (
        percentile_stats(
            min_plan_dist[
                planning_used
                & (~v2_accept)
            ]
        )
    ),

    "abs_mag_stats": (
        percentile_stats(
            abs_mag
        )
    ),

    "sign_margin_stats": (
        percentile_stats(
            sign_margin
        )
    ),

    "sign_entropy_stats": (
        percentile_stats(
            sign_entropy
        )
    ),
}


# =============================================================================
# Horizon audit
# =============================================================================

summary[
    "by_horizon"
] = {}


for h in range(5):

    mask = (
        horizon == h
    )

    n = int(
        mask.sum()
    )

    near_n = int(
        (
            mask
            & near_5m
        ).sum()
    )

    v2_n = int(
        (
            mask
            & v2_accept
        ).sum()
    )

    v2_near_n = int(
        (
            mask
            & v2_accept
            & near_5m
        ).sum()
    )

    finite_dist = min_plan_dist[
        mask
        & planning_used
    ]

    summary[
        "by_horizon"
    ][str(h)] = {
        "candidates":
            n,

        "candidate_rate":
            rate(
                n,
                len(records),
            ),

        "planning_used":
            bool(
                len(
                    PLANNING_STEPS_BY_OCC_HORIZON[
                        h
                    ]
                )
                > 0
            ),

        "v1_accepted":
            int(
                (
                    mask
                    & v1_accept
                ).sum()
            ),

        "v1_accept_rate":
            rate(
                int(
                    (
                        mask
                        & v1_accept
                    ).sum()
                ),
                n,
            ),

        "v2_accepted":
            v2_n,

        "v2_accept_rate":
            rate(
                v2_n,
                n,
            ),

        "near_5m":
            near_n,

        "near_5m_rate":
            rate(
                near_n,
                n,
            ),

        "v2_accepted_near_5m":
            v2_near_n,

        "v2_near_5m_fraction_of_accepts":
            rate(
                v2_near_n,
                v2_n,
            ),

        "distance_stats_m":
            percentile_stats(
                finite_dist
            ),
    }


# =============================================================================
# Distance threshold audit
# =============================================================================

summary[
    "distance_thresholds"
] = {}


for d in [
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
]:

    mask = (
        planning_used
        &
        (
            min_plan_dist
            < d
        )
    )

    n = int(
        mask.sum()
    )

    n_v2 = int(
        (
            mask
            & v2_accept
        ).sum()
    )

    summary[
        "distance_thresholds"
    ][str(d)] = {
        "candidate_count":
            n,

        "candidate_rate":
            rate(
                n,
                len(records),
            ),

        "v2_accepted":
            n_v2,

        "v2_accept_rate":
            rate(
                n_v2,
                n,
            ),
    }


# =============================================================================
# Magnitude-boundary audit
# =============================================================================

summary[
    "magnitude_boundary"
] = {}


for threshold in [
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    5e-2,
]:

    mask = (
        abs_mag
        <
        threshold
    )

    n = int(
        mask.sum()
    )

    near_n = int(
        (
            mask
            & near_5m
        ).sum()
    )

    summary[
        "magnitude_boundary"
    ][f"{threshold:.0e}"] = {
        "count":
            n,

        "rate":
            rate(
                n,
                len(records),
            ),

        "near_5m":
            near_n,

        "near_5m_rate":
            rate(
                near_n,
                n,
            ),
    }


# =============================================================================
# Sign-head confidence audit
# =============================================================================

summary[
    "sign_boundary"
] = {}


for threshold in [
    0.01,
    0.05,
    0.10,
    0.20,
]:

    mask = (
        sign_margin
        <
        threshold
    )

    n = int(
        mask.sum()
    )

    near_n = int(
        (
            mask
            & near_5m
        ).sum()
    )

    summary[
        "sign_boundary"
    ][str(threshold)] = {
        "count":
            n,

        "rate":
            rate(
                n,
                len(records),
            ),

        "near_5m":
            near_n,

        "near_5m_rate":
            rate(
                near_n,
                n,
            ),
    }


# =============================================================================
# Decision × Planning relevance cross table
# =============================================================================

summary[
    "decision_planning_cross"
] = {
    "v2_accept_near5":
        int(
            (
                v2_accept
                & near_5m
            ).sum()
        ),

    "v2_accept_not_near5":
        int(
            (
                v2_accept
                & (~near_5m)
            ).sum()
        ),

    "v2_reject_near5":
        int(
            (
                (~v2_accept)
                & near_5m
            ).sum()
        ),

    "v2_reject_not_near5":
        int(
            (
                (~v2_accept)
                & (~near_5m)
            ).sum()
        ),
}


# =============================================================================
# Print audit
# =============================================================================

print()
print("=" * 100)
print("FROZEN DECISION AUDIT")
print("=" * 100)

print(
    "Candidates        :",
    len(records),
)

print(
    "V1 accepted       :",
    int(
        v1_accept.sum()
    ),
)

print(
    "V2 accepted       :",
    int(
        v2_accept.sum()
    ),
)

print(
    "Candidate audit   : PASS (18555)"
)

print(
    "V1 decision audit : PASS (5691)"
)

print(
    "V2 decision audit : PASS (6156)"
)


print()
print("=" * 100)
print("PLANNING RELEVANCE AUDIT")
print("=" * 100)

print(
    "Planning-used cand:",
    int(
        planning_used.sum()
    ),
    f"({planning_used.mean():.6%})",
)

print(
    "Within 1m        :",
    int(
        near_1m.sum()
    ),
    f"({near_1m.mean():.6%})",
)

print(
    "Within 2m        :",
    int(
        near_2m.sum()
    ),
    f"({near_2m.mean():.6%})",
)

print(
    "Within 3m        :",
    int(
        near_3m.sum()
    ),
    f"({near_3m.mean():.6%})",
)

print(
    "Within 5m        :",
    int(
        near_5m.sum()
    ),
    f"({near_5m.mean():.6%})",
)

v2_near = int(
    (
        v2_accept
        & near_5m
    ).sum()
)

v2_far = int(
    (
        v2_accept
        & (~near_5m)
    ).sum()
)

print()
print(
    "V2 accept near5  :",
    v2_near,
    f"({v2_near/v2_accept_count:.6%} "
    "of V2 accepts)",
)

print(
    "V2 accept outside:",
    v2_far,
    f"({v2_far/v2_accept_count:.6%} "
    "of V2 accepts)",
)

print(
    "Samples any near5:",
    summary[
        "global"
    ][
        "samples_with_any_near_5m_candidate"
    ],
    "/ 96",
)

print(
    "Samples V2 near5 :",
    summary[
        "global"
    ][
        "samples_with_v2_accepted_near_5m"
    ],
    "/ 96",
)


print()
print("=" * 100)
print("BY HORIZON")
print("=" * 100)

print(
    f"{'h':>2s} "
    f"{'cand':>7s} "
    f"{'V2_acc':>8s} "
    f"{'acc_rate':>10s} "
    f"{'near5':>7s} "
    f"{'V2_near':>8s} "
    f"{'near/acc':>10s}"
)

for h in range(5):

    d = summary[
        "by_horizon"
    ][str(h)]

    near_acc_rate = (
        d[
            "v2_near_5m_fraction_of_accepts"
        ]
    )

    print(
        f"{h:2d} "
        f"{d['candidates']:7d} "
        f"{d['v2_accepted']:8d} "
        f"{(d['v2_accept_rate'] or 0):10.6f} "
        f"{d['near_5m']:7d} "
        f"{d['v2_accepted_near_5m']:8d} "
        f"{(near_acc_rate or 0):10.6f}"
    )


print()
print("=" * 100)
print("DECISION MARGIN AUDIT")
print("=" * 100)

for threshold in [
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    5e-2,
]:

    d = summary[
        "magnitude_boundary"
    ][f"{threshold:.0e}"]

    print(
        f"|mag| < {threshold:.0e}: "
        f"{d['count']:5d} "
        f"({d['rate']:.6%}) | "
        f"near5={d['near_5m']:4d}"
    )


print()
print("=" * 100)
print("DECISION × PLANNING CROSS TABLE")
print("=" * 100)

cross = summary[
    "decision_planning_cross"
]

print(
    "V2 accept & near5     :",
    cross[
        "v2_accept_near5"
    ],
)

print(
    "V2 accept & not near5 :",
    cross[
        "v2_accept_not_near5"
    ],
)

print(
    "V2 reject & near5     :",
    cross[
        "v2_reject_near5"
    ],
)

print(
    "V2 reject & not near5 :",
    cross[
        "v2_reject_not_near5"
    ],
)


# =============================================================================
# Save CSV
# =============================================================================

fieldnames = list(
    records[0].keys()
)

with open(
    CSV_OUT,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        records
    )


# =============================================================================
# Save compact arrays for V2-3B
# =============================================================================

np.savez_compressed(
    NPZ_OUT,

    candidate_idx=np.arange(
        len(records),
        dtype=np.int64,
    ),

    sample_order=sample_order,
    horizon=horizon,
    patch_r=patch_r,
    patch_c=patch_c,

    n_add_cells=n_add_cells,

    planning_used=(
        planning_used.astype(
            np.uint8
        )
    ),

    min_plan_dist_m=(
        min_plan_dist.astype(
            np.float32
        )
    ),

    add_cells_within_5m=(
        add_cells_within_5m
    ),

    near_1m=near_1m.astype(
        np.uint8
    ),

    near_2m=near_2m.astype(
        np.uint8
    ),

    near_3m=near_3m.astype(
        np.uint8
    ),

    near_5m=near_5m.astype(
        np.uint8
    ),

    v1_accept=v1_accept.astype(
        np.uint8
    ),

    v2_mag=mag.astype(
        np.float32
    ),

    v2_accept=v2_accept.astype(
        np.uint8
    ),

    abs_mag=abs_mag.astype(
        np.float32
    ),

    sign_logit=sign_logit.astype(
        np.float32
    ),

    sign_prob=sign_prob.astype(
        np.float32
    ),

    sign_margin=sign_margin.astype(
        np.float32
    ),

    sign_entropy=sign_entropy.astype(
        np.float32
    ),
)


# =============================================================================
# Save summary
# =============================================================================

with open(
    JSON_OUT,
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
    CSV_OUT,
)

print(
    "NPZ :",
    NPZ_OUT,
)

print(
    "JSON:",
    JSON_OUT,
)

print()
print(
    "[V2-3A DECISION-RISK AUDIT] COMPLETE"
)

print("=" * 100)

