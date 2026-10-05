#!/usr/bin/env python3

import os
import glob
import numpy as np
from collections import defaultdict


# ============================================================
# STCV-Occ V2-0
# Multi-scale Metric-aligned Marginal Utility Oracle Audit
#
# Train scenes ONLY.
# No model training.
#
# Utility:
#
#   U_MS(alpha)
#     = alpha * DeltaIoU_30
#       + (1-alpha) * DeltaIoU_100
#
# alpha=0   -> DeltaIoU100
# alpha=1   -> DeltaIoU30
# ============================================================


ROOT = "/root/autodl-tmp/UniV2X"

CACHE_DIR = os.path.join(
    ROOT,
    "G0-G1-G2/cache_full",
)

VALID_FILE = os.path.join(
    ROOT,
    "G0-G1-G2/G3/cache_valid_samples.txt",
)

SEED = 2026
PATCH = 4
GRID = 200

RANGES = {
    "30x30": (70, 130),
    "100x100": (0, 200),
}

ALPHAS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


# ============================================================
# Basic metrics
# ============================================================

def occ_stats(pred, gt, lo, hi):

    p = pred[
        lo:hi,
        lo:hi,
    ].astype(bool)

    g = gt[
        lo:hi,
        lo:hi,
    ].astype(bool)

    tp = int(
        (p & g).sum()
    )

    fp = int(
        (p & (~g)).sum()
    )

    fn = int(
        ((~p) & g).sum()
    )

    return tp, fp, fn


def occ_iou(tp, fp, fn):

    den = tp + fp + fn

    if den == 0:
        return 0.0

    return tp / den


def delta_iou(tp, fp, fn, ba, ha):
    """
    Exact occupied-class IoU marginal change
    for one patch relative to Ego prediction.

    Before:
        IoU = TP / (TP + FP + FN)

    Accept patch:
        TP -> TP + BA
        FN -> FN - BA
        FP -> FP + HA

    Therefore:
        denominator -> D + HA
    """

    D = tp + fp + fn

    if D == 0:
        # Consistent with local occupied-class IoU:
        # if no GT/pred foreground exists, IoU is 0.
        return 0.0

    before = tp / D

    after = (
        (tp + ba)
        /
        (D + ha)
    )

    return after - before


def accumulate(acc, strategy, range_name, pred, gt):

    lo, hi = RANGES[range_name]

    tp, fp, fn = occ_stats(
        pred,
        gt,
        lo,
        hi,
    )

    acc[strategy][range_name]["tp"] += tp
    acc[strategy][range_name]["fp"] += fp
    acc[strategy][range_name]["fn"] += fn


# ============================================================
# Reconstruct exact G3 scene split
# ============================================================

scene_to_samples = defaultdict(list)
valid_sample_ids = set()

with open(VALID_FILE, "r") as f:

    next(f)

    for line in f:

        parts = line.strip().split("\t")

        if len(parts) < 3:
            continue

        sample_idx = parts[0]
        scene_token = parts[1]

        valid_sample_ids.add(
            sample_idx
        )

        scene_to_samples[
            scene_token
        ].append(sample_idx)


scenes = sorted(
    scene_to_samples.keys()
)

np.random.seed(SEED)
np.random.shuffle(scenes)

train_scenes = set(
    scenes[:13]
)

val_scenes = set(
    scenes[13:17]
)

test_scenes = set(
    scenes[17:21]
)


# ============================================================
# Strategies
# ============================================================

alpha_names = {
    a: f"MS_alpha_{a:.2f}"
    for a in ALPHAS
}

STRATEGIES = [
    "Ego",
    "OfficialOR",
    "V1_BAminusHA",
] + [
    alpha_names[a]
    for a in ALPHAS
]


# ============================================================
# Accumulators
# ============================================================

acc = {}

for strategy in STRATEGIES:

    acc[strategy] = {}

    for range_name in RANGES:

        acc[strategy][range_name] = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }


decision_stats = {}

for a in ALPHAS:

    decision_stats[a] = {
        "accepted": 0,
        "flip_vs_v1": 0,
        "flip_vs_d100": 0,
    }


total_candidates = 0
v1_accepted = 0

processed_samples = 0
processed_horizons = 0

zero_den_30 = 0
zero_den_100 = 0

# Distribution audit for raw utility scales
all_du30 = []
all_du100 = []


# ============================================================
# Start
# ============================================================

print("=" * 110)
print(
    "STCV-Occ V2-0: "
    "Multi-scale Metric-aligned Utility Oracle Audit"
)
print("TRAIN SCENES ONLY")
print("=" * 110)

print(
    f"Valid samples : {len(valid_sample_ids)}"
)

print(
    f"Total scenes  : {len(scenes)}"
)

print(
    f"Train scenes  : {len(train_scenes)}"
)

print(
    f"Val scenes    : {len(val_scenes)}"
)

print(
    f"Test scenes   : {len(test_scenes)}"
)

print(
    f"Alphas        : {ALPHAS}"
)


# ============================================================
# Main loop
# ============================================================

cache_files = sorted(
    glob.glob(
        os.path.join(
            CACHE_DIR,
            "sample_*.npz",
        )
    )
)


for path in cache_files:

    with np.load(
        path,
        allow_pickle=True,
    ) as d:

        sample_idx = str(
            d["sample_idx"].item()
        )

        scene_token = str(
            d["scene_token"].item()
        )

        if sample_idx not in valid_sample_ids:
            continue

        if scene_token not in train_scenes:
            continue

        Ov = d["Ov"].astype(bool)
        Oi = d["Oi"].astype(bool)

        Oofficial = d[
            "Oofficial"
        ].astype(bool)

        GT = d["GT"].astype(bool)

        M_gt = d[
            "gt_cell_valid_mask"
        ].astype(bool)

        M_warp_2d = d[
            "warp_valid_mask"
        ].astype(bool)

        M_warp = np.broadcast_to(
            M_warp_2d,
            (5, GRID, GRID),
        )

        # Same candidate domain as V1/G3
        M_valid = (
            M_gt
            & M_warp
        )

        ADD = (
            (~Ov)
            & Oi
            & M_valid
        )

        BA = (
            ADD
            & GT
        )

        HA = (
            ADD
            & (~GT)
        )

        processed_samples += 1


        # ====================================================
        # Five occupancy horizons
        # ====================================================

        for h in range(5):

            ov = Ov[h]
            gt = GT[h]

            add_h = ADD[h]
            ba_h = BA[h]
            ha_h = HA[h]

            # -----------------------------------------------
            # Base Ego metrics: 100m
            # -----------------------------------------------

            tp100, fp100, fn100 = occ_stats(
                ov,
                gt,
                0,
                200,
            )

            if (
                tp100
                + fp100
                + fn100
            ) == 0:
                zero_den_100 += 1

            # -----------------------------------------------
            # Base Ego metrics: 30m
            # -----------------------------------------------

            tp30, fp30, fn30 = occ_stats(
                ov,
                gt,
                70,
                130,
            )

            if (
                tp30
                + fp30
                + fn30
            ) == 0:
                zero_den_30 += 1

            # -----------------------------------------------
            # Strategy predictions
            # -----------------------------------------------

            pred_v1 = ov.copy()

            pred_ms = {
                a: ov.copy()
                for a in ALPHAS
            }


            # ===============================================
            # P4 candidate patches
            # ===============================================

            for pr in range(50):

                y0 = (
                    pr * PATCH
                )

                y1 = (
                    y0 + PATCH
                )

                for pc in range(50):

                    x0 = (
                        pc * PATCH
                    )

                    x1 = (
                        x0 + PATCH
                    )

                    add_patch = add_h[
                        y0:y1,
                        x0:x1,
                    ]

                    if not add_patch.any():
                        continue

                    total_candidates += 1

                    ba_patch = ba_h[
                        y0:y1,
                        x0:x1,
                    ]

                    ha_patch = ha_h[
                        y0:y1,
                        x0:x1,
                    ]

                    # ---------------------------------------
                    # 100m BA / HA
                    # ---------------------------------------

                    ba100 = int(
                        ba_patch.sum()
                    )

                    ha100 = int(
                        ha_patch.sum()
                    )

                    du100 = delta_iou(
                        tp100,
                        fp100,
                        fn100,
                        ba100,
                        ha100,
                    )

                    # ---------------------------------------
                    # 30m BA / HA
                    #
                    # Patch may partially overlap 30m range.
                    # ---------------------------------------

                    iy0 = max(
                        y0,
                        70,
                    )

                    iy1 = min(
                        y1,
                        130,
                    )

                    ix0 = max(
                        x0,
                        70,
                    )

                    ix1 = min(
                        x1,
                        130,
                    )

                    if (
                        iy0 < iy1
                        and ix0 < ix1
                    ):

                        py0 = (
                            iy0 - y0
                        )

                        py1 = (
                            iy1 - y0
                        )

                        px0 = (
                            ix0 - x0
                        )

                        px1 = (
                            ix1 - x0
                        )

                        ba30 = int(
                            ba_patch[
                                py0:py1,
                                px0:px1,
                            ].sum()
                        )

                        ha30 = int(
                            ha_patch[
                                py0:py1,
                                px0:px1,
                            ].sum()
                        )

                        du30 = delta_iou(
                            tp30,
                            fp30,
                            fn30,
                            ba30,
                            ha30,
                        )

                    else:

                        ba30 = 0
                        ha30 = 0
                        du30 = 0.0


                    all_du30.append(
                        du30
                    )

                    all_du100.append(
                        du100
                    )


                    # =======================================
                    # V1 decision
                    # =======================================

                    accept_v1 = (
                        ba100 - ha100
                    ) > 0

                    if accept_v1:

                        v1_accepted += 1

                        region = pred_v1[
                            y0:y1,
                            x0:x1,
                        ]

                        region[
                            add_patch
                        ] = True


                    # =======================================
                    # DeltaIoU100 reference
                    # alpha = 0
                    # =======================================

                    accept_d100 = (
                        du100 > 0.0
                    )


                    # =======================================
                    # Multi-scale decisions
                    # =======================================

                    for a in ALPHAS:

                        utility_ms = (
                            a * du30
                            +
                            (1.0 - a)
                            * du100
                        )

                        accept_ms = (
                            utility_ms > 0.0
                        )

                        if accept_ms:

                            decision_stats[
                                a
                            ]["accepted"] += 1

                            region = pred_ms[a][
                                y0:y1,
                                x0:x1,
                            ]

                            region[
                                add_patch
                            ] = True


                        if (
                            accept_ms
                            != accept_v1
                        ):

                            decision_stats[
                                a
                            ]["flip_vs_v1"] += 1


                        if (
                            accept_ms
                            != accept_d100
                        ):

                            decision_stats[
                                a
                            ]["flip_vs_d100"] += 1


            # ===============================================
            # Accumulate dataset-level official-style
            # occupied-class TP/FP/FN
            # ===============================================

            for r in RANGES:

                accumulate(
                    acc,
                    "Ego",
                    r,
                    ov,
                    gt,
                )

                accumulate(
                    acc,
                    "OfficialOR",
                    r,
                    Oofficial[h],
                    gt,
                )

                accumulate(
                    acc,
                    "V1_BAminusHA",
                    r,
                    pred_v1,
                    gt,
                )

                for a in ALPHAS:

                    accumulate(
                        acc,
                        alpha_names[a],
                        r,
                        pred_ms[a],
                        gt,
                    )

            processed_horizons += 1


# ============================================================
# Utility scale audit
# ============================================================

all_du30 = np.asarray(
    all_du30,
    dtype=np.float64,
)

all_du100 = np.asarray(
    all_du100,
    dtype=np.float64,
)


print()
print("=" * 110)
print("DATA AUDIT")
print("=" * 110)

print(
    f"Processed train samples : "
    f"{processed_samples}"
)

print(
    f"Processed horizons      : "
    f"{processed_horizons}"
)

print(
    f"Candidate P4 patches    : "
    f"{total_candidates}"
)

print(
    f"V1 accepted             : "
    f"{v1_accepted} "
    f"({v1_accepted / total_candidates:.6f})"
)

print(
    f"Zero denominator 30m    : "
    f"{zero_den_30}"
)

print(
    f"Zero denominator 100m   : "
    f"{zero_den_100}"
)


print()
print("=" * 110)
print("RAW DeltaIoU SCALE AUDIT")
print("=" * 110)

for name, arr in [
    ("DeltaIoU30", all_du30),
    ("DeltaIoU100", all_du100),
]:

    abs_arr = np.abs(arr)

    print()
    print(name)

    print(
        f"  mean             : "
        f"{arr.mean():.10f}"
    )

    print(
        f"  std              : "
        f"{arr.std():.10f}"
    )

    print(
        f"  median |delta|   : "
        f"{np.median(abs_arr):.10f}"
    )

    print(
        f"  p90 |delta|      : "
        f"{np.percentile(abs_arr, 90):.10f}"
    )

    print(
        f"  p99 |delta|      : "
        f"{np.percentile(abs_arr, 99):.10f}"
    )

    print(
        f"  min/max          : "
        f"{arr.min():.10f} / "
        f"{arr.max():.10f}"
    )


# ============================================================
# Decision audit
# ============================================================

print()
print("=" * 110)
print("MULTI-SCALE PATCH DECISION AUDIT")
print("=" * 110)

print(
    "alpha | accepted | accept_rate | "
    "flip_vs_V1 | flip_vs_Delta100"
)

print("-" * 80)


for a in ALPHAS:

    s = decision_stats[a]

    print(
        f"{a:5.2f} | "
        f"{s['accepted']:8d} | "
        f"{s['accepted'] / total_candidates:11.6f} | "
        f"{s['flip_vs_v1'] / total_candidates:11.6f} | "
        f"{s['flip_vs_d100'] / total_candidates:16.6f}"
    )


# ============================================================
# Dataset-level IoU
# ============================================================

results = {}

print()
print("=" * 110)
print("TRAIN-SPLIT DATASET-LEVEL OCCUPIED-CLASS IoU")
print("=" * 110)

header = (
    f"{'Strategy':24s} "
    f"{'30x30 IoU':>12s} "
    f"{'100x100 IoU':>14s} "
    f"{'Mean':>12s}"
)

print(header)
print("-" * len(header))


for strategy in STRATEGIES:

    vals = {}

    for r in [
        "30x30",
        "100x100",
    ]:

        s = acc[
            strategy
        ][r]

        vals[r] = occ_iou(
            s["tp"],
            s["fp"],
            s["fn"],
        )

    mean_iou = (
        vals["30x30"]
        + vals["100x100"]
    ) / 2.0

    vals["mean"] = mean_iou

    results[
        strategy
    ] = vals

    print(
        f"{strategy:24s} "
        f"{vals['30x30']:12.6f} "
        f"{vals['100x100']:14.6f} "
        f"{mean_iou:12.6f}"
    )


# ============================================================
# Gains
# ============================================================

print()
print("=" * 110)
print("GAIN OVER V1 P4 ORACLE")
print("=" * 110)

v1 = results[
    "V1_BAminusHA"
]


for a in ALPHAS:

    name = alpha_names[a]

    r = results[name]

    print()
    print(
        f"alpha = {a:.2f}"
    )

    print(
        f"  30x30   : "
        f"{r['30x30'] - v1['30x30']:+.6f}"
    )

    print(
        f"  100x100 : "
        f"{r['100x100'] - v1['100x100']:+.6f}"
    )

    print(
        f"  Mean     : "
        f"{r['mean'] - v1['mean']:+.6f}"
    )


# ============================================================
# Compare against alpha=0 = DeltaIoU100
# ============================================================

print()
print("=" * 110)
print("GAIN OVER SINGLE-SCALE DeltaIoU100 (alpha=0)")
print("=" * 110)

base = results[
    alpha_names[0.00]
]


for a in ALPHAS[1:]:

    name = alpha_names[a]

    r = results[name]

    print()
    print(
        f"alpha = {a:.2f}"
    )

    print(
        f"  30x30   : "
        f"{r['30x30'] - base['30x30']:+.6f}"
    )

    print(
        f"  100x100 : "
        f"{r['100x100'] - base['100x100']:+.6f}"
    )

    print(
        f"  Mean     : "
        f"{r['mean'] - base['mean']:+.6f}"
    )


# ============================================================
# Diagnostic-only best alpha
#
# IMPORTANT:
# This does NOT select the final training alpha.
# ============================================================

best_alpha = max(
    ALPHAS,
    key=lambda a: results[
        alpha_names[a]
    ]["mean"],
)

best = results[
    alpha_names[
        best_alpha
    ]
]


print()
print("=" * 110)
print("TRAIN-ONLY DIAGNOSTIC SUMMARY")
print("=" * 110)

print(
    f"Best alpha by simple mean IoU "
    f"(diagnostic only): "
    f"{best_alpha:.2f}"
)

print(
    f"30x30 IoU : "
    f"{best['30x30']:.6f}"
)

print(
    f"100x100 IoU: "
    f"{best['100x100']:.6f}"
)

print(
    f"Mean IoU   : "
    f"{best['mean']:.6f}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "This Train-only sweep is an Oracle design audit."
)

print(
    "Do NOT freeze alpha from this result for the final model."
)

print(
    "Final alpha, if needed, must be selected using internal validation only."
)

print(
    "No model was trained."
)

print()
print(
    "Multi-scale DeltaIoU Oracle audit finished."
)
