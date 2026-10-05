#!/usr/bin/env python3

import os
import glob
import numpy as np
from collections import defaultdict


# ============================================================
# STCV-Occ V2-1A
# Continuous DeltaIoU100 Target Distribution / Scale Audit
#
# Purpose:
#   1. Reconstruct exact G3 Train / Val / Test split
#   2. Build continuous target:
#
#        y_R = DeltaIoU100_R
#
#   3. Audit:
#      - positive / zero / negative proportions
#      - magnitude distribution
#      - horizon distribution
#      - V1 vs V2 decision changes
#      - potential target scaling
#
# No model training.
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


# ============================================================
# Metric helpers
# ============================================================

def occ_stats(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = int((pred & gt).sum())
    fp = int((pred & (~gt)).sum())
    fn = int(((~pred) & gt).sum())

    return tp, fp, fn


def delta_iou(tp, fp, fn, ba, ha):
    """
    Per-sample / per-horizon occupied-class IoU marginal utility.

    Ego:
        IoU = TP / (TP + FP + FN)

    Accept patch:
        TP' = TP + BA
        FP' = FP + HA
        FN' = FN - BA

    Therefore:
        IoU' = (TP + BA) / (TP + FP + FN + HA)
    """

    D = tp + fp + fn

    if D == 0:
        return 0.0

    before = tp / D
    after = (tp + ba) / (D + ha)

    return after - before


# ============================================================
# Exact G3 scene split reconstruction
# ============================================================

scene_to_samples = defaultdict(list)
valid_sample_ids = set()

with open(VALID_FILE, "r") as f:

    header = next(f, None)

    for line in f:

        parts = line.strip().split("\t")

        if len(parts) < 3:
            continue

        sample_idx = parts[0]
        scene_token = parts[1]

        valid_sample_ids.add(sample_idx)
        scene_to_samples[scene_token].append(sample_idx)


scenes = sorted(scene_to_samples.keys())

np.random.seed(SEED)
np.random.shuffle(scenes)

train_scenes = set(scenes[:13])
val_scenes = set(scenes[13:17])
test_scenes = set(scenes[17:21])


def get_split(scene_token):

    if scene_token in train_scenes:
        return "train"

    if scene_token in val_scenes:
        return "val"

    if scene_token in test_scenes:
        return "test"

    return None


# ============================================================
# Storage
# ============================================================

splits = ["train", "val", "test"]

records = {
    s: {
        "target": [],
        "v1_utility": [],
        "ba": [],
        "ha": [],
        "horizon": [],
    }
    for s in splits
}

sample_count = {
    s: 0
    for s in splits
}

horizon_count = {
    s: 0
    for s in splits
}

zero_den = {
    s: 0
    for s in splits
}


# ============================================================
# Main extraction
# ============================================================

cache_files = sorted(
    glob.glob(
        os.path.join(
            CACHE_DIR,
            "sample_*.npz",
        )
    )
)

print("=" * 110)
print("STCV-Occ V2-1A: DeltaIoU100 Continuous Target Audit")
print("=" * 110)

print(f"Total valid samples : {len(valid_sample_ids)}")
print(f"Total scenes        : {len(scenes)}")
print(f"Train scenes        : {len(train_scenes)}")
print(f"Val scenes          : {len(val_scenes)}")
print(f"Test scenes         : {len(test_scenes)}")


for path in cache_files:

    with np.load(
        path,
        allow_pickle=True,
    ) as d:

        sample_idx = str(
            d["sample_idx"].item()
        )

        if sample_idx not in valid_sample_ids:
            continue

        scene_token = str(
            d["scene_token"].item()
        )

        split = get_split(scene_token)

        if split is None:
            continue

        Ov = d["Ov"].astype(bool)
        Oi = d["Oi"].astype(bool)
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

        # Same candidate definition as G3 / V1
        M_valid = M_gt & M_warp

        ADD = (
            (~Ov)
            & Oi
            & M_valid
        )

        BA = ADD & GT
        HA = ADD & (~GT)

        sample_count[split] += 1


        # ====================================================
        # Horizon
        # ====================================================

        for h in range(5):

            ov = Ov[h]
            gt = GT[h]

            add_h = ADD[h]
            ba_h = BA[h]
            ha_h = HA[h]

            tp, fp, fn = occ_stats(
                ov,
                gt,
            )

            if tp + fp + fn == 0:
                zero_den[split] += 1

            horizon_count[split] += 1


            # ===============================================
            # P4 candidates
            # ===============================================

            for pr in range(50):

                y0 = pr * PATCH
                y1 = y0 + PATCH

                for pc in range(50):

                    x0 = pc * PATCH
                    x1 = x0 + PATCH

                    add_patch = add_h[
                        y0:y1,
                        x0:x1,
                    ]

                    if not add_patch.any():
                        continue

                    ba = int(
                        ba_h[
                            y0:y1,
                            x0:x1,
                        ].sum()
                    )

                    ha = int(
                        ha_h[
                            y0:y1,
                            x0:x1,
                        ].sum()
                    )

                    u_v1 = ba - ha

                    y = delta_iou(
                        tp,
                        fp,
                        fn,
                        ba,
                        ha,
                    )

                    records[
                        split
                    ]["target"].append(y)

                    records[
                        split
                    ]["v1_utility"].append(u_v1)

                    records[
                        split
                    ]["ba"].append(ba)

                    records[
                        split
                    ]["ha"].append(ha)

                    records[
                        split
                    ]["horizon"].append(h)


# ============================================================
# Convert arrays
# ============================================================

for split in splits:

    for key in records[split]:

        records[split][key] = np.asarray(
            records[split][key]
        )


# ============================================================
# Distribution helpers
# ============================================================

def safe_ratio(a, b):

    if b == 0:
        return 0.0

    return a / b


def print_quantiles(arr):

    qs = [
        0,
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        100,
    ]

    for q in qs:

        print(
            f"  p{q:>3} : "
            f"{np.percentile(arr, q): .10f}"
        )


# ============================================================
# Split audit
# ============================================================

for split in splits:

    y = records[split]["target"].astype(np.float64)
    u = records[split]["v1_utility"].astype(np.float64)

    n = len(y)

    v1_pos = u > 0
    v2_pos = y > 0

    pos = int((y > 0).sum())
    neg = int((y < 0).sum())
    zero = int((y == 0).sum())

    v1_pos_v2_pos = int(
        (v1_pos & v2_pos).sum()
    )

    v1_pos_v2_nonpos = int(
        (v1_pos & (~v2_pos)).sum()
    )

    v1_nonpos_v2_pos = int(
        ((~v1_pos) & v2_pos).sum()
    )

    both_nonpos = int(
        ((~v1_pos) & (~v2_pos)).sum()
    )

    flip = int(
        (v1_pos != v2_pos).sum()
    )

    print()
    print("=" * 110)
    print(f"{split.upper()} SPLIT")
    print("=" * 110)

    print(
        f"Samples               : "
        f"{sample_count[split]}"
    )

    print(
        f"Horizons              : "
        f"{horizon_count[split]}"
    )

    print(
        f"Zero IoU denominator  : "
        f"{zero_den[split]}"
    )

    print(
        f"Candidate patches     : "
        f"{n}"
    )

    print()
    print("SIGN DISTRIBUTION")
    print("-" * 70)

    print(
        f"Positive DeltaIoU     : "
        f"{pos:8d} "
        f"({safe_ratio(pos, n):.6f})"
    )

    print(
        f"Zero DeltaIoU         : "
        f"{zero:8d} "
        f"({safe_ratio(zero, n):.6f})"
    )

    print(
        f"Negative DeltaIoU     : "
        f"{neg:8d} "
        f"({safe_ratio(neg, n):.6f})"
    )

    print()
    print("RAW TARGET STATISTICS")
    print("-" * 70)

    print(
        f"mean                  : "
        f"{y.mean(): .10f}"
    )

    print(
        f"std                   : "
        f"{y.std(): .10f}"
    )

    print(
        f"min                   : "
        f"{y.min(): .10f}"
    )

    print(
        f"max                   : "
        f"{y.max(): .10f}"
    )

    print(
        f"mean |target|         : "
        f"{np.abs(y).mean(): .10f}"
    )

    print(
        f"median |target|       : "
        f"{np.median(np.abs(y)): .10f}"
    )

    print()
    print("RAW TARGET QUANTILES")
    print("-" * 70)

    print_quantiles(y)

    print()
    print("ABSOLUTE TARGET QUANTILES")
    print("-" * 70)

    print_quantiles(
        np.abs(y)
    )

    print()
    print("TAIL AUDIT")
    print("-" * 70)

    for threshold in [
        0.001,
        0.0025,
        0.005,
        0.010,
        0.020,
        0.050,
    ]:

        count = int(
            (
                np.abs(y)
                >= threshold
            ).sum()
        )

        print(
            f"|target| >= {threshold:7.4f} : "
            f"{count:8d} "
            f"({safe_ratio(count, n):.6f})"
        )

    print()
    print("V1 -> V2 DECISION AUDIT")
    print("-" * 70)

    print(
        f"V1 positive           : "
        f"{int(v1_pos.sum()):8d} "
        f"({v1_pos.mean():.6f})"
    )

    print(
        f"V2 positive           : "
        f"{int(v2_pos.sum()):8d} "
        f"({v2_pos.mean():.6f})"
    )

    print(
        f"V1+ / V2+             : "
        f"{v1_pos_v2_pos:8d}"
    )

    print(
        f"V1+ / V2<=0           : "
        f"{v1_pos_v2_nonpos:8d}"
    )

    print(
        f"V1<=0 / V2+           : "
        f"{v1_nonpos_v2_pos:8d}"
    )

    print(
        f"V1<=0 / V2<=0         : "
        f"{both_nonpos:8d}"
    )

    print(
        f"Decision flips        : "
        f"{flip:8d} "
        f"({safe_ratio(flip, n):.6f})"
    )


    # ========================================================
    # Horizon audit
    # ========================================================

    print()
    print("HORIZON TARGET AUDIT")
    print("-" * 100)

    print(
        "h | patches | positive | pos_rate | "
        "mean_target | std_target | median_abs | p90_abs"
    )

    print("-" * 100)

    horizons = records[
        split
    ]["horizon"]

    for h in range(5):

        mask = horizons == h

        yh = y[mask]

        if len(yh) == 0:
            continue

        h_pos = int(
            (yh > 0).sum()
        )

        print(
            f"{h:d} | "
            f"{len(yh):7d} | "
            f"{h_pos:8d} | "
            f"{safe_ratio(h_pos, len(yh)):8.6f} | "
            f"{yh.mean(): .8f} | "
            f"{yh.std(): .8f} | "
            f"{np.median(np.abs(yh)): .8f} | "
            f"{np.percentile(np.abs(yh), 90): .8f}"
        )


    # ========================================================
    # Candidate target scaling diagnostics
    # ========================================================

    print()
    print("TARGET SCALING DIAGNOSTIC")
    print("-" * 100)

    print(
        "scale | std | median_abs | p90_abs | "
        "p99_abs | min | max"
    )

    print("-" * 100)

    for scale in [
        1.0,
        10.0,
        100.0,
        1000.0,
    ]:

        ys = y * scale
        abs_ys = np.abs(ys)

        print(
            f"{scale:7.1f} | "
            f"{ys.std(): .6f} | "
            f"{np.median(abs_ys): .6f} | "
            f"{np.percentile(abs_ys, 90): .6f} | "
            f"{np.percentile(abs_ys, 99): .6f} | "
            f"{ys.min(): .6f} | "
            f"{ys.max(): .6f}"
        )


# ============================================================
# Global split comparison
# ============================================================

print()
print("=" * 110)
print("TRAIN / VAL / TEST SUMMARY")
print("=" * 110)

print(
    "split | candidates | positive_rate | "
    "mean | std | median_abs | p90_abs | p99_abs"
)

print("-" * 110)

for split in splits:

    y = records[
        split
    ]["target"].astype(np.float64)

    print(
        f"{split:5s} | "
        f"{len(y):10d} | "
        f"{(y > 0).mean():13.6f} | "
        f"{y.mean(): .8f} | "
        f"{y.std(): .8f} | "
        f"{np.median(np.abs(y)): .8f} | "
        f"{np.percentile(np.abs(y), 90): .8f} | "
        f"{np.percentile(np.abs(y), 99): .8f}"
    )


# ============================================================
# Save compact target audit arrays
# ============================================================

OUT_NPZ = os.path.join(
    ROOT,
    "G0-G1-G2/V2/V2-1_magnitude_learning/"
    "v2_1a_target_audit_arrays.npz",
)

np.savez_compressed(
    OUT_NPZ,

    train_target=records["train"]["target"],
    train_v1_utility=records["train"]["v1_utility"],
    train_ba=records["train"]["ba"],
    train_ha=records["train"]["ha"],
    train_horizon=records["train"]["horizon"],

    val_target=records["val"]["target"],
    val_v1_utility=records["val"]["v1_utility"],
    val_ba=records["val"]["ba"],
    val_ha=records["val"]["ha"],
    val_horizon=records["val"]["horizon"],

    test_target=records["test"]["target"],
    test_v1_utility=records["test"]["v1_utility"],
    test_ba=records["test"]["ba"],
    test_ha=records["test"]["ha"],
    test_horizon=records["test"]["horizon"],
)

print()
print("=" * 110)
print("OUTPUT")
print("=" * 110)

print(
    f"Saved audit arrays: {OUT_NPZ}"
)

print()
print(
    "V2-1A target audit finished."
)
