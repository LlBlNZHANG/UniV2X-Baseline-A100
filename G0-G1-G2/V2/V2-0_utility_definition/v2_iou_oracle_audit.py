#!/usr/bin/env python3

import os
import glob
import numpy as np
from collections import defaultdict


ROOT = "/root/autodl-tmp/UniV2X"
CACHE_DIR = os.path.join(ROOT, "G0-G1-G2/cache_full")
VALID_FILE = os.path.join(ROOT, "G0-G1-G2/G3/cache_valid_samples.txt")

SEED = 2026
PATCH = 4
GRID = 200

RANGES = {
    "30x30": (70, 130),
    "100x100": (0, 200),
}

STRATEGIES = [
    "Ego",
    "OfficialOR",
    "V1_BAminusHA",
    "V2_DeltaIoU100",
    "Ref_DeltaIoU30",
]


# ============================================================
# Helpers
# ============================================================

def occ_stats(pred, gt, lo, hi):
    pred = pred[lo:hi, lo:hi].astype(bool)
    gt = gt[lo:hi, lo:hi].astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    return int(tp), int(fp), int(fn)


def occ_iou(tp, fp, fn):
    den = tp + fp + fn
    if den == 0:
        return 0.0
    return tp / den


def delta_iou(base_tp, base_fp, base_fn, ba, ha):
    """
    Exact local occupied-class IoU change caused by accepting additions.

    Before:
        IoU = TP / (TP + FP + FN)

    After accepting patch:
        TP -> TP + BA
        FN -> FN - BA
        FP -> FP + HA

    Therefore denominator:
        D -> D + HA
    """
    D = base_tp + base_fp + base_fn

    if D == 0:
        # If BA > 0 then GT itself would contribute FN, hence D should not be 0.
        # Pure HA additions from an empty scene keep occupied IoU at 0.
        return 0.0

    before = base_tp / D
    after = (base_tp + ba) / (D + ha)

    return after - before


def accumulate(acc, strategy, range_name, pred, gt):
    lo, hi = RANGES[range_name]
    tp, fp, fn = occ_stats(pred, gt, lo, hi)

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

        valid_sample_ids.add(sample_idx)
        scene_to_samples[scene_token].append(sample_idx)


scenes = sorted(scene_to_samples.keys())

np.random.seed(SEED)
np.random.shuffle(scenes)

train_scenes = set(scenes[:13])
val_scenes = set(scenes[13:17])
test_scenes = set(scenes[17:21])


print("=" * 100)
print("STCV-Occ V2-0: Metric-aligned Utility Oracle Audit")
print("TRAIN SCENES ONLY")
print("=" * 100)

print(f"Total valid samples : {len(valid_sample_ids)}")
print(f"Total scenes        : {len(scenes)}")
print(f"Train scenes        : {len(train_scenes)}")
print(f"Val scenes          : {len(val_scenes)}")
print(f"Test scenes         : {len(test_scenes)}")


# ============================================================
# Accumulators
# ============================================================

acc = {}

for strategy in STRATEGIES:
    acc[strategy] = {}

    for r in RANGES:
        acc[strategy][r] = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }


patch_stats = {
    "total": 0,
    "v1_accept": 0,
    "d100_accept": 0,
    "d30_accept": 0,
    "v1_vs_d100_flip": 0,
    "v1_vs_d30_flip": 0,
}

processed_samples = 0
processed_horizons = 0

gt_mask_all_one_samples = 0
gt_mask_not_all_one_samples = 0


# ============================================================
# Main
# ============================================================

cache_files = sorted(
    glob.glob(os.path.join(CACHE_DIR, "sample_*.npz"))
)

for path in cache_files:

    with np.load(path, allow_pickle=True) as d:

        sample_idx = str(d["sample_idx"].item())
        scene_token = str(d["scene_token"].item())

        if sample_idx not in valid_sample_ids:
            continue

        if scene_token not in train_scenes:
            continue

        Ov = d["Ov"].astype(bool)
        Oi = d["Oi"].astype(bool)
        Oofficial = d["Oofficial"].astype(bool)
        GT = d["GT"].astype(bool)

        M_gt = d["gt_cell_valid_mask"].astype(bool)
        M_warp_2d = d["warp_valid_mask"].astype(bool)

        if M_gt.all():
            gt_mask_all_one_samples += 1
        else:
            gt_mask_not_all_one_samples += 1

        M_warp = np.broadcast_to(
            M_warp_2d,
            (5, GRID, GRID)
        )

        M_valid = M_gt & M_warp

        ADD = (~Ov) & Oi & M_valid
        BA = ADD & GT
        HA = ADD & (~GT)

        processed_samples += 1

        for h in range(5):

            ov = Ov[h]
            gt = GT[h]

            # ------------------------------------------------
            # Baseline TP / FP / FN for local marginal utility
            # ------------------------------------------------

            tp100, fp100, fn100 = occ_stats(
                ov, gt, 0, 200
            )

            tp30, fp30, fn30 = occ_stats(
                ov, gt, 70, 130
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            pred_v1 = ov.copy()
            pred_d100 = ov.copy()
            pred_d30 = ov.copy()

            add_h = ADD[h]
            ba_h = BA[h]
            ha_h = HA[h]

            # ------------------------------------------------
            # P4 candidate patches
            # ------------------------------------------------

            for pr in range(50):
                y0 = pr * PATCH
                y1 = y0 + PATCH

                for pc in range(50):
                    x0 = pc * PATCH
                    x1 = x0 + PATCH

                    add_patch = add_h[y0:y1, x0:x1]

                    if not add_patch.any():
                        continue

                    patch_stats["total"] += 1

                    ba_patch = ba_h[y0:y1, x0:x1]
                    ha_patch = ha_h[y0:y1, x0:x1]

                    ba100 = int(ba_patch.sum())
                    ha100 = int(ha_patch.sum())

                    # ========================================
                    # V1
                    # ========================================

                    accept_v1 = (
                        ba100 - ha100
                    ) > 0

                    if accept_v1:
                        patch_stats["v1_accept"] += 1

                        region = pred_v1[y0:y1, x0:x1]
                        region[add_patch] = True

                    # ========================================
                    # V2 candidate:
                    # exact local ΔIoU over 100x100 scope
                    # ========================================

                    du100 = delta_iou(
                        tp100,
                        fp100,
                        fn100,
                        ba100,
                        ha100,
                    )

                    accept_d100 = du100 > 0.0

                    if accept_d100:
                        patch_stats["d100_accept"] += 1

                        region = pred_d100[y0:y1, x0:x1]
                        region[add_patch] = True

                    if accept_d100 != accept_v1:
                        patch_stats["v1_vs_d100_flip"] += 1

                    # ========================================
                    # Reference:
                    # exact local ΔIoU over 30x30 scope
                    # ========================================

                    iy0 = max(y0, 70)
                    iy1 = min(y1, 130)

                    ix0 = max(x0, 70)
                    ix1 = min(x1, 130)

                    if iy0 < iy1 and ix0 < ix1:

                        py0 = iy0 - y0
                        py1 = iy1 - y0

                        px0 = ix0 - x0
                        px1 = ix1 - x0

                        ba30 = int(
                            ba_patch[
                                py0:py1,
                                px0:px1
                            ].sum()
                        )

                        ha30 = int(
                            ha_patch[
                                py0:py1,
                                px0:px1
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
                        du30 = 0.0

                    accept_d30 = du30 > 0.0

                    if accept_d30:
                        patch_stats["d30_accept"] += 1

                        region = pred_d30[y0:y1, x0:x1]
                        region[add_patch] = True

                    if accept_d30 != accept_v1:
                        patch_stats["v1_vs_d30_flip"] += 1

            # ------------------------------------------------
            # Accumulate EXACT dataset-level occupied IoU stats
            # ------------------------------------------------

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

                accumulate(
                    acc,
                    "V2_DeltaIoU100",
                    r,
                    pred_d100,
                    gt,
                )

                accumulate(
                    acc,
                    "Ref_DeltaIoU30",
                    r,
                    pred_d30,
                    gt,
                )

            processed_horizons += 1


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 100)
print("DATA AUDIT")
print("=" * 100)

print(f"Processed train samples       : {processed_samples}")
print(f"Processed horizons            : {processed_horizons}")
print(f"M_gt all-one samples          : {gt_mask_all_one_samples}")
print(f"M_gt non-all-one samples      : {gt_mask_not_all_one_samples}")


print("\n" + "=" * 100)
print("PATCH DECISION AUDIT")
print("=" * 100)

total = patch_stats["total"]

print(f"Candidate patches             : {total}")

for key in [
    "v1_accept",
    "d100_accept",
    "d30_accept",
]:
    n = patch_stats[key]

    print(
        f"{key:28s}: "
        f"{n:8d} "
        f"({n / total:.4f})"
    )

for key in [
    "v1_vs_d100_flip",
    "v1_vs_d30_flip",
]:
    n = patch_stats[key]

    print(
        f"{key:28s}: "
        f"{n:8d} "
        f"({n / total:.4f})"
    )


print("\n" + "=" * 100)
print("TRAIN-SPLIT DATASET-LEVEL OCCUPIED-CLASS IoU")
print("=" * 100)

header = (
    f"{'Strategy':24s} "
    f"{'30x30 IoU':>12s} "
    f"{'100x100 IoU':>14s}"
)

print(header)
print("-" * len(header))

results = {}

for strategy in STRATEGIES:

    results[strategy] = {}

    vals = []

    for r in ["30x30", "100x100"]:

        s = acc[strategy][r]

        iou = occ_iou(
            s["tp"],
            s["fp"],
            s["fn"],
        )

        results[strategy][r] = iou
        vals.append(iou)

    print(
        f"{strategy:24s} "
        f"{vals[0]:12.6f} "
        f"{vals[1]:14.6f}"
    )


print("\n" + "=" * 100)
print("GAIN OVER V1 ORACLE")
print("=" * 100)

for strategy in [
    "V2_DeltaIoU100",
    "Ref_DeltaIoU30",
]:

    print(f"\n{strategy}")

    for r in ["30x30", "100x100"]:

        gain = (
            results[strategy][r]
            - results["V1_BAminusHA"][r]
        )

        print(
            f"  {r:8s}: "
            f"{gain:+.6f}"
        )


print("\nInterpretation:")
print(
    "1. V1_BAminusHA reproduces the original P4 decision principle."
)

print(
    "2. V2_DeltaIoU100 uses exact local occupied-IoU marginal utility."
)

print(
    "3. Ref_DeltaIoU30 is diagnostic only; it is NOT yet a proposed final target."
)

print(
    "4. No model is trained in this script."
)

print("\nV2 IoU oracle audit finished.")
