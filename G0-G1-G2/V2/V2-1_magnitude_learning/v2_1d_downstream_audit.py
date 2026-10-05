#!/usr/bin/env python3

import os
import glob
import numpy as np
from collections import defaultdict

ROOT = "/root/autodl-tmp/UniV2X"

CACHE_DIR = os.path.join(
    ROOT, "G0-G1-G2/cache_full"
)

VALID_FILE = os.path.join(
    ROOT, "G0-G1-G2/G3/cache_valid_samples.txt"
)

TARGET_FILE = os.path.join(
    ROOT,
    "G0-G1-G2/V2/V2-1_magnitude_learning/"
    "v2_1a_target_audit_arrays.npz"
)

RESULT_ROOT = os.path.join(
    ROOT,
    "G0-G1-G2/V2/V2-1_magnitude_learning/results"
)

PATCH = 4
GRID = 200
SEED = 2026

VARIANTS = {
    "SmoothL1": "main_lambda_0p50",
    "MSE": "loss_mse_lambda_0p50",
    "WeightedSmoothL1": "loss_weighted_smoothl1_lambda_0p50",
}

RANGES = {
    "30m": (70, 130),
    "100m": (0, 200),
}


def occ_stats(pred, gt_raw, lo, hi):

    p = pred[lo:hi, lo:hi].astype(bool)
    g_raw = gt_raw[lo:hi, lo:hi]

    valid = (g_raw == 0) | (g_raw == 1)
    g = (g_raw == 1)

    tp = int((p & g & valid).sum())
    fp = int((p & (~g) & valid).sum())
    fn = int(((~p) & g & valid).sum())

    return tp, fp, fn


def iou(tp, fp, fn):

    d = tp + fp + fn

    return (
        tp / d
        if d > 0
        else 0.0
    )


def delta_iou(tp, fp, fn, ba, ha):

    d = tp + fp + fn

    if d == 0:
        return 0.0

    return (
        (tp + ba) / (d + ha)
        - tp / d
    )


def apply_patch(
    pred,
    add_patch,
    y0,
    y1,
    x0,
    x1,
):

    region = pred[
        y0:y1,
        x0:x1,
    ]

    region[add_patch] = True


# ============================================================
# Exact scene split
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

val_scenes = set(
    scenes[13:17]
)


# ============================================================
# Targets + predictions
# ============================================================

target_data = np.load(
    TARGET_FILE,
    allow_pickle=True,
)

val_target = target_data[
    "val_target"
].astype(np.float64)

assert len(val_target) == 18555


predictions = {}

for name, dirname in VARIANTS.items():

    p = np.load(
        os.path.join(
            RESULT_ROOT,
            dirname,
            "val_predictions.npz",
        ),
        allow_pickle=True,
    )

    mag = p[
        "magnitude_pred"
    ].astype(np.float64)

    raw = p[
        "raw_delta_iou"
    ].astype(np.float64)

    assert len(mag) == len(val_target)

    if not np.allclose(
        raw,
        val_target,
        atol=1e-8,
        rtol=0,
    ):
        raise RuntimeError(
            f"{name}: target ordering mismatch"
        )

    predictions[name] = mag


# ============================================================
# Accumulators
# ============================================================

STRATEGIES = [
    "Ego",
    "OfficialOR",
    "V1_Oracle",
    "V2_Oracle",
] + list(VARIANTS.keys())


acc = {
    s: {
        r: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for r in RANGES
    }
    for s in STRATEGIES
}


accept_count = {
    name: 0
    for name in VARIANTS
}


def accumulate(
    strategy,
    pred,
    gt,
):

    for r, (lo, hi) in RANGES.items():

        tp, fp, fn = occ_stats(
            pred,
            gt,
            lo,
            hi,
        )

        acc[strategy][r]["tp"] += tp
        acc[strategy][r]["fp"] += fp
        acc[strategy][r]["fn"] += fn


# ============================================================
# Reconstruct validation candidates
# ============================================================

candidate_idx = 0
val_samples = 0

for path in sorted(
    glob.glob(
        os.path.join(
            CACHE_DIR,
            "sample_*.npz",
        )
    )
):

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

        if scene_token not in val_scenes:
            continue

        val_samples += 1

        Ov = d["Ov"].astype(bool)
        Oi = d["Oi"].astype(bool)
        Oofficial = d["Oofficial"].astype(bool)
        GT = d["GT"].astype(np.int16)

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

        M_valid = M_gt & M_warp

        ADD = (
            (~Ov)
            & Oi
            & M_valid
        )

        BA = (
            ADD
            & (GT == 1)
        )

        HA = (
            ADD
            & (GT == 0)
        )


        for h in range(5):

            ov = Ov[h]
            gt = GT[h]

            add_h = ADD[h]
            ba_h = BA[h]
            ha_h = HA[h]

            tp100, fp100, fn100 = occ_stats(
                ov,
                gt,
                0,
                200,
            )

            pred_v1 = ov.copy()
            pred_oracle = ov.copy()

            pred_learned = {
                name: ov.copy()
                for name in VARIANTS
            }


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

                    u_v2 = delta_iou(
                        tp100,
                        fp100,
                        fn100,
                        ba,
                        ha,
                    )


                    if not np.isclose(
                        u_v2,
                        val_target[candidate_idx],
                        atol=1e-8,
                        rtol=0,
                    ):
                        raise RuntimeError(
                            f"Candidate ordering mismatch "
                            f"at {candidate_idx}"
                        )


                    if u_v1 > 0:

                        apply_patch(
                            pred_v1,
                            add_patch,
                            y0,
                            y1,
                            x0,
                            x1,
                        )


                    if u_v2 > 0:

                        apply_patch(
                            pred_oracle,
                            add_patch,
                            y0,
                            y1,
                            x0,
                            x1,
                        )


                    for name in VARIANTS:

                        # Primary V2-1 rule
                        # magnitude prediction > 0
                        if (
                            predictions[name][
                                candidate_idx
                            ]
                            > 0.0
                        ):

                            accept_count[
                                name
                            ] += 1

                            apply_patch(
                                pred_learned[name],
                                add_patch,
                                y0,
                                y1,
                                x0,
                                x1,
                            )


                    candidate_idx += 1


            accumulate(
                "Ego",
                ov,
                gt,
            )

            accumulate(
                "OfficialOR",
                Oofficial[h],
                gt,
            )

            accumulate(
                "V1_Oracle",
                pred_v1,
                gt,
            )

            accumulate(
                "V2_Oracle",
                pred_oracle,
                gt,
            )

            for name in VARIANTS:

                accumulate(
                    name,
                    pred_learned[name],
                    gt,
                )


if candidate_idx != len(val_target):

    raise RuntimeError(
        f"Candidate count mismatch: "
        f"{candidate_idx} vs "
        f"{len(val_target)}"
    )


# ============================================================
# Print
# ============================================================

print("=" * 100)
print(
    "STCV-Occ V2-1D "
    "LOSS DOWNSTREAM VALIDATION AUDIT"
)
print("=" * 100)

print(
    f"Val samples       : {val_samples}"
)

print(
    f"Candidate patches : {candidate_idx}"
)

print(
    "Candidate ordering: PASS"
)

print()

print(
    f"{'Strategy':24s} "
    f"{'Accept':>9s} "
    f"{'Rate':>10s} "
    f"{'30m IoU':>12s} "
    f"{'100m IoU':>12s} "
    f"{'Mean':>12s}"
)

print("-" * 85)


results = {}

for strategy in STRATEGIES:

    vals = {}

    for r in RANGES:

        x = acc[strategy][r]

        vals[r] = iou(
            x["tp"],
            x["fp"],
            x["fn"],
        )

    mean = (
        vals["30m"]
        + vals["100m"]
    ) / 2.0

    results[strategy] = {
        "30m": vals["30m"],
        "100m": vals["100m"],
        "mean": mean,
    }

    if strategy in VARIANTS:

        n_accept = accept_count[
            strategy
        ]

        rate = (
            n_accept
            / candidate_idx
        )

    else:

        n_accept = 0
        rate = 0.0

    print(
        f"{strategy:24s} "
        f"{n_accept:9d} "
        f"{rate:10.6f} "
        f"{vals['30m']:12.6f} "
        f"{vals['100m']:12.6f} "
        f"{mean:12.6f}"
    )


print()
print("=" * 100)
print("GAIN OVER SMOOTHL1 BASELINE")
print("=" * 100)

base = results[
    "SmoothL1"
]

for name in [
    "MSE",
    "WeightedSmoothL1",
]:

    r = results[name]

    print()
    print(name)

    print(
        f"30m  : "
        f"{r['30m'] - base['30m']:+.6f}"
    )

    print(
        f"100m : "
        f"{r['100m'] - base['100m']:+.6f}"
    )

    print(
        f"Mean : "
        f"{r['mean'] - base['mean']:+.6f}"
    )


print()
print(
    "TEST NOT EVALUATED."
)

print(
    "V2-1D downstream audit finished."
)
