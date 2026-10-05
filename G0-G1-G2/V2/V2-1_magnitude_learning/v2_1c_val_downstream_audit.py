#!/usr/bin/env python3

import os
import glob
import json
import numpy as np
from collections import defaultdict


# ============================================================
# STCV-Occ V2-1C
# Validation Downstream Occupancy Audit
#
# IMPORTANT:
#   - VAL ONLY
#   - TEST remains sealed
#   - Main V2-1 decision:
#
#         accept iff magnitude_pred > 0
#
#   - sign>=0.5 and joint are diagnostic only
#
# Metrics:
#   official-style accumulated foreground IoU
#   on 30x30 and 100x100 ranges
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

TARGET_FILE = os.path.join(
    ROOT,
    "G0-G1-G2/V2/V2-1_magnitude_learning/"
    "v2_1a_target_audit_arrays.npz",
)

RESULT_ROOT = os.path.join(
    ROOT,
    "G0-G1-G2/V2/V2-1_magnitude_learning/results",
)

PATCH = 4
GRID = 200
SEED = 2026

LAMBDAS = {
    "0.00": "main_lambda_0p00",
    "0.25": "main_lambda_0p25",
    "0.50": "main_lambda_0p50",
    "1.00": "main_lambda_1p00",
}

RANGES = {
    "30x30": (70, 130),
    "100x100": (0, 200),
}


# ============================================================
# Metrics
# ============================================================

def stats(pred, gt_raw, lo, hi):

    p = pred[
        lo:hi,
        lo:hi,
    ].astype(bool)

    g_raw = gt_raw[
        lo:hi,
        lo:hi,
    ]

    # Official metric ignores labels outside valid classes.
    valid = (
        (g_raw == 0)
        | (g_raw == 1)
    )

    g = (
        g_raw == 1
    )

    tp = int(
        (
            p
            & g
            & valid
        ).sum()
    )

    fp = int(
        (
            p
            & (~g)
            & valid
        ).sum()
    )

    fn = int(
        (
            (~p)
            & g
            & valid
        ).sum()
    )

    return tp, fp, fn


def iou(tp, fp, fn):

    d = tp + fp + fn

    if d == 0:
        return 0.0

    return tp / d


def add_stats(
    accumulator,
    strategy,
    pred,
    gt,
):

    for name, (
        lo,
        hi,
    ) in RANGES.items():

        tp, fp, fn = stats(
            pred,
            gt,
            lo,
            hi,
        )

        accumulator[
            strategy
        ][name]["tp"] += tp

        accumulator[
            strategy
        ][name]["fp"] += fp

        accumulator[
            strategy
        ][name]["fn"] += fn


def delta_iou(
    tp,
    fp,
    fn,
    ba,
    ha,
):

    D = tp + fp + fn

    if D == 0:
        return 0.0

    return (
        (tp + ba)
        / (D + ha)
        - tp / D
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

    region[
        add_patch
    ] = True


# ============================================================
# Reconstruct exact G3 split
# ============================================================

scene_to_samples = defaultdict(list)
valid_sample_ids = set()

with open(
    VALID_FILE,
    "r",
) as f:

    next(f)

    for line in f:

        parts = (
            line
            .strip()
            .split("\t")
        )

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
# Load V2 target arrays
# ============================================================

target_data = np.load(
    TARGET_FILE,
    allow_pickle=True,
)

val_target = target_data[
    "val_target"
].astype(np.float64)

val_v1_utility = target_data[
    "val_v1_utility"
].astype(np.float64)

assert len(val_target) == 18555
assert len(val_v1_utility) == 18555


# ============================================================
# Load predictor outputs
# ============================================================

predictions = {}

for lam, dirname in LAMBDAS.items():

    pred_path = os.path.join(
        RESULT_ROOT,
        dirname,
        "val_predictions.npz",
    )

    summary_path = os.path.join(
        RESULT_ROOT,
        dirname,
        "summary.json",
    )

    p = np.load(
        pred_path,
        allow_pickle=True,
    )

    with open(
        summary_path,
        "r",
    ) as f:

        summary = json.load(f)

    mag = p[
        "magnitude_pred"
    ].astype(np.float64)

    sign_prob = p[
        "sign_prob"
    ].astype(np.float64)

    raw_delta = p[
        "raw_delta_iou"
    ].astype(np.float64)

    assert len(mag) == 18555
    assert len(sign_prob) == 18555
    assert len(raw_delta) == 18555

    if not np.allclose(
        raw_delta,
        val_target,
        atol=1e-8,
        rtol=0,
    ):
        raise RuntimeError(
            f"lambda={lam}: "
            "prediction/target ordering mismatch"
        )

    predictions[lam] = {
        "mag": mag,
        "sign": sign_prob,
        "best_epoch": int(
            summary["best_epoch"]
        ),
    }


# ============================================================
# Strategies
# ============================================================

strategies = [
    "Ego",
    "OfficialOR",
    "V1_Oracle",
    "V2_DeltaIoU_Oracle",
]

for lam in LAMBDAS:

    strategies.extend([
        f"L{lam}_MagZero",
        f"L{lam}_Sign050",
        f"L{lam}_Joint",
    ])


acc = {}

for strategy in strategies:

    acc[strategy] = {}

    for range_name in RANGES:

        acc[strategy][
            range_name
        ] = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }


decision = {}

for lam in LAMBDAS:

    decision[lam] = {
        "mag_accept": 0,
        "sign_accept": 0,
        "joint_accept": 0,
    }


# ============================================================
# Main reconstruction
# ============================================================

candidate_idx = 0
val_samples = 0
val_horizons = 0
val_samples_with_candidate = 0

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

        if (
            sample_idx
            not in valid_sample_ids
        ):
            continue

        scene_token = str(
            d["scene_token"].item()
        )

        if (
            scene_token
            not in val_scenes
        ):
            continue

        val_samples += 1

        Ov = d[
            "Ov"
        ].astype(bool)

        Oi = d[
            "Oi"
        ].astype(bool)

        Oofficial = d[
            "Oofficial"
        ].astype(bool)

        GT = d[
            "GT"
        ].astype(np.int16)

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
            & (GT == 1)
        )

        HA = (
            ADD
            & (GT == 0)
        )

        sample_has_candidate = bool(
            ADD.any()
        )

        if sample_has_candidate:
            val_samples_with_candidate += 1


        # ====================================================
        # Horizons
        # ====================================================

        for h in range(5):

            val_horizons += 1

            ov = Ov[h]
            gt = GT[h]

            add_h = ADD[h]
            ba_h = BA[h]
            ha_h = HA[h]

            # Full-BEV Ego statistics for DeltaIoU100
            tp100, fp100, fn100 = stats(
                ov,
                gt,
                0,
                200,
            )

            pred_v1 = ov.copy()
            pred_oracle = ov.copy()

            pred_mag = {
                lam: ov.copy()
                for lam in LAMBDAS
            }

            pred_sign = {
                lam: ov.copy()
                for lam in LAMBDAS
            }

            pred_joint = {
                lam: ov.copy()
                for lam in LAMBDAS
            }


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

                    u_v1 = (
                        ba - ha
                    )

                    u_v2 = delta_iou(
                        tp100,
                        fp100,
                        fn100,
                        ba,
                        ha,
                    )


                    # ---------------------------------------
                    # Ordering assertions
                    # ---------------------------------------

                    if (
                        candidate_idx
                        >= len(val_target)
                    ):
                        raise RuntimeError(
                            "Too many reconstructed candidates"
                        )

                    if (
                        abs(
                            u_v1
                            - val_v1_utility[
                                candidate_idx
                            ]
                        )
                        > 1e-8
                    ):
                        raise RuntimeError(
                            f"V1 utility ordering mismatch "
                            f"at candidate {candidate_idx}"
                        )

                    if not np.isclose(
                        u_v2,
                        val_target[
                            candidate_idx
                        ],
                        atol=1e-8,
                        rtol=0,
                    ):
                        raise RuntimeError(
                            f"DeltaIoU ordering mismatch "
                            f"at candidate {candidate_idx}: "
                            f"{u_v2} vs "
                            f"{val_target[candidate_idx]}"
                        )


                    # ---------------------------------------
                    # V1 Oracle
                    # ---------------------------------------

                    if u_v1 > 0:

                        apply_patch(
                            pred_v1,
                            add_patch,
                            y0,
                            y1,
                            x0,
                            x1,
                        )


                    # ---------------------------------------
                    # V2 metric-aligned Oracle
                    # ---------------------------------------

                    if u_v2 > 0:

                        apply_patch(
                            pred_oracle,
                            add_patch,
                            y0,
                            y1,
                            x0,
                            x1,
                        )


                    # ---------------------------------------
                    # Learned predictors
                    # ---------------------------------------

                    for lam in LAMBDAS:

                        mag = predictions[
                            lam
                        ]["mag"][
                            candidate_idx
                        ]

                        sign_prob = predictions[
                            lam
                        ]["sign"][
                            candidate_idx
                        ]

                        accept_mag = (
                            mag > 0.0
                        )

                        accept_sign = (
                            sign_prob >= 0.5
                        )

                        accept_joint = (
                            accept_mag
                            and accept_sign
                        )


                        if accept_mag:

                            decision[
                                lam
                            ]["mag_accept"] += 1

                            apply_patch(
                                pred_mag[lam],
                                add_patch,
                                y0,
                                y1,
                                x0,
                                x1,
                            )


                        if accept_sign:

                            decision[
                                lam
                            ]["sign_accept"] += 1

                            apply_patch(
                                pred_sign[lam],
                                add_patch,
                                y0,
                                y1,
                                x0,
                                x1,
                            )


                        if accept_joint:

                            decision[
                                lam
                            ]["joint_accept"] += 1

                            apply_patch(
                                pred_joint[lam],
                                add_patch,
                                y0,
                                y1,
                                x0,
                                x1,
                            )


                    candidate_idx += 1


            # ===============================================
            # Accumulate official-style TP / FP / FN
            # ===============================================

            add_stats(
                acc,
                "Ego",
                ov,
                gt,
            )

            add_stats(
                acc,
                "OfficialOR",
                Oofficial[h],
                gt,
            )

            add_stats(
                acc,
                "V1_Oracle",
                pred_v1,
                gt,
            )

            add_stats(
                acc,
                "V2_DeltaIoU_Oracle",
                pred_oracle,
                gt,
            )

            for lam in LAMBDAS:

                add_stats(
                    acc,
                    f"L{lam}_MagZero",
                    pred_mag[lam],
                    gt,
                )

                add_stats(
                    acc,
                    f"L{lam}_Sign050",
                    pred_sign[lam],
                    gt,
                )

                add_stats(
                    acc,
                    f"L{lam}_Joint",
                    pred_joint[lam],
                    gt,
                )


# ============================================================
# Final checks
# ============================================================

if (
    candidate_idx
    != len(val_target)
):

    raise RuntimeError(
        f"Candidate count mismatch: "
        f"{candidate_idx} vs "
        f"{len(val_target)}"
    )


print("=" * 110)
print(
    "STCV-Occ V2-1C "
    "VALIDATION DOWNSTREAM OCCUPANCY AUDIT"
)
print("=" * 110)

print(
    f"Val samples               : "
    f"{val_samples}"
)

print(
    f"Val samples with candidate: "
    f"{val_samples_with_candidate}"
)

print(
    f"Val horizons              : "
    f"{val_horizons}"
)

print(
    f"Candidate patches         : "
    f"{candidate_idx}"
)

print(
    "Candidate ordering audit : PASS"
)


# ============================================================
# Results
# ============================================================

results = {}

print()
print("=" * 110)
print("DATASET-LEVEL FOREGROUND IoU")
print("=" * 110)

print(
    f"{'Strategy':30s} "
    f"{'30m IoU':>12s} "
    f"{'100m IoU':>12s} "
    f"{'Mean':>12s}"
)

print("-" * 72)


for strategy in strategies:

    vals = {}

    for range_name in [
        "30x30",
        "100x100",
    ]:

        s = acc[
            strategy
        ][range_name]

        vals[
            range_name
        ] = iou(
            s["tp"],
            s["fp"],
            s["fn"],
        )

    vals["mean"] = (
        vals["30x30"]
        + vals["100x100"]
    ) / 2.0

    results[
        strategy
    ] = vals

    print(
        f"{strategy:30s} "
        f"{vals['30x30']:12.6f} "
        f"{vals['100x100']:12.6f} "
        f"{vals['mean']:12.6f}"
    )


# ============================================================
# Main magnitude strategy comparison
# ============================================================

print()
print("=" * 110)
print(
    "MAIN V2-1 COMPARISON: "
    "accept iff magnitude_pred > 0"
)
print("=" * 110)

print(
    f"{'lambda':>8s} "
    f"{'epoch':>8s} "
    f"{'accept':>10s} "
    f"{'rate':>10s} "
    f"{'30m':>12s} "
    f"{'100m':>12s} "
    f"{'mean':>12s}"
)

print("-" * 90)


for lam in LAMBDAS:

    strategy = (
        f"L{lam}_MagZero"
    )

    r = results[
        strategy
    ]

    n_accept = decision[
        lam
    ]["mag_accept"]

    print(
        f"{lam:>8s} "
        f"{predictions[lam]['best_epoch']:8d} "
        f"{n_accept:10d} "
        f"{n_accept / candidate_idx:10.6f} "
        f"{r['30x30']:12.6f} "
        f"{r['100x100']:12.6f} "
        f"{r['mean']:12.6f}"
    )


# ============================================================
# Diagnostic sign / joint acceptance
# ============================================================

print()
print("=" * 110)
print("DECISION RATE DIAGNOSTIC")
print("=" * 110)

print(
    f"{'lambda':>8s} "
    f"{'Mag>0':>12s} "
    f"{'Sign>=.5':>12s} "
    f"{'Joint':>12s}"
)

print("-" * 52)

for lam in LAMBDAS:

    d = decision[
        lam
    ]

    print(
        f"{lam:>8s} "
        f"{d['mag_accept']/candidate_idx:12.6f} "
        f"{d['sign_accept']/candidate_idx:12.6f} "
        f"{d['joint_accept']/candidate_idx:12.6f}"
    )


# ============================================================
# Diagnostic best lambda by 100m / Pareto inspection
# ============================================================

best_100 = max(
    LAMBDAS.keys(),
    key=lambda lam:
        results[
            f"L{lam}_MagZero"
        ]["100x100"],
)

best_30 = max(
    LAMBDAS.keys(),
    key=lambda lam:
        results[
            f"L{lam}_MagZero"
        ]["30x30"],
)


print()
print("=" * 110)
print("VAL-ONLY DIAGNOSTIC SUMMARY")
print("=" * 110)

print(
    f"Best lambda by 100m IoU : "
    f"{best_100}"
)

print(
    f"Best lambda by 30m IoU  : "
    f"{best_30}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "Test split was NOT evaluated."
)

print(
    "Primary V2-1 decision rule in this audit "
    "is magnitude_pred > 0."
)

print(
    "Sign>=0.5 and Joint are diagnostics only."
)

print()
print(
    "V2-1C validation downstream audit finished."
)
