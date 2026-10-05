#!/usr/bin/env python3

import os
import glob
import sys
import numpy as np
from collections import defaultdict

import torch

THIS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(0, THIS_DIR)

from train_v2_magnitude_utility import (
    V2UtilityPredictor,
)

ROOT = "/root/autodl-tmp/UniV2X"

CACHE_DIR = os.path.join(
    ROOT,
    "G0-G1-G2/cache_full",
)

VALID_FILE = os.path.join(
    ROOT,
    "G0-G1-G2/G3/cache_valid_samples.txt",
)

VAL_GROUP_FILE = os.path.join(
    THIS_DIR,
    "v2_1e_group_val.npz",
)

CKPT_DIR = os.path.join(
    THIS_DIR,
    "checkpoints/set_refine_lambda_0p02/epochs",
)

PATCH = 4
GRID = 200
SEED = 2026


# ============================================================
# Official-style foreground IoU statistics
# ============================================================

RANGES = {
    "30m": (70, 130),
    "100m": (0, 200),
}


def occ_stats(pred, gt_raw, lo, hi):

    p = pred[
        lo:hi,
        lo:hi,
    ].astype(bool)

    g_raw = gt_raw[
        lo:hi,
        lo:hi,
    ]

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
# Exact scene split
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

val_scenes = set(
    scenes[13:17]
)


# ============================================================
# Load candidate features in exact ordering
# ============================================================

d = np.load(
    VAL_GROUP_FILE,
    allow_pickle=True,
)

features = d[
    "features"
].astype(np.float32)

assert len(features) == 18555


# ============================================================
# Predict every epoch
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

checkpoint_paths = sorted(
    glob.glob(
        os.path.join(
            CKPT_DIR,
            "epoch_*.pth",
        )
    )
)

if not checkpoint_paths:
    raise RuntimeError(
        f"No epoch checkpoints found in {CKPT_DIR}"
    )


epoch_predictions = {}

X = torch.from_numpy(
    features
)


for ckpt_path in checkpoint_paths:

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
    )

    epoch = int(
        ckpt["epoch"]
    )

    model = V2UtilityPredictor(
        dropout=0.1
    ).to(device)

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    preds = []

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            4096,
        ):

            xb = X[
                start:start + 4096
            ].to(device)

            mag_pred, _ = model(
                xb
            )

            preds.append(
                mag_pred
                .detach()
                .cpu()
                .numpy()
            )

    pred = np.concatenate(
        preds
    ).astype(np.float64)

    assert len(pred) == 18555

    epoch_predictions[
        epoch
    ] = pred

    print(
        f"Loaded epoch {epoch:03d}: "
        f"accept={(pred > 0).mean():.6f}, "
        f"pred_std={pred.std():.6f}"
    )


epochs = sorted(
    epoch_predictions.keys()
)


# ============================================================
# Accumulators
# ============================================================

strategies = (
    [
        "Ego",
        "OfficialOR",
    ]
    + [
        f"Epoch{e:03d}"
        for e in epochs
    ]
)

acc = {
    strategy: {
        r: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for r in RANGES
    }
    for strategy in strategies
}


def accumulate(
    strategy,
    pred,
    gt,
):

    for r, (
        lo,
        hi,
    ) in RANGES.items():

        tp, fp, fn = occ_stats(
            pred,
            gt,
            lo,
            hi,
        )

        acc[
            strategy
        ][r]["tp"] += tp

        acc[
            strategy
        ][r]["fp"] += fp

        acc[
            strategy
        ][r]["fn"] += fn


# ============================================================
# Full Validation reconstruction
# ============================================================

candidate_idx = 0
val_samples = 0
val_horizons = 0

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
    ) as sample:

        sample_idx = str(
            sample["sample_idx"].item()
        )

        scene_token = str(
            sample["scene_token"].item()
        )

        if (
            sample_idx
            not in valid_sample_ids
        ):
            continue

        if (
            scene_token
            not in val_scenes
        ):
            continue

        val_samples += 1

        Ov = sample[
            "Ov"
        ].astype(bool)

        Oi = sample[
            "Oi"
        ].astype(bool)

        Oofficial = sample[
            "Oofficial"
        ].astype(bool)

        GT = sample[
            "GT"
        ].astype(np.int16)

        M_gt = sample[
            "gt_cell_valid_mask"
        ].astype(bool)

        M_warp_2d = sample[
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


        for h in range(5):

            val_horizons += 1

            ov = Ov[h]
            gt = GT[h]
            add_h = ADD[h]

            learned = {
                epoch: ov.copy()
                for epoch in epochs
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


                    if (
                        candidate_idx
                        >= 18555
                    ):
                        raise RuntimeError(
                            "Too many candidates reconstructed"
                        )


                    for epoch in epochs:

                        if (
                            epoch_predictions[
                                epoch
                            ][candidate_idx]
                            > 0.0
                        ):

                            apply_patch(
                                learned[
                                    epoch
                                ],
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

            for epoch in epochs:

                accumulate(
                    f"Epoch{epoch:03d}",
                    learned[epoch],
                    gt,
                )


if candidate_idx != 18555:

    raise RuntimeError(
        f"Candidate mismatch: "
        f"{candidate_idx} != 18555"
    )


# ============================================================
# Results
# ============================================================

print()
print("=" * 105)
print(
    "STCV-Occ V2-1E "
    "FULL VALIDATION EPOCH SWEEP"
)
print("=" * 105)

print(
    f"Val samples       : {val_samples}"
)

print(
    f"Val horizons      : {val_horizons}"
)

print(
    f"Candidate patches : {candidate_idx}"
)

print(
    "Ordering audit    : PASS"
)

print()

print(
    f"{'Strategy':18s} "
    f"{'30m IoU':>12s} "
    f"{'100m IoU':>12s} "
    f"{'Mean':>12s} "
    f"{'Accept':>12s}"
)

print("-" * 72)


results = {}

for strategy in strategies:

    vals = {}

    for r in [
        "30m",
        "100m",
    ]:

        s = acc[
            strategy
        ][r]

        vals[r] = iou(
            s["tp"],
            s["fp"],
            s["fn"],
        )

    mean = (
        vals["30m"]
        + vals["100m"]
    ) / 2.0

    if strategy.startswith(
        "Epoch"
    ):

        epoch = int(
            strategy.replace(
                "Epoch",
                ""
            )
        )

        accept = float(
            (
                epoch_predictions[
                    epoch
                ] > 0
            ).mean()
        )

    else:
        accept = 0.0

    results[
        strategy
    ] = {
        "30m":
            vals["30m"],

        "100m":
            vals["100m"],

        "mean":
            mean,

        "accept":
            accept,
    }

    print(
        f"{strategy:18s} "
        f"{vals['30m']:12.6f} "
        f"{vals['100m']:12.6f} "
        f"{mean:12.6f} "
        f"{accept:12.6f}"
    )


# ============================================================
# Selection
# ============================================================

epoch_names = [
    f"Epoch{e:03d}"
    for e in epochs
]

best_100 = max(
    epoch_names,
    key=lambda x:
        results[x]["100m"],
)

best_30 = max(
    epoch_names,
    key=lambda x:
        results[x]["30m"],
)

best_mean = max(
    epoch_names,
    key=lambda x:
        results[x]["mean"],
)


baseline = results[
    "Epoch000"
]

winner = results[
    best_100
]


print()
print("=" * 105)
print("VAL-ONLY MODEL SELECTION")
print("=" * 105)

print(
    f"Best epoch by 100m : "
    f"{best_100}"
)

print(
    f"Best epoch by 30m  : "
    f"{best_30}"
)

print(
    f"Best epoch by mean : "
    f"{best_mean}"
)

print()

print(
    f"100m gain vs epoch0: "
    f"{winner['100m'] - baseline['100m']:+.6f}"
)

print(
    f"30m change vs epoch0: "
    f"{winner['30m'] - baseline['30m']:+.6f}"
)

print(
    f"Mean change vs epoch0: "
    f"{winner['mean'] - baseline['mean']:+.6f}"
)

print()
print(
    "TEST NOT EVALUATED."
)
