#!/usr/bin/env python3

import os
import glob
import numpy as np
from collections import defaultdict


# ============================================================
# STCV-Occ V2-1E-0
# Build Sample-Horizon Group Dataset for Set-level IoU Learning
#
# Features:
#   EXACT SAME 127-D V1/G3 feature.
#
# Candidate definition:
#   EXACT SAME as G3 / V2-0 / V2-1.
#
# Each candidate stores:
#   feature
#   raw DeltaIoU100 target
#   100*DeltaIoU100 magnitude target
#   sign target
#   BA / HA
#   group id
#
# Each group = one sample + one horizon with >=1 candidate.
# Each group stores:
#   TP / FP / FN
#   horizon
#   sample index
#
# NO MODEL TRAINING.
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

G3_SOFT_DIR = os.path.join(
    ROOT,
    "G0-G1-G2/G3/soft",
)

TARGET_FILE = os.path.join(
    ROOT,
    "G0-G1-G2/V2/V2-1_magnitude_learning/"
    "v2_1a_target_audit_arrays.npz",
)

OUT_DIR = os.path.join(
    ROOT,
    "G0-G1-G2/V2/V2-1_magnitude_learning",
)

SEED = 2026
PATCH = 4
GRID = 200
TARGET_SCALE = 100.0


# ============================================================
# Helpers
# ============================================================

def occ_stats(pred, gt_raw):

    p = pred.astype(bool)

    valid = (
        (gt_raw == 0)
        | (gt_raw == 1)
    )

    g = (
        gt_raw == 1
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


# ============================================================
# Exact G3 scene split
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


def get_split(scene_token):

    if scene_token in train_scenes:
        return "train"

    if scene_token in val_scenes:
        return "val"

    if scene_token in test_scenes:
        return "test"

    return None


# ============================================================
# Existing V1 features
# ============================================================

feature_data = {}

for split in [
    "train",
    "val",
    "test",
]:

    path = os.path.join(
        G3_SOFT_DIR,
        f"g3_neighborhood_{split}.npz",
    )

    d = np.load(
        path,
        allow_pickle=True,
    )

    feature_data[split] = {
        "features":
            d["features"].astype(
                np.float32
            ),
        "labels":
            d["labels"].astype(
                np.int64
            ),
    }


# ============================================================
# Existing V2 target arrays
# ============================================================

target_data = np.load(
    TARGET_FILE,
    allow_pickle=True,
)

# ============================================================
# IMPORTANT SPEED FIX
#
# np.load(.npz) is lazy. Never index compressed NPZ arrays
# inside the candidate loop; otherwise the same arrays may be
# decompressed repeatedly tens of thousands of times.
# Load all required arrays into RAM exactly once.
# ============================================================

target_cache = {}

for split in [
    "train",
    "val",
    "test",
]:

    target_cache[split] = {
        "target": np.asarray(
            target_data[
                f"{split}_target"
            ]
        ).copy(),

        "v1_utility": np.asarray(
            target_data[
                f"{split}_v1_utility"
            ]
        ).copy(),
    }

target_data.close()


# ============================================================
# Storage
# ============================================================

splits = [
    "train",
    "val",
    "test",
]

candidate = {}

group = {}

cursor = {}

for split in splits:

    candidate[split] = {
        "ba": [],
        "ha": [],
        "group_id": [],
        "horizon": [],
    }

    group[split] = {
        "tp": [],
        "fp": [],
        "fn": [],
        "horizon": [],
        "sample_idx": [],
        "scene_token": [],
        "candidate_count": [],
    }

    cursor[split] = 0


# ============================================================
# Reconstruct candidate ordering
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
print(
    "STCV-Occ V2-1E-0 "
    "GROUP DATASET CONSTRUCTION"
)
print("=" * 110)


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

        split = get_split(
            scene_token
        )

        if split is None:
            continue

        Ov = d[
            "Ov"
        ].astype(bool)

        Oi = d[
            "Oi"
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


        # ====================================================
        # Horizon = one potential set/group
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

            # Tentative group id.
            # Only committed if >=1 candidate exists.
            gid = len(
                group[split]["tp"]
            )

            group_candidate_count = 0


            # ===============================================
            # P4 candidate patches
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
                        tp,
                        fp,
                        fn,
                        ba,
                        ha,
                    )

                    idx = cursor[
                        split
                    ]

                    old_target = (
                        target_cache[
                            split
                        ]["target"][idx]
                    )

                    old_v1_u = (
                        target_cache[
                            split
                        ]["v1_utility"][idx]
                    )

                    old_v1_label = (
                        feature_data[
                            split
                        ]["labels"][idx]
                    )


                    # ---------------------------------------
                    # Exact ordering audit
                    # ---------------------------------------

                    if (
                        abs(
                            float(old_v1_u)
                            - float(u_v1)
                        )
                        > 1e-8
                    ):

                        raise RuntimeError(
                            f"{split}: V1 utility "
                            f"ordering mismatch "
                            f"at {idx}"
                        )


                    if not np.isclose(
                        float(old_target),
                        float(u_v2),
                        atol=1e-8,
                        rtol=0,
                    ):

                        raise RuntimeError(
                            f"{split}: DeltaIoU "
                            f"ordering mismatch "
                            f"at {idx}: "
                            f"{old_target} vs "
                            f"{u_v2}"
                        )


                    reconstructed_v1_label = int(
                        u_v1 > 0
                    )

                    if (
                        reconstructed_v1_label
                        != int(
                            old_v1_label
                        )
                    ):

                        raise RuntimeError(
                            f"{split}: feature/label "
                            f"alignment mismatch "
                            f"at {idx}"
                        )


                    candidate[
                        split
                    ]["ba"].append(
                        ba
                    )

                    candidate[
                        split
                    ]["ha"].append(
                        ha
                    )

                    candidate[
                        split
                    ]["group_id"].append(
                        gid
                    )

                    candidate[
                        split
                    ]["horizon"].append(
                        h
                    )

                    cursor[
                        split
                    ] += 1

                    group_candidate_count += 1


            # ===============================================
            # Commit non-empty group
            # ===============================================

            if group_candidate_count > 0:

                group[
                    split
                ]["tp"].append(
                    tp
                )

                group[
                    split
                ]["fp"].append(
                    fp
                )

                group[
                    split
                ]["fn"].append(
                    fn
                )

                group[
                    split
                ]["horizon"].append(
                    h
                )

                group[
                    split
                ]["sample_idx"].append(
                    sample_idx
                )

                group[
                    split
                ]["scene_token"].append(
                    scene_token
                )

                group[
                    split
                ]["candidate_count"].append(
                    group_candidate_count
                )


# ============================================================
# Save + audit
# ============================================================

expected_count = {
    "train": 86143,
    "val": 18555,
    "test": 14750,
}


for split in splits:

    n = cursor[
        split
    ]

    if (
        n
        != expected_count[
            split
        ]
    ):

        raise RuntimeError(
            f"{split}: candidate count "
            f"{n} != "
            f"{expected_count[split]}"
        )


    features = feature_data[
        split
    ]["features"]

    v1_labels = feature_data[
        split
    ]["labels"]

    raw_target = target_cache[
        split
    ]["target"].astype(
        np.float32
    )

    magnitude_target = (
        raw_target
        * TARGET_SCALE
    ).astype(
        np.float32
    )

    sign_target = (
        raw_target > 0
    ).astype(
        np.float32
    )

    ba = np.asarray(
        candidate[
            split
        ]["ba"],
        dtype=np.int16,
    )

    ha = np.asarray(
        candidate[
            split
        ]["ha"],
        dtype=np.int16,
    )

    group_id = np.asarray(
        candidate[
            split
        ]["group_id"],
        dtype=np.int32,
    )

    horizon = np.asarray(
        candidate[
            split
        ]["horizon"],
        dtype=np.int8,
    )


    group_tp = np.asarray(
        group[
            split
        ]["tp"],
        dtype=np.int32,
    )

    group_fp = np.asarray(
        group[
            split
        ]["fp"],
        dtype=np.int32,
    )

    group_fn = np.asarray(
        group[
            split
        ]["fn"],
        dtype=np.int32,
    )

    group_horizon = np.asarray(
        group[
            split
        ]["horizon"],
        dtype=np.int8,
    )

    group_sample_idx = np.asarray(
        group[
            split
        ]["sample_idx"],
        dtype=object,
    )

    group_scene_token = np.asarray(
        group[
            split
        ]["scene_token"],
        dtype=object,
    )

    group_candidate_count = np.asarray(
        group[
            split
        ]["candidate_count"],
        dtype=np.int32,
    )


    # ========================================================
    # Group pointer
    #
    # Candidate construction is group-contiguous.
    # group_ptr[g]: first candidate index of group g
    # group_ptr[g+1]: exclusive end.
    # ========================================================

    group_ptr = np.concatenate([
        np.asarray(
            [0],
            dtype=np.int64,
        ),
        np.cumsum(
            group_candidate_count,
            dtype=np.int64,
        ),
    ])


    # ========================================================
    # Assertions
    # ========================================================

    assert len(features) == n
    assert len(v1_labels) == n
    assert len(raw_target) == n
    assert len(ba) == n
    assert len(ha) == n
    assert len(group_id) == n

    assert (
        group_ptr[-1]
        == n
    )

    assert (
        len(group_tp)
        ==
        len(group_fp)
        ==
        len(group_fn)
        ==
        len(group_candidate_count)
    )

    # group IDs must exactly match contiguous group_ptr ranges.
    for gid in range(
        len(group_candidate_count)
    ):

        s = group_ptr[gid]
        e = group_ptr[gid + 1]

        if not np.all(
            group_id[s:e]
            == gid
        ):

            raise RuntimeError(
                f"{split}: non-contiguous "
                f"group id {gid}"
            )


    # ========================================================
    # Save
    # ========================================================

    out_path = os.path.join(
        OUT_DIR,
        f"v2_1e_group_{split}.npz",
    )

    np.savez_compressed(
        out_path,

        # Candidate-level
        features=
            features,
        raw_target=
            raw_target,
        magnitude_target=
            magnitude_target,
        sign_target=
            sign_target,
        v1_label=
            v1_labels,
        ba=
            ba,
        ha=
            ha,
        group_id=
            group_id,
        horizon=
            horizon,

        # Group-level
        group_ptr=
            group_ptr,
        group_tp=
            group_tp,
        group_fp=
            group_fp,
        group_fn=
            group_fn,
        group_horizon=
            group_horizon,
        group_sample_idx=
            group_sample_idx,
        group_scene_token=
            group_scene_token,
        group_candidate_count=
            group_candidate_count,
    )


    # ========================================================
    # Summary
    # ========================================================

    n_groups = len(
        group_candidate_count
    )

    D = (
        group_tp
        + group_fp
        + group_fn
    )

    print()
    print("=" * 110)
    print(
        f"{split.upper()} GROUP AUDIT"
    )
    print("=" * 110)

    print(
        f"Candidates             : "
        f"{n}"
    )

    print(
        f"Groups with candidates : "
        f"{n_groups}"
    )

    print(
        f"Mean patches / group   : "
        f"{group_candidate_count.mean():.4f}"
    )

    print(
        f"Median patches / group : "
        f"{np.median(group_candidate_count):.1f}"
    )

    print(
        f"Min patches / group    : "
        f"{group_candidate_count.min()}"
    )

    print(
        f"Max patches / group    : "
        f"{group_candidate_count.max()}"
    )

    print(
        f"Groups with D=0        : "
        f"{int((D == 0).sum())}"
    )

    print(
        f"Positive targets       : "
        f"{int(sign_target.sum())} "
        f"({sign_target.mean():.6f})"
    )

    print(
        f"Group ordering audit   : PASS"
    )

    print(
        f"Candidate alignment    : PASS"
    )

    print(
        f"Saved                  : "
        f"{out_path}"
    )


print()
print("=" * 110)
print(
    "V2-1E-0 GROUP DATASET CONSTRUCTION COMPLETE"
)
print("=" * 110)

print(
    "No model was trained."
)

print(
    "No Test model performance was evaluated."
)
