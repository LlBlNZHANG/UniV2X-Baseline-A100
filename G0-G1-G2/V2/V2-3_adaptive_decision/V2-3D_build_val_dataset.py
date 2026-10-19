#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/root/autodl-tmp/UniV2X")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from projects.mmdet3d_plugin.univ2x.dense_heads.stcv_occ import (
    STCVFeatureExtractor,
)


BASE = (
    ROOT
    / "G0-G1-G2/V2/V2-3_adaptive_decision"
)

CSV_IN = (
    BASE
    / "V2-3B_counterfactual_candidate_planning_utility.csv"
)

G0_CACHE = (
    ROOT
    / "G0-G1-G2/cache_full"
)

OUT = (
    BASE
    / "V2-3D_val_planning_utility_dataset.npz"
)


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


# ============================================================================
# Load V2-3B labels
# ============================================================================

assert CSV_IN.exists(), CSV_IN

records = []

with open(
    CSV_IN,
    "r",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        records.append(row)


assert len(records) == 306, (
    f"Expected 306 Val near5 candidates, got {len(records)}"
)

print("=" * 100)
print("V2-3D-0 BUILD VAL PLANNING-UTILITY DATASET")
print("=" * 100)

print("V2-3B records :", len(records))


# ============================================================================
# Build sample_idx -> G0 cache file mapping
# ============================================================================

sample_to_file = {}

for p in sorted(
    G0_CACHE.glob("sample_*.npz")
):

    with np.load(
        p,
        allow_pickle=False,
    ) as d:

        sample_idx = str(
            d["sample_idx"].item()
        )

    assert sample_idx not in sample_to_file

    sample_to_file[
        sample_idx
    ] = p


print(
    "G0 cache samples:",
    len(sample_to_file),
)

assert len(sample_to_file) == 675


# ============================================================================
# Exact feature extractor
# ============================================================================

extractor = STCVFeatureExtractor()

features = []

candidate_idx = []
sample_idx_arr = []

horizon = []
patch_r = []
patch_c = []

min_plan_dist = []

v2_mag = []
v2_sign_prob = []
v2_sign_margin = []
v2_accept = []

delta_L2 = []
plan_utility = []
plan_label = []

delta_safety_sum = []

feature_cache = {}


for i, row in enumerate(records):

    sid = str(
        row["sample_idx"]
    )

    h = int(
        row["horizon"]
    )

    r = int(
        row["patch_r"]
    )

    c = int(
        row["patch_c"]
    )

    assert sid in sample_to_file, sid

    # ------------------------------------------------------------------------
    # Load a sample once
    # ------------------------------------------------------------------------

    if sid not in feature_cache:

        p = sample_to_file[
            sid
        ]

        with np.load(
            p,
            allow_pickle=False,
        ) as d:

            pv = torch.from_numpy(
                norm(
                    d["Pv"],
                    3,
                )
            ).float()

            pi = torch.from_numpy(
                norm(
                    d["Pi_aligned"],
                    3,
                )
            ).float()

            warp = torch.from_numpy(
                norm(
                    d["warp_valid_mask"],
                    2,
                )
            ).bool()

        feature_cache[
            sid
        ] = (
            pv,
            pi,
            warp,
        )

    pv, pi, warp = feature_cache[
        sid
    ]

    feat = extractor.extract_127_features(
        pv,
        pi,
        warp,
        h,
        r,
        c,
    )

    assert feat.shape == (127,)

    features.append(
        feat.cpu().numpy()
    )

    candidate_idx.append(
        int(
            row["candidate_idx"]
        )
    )

    sample_idx_arr.append(
        sid
    )

    horizon.append(h)
    patch_r.append(r)
    patch_c.append(c)

    min_plan_dist.append(
        float(
            row["min_plan_dist_m"]
        )
    )

    v2_mag.append(
        float(
            row["v2_mag"]
        )
    )

    # V2-3B used these names.
    # Accept both possible spellings for robustness.
    if "sign_prob" in row:
        sp = float(
            row["sign_prob"]
        )
    else:
        sp = float(
            row["v2_sign_prob"]
        )

    if "sign_margin" in row:
        sm = float(
            row["sign_margin"]
        )
    else:
        sm = abs(
            sp - 0.5
        )

    v2_sign_prob.append(sp)
    v2_sign_margin.append(sm)

    v2_accept.append(
        int(
            row["v2_accept"]
        )
    )

    dl2 = float(
        row["delta_L2_mean"]
    )

    u = -dl2

    if u > 1e-6:
        label = 1
    elif u < -1e-6:
        label = -1
    else:
        label = 0

    delta_L2.append(dl2)
    plan_utility.append(u)
    plan_label.append(label)

    delta_safety_sum.append(
        float(
            row[
                "delta_safety_sum"
            ]
        )
    )

    if (
        i < 5
        or (i + 1) % 50 == 0
        or i + 1 == len(records)
    ):

        print(
            f"[{i+1:03d}/{len(records):03d}] "
            f"cand={candidate_idx[-1]:05d} "
            f"sample={sid} "
            f"h={h} "
            f"r={r} c={c} "
            f"d={min_plan_dist[-1]:.3f} "
            f"Uplan={u:+.8f}"
        )


features = np.asarray(
    features,
    dtype=np.float32,
)

assert features.shape == (
    306,
    127,
)


# ============================================================================
# Final audit
# ============================================================================

plan_utility = np.asarray(
    plan_utility,
    dtype=np.float32,
)

plan_label = np.asarray(
    plan_label,
    dtype=np.int8,
)

beneficial = (
    plan_label == 1
)

harmful = (
    plan_label == -1
)

neutral = (
    plan_label == 0
)


print()
print("=" * 100)
print("VAL DATASET AUDIT")
print("=" * 100)

print(
    "Candidates :",
    len(plan_label),
)

print(
    "Feature dim:",
    features.shape,
)

print(
    "Beneficial :",
    int(
        beneficial.sum()
    ),
)

print(
    "Harmful    :",
    int(
        harmful.sum()
    ),
)

print(
    "Neutral    :",
    int(
        neutral.sum()
    ),
)

print(
    "Uplan mean :",
    f"{plan_utility.mean():+.10f}",
)

print(
    "Uplan min  :",
    f"{plan_utility.min():+.10f}",
)

print(
    "Uplan max  :",
    f"{plan_utility.max():+.10f}",
)


# Historical V2-3B exact checks.
assert int(
    beneficial.sum()
) == 133

assert int(
    harmful.sum()
) == 167

assert int(
    neutral.sum()
) == 6

assert int(
    np.asarray(
        v2_accept,
        dtype=np.uint8,
    ).sum()
) == 11


# ============================================================================
# Save
# ============================================================================

np.savez_compressed(

    OUT,

    features_127=features,

    candidate_idx=np.asarray(
        candidate_idx,
        dtype=np.int64,
    ),

    sample_idx=np.asarray(
        sample_idx_arr,
    ),

    horizon=np.asarray(
        horizon,
        dtype=np.int64,
    ),

    patch_r=np.asarray(
        patch_r,
        dtype=np.int64,
    ),

    patch_c=np.asarray(
        patch_c,
        dtype=np.int64,
    ),

    min_plan_dist_m=np.asarray(
        min_plan_dist,
        dtype=np.float32,
    ),

    v2_mag=np.asarray(
        v2_mag,
        dtype=np.float32,
    ),

    v2_sign_prob=np.asarray(
        v2_sign_prob,
        dtype=np.float32,
    ),

    v2_sign_margin=np.asarray(
        v2_sign_margin,
        dtype=np.float32,
    ),

    v2_accept=np.asarray(
        v2_accept,
        dtype=np.uint8,
    ),

    delta_L2_mean=np.asarray(
        delta_L2,
        dtype=np.float32,
    ),

    plan_utility=plan_utility,

    plan_label=plan_label,

    delta_safety_sum=np.asarray(
        delta_safety_sum,
        dtype=np.float32,
    ),
)


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print("NPZ:", OUT)

print()
print(
    "[V2-3D-0 VAL DATASET] COMPLETE"
)

print("=" * 100)
