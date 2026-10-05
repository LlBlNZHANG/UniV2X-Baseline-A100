import numpy as np
from scipy.stats import spearmanr


# ============================================================
# STCV-Occ V2-0
# Risk / Utility Definition Audit
#
# Purpose:
#   1. Audit whether horizon is related to harmful cooperation
#   2. Audit whether spatial distance is related to harmful cooperation
#   3. Audit sensitivity to asymmetric harmful cost beta
#
# IMPORTANT:
#   - Train split only
#   - No model training
#   - No modification to STCV-v1
# ============================================================


DATA_PATH = (
    "G0-G1-G2/G3/soft/"
    "g3_neighborhood_train.npz"
)

# ------------------------------------------------------------
# Official UniV2X Occupancy geometry
# ------------------------------------------------------------

PC_RANGE = [
    -51.2, -51.2, -5.0,
     51.2,  51.2,  3.0
]

CELL_SIZE = 0.512

GRID_H = 200
GRID_W = 200

PATCH_SIZE = 4
PATCH_GRID = GRID_H // PATCH_SIZE   # 50

# Ego origin in BEV world coordinate
EGO_X = 0.0
EGO_Y = 0.0

# 5m distance bins
DISTANCE_BINS = np.arange(
    0.0,
    75.0 + 5.0,
    5.0
)


# ============================================================
# Load G3 train metadata
# ============================================================

print("=" * 100)
print("STCV-Occ V2-0: Risk / Utility Definition Audit")
print("TRAIN SPLIT ONLY")
print("=" * 100)

data = np.load(
    DATA_PATH,
    allow_pickle=True
)

meta = data["metadata"]
labels = data["labels"]


# ============================================================
# Extract metadata
# ============================================================

BA = np.asarray(
    [int(m["BA"]) for m in meta],
    dtype=np.float64
)

HA = np.asarray(
    [int(m["HA"]) for m in meta],
    dtype=np.float64
)

ADD = np.asarray(
    [int(m["ADD"]) for m in meta],
    dtype=np.float64
)

UTILITY_V1 = np.asarray(
    [int(m["utility"]) for m in meta],
    dtype=np.float64
)

HORIZON = np.asarray(
    [int(m["horizon"]) for m in meta],
    dtype=np.int64
)

PATCH_ROW = np.asarray(
    [int(m["patch_row"]) for m in meta],
    dtype=np.int64
)

PATCH_COL = np.asarray(
    [int(m["patch_col"]) for m in meta],
    dtype=np.int64
)


# ============================================================
# Basic consistency audit
# ============================================================

print("\n[1] V1 DATA CONSISTENCY")

err_add = np.sum(
    ADD != (BA + HA)
)

err_utility = np.sum(
    UTILITY_V1 != (BA - HA)
)

err_label = np.sum(
    labels != (UTILITY_V1 > 0).astype(np.int32)
)

print(
    f"Samples                 : {len(meta):,}"
)

print(
    f"ADD != BA + HA          : {err_add}"
)

print(
    f"utility != BA - HA      : {err_utility}"
)

print(
    f"label != 1[utility > 0] : {err_label}"
)

if (
    err_add != 0
    or err_utility != 0
    or err_label != 0
):
    raise RuntimeError(
        "V1 metadata consistency check FAILED."
    )

print("Consistency             : PASS")


# ============================================================
# P4 patch center -> world coordinate
#
# patch_col c:
#   cells = 4c ... 4c+3
#
# patch center:
#   x = xmin + (4c + 2.0) * 0.512
#
# same for y
# ============================================================

CENTER_X = (
    PC_RANGE[0]
    + (
        PATCH_COL * PATCH_SIZE
        + PATCH_SIZE / 2.0
    )
    * CELL_SIZE
)

CENTER_Y = (
    PC_RANGE[1]
    + (
        PATCH_ROW * PATCH_SIZE
        + PATCH_SIZE / 2.0
    )
    * CELL_SIZE
)

DISTANCE = np.sqrt(
    (CENTER_X - EGO_X) ** 2
    + (CENTER_Y - EGO_Y) ** 2
)

PATCH_WAR = HA / ADD

LABEL_V1 = (
    UTILITY_V1 > 0
)


# ============================================================
# V2-A magnitude target diagnostic
#
# Preserve original marginal utility:
#
#       U = BA - HA
#
# P4 max cell count = 16
#
#       u_mag = (BA - HA) / 16
#
# Range:
#       [-1, 1]
# ============================================================

U_MAG = (
    UTILITY_V1
    / float(PATCH_SIZE * PATCH_SIZE)
)


print("\n" + "=" * 100)
print("[2] V2-A MAGNITUDE TARGET")
print("=" * 100)

print(
    f"Positive patches        : "
    f"{LABEL_V1.sum():,} "
    f"({LABEL_V1.mean():.4f})"
)

print(
    f"Negative/zero patches   : "
    f"{(~LABEL_V1).sum():,} "
    f"({(~LABEL_V1).mean():.4f})"
)

print(
    f"Raw utility mean        : "
    f"{UTILITY_V1.mean():.6f}"
)

print(
    f"Raw utility median      : "
    f"{np.median(UTILITY_V1):.6f}"
)

print(
    f"Raw utility min/max     : "
    f"{UTILITY_V1.min():.0f} / "
    f"{UTILITY_V1.max():.0f}"
)

print(
    f"u_mag mean              : "
    f"{U_MAG.mean():.6f}"
)

print(
    f"u_mag std               : "
    f"{U_MAG.std():.6f}"
)

print(
    f"u_mag min/max           : "
    f"{U_MAG.min():.6f} / "
    f"{U_MAG.max():.6f}"
)

print("\nQuantiles of u_mag:")

for q in [
    0,
    5,
    10,
    25,
    50,
    75,
    90,
    95,
    100
]:
    val = np.percentile(
        U_MAG,
        q
    )

    print(
        f"p{q:>3}: {val: .6f}"
    )

print("\nMagnitude concentration:")

print(
    "|u| <= 1/16 : "
    f"{np.mean(np.abs(U_MAG) <= 1/16):.4f}"
)

print(
    "|u| <= 2/16 : "
    f"{np.mean(np.abs(U_MAG) <= 2/16):.4f}"
)

print(
    "|u| >= 8/16 : "
    f"{np.mean(np.abs(U_MAG) >= 8/16):.4f}"
)


# ============================================================
# Horizon-wise statistics
# ============================================================

print("\n" + "=" * 100)
print("[3] HORIZON-WISE NEGATIVE COOPERATION")
print("=" * 100)

print(
    "h | patches | pos_rate | "
    "ADD     | BA      | HA      | "
    "WAR     | mean_U"
)

print("-" * 100)

for h in range(5):

    mask = (
        HORIZON == h
    )

    n = int(
        mask.sum()
    )

    add = ADD[mask].sum()
    ba = BA[mask].sum()
    ha = HA[mask].sum()

    war = (
        ha / add
        if add > 0
        else np.nan
    )

    pos_rate = (
        LABEL_V1[mask].mean()
        if n > 0
        else np.nan
    )

    mean_u = (
        UTILITY_V1[mask].mean()
        if n > 0
        else np.nan
    )

    print(
        f"{h} | "
        f"{n:7d} | "
        f"{pos_rate:8.4f} | "
        f"{add:7.0f} | "
        f"{ba:7.0f} | "
        f"{ha:7.0f} | "
        f"{war:7.4f} | "
        f"{mean_u:8.4f}"
    )


# ============================================================
# Distance-wise statistics
# ============================================================

print("\n" + "=" * 100)
print("[4] DISTANCE-WISE NEGATIVE COOPERATION")
print("=" * 100)

print(
    "dist(m) | patches | pos_rate | "
    "ADD     | BA      | HA      | "
    "WAR     | mean_U"
)

print("-" * 105)

for i in range(
    len(DISTANCE_BINS) - 1
):

    lo = DISTANCE_BINS[i]
    hi = DISTANCE_BINS[i + 1]

    mask = (
        (DISTANCE >= lo)
        & (DISTANCE < hi)
    )

    n = int(
        mask.sum()
    )

    if n == 0:
        continue

    add = ADD[mask].sum()
    ba = BA[mask].sum()
    ha = HA[mask].sum()

    war = (
        ha / add
        if add > 0
        else np.nan
    )

    pos_rate = (
        LABEL_V1[mask].mean()
    )

    mean_u = (
        UTILITY_V1[mask].mean()
    )

    print(
        f"{lo:4.0f}-{hi:<4.0f} | "
        f"{n:7d} | "
        f"{pos_rate:8.4f} | "
        f"{add:7.0f} | "
        f"{ba:7.0f} | "
        f"{ha:7.0f} | "
        f"{war:7.4f} | "
        f"{mean_u:8.4f}"
    )


# ============================================================
# Correlation
# ============================================================

print("\n" + "=" * 100)
print("[5] PATCH-LEVEL CORRELATION")
print("=" * 100)

rho_war, p_war = spearmanr(
    DISTANCE,
    PATCH_WAR
)

rho_u, p_u = spearmanr(
    DISTANCE,
    UTILITY_V1
)

print(
    "distance vs patch WAR : "
    f"rho={rho_war:.6f}, "
    f"p={p_war:.6e}"
)

print(
    "distance vs utility   : "
    f"rho={rho_u:.6f}, "
    f"p={p_u:.6e}"
)


# ============================================================
# Fixed beta sensitivity
#
# Candidate V2 form:
#
#       U_beta = BA - beta * HA
#
# This section ONLY checks sensitivity.
# It does NOT select beta.
# ============================================================

print("\n" + "=" * 100)
print("[6] ASYMMETRIC HARMFUL-COST BETA SENSITIVITY")
print("=" * 100)

print(
    "beta | positive_ratio | "
    "flip_vs_V1 | "
    "V1_positive_kept | "
    "V1_positive_removed"
)

print("-" * 90)

betas = [
    1.00,
    1.10,
    1.25,
    1.50,
    2.00
]

v1_positive_count = (
    LABEL_V1.sum()
)

for beta in betas:

    utility_beta = (
        BA
        - beta * HA
    )

    label_beta = (
        utility_beta > 0
    )

    positive_ratio = (
        label_beta.mean()
    )

    flip_ratio = np.mean(
        label_beta
        != LABEL_V1
    )

    positive_kept = (
        np.sum(
            label_beta
            & LABEL_V1
        )
        / v1_positive_count
    )

    positive_removed = (
        np.sum(
            (~label_beta)
            & LABEL_V1
        )
        / v1_positive_count
    )

    print(
        f"{beta:4.2f} | "
        f"{positive_ratio:14.4f} | "
        f"{flip_ratio:11.4f} | "
        f"{positive_kept:17.4f} | "
        f"{positive_removed:20.4f}"
    )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 100)
print("[7] CURRENT STATUS")
print("=" * 100)

print(
    """
This script does NOT train STCV-v2.

Current V2 pipeline:

V2-0:
    Utility definition audit
    <-- CURRENT STAGE

V2-1:
    Magnitude-aware utility learning

V2-2:
    Uncertainty-aware adaptive decision

V2-3:
    Runtime integration

V2-4:
    Official evaluation

Current V2-A candidate target:

    u_mag = (BA - HA) / 16

Risk-aware utility is NOT frozen yet.

We will decide whether distance / horizon / asymmetric HA cost
should enter V2 utility only after reviewing this audit.
"""
)

print("V2-0 audit finished.")
