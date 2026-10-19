#!/usr/bin/env python3

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


ROOT = Path("/root/autodl-tmp/UniV2X")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE = (
    ROOT
    / "G0-G1-G2/V2/V2-3_adaptive_decision"
)

TRAIN_SCRIPT = (
    BASE
    / "V2-3D_train_planning_utility_baseline.py"
)

VAL_FILE = (
    BASE
    / "V2-3D_val_planning_utility_dataset.npz"
)

CKPT_FILE = (
    BASE
    / "checkpoints/V2-3D-1_baseline/best.pth"
)

OUT_DIR = (
    BASE
    / "results/V2-3D-2_uncertainty"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NPZ_OUT = (
    OUT_DIR
    / "V2-3D-2_val_mc_dropout_predictions.npz"
)

JSON_OUT = (
    OUT_DIR
    / "V2-3D-2_uncertainty_summary.json"
)


# ============================================================================
# Import exact predictor class
# ============================================================================

spec = importlib.util.spec_from_file_location(
    "v23d1",
    TRAIN_SCRIPT,
)

m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


# ============================================================================
# Load
# ============================================================================

assert VAL_FILE.exists()
assert CKPT_FILE.exists()

with np.load(
    VAL_FILE,
    allow_pickle=False,
) as d:

    val = {
        k: d[k].copy()
        for k in d.files
    }


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Fixed MC-dropout seed for reproducible uncertainty estimation.
SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ckpt = torch.load(
    CKPT_FILE,
    map_location=device,
)

extra_mean = np.asarray(
    ckpt["extra_mean"],
    dtype=np.float32,
)

extra_std = np.asarray(
    ckpt["extra_std"],
    dtype=np.float32,
)

utility_scale = float(
    ckpt["utility_scale"]
)


x_val, _, _ = m.build_input(
    val,
    extra_mean=extra_mean,
    extra_std=extra_std,
)


y_true = val[
    "plan_utility"
].astype(
    np.float64
)

label_raw = val[
    "plan_label"
].astype(
    np.int64
)


model = m.PlanningUtilityPredictor(
    dropout=0.1
).to(
    device
)

model.load_state_dict(
    ckpt["model"],
    strict=True,
)


# ============================================================================
# MC Dropout
# ============================================================================

N_MC = 50

# Enable dropout deliberately.
# LayerNorm has no running statistics, so train() here is safe for MC dropout.
model.train()

x = torch.from_numpy(
    x_val
).to(
    device
)


utility_mc = []
sign_mc = []


with torch.no_grad():

    for i in range(N_MC):

        utility_scaled, sign_logit = model(
            x
        )

        utility = (
            utility_scaled
            /
            utility_scale
        )

        sign_prob = torch.sigmoid(
            sign_logit
        )

        utility_mc.append(
            utility
            .detach()
            .cpu()
            .numpy()
        )

        sign_mc.append(
            sign_prob
            .detach()
            .cpu()
            .numpy()
        )


utility_mc = np.stack(
    utility_mc,
    axis=0,
)

sign_mc = np.stack(
    sign_mc,
    axis=0,
)


utility_mean = utility_mc.mean(
    axis=0
)

utility_std = utility_mc.std(
    axis=0,
    ddof=0,
)

sign_mean = sign_mc.mean(
    axis=0
)

sign_std = sign_mc.std(
    axis=0,
    ddof=0,
)


abs_error = np.abs(
    utility_mean
    -
    y_true
)


# ============================================================================
# Uncertainty ↔ prediction-error relationship
# ============================================================================

rho_error = float(
    spearmanr(
        utility_std,
        abs_error,
    )[0]
)

if np.isnan(rho_error):
    rho_error = 0.0


# ============================================================================
# Quartile audit
# ============================================================================

order = np.argsort(
    utility_std
)

quartiles = {}

n = len(order)

for q in range(4):

    start = int(
        q * n / 4
    )

    end = int(
        (q + 1) * n / 4
    )

    idx = order[
        start:end
    ]

    quartiles[
        f"Q{q+1}"
    ] = {
        "n":
            int(
                len(idx)
            ),

        "uncertainty_mean":
            float(
                utility_std[
                    idx
                ].mean()
            ),

        "mae":
            float(
                abs_error[
                    idx
                ].mean()
            ),

        "true_utility_mean":
            float(
                y_true[
                    idx
                ].mean()
            ),
    }


# ============================================================================
# Selective prediction:
# retain lowest-uncertainty candidates
# ============================================================================

coverage_stats = {}

for coverage in [
    0.25,
    0.50,
    0.75,
    1.00,
]:

    k = max(
        1,
        int(
            round(
                coverage
                *
                n
            )
        ),
    )

    idx = order[:k]

    coverage_stats[
        str(coverage)
    ] = {
        "n":
            int(k),

        "mae":
            float(
                abs_error[
                    idx
                ].mean()
            ),

        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        (
                            utility_mean[
                                idx
                            ]
                            -
                            y_true[
                                idx
                            ]
                        )
                        ** 2
                    )
                )
            ),

        "uncertainty_mean":
            float(
                utility_std[
                    idx
                ].mean()
            ),
    }


# ============================================================================
# Beneficial classification
# ============================================================================

non_neutral = (
    label_raw != 0
)

binary = (
    label_raw[
        non_neutral
    ]
    == 1
).astype(
    np.int64
)


mean_score = utility_mean[
    non_neutral
]

std_score = utility_std[
    non_neutral
]


base_auroc = float(
    roc_auc_score(
        binary,
        mean_score,
    )
)

base_auprc = float(
    average_precision_score(
        binary,
        mean_score,
    )
)


# ============================================================================
# Lower confidence bound:
#
#   LCB = mu - beta * sigma
#
# beta sweep is diagnostic only.
# We do NOT freeze final beta here.
# ============================================================================

beta_results = {}

for beta in [
    0.0,
    0.25,
    0.5,
    1.0,
    1.5,
    2.0,
]:

    lcb = (
        mean_score
        -
        beta
        *
        std_score
    )

    auroc = float(
        roc_auc_score(
            binary,
            lcb,
        )
    )

    auprc = float(
        average_precision_score(
            binary,
            lcb,
        )
    )

    accept = (
        lcb > 0.0
    )

    if accept.sum() > 0:

        precision = float(
            binary[
                accept
            ].mean()
        )

        true_u_mean = float(
            y_true[
                non_neutral
            ][
                accept
            ].mean()
        )

    else:

        precision = 0.0
        true_u_mean = 0.0

    beta_results[
        str(beta)
    ] = {
        "auroc":
            auroc,

        "auprc":
            auprc,

        "accept_count":
            int(
                accept.sum()
            ),

        "accept_rate":
            float(
                accept.mean()
            ),

        "accepted_beneficial_rate":
            precision,

        "accepted_true_utility_mean":
            true_u_mean,
    }


# ============================================================================
# Save
# ============================================================================

summary = {
    "stage":
        "V2-3D-2 MC-dropout uncertainty audit",

    "mc_passes":
        N_MC,

    "val_candidates":
        int(
            len(y_true)
        ),

    "non_neutral_candidates":
        int(
            non_neutral.sum()
        ),

    "uncertainty_error_spearman":
        rho_error,

    "utility_mean_mae":
        float(
            abs_error.mean()
        ),

    "base_mean_prediction": {
        "auroc":
            base_auroc,

        "auprc":
            base_auprc,
    },

    "quartiles":
        quartiles,

    "coverage":
        coverage_stats,

    "lcb_beta_diagnostic":
        beta_results,

    "test_used":
        False,
}


with open(
    JSON_OUT,
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


np.savez_compressed(
    NPZ_OUT,

    utility_true=(
        y_true.astype(
            np.float32
        )
    ),

    label_true=(
        label_raw.astype(
            np.int8
        )
    ),

    utility_mean=(
        utility_mean.astype(
            np.float32
        )
    ),

    utility_std=(
        utility_std.astype(
            np.float32
        )
    ),

    sign_mean=(
        sign_mean.astype(
            np.float32
        )
    ),

    sign_std=(
        sign_std.astype(
            np.float32
        )
    ),

    abs_error=(
        abs_error.astype(
            np.float32
        )
    ),
)


# ============================================================================
# Print
# ============================================================================

print("=" * 90)
print("V2-3D-2 PLANNING UTILITY UNCERTAINTY AUDIT")
print("=" * 90)

print(
    "Val candidates              :",
    len(y_true),
)

print(
    "Non-neutral candidates      :",
    int(
        non_neutral.sum()
    ),
)

print(
    "MC dropout passes           :",
    N_MC,
)

print(
    "Test split used             : NO"
)

print()

print(
    "Mean-prediction MAE         :",
    f"{abs_error.mean():.8f}",
)

print(
    "Uncertainty-error Spearman  :",
    f"{rho_error:+.6f}",
)

print()

print("UNCERTAINTY QUARTILES")

for q, d in quartiles.items():

    print(
        f"{q}: "
        f"n={d['n']:3d} "
        f"sigma={d['uncertainty_mean']:.7f} "
        f"MAE={d['mae']:.7f}"
    )

print()
print("SELECTIVE COVERAGE")

for coverage, d in coverage_stats.items():

    print(
        f"coverage={float(coverage):.2f} "
        f"n={d['n']:3d} "
        f"MAE={d['mae']:.7f} "
        f"RMSE={d['rmse']:.7f}"
    )

print()
print("MEAN UTILITY CLASSIFICATION")

print(
    f"AUROC={base_auroc:.6f} "
    f"AUPRC={base_auprc:.6f}"
)

print()
print("LCB DIAGNOSTIC")

print(
    f"{'beta':>6s} "
    f"{'AUROC':>10s} "
    f"{'AUPRC':>10s} "
    f"{'Accept':>8s} "
    f"{'Rate':>10s} "
    f"{'Beneficial':>12s} "
    f"{'TrueUmean':>12s}"
)

for beta, d in beta_results.items():

    print(
        f"{float(beta):6.2f} "
        f"{d['auroc']:10.6f} "
        f"{d['auprc']:10.6f} "
        f"{d['accept_count']:8d} "
        f"{d['accept_rate']:10.6f} "
        f"{d['accepted_beneficial_rate']:12.6f} "
        f"{d['accepted_true_utility_mean']:+12.7f}"
    )

print()
print("=" * 90)
print("SAVED")
print("=" * 90)

print("NPZ :", NPZ_OUT)
print("JSON:", JSON_OUT)

print()
print(
    "[V2-3D-2 UNCERTAINTY AUDIT] COMPLETE"
)

print("=" * 90)
