#!/usr/bin/env python3

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path("/root/autodl-tmp/UniV2X")

BASE = (
    ROOT
    / "G0-G1-G2/V2/V2-4_learnable_adaptive_fusion"
)

TRAIN_DATA = (
    BASE
    / "V2-4B_train_learnable_fusion_dataset.npz"
)

VAL_DATA = (
    BASE
    / "V2-4B_val_learnable_fusion_dataset.npz"
)

TRAIN_CURVE = (
    BASE
    / "V2-4C-2A_train_gate_utility_curves.npz"
)

VAL_CURVE = (
    BASE
    / "V2-4C-2A_val_gate_utility_curves.npz"
)

OLD_C_SCRIPT = (
    BASE
    / "V2-4C_train_utility_conditioned_gate.py"
)

OUT_DIR = (
    BASE
    / "results/V2-4C-2B_curve_predictor"
)

CKPT_DIR = (
    BASE
    / "checkpoints/V2-4C-2B_curve_predictor"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CKPT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CKPT_FILE = (
    CKPT_DIR
    / "best.pth"
)

SEED = 2026

BATCH_SIZE = 1024
EPOCHS = 100
PATIENCE = 15

LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.10

UTILITY_SCALE = 100.0

LAMBDA_RANK = 0.25


# =============================================================================
# Reuse exact evaluator from V2-4C
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "v24c",
    OLD_C_SCRIPT,
)

v24c = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    v24c
)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_npz(path):

    assert path.exists(), path

    with np.load(
        path,
        allow_pickle=False,
    ) as d:

        return {
            k: d[k].copy()
            for k in d.files
        }


# =============================================================================
# Model
# =============================================================================

class GateUtilityCurvePredictor(nn.Module):

    def __init__(
        self,
        dropout=0.1,
    ):

        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(129, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.residual = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.down_256 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.down_128 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Predict g=0.1 ... 1.0.
        # g=0 utility is fixed to zero.
        self.utility_head = nn.Linear(
            128,
            10,
        )

    def forward(self, x):

        h = self.input_proj(x)

        h = (
            h
            +
            self.residual(h)
        )

        h = self.down_256(h)
        h = self.down_128(h)

        return self.utility_head(h)


# =============================================================================
# Target preparation
# =============================================================================

def build_input(
    data,
    mean=None,
    std=None,
):

    x127 = data[
        "features_127"
    ].astype(
        np.float32
    )

    extra = np.stack(
        [
            data[
                "v2_mag"
            ].astype(
                np.float32
            ),

            data[
                "v2_sign_prob"
            ].astype(
                np.float32
            ),
        ],
        axis=1,
    )

    if mean is None:

        mean = extra.mean(
            axis=0,
            keepdims=True,
        )

        std = extra.std(
            axis=0,
            keepdims=True,
        )

        std = np.maximum(
            std,
            1e-6,
        )

    extra = (
        extra
        -
        mean
    ) / std

    x = np.concatenate(
        [
            x127,
            extra,
        ],
        axis=1,
    ).astype(
        np.float32
    )

    assert x.shape[1] == 129

    return (
        x,
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


def full_curve_from_pred(pred10):

    zero = torch.zeros(
        (
            pred10.shape[0],
            1,
        ),
        device=pred10.device,
        dtype=pred10.dtype,
    )

    return torch.cat(
        [
            zero,
            pred10,
        ],
        dim=1,
    )


# =============================================================================
# Loss
# =============================================================================

def compute_policy_temperature(
    target_curve_scaled,
):

    spread = (
        target_curve_scaled.max(
            axis=1
        )
        -
        target_curve_scaled.min(
            axis=1
        )
    )

    positive = spread[
        spread > 1e-8
    ]

    temp = float(
        np.median(
            positive
        )
    )

    # Avoid pathological overly sharp/flat targets.
    temp = float(
        np.clip(
            temp,
            0.05,
            0.50,
        )
    )

    return temp


def curve_loss(
    pred10,
    target11,
    policy_temp,
):

    pred11 = full_curve_from_pred(
        pred10
    )

    # -------------------------------------------------------------
    # Numerical curve regression.
    # -------------------------------------------------------------

    loss_reg = F.smooth_l1_loss(
        pred11,
        target11,
        beta=0.10,
    )

    # -------------------------------------------------------------
    # Listwise utility ranking.
    #
    # q gives larger probability to gates with larger true utility.
    # Predictor should reproduce that preference structure.
    # -------------------------------------------------------------

    with torch.no_grad():

        q = torch.softmax(
            target11
            /
            policy_temp,
            dim=1,
        )

    log_p = torch.log_softmax(
        pred11
        /
        policy_temp,
        dim=1,
    )

    loss_rank = F.kl_div(
        log_p,
        q,
        reduction="batchmean",
    )

    loss = (
        loss_reg
        +
        LAMBDA_RANK
        *
        loss_rank
    )

    return (
        loss,
        loss_reg,
        loss_rank,
        pred11,
    )


# =============================================================================
# Prediction
# =============================================================================

@torch.no_grad()
def predict_curves(
    model,
    x,
    device,
):

    model.eval()

    out = []

    for start in range(
        0,
        len(x),
        4096,
    ):

        xb = torch.from_numpy(
            x[
                start:
                start + 4096
            ]
        ).to(
            device
        )

        pred10 = model(
            xb
        )

        pred11 = (
            full_curve_from_pred(
                pred10
            )
        )

        out.append(
            pred11
            .cpu()
            .numpy()
        )

    return np.concatenate(
        out,
        axis=0,
    ).astype(
        np.float32
    )


def gate_from_argmax(
    predicted_curve,
    gate_grid,
):

    idx = np.argmax(
        predicted_curve,
        axis=1,
    )

    return gate_grid[
        idx
    ].astype(
        np.float32
    )


# =============================================================================
# Diagnostics
# =============================================================================

def mean_selected_regret(
    true_curve,
    gates,
    gate_grid,
):

    # gates are exact anchors here.
    idx = np.rint(
        gates * 10.0
    ).astype(
        np.int64
    )

    selected = true_curve[
        np.arange(
            len(true_curve)
        ),
        idx,
    ]

    oracle = true_curve.max(
        axis=1
    )

    regret = (
        oracle
        -
        selected
    )

    return (
        float(
            regret.mean()
        ),
        float(
            np.median(
                regret
            )
        ),
    )


# =============================================================================
# Main
# =============================================================================

def main():

    set_seed(
        SEED
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA required."
        )

    device = torch.device(
        "cuda"
    )

    train = load_npz(
        TRAIN_DATA
    )

    val = load_npz(
        VAL_DATA
    )

    train_curve = load_npz(
        TRAIN_CURVE
    )

    val_curve = load_npz(
        VAL_CURVE
    )

    assert len(
        train[
            "features_127"
        ]
    ) == 86143

    assert len(
        val[
            "features_127"
        ]
    ) == 18555

    assert np.array_equal(
        train[
            "sample_idx"
        ].astype(str),
        train_curve[
            "sample_idx"
        ].astype(str),
    )

    assert np.array_equal(
        val[
            "sample_idx"
        ].astype(str),
        val_curve[
            "sample_idx"
        ].astype(str),
    )

    (
        x_train,
        extra_mean,
        extra_std,
    ) = build_input(
        train
    )

    (
        x_val,
        _,
        _,
    ) = build_input(
        val,
        extra_mean,
        extra_std,
    )

    y_train = (
        UTILITY_SCALE
        *
        train_curve[
            "utility_curve"
        ].astype(
            np.float32
        )
    )

    y_val = (
        UTILITY_SCALE
        *
        val_curve[
            "utility_curve"
        ].astype(
            np.float32
        )
    )

    assert y_train.shape == (
        86143,
        11,
    )

    assert y_val.shape == (
        18555,
        11,
    )

    # U(g=0) must be exactly zero.
    assert float(
        np.abs(
            y_train[:, 0]
        ).max()
    ) == 0.0

    assert float(
        np.abs(
            y_val[:, 0]
        ).max()
    ) == 0.0

    gate_grid = train_curve[
        "gate_grid"
    ].astype(
        np.float32
    )

    assert np.allclose(
        gate_grid,
        np.arange(
            11,
            dtype=np.float32,
        ) / 10.0,
    )

    policy_temp = (
        compute_policy_temperature(
            y_train
        )
    )

    print("=" * 110)
    print(
        "V2-4C-2B GATE UTILITY CURVE LEARNING"
    )
    print("=" * 110)

    print(
        "Train candidates       :",
        len(x_train),
    )

    print(
        "Val candidates         :",
        len(x_val),
    )

    print(
        "Input                  :",
        "127 + V2 magnitude + V2 sign = 129-D",
    )

    print(
        "Output                 :",
        "10 utilities for g=0.1...1.0; U(g=0)=0 fixed",
    )

    print(
        "Utility scale          :",
        UTILITY_SCALE,
    )

    print(
        "Policy temperature     :",
        policy_temp,
    )

    print(
        "Loss                   :",
        "SmoothL1(curve) + 0.25*KL(utility ranking)",
    )

    print(
        "Oracle best gate label :",
        "NOT used",
    )

    print(
        "Test used              :",
        "NO",
    )

    # =========================================================================
    # Full Val evaluator: exact same code as V2-4C
    # =========================================================================

    val_scene_tokens = set(
        np.unique(
            val[
                "scene_token"
            ].astype(str)
        ).tolist()
    )

    valid_file = (
        ROOT
        / "G0-G1-G2/G3/cache_valid_samples.txt"
    )

    full_val_ids = []

    with open(
        valid_file,
        "r",
    ) as f:

        _ = next(
            f,
            None,
        )

        for line in f:

            parts = (
                line.strip()
                .split("\t")
            )

            if len(parts) < 3:
                continue

            if (
                parts[1]
                in
                val_scene_tokens
            ):

                full_val_ids.append(
                    parts[0]
                )

    full_val_ids = np.asarray(
        full_val_ids
    )

    assert len(
        np.unique(
            full_val_ids
        )
    ) == 96

    ego_counts = (
        v24c.build_ego_baseline_counts(
            full_val_ids
        )
    )

    range30_mask = (
        v24c.build_range30_mask(
            val[
                "patch_r"
            ],
            val[
                "patch_c"
            ],
        )
    )

    # Frozen comparison references.
    v2_result = (
        v24c.evaluate_gates(
            val,
            val[
                "v2_accept"
            ].astype(
                np.float32
            ),
            ego_counts,
            range30_mask,
        )
    )

    hard_oracle_result = (
        v24c.evaluate_gates(
            val,
            val_curve[
                "hard_best_gate"
            ].astype(
                np.float32
            ),
            ego_counts,
            range30_mask,
        )
    )

    soft_oracle_result = (
        v24c.evaluate_gates(
            val,
            val_curve[
                "soft_best_gate"
            ].astype(
                np.float32
            ),
            ego_counts,
            range30_mask,
        )
    )

    print()
    print(
        "Reference V2 Hard      :",
        f"30m={v2_result['30m']['iou']:.6f} "
        f"100m={v2_result['100m']['iou']:.6f}",
    )

    print(
        "Reference Hard Oracle  :",
        f"30m={hard_oracle_result['30m']['iou']:.6f} "
        f"100m={hard_oracle_result['100m']['iou']:.6f}",
    )

    print(
        "Reference Soft Oracle  :",
        f"30m={soft_oracle_result['30m']['iou']:.6f} "
        f"100m={soft_oracle_result['100m']['iou']:.6f}",
    )

    # =========================================================================
    # Dataset
    # =========================================================================

    dataset = TensorDataset(
        torch.from_numpy(
            x_train
        ),
        torch.from_numpy(
            y_train
        ),
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        SEED
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )

    model = (
        GateUtilityCurvePredictor(
            dropout=DROPOUT
        )
        .to(
            device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model parameters       :",
        n_params,
    )

    best_iou100 = -1.0
    best_iou30 = -1.0
    best_epoch = -1

    bad_epochs = 0
    history = []

    # =========================================================================
    # Train
    # =========================================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        loss_sum = 0.0
        reg_sum = 0.0
        rank_sum = 0.0
        n_seen = 0

        for xb, yb in loader:

            xb = xb.to(
                device,
                non_blocking=True,
            )

            yb = yb.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred10 = model(
                xb
            )

            (
                loss,
                loss_reg,
                loss_rank,
                _,
            ) = curve_loss(
                pred10,
                yb,
                policy_temp,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            bs = len(xb)

            loss_sum += (
                float(
                    loss.item()
                )
                *
                bs
            )

            reg_sum += (
                float(
                    loss_reg.item()
                )
                *
                bs
            )

            rank_sum += (
                float(
                    loss_rank.item()
                )
                *
                bs
            )

            n_seen += bs

        # ---------------------------------------------------------------------
        # Validation predicted utility curve
        # ---------------------------------------------------------------------

        pred_val = predict_curves(
            model,
            x_val,
            device,
        )

        pred_gate = gate_from_argmax(
            pred_val,
            gate_grid,
        )

        val_result = (
            v24c.evaluate_gates(
                val,
                pred_gate,
                ego_counts,
                range30_mask,
            )
        )

        iou30 = (
            val_result[
                "30m"
            ][
                "iou"
            ]
        )

        iou100 = (
            val_result[
                "100m"
            ][
                "iou"
            ]
        )

        mae = float(
            np.mean(
                np.abs(
                    pred_val
                    -
                    y_val
                )
            )
        )

        regret_mean, regret_med = (
            mean_selected_regret(
                y_val,
                pred_gate,
                gate_grid,
            )
        )

        gate0_rate = float(
            (
                pred_gate
                ==
                0.0
            ).mean()
        )

        interior_rate = float(
            (
                (pred_gate > 0.0)
                &
                (pred_gate < 1.0)
            ).mean()
        )

        gate1_rate = float(
            (
                pred_gate
                ==
                1.0
            ).mean()
        )

        row = {
            "epoch":
                epoch,

            "loss":
                loss_sum
                /
                n_seen,

            "reg":
                reg_sum
                /
                n_seen,

            "rank":
                rank_sum
                /
                n_seen,

            "val_curve_mae":
                mae,

            "val_regret_mean_scaled":
                regret_mean,

            "val_regret_median_scaled":
                regret_med,

            "val_iou30":
                iou30,

            "val_iou100":
                iou100,

            "gate0_rate":
                gate0_rate,

            "interior_rate":
                interior_rate,

            "gate1_rate":
                gate1_rate,
        }

        history.append(
            row
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['loss']:.6f} "
            f"reg={row['reg']:.6f} "
            f"rank={row['rank']:.6f} | "
            f"MAE={mae:.5f} "
            f"regret={regret_mean:.5f} | "
            f"IoU30={iou30:.6f} "
            f"IoU100={iou100:.6f} | "
            f"g0={gate0_rate:.3f} "
            f"mid={interior_rate:.3f} "
            f"g1={gate1_rate:.3f}"
        )

        improved = (
            iou100
            >
            best_iou100
            +
            1e-10
        )

        tied = (
            abs(
                iou100
                -
                best_iou100
            )
            <=
            1e-10
        )

        if (
            improved
            or
            (
                tied
                and
                iou30
                >
                best_iou30
                +
                1e-10
            )
        ):

            best_iou100 = float(
                iou100
            )

            best_iou30 = float(
                iou30
            )

            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model":
                        model.state_dict(),

                    "extra_mean":
                        extra_mean,

                    "extra_std":
                        extra_std,

                    "policy_temp":
                        policy_temp,

                    "utility_scale":
                        UTILITY_SCALE,

                    "lambda_rank":
                        LAMBDA_RANK,

                    "best_iou30":
                        best_iou30,

                    "best_iou100":
                        best_iou100,

                    "seed":
                        SEED,
                },
                CKPT_FILE,
            )

        else:

            bad_epochs += 1

        if (
            bad_epochs
            >=
            PATIENCE
        ):

            print()
            print(
                "Early stopping:"
                f" epoch={epoch},"
                f" best={best_epoch}"
            )

            break

    # =========================================================================
    # Best checkpoint
    # =========================================================================

    ckpt = torch.load(
        CKPT_FILE,
        map_location=device,
    )

    model.load_state_dict(
        ckpt[
            "model"
        ]
    )

    pred_val = predict_curves(
        model,
        x_val,
        device,
    )

    pred_gate = gate_from_argmax(
        pred_val,
        gate_grid,
    )

    final_result = (
        v24c.evaluate_gates(
            val,
            pred_gate,
            ego_counts,
            range30_mask,
        )
    )

    regret_mean, regret_median = (
        mean_selected_regret(
            y_val,
            pred_gate,
            gate_grid,
        )
    )

    # =========================================================================
    # Save
    # =========================================================================

    np.savez_compressed(
        OUT_DIR
        /
        "V2-4C-2B_val_predictions.npz",

        predicted_utility_curve=(
            pred_val
        ),

        true_utility_curve=(
            y_val
        ),

        predicted_gate=(
            pred_gate
        ),

        gate_grid=(
            gate_grid
        ),

        sample_idx=(
            val[
                "sample_idx"
            ]
        ),

        horizon=(
            val[
                "horizon"
            ]
        ),

        patch_r=(
            val[
                "patch_r"
            ]
        ),

        patch_c=(
            val[
                "patch_c"
            ]
        ),
    )

    summary = {
        "stage":
            "V2-4C-2B Gate Utility Curve Learning",

        "best_epoch":
            int(
                ckpt[
                    "epoch"
                ]
            ),

        "policy_temperature":
            float(
                policy_temp
            ),

        "val_iou30":
            float(
                final_result[
                    "30m"
                ][
                    "iou"
                ]
            ),

        "val_iou100":
            float(
                final_result[
                    "100m"
                ][
                    "iou"
                ]
            ),

        "delta_vs_v2_30":
            float(
                final_result[
                    "30m"
                ][
                    "iou"
                ]
                -
                v2_result[
                    "30m"
                ][
                    "iou"
                ]
            ),

        "delta_vs_v2_100":
            float(
                final_result[
                    "100m"
                ][
                    "iou"
                ]
                -
                v2_result[
                    "100m"
                ][
                    "iou"
                ]
            ),

        "mean_selected_regret_scaled":
            float(
                regret_mean
            ),

        "median_selected_regret_scaled":
            float(
                regret_median
            ),

        "gate0_rate":
            float(
                (
                    pred_gate
                    ==
                    0.0
                ).mean()
            ),

        "interior_gate_rate":
            float(
                (
                    (pred_gate > 0.0)
                    &
                    (pred_gate < 1.0)
                ).mean()
            ),

        "gate1_rate":
            float(
                (
                    pred_gate
                    ==
                    1.0
                ).mean()
            ),

        "oracle_best_gate_used_as_label":
            False,

        "test_used":
            False,

        "history":
            history,
    }

    with open(
        OUT_DIR
        /
        "V2-4C-2B_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # =========================================================================
    # Print
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "V2-4C-2B FINAL VALIDATION"
    )
    print("=" * 110)

    print(
        "Best epoch             :",
        ckpt[
            "epoch"
        ],
    )

    print()
    print(
        "V2 Hard               :",
        f"30m={v2_result['30m']['iou']:.6f} "
        f"100m={v2_result['100m']['iou']:.6f}",
    )

    print(
        "Hard Oracle           :",
        f"30m={hard_oracle_result['30m']['iou']:.6f} "
        f"100m={hard_oracle_result['100m']['iou']:.6f}",
    )

    print(
        "Soft Oracle           :",
        f"30m={soft_oracle_result['30m']['iou']:.6f} "
        f"100m={soft_oracle_result['100m']['iou']:.6f}",
    )

    print(
        "Curve Pred Argmax     :",
        f"30m={final_result['30m']['iou']:.6f} "
        f"100m={final_result['100m']['iou']:.6f}",
    )

    print()
    print(
        "Delta vs V2 30m      :",
        f"{final_result['30m']['iou'] - v2_result['30m']['iou']:+.8f}",
    )

    print(
        "Delta vs V2 100m     :",
        f"{final_result['100m']['iou'] - v2_result['100m']['iou']:+.8f}",
    )

    print()
    print(
        "Mean selected regret :",
        f"{regret_mean:.6f}",
        "(scaled utility)",
    )

    print(
        "Gate=0 rate          :",
        f"{(pred_gate == 0).mean():.6%}",
    )

    print(
        "Interior gate rate   :",
        f"{((pred_gate > 0) & (pred_gate < 1)).mean():.6%}",
    )

    print(
        "Gate=1 rate          :",
        f"{(pred_gate == 1).mean():.6%}",
    )

    print()
    print(
        "Test used             : NO"
    )

    print(
        "[V2-4C-2B CURVE LEARNING] COMPLETE"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
