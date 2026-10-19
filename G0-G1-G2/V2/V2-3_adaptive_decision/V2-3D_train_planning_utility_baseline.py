#!/usr/bin/env python3

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from scipy.stats import spearmanr


# =============================================================================
# Paths
# =============================================================================

ROOT = Path("/root/autodl-tmp/UniV2X")

BASE = (
    ROOT
    / "G0-G1-G2/V2/V2-3_adaptive_decision"
)

TRAIN_FILE = (
    BASE
    / "V2-3C_train_planning_utility_dataset.npz"
)

VAL_FILE = (
    BASE
    / "V2-3D_val_planning_utility_dataset.npz"
)

CKPT_DIR = (
    BASE
    / "checkpoints"
    / "V2-3D-1_baseline"
)

RESULT_DIR = (
    BASE
    / "results"
    / "V2-3D-1_baseline"
)

CKPT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CKPT_FILE = (
    CKPT_DIR
    / "best.pth"
)

RESULT_FILE = (
    RESULT_DIR
    / "V2-3D-1_baseline_summary.json"
)

PRED_FILE = (
    RESULT_DIR
    / "V2-3D-1_val_predictions.npz"
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


# =============================================================================
# Data
# =============================================================================

def load_npz(path):

    assert path.exists(), path

    with np.load(
        path,
        allow_pickle=False,
    ) as d:

        data = {
            key: d[key].copy()
            for key in d.files
        }

    return data


def build_input(
    data,
    extra_mean=None,
    extra_std=None,
):
    """
    131-D input:

      127-D original STCV feature
        +
      4-D planning/decision context:
        1. min_plan_dist_m
        2. v2_mag
        3. v2_sign_prob
        4. v2_sign_margin

    Extra features are standardized using TRAIN statistics only.
    """

    x127 = data[
        "features_127"
    ].astype(
        np.float32
    )

    assert x127.ndim == 2
    assert x127.shape[1] == 127

    extra = np.stack(
        [
            data[
                "min_plan_dist_m"
            ].astype(
                np.float32
            ),

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

            data[
                "v2_sign_margin"
            ].astype(
                np.float32
            ),
        ],
        axis=1,
    )

    if extra_mean is None:

        extra_mean = extra.mean(
            axis=0,
            keepdims=True,
        )

        extra_std = extra.std(
            axis=0,
            keepdims=True,
        )

        extra_std = np.maximum(
            extra_std,
            1e-6,
        )

    extra_norm = (
        extra
        -
        extra_mean
    ) / extra_std

    x = np.concatenate(
        [
            x127,
            extra_norm,
        ],
        axis=1,
    ).astype(
        np.float32
    )

    assert x.shape[1] == 131

    return (
        x,
        extra_mean.astype(
            np.float32
        ),
        extra_std.astype(
            np.float32
        ),
    )


# =============================================================================
# Model
# =============================================================================

class PlanningUtilityPredictor(nn.Module):

    def __init__(
        self,
        dropout=0.1,
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                131,
                256,
            ),

            nn.LayerNorm(
                256
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                256,
                128,
            ),

            nn.LayerNorm(
                128
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                64,
            ),

            nn.LayerNorm(
                64
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),
        )

        # Regression:
        # predicts scaled planning utility.
        self.utility_head = nn.Linear(
            64,
            1,
        )

        # Auxiliary sign head:
        # beneficial vs harmful.
        self.sign_head = nn.Linear(
            64,
            1,
        )

    def forward(
        self,
        x,
    ):

        h = self.encoder(
            x
        )

        utility = (
            self.utility_head(
                h
            )
            .squeeze(1)
        )

        sign_logit = (
            self.sign_head(
                h
            )
            .squeeze(1)
        )

        return (
            utility,
            sign_logit,
        )


# =============================================================================
# Metrics
# =============================================================================

def safe_spearman(
    target,
    pred,
):

    # scipy-version compatible:
    # old scipy -> SpearmanrResult(correlation, pvalue)
    # new scipy -> SignificanceResult(statistic, pvalue)
    rho = spearmanr(
        target,
        pred,
    )[0]

    if np.isnan(rho):
        return 0.0

    return float(rho)


def regression_metrics(
    target,
    pred,
):

    err = (
        pred
        -
        target
    )

    return {
        "mae": float(
            np.mean(
                np.abs(err)
            )
        ),

        "rmse": float(
            np.sqrt(
                np.mean(
                    err ** 2
                )
            )
        ),

        "spearman": safe_spearman(
            target,
            pred,
        ),

        "target_mean": float(
            np.mean(target)
        ),

        "pred_mean": float(
            np.mean(pred)
        ),
    }


def classification_metrics(
    labels,
    scores,
    threshold=0.5,
):

    pred = (
        scores
        >= threshold
    ).astype(
        np.int64
    )

    return {
        "auroc": float(
            roc_auc_score(
                labels,
                scores,
            )
        ),

        "auprc": float(
            average_precision_score(
                labels,
                scores,
            )
        ),

        "accuracy": float(
            accuracy_score(
                labels,
                pred,
            )
        ),

        "precision": float(
            precision_score(
                labels,
                pred,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                labels,
                pred,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                labels,
                pred,
                zero_division=0,
            )
        ),

        "threshold": float(
            threshold
        ),

        "positive_rate_gt": float(
            labels.mean()
        ),

        "positive_rate_pred": float(
            pred.mean()
        ),
    }


def find_best_f1_threshold(
    labels,
    scores,
):

    best_t = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(
        0.01,
        0.99,
        99,
    ):

        pred = (
            scores
            >= threshold
        ).astype(
            np.int64
        )

        score = f1_score(
            labels,
            pred,
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = float(
                score
            )

            best_t = float(
                threshold
            )

    return (
        best_t,
        best_f1,
    )


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def predict(
    model,
    x,
    device,
    utility_scale,
    batch_size=2048,
):

    model.eval()

    utility_all = []
    sign_all = []

    for start in range(
        0,
        len(x),
        batch_size,
    ):

        xb = torch.from_numpy(
            x[
                start:
                start + batch_size
            ]
        ).to(
            device,
        )

        (
            utility_scaled,
            sign_logit,
        ) = model(
            xb
        )

        utility_all.append(
            (
                utility_scaled
                /
                utility_scale
            )
            .detach()
            .cpu()
            .numpy()
        )

        sign_all.append(
            sign_logit
            .detach()
            .cpu()
            .numpy()
        )

    utility_pred = np.concatenate(
        utility_all
    )

    sign_logit = np.concatenate(
        sign_all
    )

    sign_prob = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -sign_logit
            )
        )
    )

    return (
        utility_pred,
        sign_logit,
        sign_prob,
    )


# =============================================================================
# Training
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lambda-sign",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--utility-scale",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    # =========================================================================
    # Load datasets
    # =========================================================================

    train = load_npz(
        TRAIN_FILE
    )

    val = load_npz(
        VAL_FILE
    )

    x_train, extra_mean, extra_std = (
        build_input(
            train
        )
    )

    x_val, _, _ = build_input(
        val,
        extra_mean=extra_mean,
        extra_std=extra_std,
    )

    y_train_utility = train[
        "plan_utility"
    ].astype(
        np.float32
    )

    y_val_utility = val[
        "plan_utility"
    ].astype(
        np.float32
    )

    y_train_label_raw = train[
        "plan_label"
    ].astype(
        np.int64
    )

    y_val_label_raw = val[
        "plan_label"
    ].astype(
        np.int64
    )

    assert len(x_train) == 3649
    assert len(x_val) == 306

    assert x_train.shape == (
        3649,
        131,
    )

    assert x_val.shape == (
        306,
        131,
    )

    # -------------------------------------------------------------------------
    # Exact historical label audit
    # -------------------------------------------------------------------------

    train_beneficial = int(
        (
            y_train_label_raw
            == 1
        ).sum()
    )

    train_harmful = int(
        (
            y_train_label_raw
            == -1
        ).sum()
    )

    train_neutral = int(
        (
            y_train_label_raw
            == 0
        ).sum()
    )

    val_beneficial = int(
        (
            y_val_label_raw
            == 1
        ).sum()
    )

    val_harmful = int(
        (
            y_val_label_raw
            == -1
        ).sum()
    )

    val_neutral = int(
        (
            y_val_label_raw
            == 0
        ).sum()
    )

    assert (
        train_beneficial,
        train_harmful,
        train_neutral,
    ) == (
        1338,
        2184,
        127,
    )

    assert (
        val_beneficial,
        val_harmful,
        val_neutral,
    ) == (
        133,
        167,
        6,
    )

    # Sign target:
    #
    # beneficial = 1
    # harmful    = 0
    # neutral    = ignored in BCE
    #
    y_train_sign = (
        y_train_label_raw
        == 1
    ).astype(
        np.float32
    )

    y_val_sign = (
        y_val_label_raw
        == 1
    ).astype(
        np.float32
    )

    train_sign_mask = (
        y_train_label_raw
        != 0
    ).astype(
        np.float32
    )

    val_sign_mask = (
        y_val_label_raw
        != 0
    ).astype(
        bool
    )

    # =========================================================================
    # Dataset / loader
    # =========================================================================

    dataset = TensorDataset(

        torch.from_numpy(
            x_train
        ),

        torch.from_numpy(
            y_train_utility
            *
            args.utility_scale
        ),

        torch.from_numpy(
            y_train_sign
        ),

        torch.from_numpy(
            train_sign_mask
        ),
    )

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )

    loader = DataLoader(

        dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=0,

        pin_memory=True,

        generator=generator,
    )

    # =========================================================================
    # Model
    # =========================================================================

    model = PlanningUtilityPredictor(
        dropout=args.dropout
    ).to(
        device
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    # Class balancing among NON-NEUTRAL training candidates.
    pos_weight_value = (
        train_harmful
        /
        train_beneficial
    )

    pos_weight = torch.tensor(
        [
            pos_weight_value
        ],
        dtype=torch.float32,
        device=device,
    )

    utility_loss_fn = nn.SmoothL1Loss(
        beta=1.0
    )

    sign_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
        reduction="none",
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=args.lr,

        weight_decay=args.weight_decay,
    )

    print("=" * 100)
    print("V2-3D-1 BASELINE PLANNING UTILITY PREDICTOR")
    print("=" * 100)

    print(
        "Train candidates       :",
        len(x_train),
    )

    print(
        "  beneficial           :",
        train_beneficial,
    )

    print(
        "  harmful              :",
        train_harmful,
    )

    print(
        "  neutral              :",
        train_neutral,
    )

    print(
        "Val candidates         :",
        len(x_val),
    )

    print(
        "  beneficial           :",
        val_beneficial,
    )

    print(
        "  harmful              :",
        val_harmful,
    )

    print(
        "  neutral              :",
        val_neutral,
    )

    print(
        "Input dim              :",
        131,
    )

    print(
        "Parameters             :",
        n_params,
    )

    print(
        "Utility target         :",
        f"{args.utility_scale} * U_plan",
    )

    print(
        "Regression loss        : SmoothL1(beta=1)"
    )

    print(
        "Sign loss              : BCEWithLogits"
    )

    print(
        "lambda_sign            :",
        args.lambda_sign,
    )

    print(
        "pos_weight             :",
        f"{pos_weight_value:.6f}",
    )

    print(
        "Checkpoint criterion   : Val utility-score AUPRC"
    )

    print(
        "Test split used        : NO"
    )

    print()

    print(
        "Extra-feature mean:",
        extra_mean.reshape(-1),
    )

    print(
        "Extra-feature std :",
        extra_std.reshape(-1),
    )

    # =========================================================================
    # Existing V2 baselines on Val
    # =========================================================================

    val_non_neutral = (
        y_val_label_raw
        != 0
    )

    val_binary = (
        y_val_label_raw[
            val_non_neutral
        ]
        == 1
    ).astype(
        np.int64
    )

    val_v2_mag = val[
        "v2_mag"
    ][
        val_non_neutral
    ]

    val_v2_sign_prob = val[
        "v2_sign_prob"
    ][
        val_non_neutral
    ]

    val_distance_score = (
        -val[
            "min_plan_dist_m"
        ][
            val_non_neutral
        ]
    )

    baseline_metrics = {

        "positive_prevalence": float(
            val_binary.mean()
        ),

        "v2_mag": {
            "auroc": float(
                roc_auc_score(
                    val_binary,
                    val_v2_mag,
                )
            ),

            "auprc": float(
                average_precision_score(
                    val_binary,
                    val_v2_mag,
                )
            ),
        },

        "v2_sign_prob": {
            "auroc": float(
                roc_auc_score(
                    val_binary,
                    val_v2_sign_prob,
                )
            ),

            "auprc": float(
                average_precision_score(
                    val_binary,
                    val_v2_sign_prob,
                )
            ),
        },

        "distance_only": {
            "auroc": float(
                roc_auc_score(
                    val_binary,
                    val_distance_score,
                )
            ),

            "auprc": float(
                average_precision_score(
                    val_binary,
                    val_distance_score,
                )
            ),
        },
    }

    print()
    print("=" * 100)
    print("VAL BASELINES BEFORE TRAINING")
    print("=" * 100)

    print(
        "Random AUPRC reference :",
        f"{baseline_metrics['positive_prevalence']:.6f}",
    )

    print(
        "V2 magnitude           :",
        f"AUROC={baseline_metrics['v2_mag']['auroc']:.6f} "
        f"AUPRC={baseline_metrics['v2_mag']['auprc']:.6f}"
    )

    print(
        "V2 sign probability    :",
        f"AUROC={baseline_metrics['v2_sign_prob']['auroc']:.6f} "
        f"AUPRC={baseline_metrics['v2_sign_prob']['auprc']:.6f}"
    )

    print(
        "Distance only          :",
        f"AUROC={baseline_metrics['distance_only']['auroc']:.6f} "
        f"AUPRC={baseline_metrics['distance_only']['auprc']:.6f}"
    )

    # =========================================================================
    # Train
    # =========================================================================

    best_val_auprc = -1.0
    best_epoch = -1
    bad_epochs = 0

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        total_loss_sum = 0.0
        utility_loss_sum = 0.0
        sign_loss_sum = 0.0

        n_seen = 0

        for (
            xb,
            utility_target,
            sign_target,
            sign_mask,
        ) in loader:

            xb = xb.to(
                device,
                non_blocking=True,
            )

            utility_target = (
                utility_target.to(
                    device,
                    non_blocking=True,
                )
            )

            sign_target = (
                sign_target.to(
                    device,
                    non_blocking=True,
                )
            )

            sign_mask = (
                sign_mask.to(
                    device,
                    non_blocking=True,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                utility_pred,
                sign_logit,
            ) = model(
                xb
            )

            loss_utility = (
                utility_loss_fn(
                    utility_pred,
                    utility_target,
                )
            )

            per_item_sign_loss = (
                sign_loss_fn(
                    sign_logit,
                    sign_target,
                )
            )

            valid_sign_count = (
                sign_mask.sum()
                .clamp_min(
                    1.0
                )
            )

            loss_sign = (
                (
                    per_item_sign_loss
                    *
                    sign_mask
                ).sum()
                /
                valid_sign_count
            )

            loss = (
                loss_utility
                +
                args.lambda_sign
                *
                loss_sign
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            bs = len(xb)

            total_loss_sum += (
                loss.item()
                *
                bs
            )

            utility_loss_sum += (
                loss_utility.item()
                *
                bs
            )

            sign_loss_sum += (
                loss_sign.item()
                *
                bs
            )

            n_seen += bs

        train_total_loss = (
            total_loss_sum
            /
            n_seen
        )

        train_utility_loss = (
            utility_loss_sum
            /
            n_seen
        )

        train_sign_loss = (
            sign_loss_sum
            /
            n_seen
        )

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        (
            val_utility_pred,
            val_sign_logit,
            val_sign_prob,
        ) = predict(

            model,

            x_val,

            device,

            args.utility_scale,
        )

        reg = regression_metrics(
            y_val_utility,
            val_utility_pred,
        )

        # Utility prediction itself is the principal score:
        # higher predicted U_plan => more likely beneficial.
        utility_score = (
            val_utility_pred[
                val_non_neutral
            ]
        )

        utility_auroc = float(
            roc_auc_score(
                val_binary,
                utility_score,
            )
        )

        utility_auprc = float(
            average_precision_score(
                val_binary,
                utility_score,
            )
        )

        sign_auroc = float(
            roc_auc_score(
                val_binary,
                val_sign_prob[
                    val_non_neutral
                ],
            )
        )

        sign_auprc = float(
            average_precision_score(
                val_binary,
                val_sign_prob[
                    val_non_neutral
                ],
            )
        )

        row = {
            "epoch": int(epoch),
            "train_total_loss": float(
                train_total_loss
            ),
            "train_utility_loss": float(
                train_utility_loss
            ),
            "train_sign_loss": float(
                train_sign_loss
            ),
            "val_mae": reg["mae"],
            "val_rmse": reg["rmse"],
            "val_spearman": reg[
                "spearman"
            ],
            "val_utility_auroc": (
                utility_auroc
            ),
            "val_utility_auprc": (
                utility_auprc
            ),
            "val_sign_auroc": (
                sign_auroc
            ),
            "val_sign_auprc": (
                sign_auprc
            ),
        }

        history.append(
            row
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_total_loss:.6f} "
            f"(U={train_utility_loss:.6f}, "
            f"S={train_sign_loss:.6f}) | "
            f"Val MAE={reg['mae']:.6f} "
            f"RMSE={reg['rmse']:.6f} "
            f"rho={reg['spearman']:+.4f} | "
            f"U-AUROC={utility_auroc:.4f} "
            f"U-AUPRC={utility_auprc:.4f} | "
            f"S-AUROC={sign_auroc:.4f} "
            f"S-AUPRC={sign_auprc:.4f}"
        )

        # ---------------------------------------------------------------------
        # Checkpoint:
        # main utility prediction AUPRC.
        # ---------------------------------------------------------------------

        if (
            utility_auprc
            >
            best_val_auprc
            +
            1e-8
        ):

            best_val_auprc = (
                utility_auprc
            )

            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "epoch":
                        int(epoch),

                    "model":
                        model.state_dict(),

                    "extra_mean":
                        extra_mean,

                    "extra_std":
                        extra_std,

                    "utility_scale":
                        float(
                            args.utility_scale
                        ),

                    "lambda_sign":
                        float(
                            args.lambda_sign
                        ),

                    "best_val_utility_auprc":
                        float(
                            best_val_auprc
                        ),

                    "args":
                        vars(args),

                    "input_dim":
                        131,

                    "architecture":
                        "131-256-128-64 dual-head",
                },
                CKPT_FILE,
            )

        else:

            bad_epochs += 1

        if (
            bad_epochs
            >= args.patience
        ):

            print()
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch = {best_epoch}"
            )

            break

    # =========================================================================
    # Load best
    # =========================================================================

    ckpt = torch.load(
        CKPT_FILE,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    (
        val_utility_pred,
        val_sign_logit,
        val_sign_prob,
    ) = predict(
        model,
        x_val,
        device,
        args.utility_scale,
    )

    # =========================================================================
    # Final validation evaluation
    # =========================================================================

    val_regression = regression_metrics(
        y_val_utility,
        val_utility_pred,
    )

    utility_scores_non_neutral = (
        val_utility_pred[
            val_non_neutral
        ]
    )

    sign_scores_non_neutral = (
        val_sign_prob[
            val_non_neutral
        ]
    )

    # Utility head:
    # threshold zero has semantic meaning:
    # predicted positive planning utility.
    val_utility_zero = (
        utility_scores_non_neutral
        >= 0.0
    ).astype(
        np.int64
    )

    utility_zero_metrics = {
        "auroc": float(
            roc_auc_score(
                val_binary,
                utility_scores_non_neutral,
            )
        ),

        "auprc": float(
            average_precision_score(
                val_binary,
                utility_scores_non_neutral,
            )
        ),

        "accuracy": float(
            accuracy_score(
                val_binary,
                val_utility_zero,
            )
        ),

        "precision": float(
            precision_score(
                val_binary,
                val_utility_zero,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                val_binary,
                val_utility_zero,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                val_binary,
                val_utility_zero,
                zero_division=0,
            )
        ),

        "threshold":
            0.0,

        "positive_rate_pred": float(
            val_utility_zero.mean()
        ),
    }

    (
        sign_best_threshold,
        sign_best_f1,
    ) = find_best_f1_threshold(
        val_binary,
        sign_scores_non_neutral,
    )

    sign_metrics_050 = (
        classification_metrics(
            val_binary,
            sign_scores_non_neutral,
            threshold=0.5,
        )
    )

    sign_metrics_best = (
        classification_metrics(
            val_binary,
            sign_scores_non_neutral,
            threshold=(
                sign_best_threshold
            ),
        )
    )

    # =========================================================================
    # Diagnostic: top utility predictions
    # =========================================================================

    order = np.argsort(
        -val_utility_pred
    )

    topk_stats = {}

    for k in [
        10,
        25,
        50,
        100,
    ]:

        k_eff = min(
            k,
            len(order),
        )

        idx = order[
            :k_eff
        ]

        positive = (
            y_val_label_raw[
                idx
            ]
            == 1
        )

        harmful = (
            y_val_label_raw[
                idx
            ]
            == -1
        )

        neutral = (
            y_val_label_raw[
                idx
            ]
            == 0
        )

        topk_stats[
            str(k)
        ] = {
            "beneficial":
                int(
                    positive.sum()
                ),

            "harmful":
                int(
                    harmful.sum()
                ),

            "neutral":
                int(
                    neutral.sum()
                ),

            "beneficial_rate":
                float(
                    positive.mean()
                ),

            "mean_true_Uplan":
                float(
                    y_val_utility[
                        idx
                    ].mean()
                ),

            "mean_pred_Uplan":
                float(
                    val_utility_pred[
                        idx
                    ].mean()
                ),
        }

    # =========================================================================
    # Output
    # =========================================================================

    summary = {

        "stage":
            "V2-3D-1 baseline planning utility predictor",

        "split": {
            "train_candidates":
                int(
                    len(x_train)
                ),

            "val_candidates":
                int(
                    len(x_val)
                ),

            "test_used":
                False,

            "train_beneficial":
                train_beneficial,

            "train_harmful":
                train_harmful,

            "train_neutral":
                train_neutral,

            "val_beneficial":
                val_beneficial,

            "val_harmful":
                val_harmful,

            "val_neutral":
                val_neutral,
        },

        "model": {
            "input_dim":
                131,

            "architecture":
                "131-256-128-64 dual-head",

            "num_parameters":
                int(
                    n_params
                ),

            "utility_scale":
                float(
                    args.utility_scale
                ),

            "lambda_sign":
                float(
                    args.lambda_sign
                ),

            "pos_weight":
                float(
                    pos_weight_value
                ),
        },

        "training": {
            "best_epoch":
                int(
                    ckpt["epoch"]
                ),

            "best_val_utility_auprc":
                float(
                    ckpt[
                        "best_val_utility_auprc"
                    ]
                ),

            "seed":
                int(
                    args.seed
                ),
        },

        "existing_baselines":
            baseline_metrics,

        "val_regression":
            val_regression,

        "val_utility_head_zero":
            utility_zero_metrics,

        "val_sign_head_050":
            sign_metrics_050,

        "val_sign_head_best_f1": {
            **sign_metrics_best,
            "best_threshold":
                float(
                    sign_best_threshold
                ),
            "best_f1":
                float(
                    sign_best_f1
                ),
        },

        "topk":
            topk_stats,

        "history":
            history,
    }

    with open(
        RESULT_FILE,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    np.savez_compressed(

        PRED_FILE,

        utility_true=(
            y_val_utility.astype(
                np.float32
            )
        ),

        label_true=(
            y_val_label_raw.astype(
                np.int8
            )
        ),

        utility_pred=(
            val_utility_pred.astype(
                np.float32
            )
        ),

        sign_logit=(
            val_sign_logit.astype(
                np.float32
            )
        ),

        sign_prob=(
            val_sign_prob.astype(
                np.float32
            )
        ),

        min_plan_dist_m=(
            val[
                "min_plan_dist_m"
            ].astype(
                np.float32
            )
        ),

        v2_mag=(
            val[
                "v2_mag"
            ].astype(
                np.float32
            )
        ),

        v2_accept=(
            val[
                "v2_accept"
            ].astype(
                np.uint8
            )
        ),

        horizon=(
            val[
                "horizon"
            ].astype(
                np.int64
            )
        ),

        patch_r=(
            val[
                "patch_r"
            ].astype(
                np.int64
            )
        ),

        patch_c=(
            val[
                "patch_c"
            ].astype(
                np.int64
            )
        ),
    )

    # =========================================================================
    # Print
    # =========================================================================

    print()
    print("=" * 100)
    print("V2-3D-1 FINAL VALIDATION")
    print("=" * 100)

    print(
        "Best epoch             :",
        ckpt["epoch"],
    )

    print(
        "Random AUPRC reference :",
        f"{baseline_metrics['positive_prevalence']:.6f}",
    )

    print()
    print(
        "Existing V2 magnitude  :",
        f"AUROC={baseline_metrics['v2_mag']['auroc']:.6f} "
        f"AUPRC={baseline_metrics['v2_mag']['auprc']:.6f}"
    )

    print(
        "Distance-only          :",
        f"AUROC={baseline_metrics['distance_only']['auroc']:.6f} "
        f"AUPRC={baseline_metrics['distance_only']['auprc']:.6f}"
    )

    print()
    print("Planning utility regression:")

    print(
        "  MAE                  :",
        f"{val_regression['mae']:.8f}",
    )

    print(
        "  RMSE                 :",
        f"{val_regression['rmse']:.8f}",
    )

    print(
        "  Spearman             :",
        f"{val_regression['spearman']:+.6f}",
    )

    print()
    print("Utility head as beneficial score:")

    print(
        "  AUROC                :",
        f"{utility_zero_metrics['auroc']:.6f}",
    )

    print(
        "  AUPRC                :",
        f"{utility_zero_metrics['auprc']:.6f}",
    )

    print(
        "  @ U_pred >= 0"
    )

    print(
        "    Precision          :",
        f"{utility_zero_metrics['precision']:.6f}",
    )

    print(
        "    Recall             :",
        f"{utility_zero_metrics['recall']:.6f}",
    )

    print(
        "    F1                 :",
        f"{utility_zero_metrics['f1']:.6f}",
    )

    print(
        "    Accept rate        :",
        f"{utility_zero_metrics['positive_rate_pred']:.6f}",
    )

    print()
    print("Sign head:")

    print(
        "  AUROC                :",
        f"{sign_metrics_050['auroc']:.6f}",
    )

    print(
        "  AUPRC                :",
        f"{sign_metrics_050['auprc']:.6f}",
    )

    print(
        "  F1 @0.5              :",
        f"{sign_metrics_050['f1']:.6f}",
    )

    print(
        "  Best F1 threshold    :",
        f"{sign_best_threshold:.4f}",
    )

    print(
        "  Best F1              :",
        f"{sign_best_f1:.6f}",
    )

    print()
    print("Top-K predicted Planning-beneficial:")

    for k in [
        "10",
        "25",
        "50",
        "100",
    ]:

        d = topk_stats[k]

        print(
            f"  Top-{k:>3s}: "
            f"beneficial={d['beneficial']:3d} "
            f"harmful={d['harmful']:3d} "
            f"neutral={d['neutral']:3d} "
            f"beneficial_rate="
            f"{d['beneficial_rate']:.4f} "
            f"true_Umean="
            f"{d['mean_true_Uplan']:+.6f}"
        )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)

    print(
        "Checkpoint:",
        CKPT_FILE,
    )

    print(
        "Summary   :",
        RESULT_FILE,
    )

    print(
        "Predictions:",
        PRED_FILE,
    )

    print()
    print(
        "[V2-3D-1 BASELINE TRAINING] COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
