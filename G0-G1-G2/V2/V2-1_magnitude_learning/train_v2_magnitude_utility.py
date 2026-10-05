#!/usr/bin/env python3

import os
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


# ============================================================
# STCV-Occ V2-1
# Magnitude-aware + Sign-aware Cooperation Utility Learning
#
# Input:
#   EXACT SAME 127-D feature as V1 G3
#
# Target:
#   magnitude:
#       u = target_scale * DeltaIoU100
#
#   sign:
#       c = 1[DeltaIoU100 > 0]
#
# Model:
#   127 -> 512 -> Residual(512) -> 256 -> 128
#                                  |       |
#                                  +-- Magnitude head
#                                  +-- Sign head
#
# Total parameters: ~0.50M
#
# Loss:
#   L = SmoothL1(mag_pred, u)
#       + lambda_sign * BCE(sign_logit, c)
#
# IMPORTANT:
#   - V1 feature definition is NOT changed.
#   - V1 scene split is NOT changed.
#   - Test is NOT used during model/hyperparameter selection.
# ============================================================


ROOT = "/root/autodl-tmp/UniV2X"

G3_SOFT_DIR = os.path.join(
    ROOT,
    "G0-G1-G2/G3/soft",
)

TARGET_FILE = os.path.join(
    ROOT,
    "G0-G1-G2/V2/V2-1_magnitude_learning/"
    "v2_1a_target_audit_arrays.npz",
)

OUT_ROOT = os.path.join(
    ROOT,
    "G0-G1-G2/V2/V2-1_magnitude_learning",
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=2026):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Model
# ============================================================

class V2UtilityPredictor(nn.Module):
    """
    STCV-Occ V2 Utility Predictor.

    Input:
        127-D feature
        = 80 local
        + 5 horizon
        + 2 spatial
        + 40 neighborhood

    Shared encoder:
        127 -> 512
        512 residual block
        512 -> 256
        256 -> 128

    Heads:
        magnitude_head : 128 -> 1
        sign_head      : 128 -> 1
    """

    def __init__(self, dropout=0.1):

        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(127, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.residual_block = nn.Sequential(
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

        self.magnitude_head = nn.Linear(
            128,
            1,
        )

        self.sign_head = nn.Linear(
            128,
            1,
        )


    def forward(self, x):

        h = self.input_proj(x)

        # One lightweight residual block
        h = h + self.residual_block(h)

        h = self.down_256(h)
        h = self.down_128(h)

        magnitude = self.magnitude_head(
            h
        ).squeeze(1)

        sign_logit = self.sign_head(
            h
        ).squeeze(1)

        return magnitude, sign_logit


# ============================================================
# Data loading
# ============================================================

def load_v1_feature_split(split):
    """
    Load EXACT V1 127-D feature dataset.

    Also returns original V1 binary labels solely for
    ordering/alignment verification.
    """

    path = os.path.join(
        G3_SOFT_DIR,
        f"g3_neighborhood_{split}.npz",
    )

    d = np.load(
        path,
        allow_pickle=True,
    )

    x = d[
        "features"
    ].astype(np.float32)

    v1_label = d[
        "labels"
    ].astype(np.int64)

    assert x.ndim == 2
    assert x.shape[1] == 127
    assert len(x) == len(v1_label)

    return x, v1_label


def load_v2_split(split, target_data, target_scale):
    """
    Pair V1 127-D features with V2 continuous DeltaIoU100 target.

    Critical alignment check:
        original V1 label
        ==
        1[v1_utility > 0]

    If this assertion passes, V2 target ordering is exactly
    aligned with the existing G3 feature ordering.
    """

    x, v1_label = load_v1_feature_split(
        split
    )

    raw_target = target_data[
        f"{split}_target"
    ].astype(np.float32)

    v1_utility = target_data[
        f"{split}_v1_utility"
    ].astype(np.float32)

    assert len(x) == len(raw_target)
    assert len(x) == len(v1_utility)

    reconstructed_v1_label = (
        v1_utility > 0
    ).astype(np.int64)

    mismatch = int(
        (
            reconstructed_v1_label
            != v1_label
        ).sum()
    )

    if mismatch != 0:

        raise RuntimeError(
            f"{split}: V1/V2 ordering mismatch: "
            f"{mismatch} samples differ."
        )

    # Continuous magnitude target
    magnitude_target = (
        raw_target * target_scale
    ).astype(np.float32)

    # Sign target:
    # positive only if metric utility is STRICTLY > 0.
    #
    # This is important because pure-HA zero-DeltaIoU
    # patches must remain rejected.
    sign_target = (
        raw_target > 0
    ).astype(np.float32)

    return (
        x,
        magnitude_target,
        sign_target,
        raw_target,
    )


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader,
    device,
):

    model.eval()

    all_mag_pred = []
    all_sign_logit = []
    all_mag_target = []
    all_sign_target = []

    for (
        x,
        mag_target,
        sign_target,
    ) in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        mag_pred, sign_logit = model(
            x
        )

        all_mag_pred.append(
            mag_pred.cpu()
        )

        all_sign_logit.append(
            sign_logit.cpu()
        )

        all_mag_target.append(
            mag_target
        )

        all_sign_target.append(
            sign_target
        )

    mag_pred = torch.cat(
        all_mag_pred
    ).numpy()

    sign_logit = torch.cat(
        all_sign_logit
    ).numpy()

    mag_target = torch.cat(
        all_mag_target
    ).numpy()

    sign_target = torch.cat(
        all_sign_target
    ).numpy()

    sign_prob = (
        1.0
        /
        (
            1.0
            + np.exp(
                -np.clip(
                    sign_logit,
                    -50,
                    50,
                )
            )
        )
    )

    return {
        "mag_pred": mag_pred,
        "mag_target": mag_target,
        "sign_logit": sign_logit,
        "sign_prob": sign_prob,
        "sign_target": sign_target.astype(
            np.int32
        ),
    }


# ============================================================
# Loss evaluation
# ============================================================

@torch.no_grad()
def evaluate_losses(
    model,
    loader,
    device,
    criterion_mag,
    criterion_sign,
    lambda_sign,
):

    model.eval()

    mag_sum = 0.0
    sign_sum = 0.0
    total_sum = 0.0
    n_seen = 0

    for (
        x,
        mag_target,
        sign_target,
    ) in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        mag_target = mag_target.to(
            device,
            non_blocking=True,
        )

        sign_target = sign_target.to(
            device,
            non_blocking=True,
        )

        mag_pred, sign_logit = model(
            x
        )

        loss_mag = criterion_mag(
            mag_pred,
            mag_target,
        )

        loss_sign = criterion_sign(
            sign_logit,
            sign_target,
        )

        loss_total = (
            loss_mag
            + lambda_sign * loss_sign
        )

        bs = len(x)

        mag_sum += (
            loss_mag.item() * bs
        )

        sign_sum += (
            loss_sign.item() * bs
        )

        total_sum += (
            loss_total.item() * bs
        )

        n_seen += bs

    return {
        "mag_loss": mag_sum / n_seen,
        "sign_loss": sign_sum / n_seen,
        "total_loss": total_sum / n_seen,
    }


# ============================================================
# Metrics
# ============================================================

def find_best_f1_threshold(
    labels,
    probs,
):

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(
        0.01,
        0.99,
        99,
    ):

        pred = (
            probs >= threshold
        ).astype(np.int32)

        score = f1_score(
            labels,
            pred,
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_f1,
    )


def safe_pearson(x, y):

    if (
        np.std(x) < 1e-12
        or np.std(y) < 1e-12
    ):
        return 0.0

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def regression_metrics(
    target,
    pred,
):

    err = pred - target

    mae = float(
        np.mean(
            np.abs(err)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                err ** 2
            )
        )
    )

    pearson = safe_pearson(
        target,
        pred,
    )

    # Sign agreement from magnitude head itself.
    target_sign = target > 0
    pred_sign = pred > 0

    sign_accuracy = float(
        np.mean(
            target_sign
            == pred_sign
        )
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "mag_sign_accuracy": sign_accuracy,
        "pred_mean": float(
            np.mean(pred)
        ),
        "pred_std": float(
            np.std(pred)
        ),
        "target_mean": float(
            np.mean(target)
        ),
        "target_std": float(
            np.std(target)
        ),
    }


def classification_metrics(
    labels,
    probs,
    threshold,
):

    pred = (
        probs >= threshold
    ).astype(np.int32)

    return {
        "auroc": float(
            roc_auc_score(
                labels,
                probs,
            )
        ),
        "auprc": float(
            average_precision_score(
                labels,
                probs,
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


def print_metric_block(
    name,
    losses,
    reg_metrics,
    cls_metrics,
):

    print()
    print("=" * 90)
    print(name)
    print("=" * 90)

    print(
        f"mag_loss             : "
        f"{losses['mag_loss']:.6f}"
    )

    print(
        f"sign_loss            : "
        f"{losses['sign_loss']:.6f}"
    )

    print(
        f"total_loss           : "
        f"{losses['total_loss']:.6f}"
    )

    print()
    print("Magnitude")

    for k, v in reg_metrics.items():
        print(
            f"{k:20s}: "
            f"{v:.6f}"
        )

    print()
    print("Sign")

    for k, v in cls_metrics.items():
        print(
            f"{k:20s}: "
            f"{v:.6f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
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
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--target-scale",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--lambda-sign",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--smoothl1-beta",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--eval-test",
        action="store_true",
        help=(
            "Evaluate Test split only after "
            "hyperparameters are frozen."
        ),
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # ========================================================
    # Output directories
    # ========================================================

    lambda_name = (
        f"{args.lambda_sign:.2f}"
        .replace(".", "p")
    )

    run_name = (
        f"main_lambda_{lambda_name}"
    )

    ckpt_dir = os.path.join(
        OUT_ROOT,
        "checkpoints",
        run_name,
    )

    result_dir = os.path.join(
        OUT_ROOT,
        "results",
        run_name,
    )

    os.makedirs(
        ckpt_dir,
        exist_ok=True,
    )

    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    ckpt_path = os.path.join(
        ckpt_dir,
        "best.pth",
    )


    # ========================================================
    # Targets
    # ========================================================

    target_data = np.load(
        TARGET_FILE,
        allow_pickle=True,
    )

    (
        x_train,
        mag_train,
        sign_train,
        raw_train,
    ) = load_v2_split(
        "train",
        target_data,
        args.target_scale,
    )

    (
        x_val,
        mag_val,
        sign_val,
        raw_val,
    ) = load_v2_split(
        "val",
        target_data,
        args.target_scale,
    )

    (
        x_test,
        mag_test,
        sign_test,
        raw_test,
    ) = load_v2_split(
        "test",
        target_data,
        args.target_scale,
    )


    # ========================================================
    # Dataset audit
    # ========================================================

    print("=" * 100)
    print(
        "STCV-Occ V2-1 "
        "Magnitude-aware Utility Learning"
    )
    print("=" * 100)

    print(
        f"Device          : {device}"
    )

    print(
        f"Train           : {len(x_train)}"
    )

    print(
        f"Val             : {len(x_val)}"
    )

    print(
        f"Test            : {len(x_test)}"
    )

    print(
        f"Feature dim     : {x_train.shape[1]}"
    )

    print(
        f"Target scale    : {args.target_scale}"
    )

    print(
        f"Lambda sign     : {args.lambda_sign}"
    )

    print(
        f"SmoothL1 beta   : {args.smoothl1_beta}"
    )

    print()

    print(
        f"Train positive  : "
        f"{int(sign_train.sum())} "
        f"({sign_train.mean():.6f})"
    )

    print(
        f"Val positive    : "
        f"{int(sign_val.sum())} "
        f"({sign_val.mean():.6f})"
    )

    print(
        f"Test positive   : "
        f"{int(sign_test.sum())} "
        f"({sign_test.mean():.6f})"
    )

    print()

    print(
        f"Train magnitude std : "
        f"{mag_train.std():.6f}"
    )

    print(
        f"Val magnitude std   : "
        f"{mag_val.std():.6f}"
    )

    print(
        f"Test magnitude std  : "
        f"{mag_test.std():.6f}"
    )

    print()

    print(
        "V1/V2 feature-target ordering "
        "alignment: PASS"
    )


    # ========================================================
    # PyTorch datasets
    # ========================================================

    train_ds = TensorDataset(
        torch.from_numpy(
            x_train
        ),
        torch.from_numpy(
            mag_train
        ),
        torch.from_numpy(
            sign_train
        ),
    )

    val_ds = TensorDataset(
        torch.from_numpy(
            x_val
        ),
        torch.from_numpy(
            mag_val
        ),
        torch.from_numpy(
            sign_val
        ),
    )

    test_ds = TensorDataset(
        torch.from_numpy(
            x_test
        ),
        torch.from_numpy(
            mag_test
        ),
        torch.from_numpy(
            sign_test
        ),
    )

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )


    # ========================================================
    # Model
    # ========================================================

    model = V2UtilityPredictor(
        dropout=args.dropout
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Parameters      : {n_params:,}"
    )

    # Exact expected architecture should be ~495K.
    if not (
        450_000
        <= n_params
        <= 550_000
    ):
        raise RuntimeError(
            "Unexpected model parameter count: "
            f"{n_params}"
        )


    # ========================================================
    # Loss
    # ========================================================

    n_pos = float(
        sign_train.sum()
    )

    n_neg = float(
        len(sign_train)
        - sign_train.sum()
    )

    pos_weight_value = (
        n_neg
        /
        max(
            n_pos,
            1.0,
        )
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    print(
        f"Sign pos_weight : "
        f"{pos_weight_value:.6f}"
    )

    criterion_mag = nn.SmoothL1Loss(
        beta=args.smoothl1_beta
    )

    criterion_sign = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


    # ========================================================
    # Training
    # ========================================================

    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        mag_sum = 0.0
        sign_sum = 0.0
        total_sum = 0.0
        n_seen = 0

        for (
            x,
            mag_target,
            sign_target,
        ) in train_loader:

            x = x.to(
                device,
                non_blocking=True,
            )

            mag_target = mag_target.to(
                device,
                non_blocking=True,
            )

            sign_target = sign_target.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                mag_pred,
                sign_logit,
            ) = model(x)

            loss_mag = criterion_mag(
                mag_pred,
                mag_target,
            )

            loss_sign = criterion_sign(
                sign_logit,
                sign_target,
            )

            loss = (
                loss_mag
                + args.lambda_sign
                * loss_sign
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            bs = len(x)

            mag_sum += (
                loss_mag.item()
                * bs
            )

            sign_sum += (
                loss_sign.item()
                * bs
            )

            total_sum += (
                loss.item()
                * bs
            )

            n_seen += bs


        train_mag_loss = (
            mag_sum / n_seen
        )

        train_sign_loss = (
            sign_sum / n_seen
        )

        train_total_loss = (
            total_sum / n_seen
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_losses = evaluate_losses(
            model,
            val_loader,
            device,
            criterion_mag,
            criterion_sign,
            args.lambda_sign,
        )

        val_pred = predict(
            model,
            val_loader,
            device,
        )

        val_reg = regression_metrics(
            val_pred["mag_target"],
            val_pred["mag_pred"],
        )

        val_auprc = float(
            average_precision_score(
                val_pred["sign_target"],
                val_pred["sign_prob"],
            )
        )

        val_auroc = float(
            roc_auc_score(
                val_pred["sign_target"],
                val_pred["sign_prob"],
            )
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_total={train_total_loss:.6f} | "
            f"train_mag={train_mag_loss:.6f} | "
            f"train_sign={train_sign_loss:.6f} | "
            f"val_total={val_losses['total_loss']:.6f} | "
            f"val_MAE={val_reg['mae']:.6f} | "
            f"val_Pearson={val_reg['pearson']:.6f} | "
            f"val_AUROC={val_auroc:.6f} | "
            f"val_AUPRC={val_auprc:.6f}"
        )


        # ----------------------------------------------------
        # Checkpoint:
        # within a fixed lambda run, use validation total loss.
        #
        # Do NOT compare total_loss across different lambda
        # settings because objective scales differ.
        # ----------------------------------------------------

        if (
            val_losses["total_loss"]
            < best_val_loss
        ):

            best_val_loss = float(
                val_losses[
                    "total_loss"
                ]
            )

            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_val_total_loss":
                        best_val_loss,
                    "num_parameters":
                        n_params,
                    "args": vars(args),
                    "architecture":
                        "127-512-res512-256-128-dualhead",
                    "target":
                        "100xDeltaIoU100",
                },
                ckpt_path,
            )

        else:

            bad_epochs += 1


        if (
            bad_epochs
            >= args.patience
        ):

            print()
            print(
                f"Early stopping at "
                f"epoch {epoch}. "
                f"Best epoch = "
                f"{best_epoch}"
            )

            break


    # ========================================================
    # Reload best checkpoint
    # ========================================================

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    print()
    print(
        f"Loaded best checkpoint: "
        f"epoch={ckpt['epoch']}, "
        f"val_total_loss="
        f"{ckpt['best_val_total_loss']:.6f}"
    )


    # ========================================================
    # Validation final evaluation
    # ========================================================

    val_losses = evaluate_losses(
        model,
        val_loader,
        device,
        criterion_mag,
        criterion_sign,
        args.lambda_sign,
    )

    val_pred = predict(
        model,
        val_loader,
        device,
    )

    val_reg = regression_metrics(
        val_pred["mag_target"],
        val_pred["mag_pred"],
    )

    (
        sign_threshold,
        val_best_f1,
    ) = find_best_f1_threshold(
        val_pred["sign_target"],
        val_pred["sign_prob"],
    )

    val_cls = classification_metrics(
        val_pred["sign_target"],
        val_pred["sign_prob"],
        sign_threshold,
    )

    print_metric_block(
        "VALIDATION",
        val_losses,
        val_reg,
        val_cls,
    )

    print()
    print(
        f"Validation-selected "
        f"sign threshold: "
        f"{sign_threshold:.4f}"
    )


    # Save validation predictions for later
    # downstream utility / IoU analysis.
    val_pred_file = os.path.join(
        result_dir,
        "val_predictions.npz",
    )

    np.savez_compressed(
        val_pred_file,
        magnitude_pred=
            val_pred["mag_pred"],
        magnitude_target=
            val_pred["mag_target"],
        sign_prob=
            val_pred["sign_prob"],
        sign_target=
            val_pred["sign_target"],
        raw_delta_iou=
            raw_val,
    )


    # ========================================================
    # Optional test
    #
    # Do not use during lambda/model selection.
    # ========================================================

    test_result = None

    if args.eval_test:

        test_losses = evaluate_losses(
            model,
            test_loader,
            device,
            criterion_mag,
            criterion_sign,
            args.lambda_sign,
        )

        test_pred = predict(
            model,
            test_loader,
            device,
        )

        test_reg = regression_metrics(
            test_pred["mag_target"],
            test_pred["mag_pred"],
        )

        test_cls = classification_metrics(
            test_pred["sign_target"],
            test_pred["sign_prob"],
            sign_threshold,
        )

        print_metric_block(
            "TEST "
            "(validation threshold frozen)",
            test_losses,
            test_reg,
            test_cls,
        )

        test_pred_file = os.path.join(
            result_dir,
            "test_predictions.npz",
        )

        np.savez_compressed(
            test_pred_file,
            magnitude_pred=
                test_pred["mag_pred"],
            magnitude_target=
                test_pred["mag_target"],
            sign_prob=
                test_pred["sign_prob"],
            sign_target=
                test_pred["sign_target"],
            raw_delta_iou=
                raw_test,
        )

        test_result = {
            "losses": test_losses,
            "regression": test_reg,
            "classification": test_cls,
        }


    # ========================================================
    # Save JSON summary
    # ========================================================

    result = {
        "model":
            "V2UtilityPredictor",
        "architecture":
            "127-512-res512-256-128-dualhead",
        "num_parameters":
            int(n_params),
        "target":
            "100xDeltaIoU100",
        "lambda_sign":
            float(args.lambda_sign),
        "smoothl1_beta":
            float(args.smoothl1_beta),
        "sign_pos_weight":
            float(pos_weight_value),
        "best_epoch":
            int(best_epoch),
        "best_val_total_loss":
            float(best_val_loss),
        "validation_sign_threshold":
            float(sign_threshold),
        "validation": {
            "losses":
                val_losses,
            "regression":
                val_reg,
            "classification":
                val_cls,
        },
        "test":
            test_result,
        "seed":
            int(args.seed),
    }

    result_file = os.path.join(
        result_dir,
        "summary.json",
    )

    with open(
        result_file,
        "w",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 100)

    print(
        f"Saved checkpoint : "
        f"{ckpt_path}"
    )

    print(
        f"Saved result     : "
        f"{result_file}"
    )

    print(
        f"Saved val preds  : "
        f"{val_pred_file}"
    )

    print()

    if not args.eval_test:

        print(
            "TEST NOT EVALUATED "
            "(correct for model-selection stage)."
        )

    print()
    print(
        "STCV-Occ V2-1 training finished."
    )


if __name__ == "__main__":
    main()
