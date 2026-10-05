#!/usr/bin/env python3

import os
import sys
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

# Reuse EXACT V2-1 model / data definition
THIS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(
    0,
    THIS_DIR,
)

from train_v2_magnitude_utility import (
    TARGET_FILE,
    OUT_ROOT,
    V2UtilityPredictor,
    set_seed,
    load_v2_split,
    predict,
    regression_metrics,
    find_best_f1_threshold,
    classification_metrics,
)


# ============================================================
# STCV-Occ V2-1D
# Magnitude Loss Improvement Audit
#
# Fixed:
#   architecture   = V2 0.50M dual-head predictor
#   features       = V1 exact 127-D features
#   target         = 100 * DeltaIoU100
#   lambda_sign    = 0.5
#   scene split    = unchanged
#
# Variable:
#   magnitude loss only
#
# Modes:
#   mse
#   weighted_smoothl1
#
# Test remains sealed.
# ============================================================


class MagnitudeLoss(nn.Module):

    def __init__(
        self,
        mode,
        beta=1.0,
        q90=None,
    ):

        super().__init__()

        self.mode = mode
        self.beta = beta
        self.q90 = q90

        if mode not in [
            "mse",
            "weighted_smoothl1",
        ]:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        if (
            mode == "weighted_smoothl1"
            and (
                q90 is None
                or q90 <= 0
            )
        ):
            raise ValueError(
                "q90 must be > 0 for "
                "weighted_smoothl1"
            )


    def forward(
        self,
        pred,
        target,
    ):

        if self.mode == "mse":

            return F.mse_loss(
                pred,
                target,
                reduction="mean",
            )

        # Per-sample SmoothL1
        base = F.smooth_l1_loss(
            pred,
            target,
            beta=self.beta,
            reduction="none",
        )

        # Importance weighting:
        # weight ∈ [1, 2]
        weight = (
            1.0
            + torch.clamp(
                torch.abs(target)
                / self.q90,
                min=0.0,
                max=1.0,
            )
        )

        # Normalize by sum of weights
        return (
            (weight * base).sum()
            / weight.sum()
        )


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

        loss = (
            loss_mag
            + lambda_sign
            * loss_sign
        )

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

    return {
        "mag_loss":
            mag_sum / n_seen,
        "sign_loss":
            sign_sum / n_seen,
        "total_loss":
            total_sum / n_seen,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mag-loss",
        required=True,
        choices=[
            "mse",
            "weighted_smoothl1",
        ],
    )

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
    # Load targets / features
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

    # Load test only for count/alignment consistency.
    # NO TEST evaluation.
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
    # Train-only q90
    # ========================================================

    q90_train = float(
        np.percentile(
            np.abs(mag_train),
            90,
        )
    )


    # ========================================================
    # Audit header
    # ========================================================

    print("=" * 105)
    print(
        "STCV-Occ V2-1D "
        "Magnitude Loss Improvement Audit"
    )
    print("=" * 105)

    print(
        f"Mode             : "
        f"{args.mag_loss}"
    )

    print(
        f"Device           : "
        f"{device}"
    )

    print(
        f"Train / Val/Test : "
        f"{len(x_train)} / "
        f"{len(x_val)} / "
        f"{len(x_test)}"
    )

    print(
        f"Feature dim      : "
        f"{x_train.shape[1]}"
    )

    print(
        f"Target scale     : "
        f"{args.target_scale}"
    )

    print(
        f"Lambda sign      : "
        f"{args.lambda_sign}"
    )

    print(
        f"Train |U| q90    : "
        f"{q90_train:.6f}"
    )

    print(
        f"Train target std : "
        f"{mag_train.std():.6f}"
    )

    print(
        f"Val target std   : "
        f"{mag_val.std():.6f}"
    )

    print()

    print(
        "V1/V2 feature-target ordering "
        "alignment: PASS"
    )


    # ========================================================
    # DataLoader
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
        f"Parameters       : "
        f"{n_params:,}"
    )

    if n_params != 495490:
        raise RuntimeError(
            f"Unexpected parameter count: "
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
        n_neg / n_pos
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    criterion_sign = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    criterion_mag = MagnitudeLoss(
        mode=args.mag_loss,
        beta=args.smoothl1_beta,
        q90=q90_train,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


    # ========================================================
    # Output
    # ========================================================

    run_name = (
        f"loss_{args.mag_loss}"
        f"_lambda_0p50"
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
    # Train
    # ========================================================

    best_val_loss = float(
        "inf"
    )

    best_epoch = -1
    bad_epochs = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        train_mag_sum = 0.0
        train_sign_sum = 0.0
        train_total_sum = 0.0
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

            train_mag_sum += (
                loss_mag.item()
                * bs
            )

            train_sign_sum += (
                loss_sign.item()
                * bs
            )

            train_total_sum += (
                loss.item()
                * bs
            )

            n_seen += bs


        train_total = (
            train_total_sum
            / n_seen
        )


        # ====================================================
        # Validation
        # ====================================================

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

        val_auroc = float(
            roc_auc_score(
                val_pred["sign_target"],
                val_pred["sign_prob"],
            )
        )

        val_auprc = float(
            average_precision_score(
                val_pred["sign_target"],
                val_pred["sign_prob"],
            )
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_total={train_total:.6f} | "
            f"val_total={val_losses['total_loss']:.6f} | "
            f"val_MAE={val_reg['mae']:.6f} | "
            f"val_RMSE={val_reg['rmse']:.6f} | "
            f"val_Pearson={val_reg['pearson']:.6f} | "
            f"pred_std={val_reg['pred_std']:.6f} | "
            f"val_AUROC={val_auroc:.6f} | "
            f"val_AUPRC={val_auprc:.6f}"
        )


        # Same within-run checkpoint rule
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
                    "epoch":
                        epoch,
                    "model":
                        model.state_dict(),
                    "best_val_total_loss":
                        best_val_loss,
                    "num_parameters":
                        n_params,
                    "args":
                        vars(args),
                    "mag_loss":
                        args.mag_loss,
                    "q90_train":
                        q90_train,
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
    # Reload best
    # ========================================================

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

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
        _
    ) = find_best_f1_threshold(
        val_pred["sign_target"],
        val_pred["sign_prob"],
    )

    val_cls = classification_metrics(
        val_pred["sign_target"],
        val_pred["sign_prob"],
        sign_threshold,
    )


    print()
    print("=" * 105)
    print("FINAL VALIDATION")
    print("=" * 105)

    print(
        f"Best epoch       : "
        f"{ckpt['epoch']}"
    )

    print(
        f"Val mag loss     : "
        f"{val_losses['mag_loss']:.6f}"
    )

    print(
        f"Val total loss   : "
        f"{val_losses['total_loss']:.6f}"
    )

    print(
        f"Val MAE          : "
        f"{val_reg['mae']:.6f}"
    )

    print(
        f"Val RMSE         : "
        f"{val_reg['rmse']:.6f}"
    )

    print(
        f"Val Pearson      : "
        f"{val_reg['pearson']:.6f}"
    )

    print(
        f"Pred std         : "
        f"{val_reg['pred_std']:.6f}"
    )

    print(
        f"Target std       : "
        f"{val_reg['target_std']:.6f}"
    )

    print(
        f"Val AUROC        : "
        f"{val_cls['auroc']:.6f}"
    )

    print(
        f"Val AUPRC        : "
        f"{val_cls['auprc']:.6f}"
    )

    print(
        f"Val F1           : "
        f"{val_cls['f1']:.6f}"
    )


    # ========================================================
    # Save predictions
    # ========================================================

    pred_file = os.path.join(
        result_dir,
        "val_predictions.npz",
    )

    np.savez_compressed(
        pred_file,
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


    result = {
        "model":
            "V2UtilityPredictor",
        "mag_loss":
            args.mag_loss,
        "lambda_sign":
            args.lambda_sign,
        "num_parameters":
            n_params,
        "q90_train":
            q90_train,
        "best_epoch":
            int(best_epoch),
        "best_val_total_loss":
            float(best_val_loss),
        "validation": {
            "losses":
                val_losses,
            "regression":
                val_reg,
            "classification":
                val_cls,
        },
        "test":
            None,
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
        f"{pred_file}"
    )

    print()
    print(
        "TEST NOT EVALUATED."
    )

    print(
        "V2-1D loss audit finished."
    )


if __name__ == "__main__":
    main()
