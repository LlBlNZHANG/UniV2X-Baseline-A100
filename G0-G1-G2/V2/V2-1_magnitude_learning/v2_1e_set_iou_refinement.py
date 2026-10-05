#!/usr/bin/env python3

import os
import sys
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

THIS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(0, THIS_DIR)

from train_v2_magnitude_utility import (
    V2UtilityPredictor,
    set_seed,
)


ROOT = "/root/autodl-tmp/UniV2X"

TRAIN_FILE = os.path.join(
    THIS_DIR,
    "v2_1e_group_train.npz",
)

VAL_FILE = os.path.join(
    THIS_DIR,
    "v2_1e_group_val.npz",
)

BASELINE_CKPT = os.path.join(
    THIS_DIR,
    "checkpoints/main_lambda_0p50/best.pth",
)


# ============================================================
# Group-complete dataset
# ============================================================

class GroupDataset(Dataset):

    def __init__(self, path):

        d = np.load(
            path,
            allow_pickle=True,
        )

        self.x = d[
            "features"
        ].astype(np.float32)

        self.mag = d[
            "magnitude_target"
        ].astype(np.float32)

        self.sign = d[
            "sign_target"
        ].astype(np.float32)

        self.ba = d[
            "ba"
        ].astype(np.float32)

        self.ha = d[
            "ha"
        ].astype(np.float32)

        self.ptr = d[
            "group_ptr"
        ].astype(np.int64)

        self.tp = d[
            "group_tp"
        ].astype(np.float32)

        self.fp = d[
            "group_fp"
        ].astype(np.float32)

        self.fn = d[
            "group_fn"
        ].astype(np.float32)

        self.n_groups = len(
            self.tp
        )

        assert len(
            self.ptr
        ) == self.n_groups + 1

        assert self.ptr[-1] == len(
            self.x
        )


    def __len__(self):

        return self.n_groups


    def __getitem__(self, gid):

        s = self.ptr[gid]
        e = self.ptr[gid + 1]

        return {
            "x":
                self.x[s:e],

            "mag":
                self.mag[s:e],

            "sign":
                self.sign[s:e],

            "ba":
                self.ba[s:e],

            "ha":
                self.ha[s:e],

            "tp":
                self.tp[gid],

            "fp":
                self.fp[gid],

            "fn":
                self.fn[gid],
        }


def collate_groups(batch):

    xs = []
    mags = []
    signs = []
    bas = []
    has = []

    group_index = []

    tp = []
    fp = []
    fn = []

    for gid, item in enumerate(batch):

        n = len(
            item["x"]
        )

        xs.append(
            torch.from_numpy(
                item["x"]
            )
        )

        mags.append(
            torch.from_numpy(
                item["mag"]
            )
        )

        signs.append(
            torch.from_numpy(
                item["sign"]
            )
        )

        bas.append(
            torch.from_numpy(
                item["ba"]
            )
        )

        has.append(
            torch.from_numpy(
                item["ha"]
            )
        )

        group_index.append(
            torch.full(
                (n,),
                gid,
                dtype=torch.long,
            )
        )

        tp.append(
            item["tp"]
        )

        fp.append(
            item["fp"]
        )

        fn.append(
            item["fn"]
        )

    return {
        "x":
            torch.cat(xs, dim=0),

        "mag":
            torch.cat(mags, dim=0),

        "sign":
            torch.cat(signs, dim=0),

        "ba":
            torch.cat(bas, dim=0),

        "ha":
            torch.cat(has, dim=0),

        "group_index":
            torch.cat(
                group_index,
                dim=0,
            ),

        "tp":
            torch.tensor(
                tp,
                dtype=torch.float32,
            ),

        "fp":
            torch.tensor(
                fp,
                dtype=torch.float32,
            ),

        "fn":
            torch.tensor(
                fn,
                dtype=torch.float32,
            ),
    }


# ============================================================
# Set-level differentiable IoU gain
#
# Complete groups are kept intact inside each minibatch.
#
# Soft acceptance:
#
#   a_i = sigmoid(mag_pred_i / T)
#
# Batch metric surrogate:
#
#   IoU_soft =
#       [sum TP_g + sum a_i BA_i]
#       ---------------------------------
#       [sum D_g  + sum a_i HA_i]
#
# We optimize IoU percentage-point gain relative to Ego:
#
#   gain_pp = 100 * (IoU_soft - IoU_ego)
#
#   L_set = -gain_pp
#
# The x100 scaling matches the magnitude target scale.
# ============================================================

def set_iou_loss(
    mag_pred,
    ba,
    ha,
    tp,
    fp,
    fn,
    temperature,
):

    soft_accept = torch.sigmoid(
        mag_pred / temperature
    )

    base_tp = tp.sum()

    base_d = (
        tp
        + fp
        + fn
    ).sum()

    # Extremely rare D=0-only batch guard.
    if base_d.item() <= 0:

        return (
            mag_pred.sum() * 0.0,
            torch.tensor(
                0.0,
                device=mag_pred.device,
            ),
            torch.tensor(
                0.0,
                device=mag_pred.device,
            ),
        )

    ego_iou = (
        base_tp
        / base_d
    )

    soft_tp = (
        base_tp
        + (
            soft_accept
            * ba
        ).sum()
    )

    soft_d = (
        base_d
        + (
            soft_accept
            * ha
        ).sum()
    )

    soft_iou = (
        soft_tp
        / soft_d
    )

    gain_pp = (
        100.0
        * (
            soft_iou
            - ego_iou
        )
    )

    loss = -gain_pp

    return (
        loss,
        soft_iou.detach(),
        gain_pp.detach(),
    )


def hard_group_iou(
    mag_pred,
    ba,
    ha,
    tp,
    fp,
    fn,
):

    accept = (
        mag_pred > 0
    ).float()

    base_tp = tp.sum()

    base_d = (
        tp
        + fp
        + fn
    ).sum()

    if base_d.item() <= 0:

        return 0.0

    final_tp = (
        base_tp
        + (
            accept
            * ba
        ).sum()
    )

    final_d = (
        base_d
        + (
            accept
            * ha
        ).sum()
    )

    return float(
        (
            final_tp
            / final_d
        ).item()
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    criterion_mag,
    criterion_sign,
    lambda_sign,
    lambda_set,
    temperature,
):

    model.eval()

    all_mag_pred = []
    all_mag_target = []
    all_sign_prob = []
    all_sign_target = []

    total_mag = 0.0
    total_sign = 0.0
    total_set = 0.0
    total_loss = 0.0

    n_patch = 0
    n_batch = 0

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    accepted_ba = 0.0
    accepted_ha = 0.0

    soft_gain_sum = 0.0

    for batch in loader:

        x = batch["x"].to(device)
        mag_t = batch["mag"].to(device)
        sign_t = batch["sign"].to(device)

        ba = batch["ba"].to(device)
        ha = batch["ha"].to(device)

        tp = batch["tp"].to(device)
        fp = batch["fp"].to(device)
        fn = batch["fn"].to(device)

        mag_pred, sign_logit = model(
            x
        )

        mag_loss = criterion_mag(
            mag_pred,
            mag_t,
        )

        sign_loss = criterion_sign(
            sign_logit,
            sign_t,
        )

        (
            s_loss,
            soft_iou,
            soft_gain,
        ) = set_iou_loss(
            mag_pred,
            ba,
            ha,
            tp,
            fp,
            fn,
            temperature,
        )

        loss = (
            mag_loss
            + lambda_sign
            * sign_loss
            + lambda_set
            * s_loss
        )

        bs = len(x)

        total_mag += (
            mag_loss.item()
            * bs
        )

        total_sign += (
            sign_loss.item()
            * bs
        )

        total_loss += (
            loss.item()
            * bs
        )

        total_set += (
            s_loss.item()
        )

        n_patch += bs
        n_batch += 1

        prob = torch.sigmoid(
            sign_logit
        )

        all_mag_pred.append(
            mag_pred.cpu().numpy()
        )

        all_mag_target.append(
            mag_t.cpu().numpy()
        )

        all_sign_prob.append(
            prob.cpu().numpy()
        )

        all_sign_target.append(
            sign_t.cpu().numpy()
        )

        accept = (
            mag_pred > 0
        ).float()

        total_tp += float(
            tp.sum().item()
        )

        total_fp += float(
            fp.sum().item()
        )

        total_fn += float(
            fn.sum().item()
        )

        accepted_ba += float(
            (
                accept
                * ba
            ).sum().item()
        )

        accepted_ha += float(
            (
                accept
                * ha
            ).sum().item()
        )

        soft_gain_sum += float(
            soft_gain.item()
        )


    mag_pred = np.concatenate(
        all_mag_pred
    ).astype(np.float64)

    mag_target = np.concatenate(
        all_mag_target
    ).astype(np.float64)

    sign_prob = np.concatenate(
        all_sign_prob
    ).astype(np.float64)

    sign_target = np.concatenate(
        all_sign_target
    ).astype(np.int64)


    mae = float(
        np.mean(
            np.abs(
                mag_pred
                - mag_target
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    mag_pred
                    - mag_target
                ) ** 2
            )
        )
    )

    if (
        np.std(mag_pred) > 0
        and np.std(mag_target) > 0
    ):
        pearson = float(
            np.corrcoef(
                mag_pred,
                mag_target,
            )[0, 1]
        )
    else:
        pearson = 0.0


    auroc = float(
        roc_auc_score(
            sign_target,
            sign_prob,
        )
    )

    auprc = float(
        average_precision_score(
            sign_target,
            sign_prob,
        )
    )


    base_d = (
        total_tp
        + total_fp
        + total_fn
    )

    hard_iou = (
        (
            total_tp
            + accepted_ba
        )
        /
        (
            base_d
            + accepted_ha
        )
    )


    return {
        "mag_loss":
            total_mag / n_patch,

        "sign_loss":
            total_sign / n_patch,

        "set_loss":
            total_set / n_batch,

        "total_loss":
            total_loss / n_patch,

        "mae":
            mae,

        "rmse":
            rmse,

        "pearson":
            pearson,

        "pred_std":
            float(
                np.std(mag_pred)
            ),

        "target_std":
            float(
                np.std(mag_target)
            ),

        "auroc":
            auroc,

        "auprc":
            auprc,

        "hard_group_iou":
            float(hard_iou),

        "accept_rate":
            float(
                np.mean(
                    mag_pred > 0
                )
            ),

        "soft_gain_pp":
            soft_gain_sum / n_batch,

        "mag_pred":
            mag_pred,

        "mag_target":
            mag_target,

        "sign_prob":
            sign_prob,

        "sign_target":
            sign_target,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lambda-set",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--lambda-sign",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--group-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
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


    train_ds = GroupDataset(
        TRAIN_FILE
    )

    val_ds = GroupDataset(
        VAL_FILE
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=
            args.group_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=
            device.type == "cuda",
        collate_fn=
            collate_groups,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=
            args.group_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=
            device.type == "cuda",
        collate_fn=
            collate_groups,
    )


    model = V2UtilityPredictor(
        dropout=0.1
    ).to(device)


    # ========================================================
    # Start from frozen V2-1 main baseline checkpoint
    # ========================================================

    baseline = torch.load(
        BASELINE_CKPT,
        map_location=device,
    )

    model.load_state_dict(
        baseline["model"]
    )


    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    if n_params != 495490:
        raise RuntimeError(
            f"Unexpected parameter count: "
            f"{n_params}"
        )


    # ========================================================
    # Loss
    # ========================================================

    train_sign = train_ds.sign

    n_pos = float(
        train_sign.sum()
    )

    n_neg = float(
        len(train_sign)
        - train_sign.sum()
    )

    pos_weight = torch.tensor(
        [
            n_neg / n_pos
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion_mag = nn.SmoothL1Loss(
        beta=1.0
    )

    criterion_sign = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


    run_name = (
        f"set_refine_lambda_"
        f"{str(args.lambda_set).replace('.', 'p')}"
    )

    ckpt_dir = os.path.join(
        THIS_DIR,
        "checkpoints",
        run_name,
    )

    result_dir = os.path.join(
        THIS_DIR,
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

    # Keep every epoch for exact full-Val downstream
    # checkpoint selection.
    epoch_ckpt_dir = os.path.join(
        ckpt_dir,
        "epochs",
    )

    os.makedirs(
        epoch_ckpt_dir,
        exist_ok=True,
    )


    print("=" * 110)

    print(
        "STCV-Occ V2-1E-1 "
        "Group-aware Set-IoU Refinement"
    )

    print("=" * 110)

    print(
        f"Device           : {device}"
    )

    print(
        f"Train groups     : "
        f"{len(train_ds)}"
    )

    print(
        f"Val groups       : "
        f"{len(val_ds)}"
    )

    print(
        f"Train candidates : "
        f"{len(train_ds.x)}"
    )

    print(
        f"Val candidates   : "
        f"{len(val_ds.x)}"
    )

    print(
        f"Parameters       : "
        f"{n_params:,}"
    )

    print(
        f"Start checkpoint : "
        f"{BASELINE_CKPT}"
    )

    print(
        f"Lambda sign      : "
        f"{args.lambda_sign}"
    )

    print(
        f"Lambda set       : "
        f"{args.lambda_set}"
    )

    print(
        f"Temperature      : "
        f"{args.temperature}"
    )

    print(
        f"LR               : "
        f"{args.lr}"
    )


    # ========================================================
    # Initial validation before any refinement
    # ========================================================

    initial = evaluate(
        model,
        val_loader,
        device,
        criterion_mag,
        criterion_sign,
        args.lambda_sign,
        args.lambda_set,
        args.temperature,
    )

    print()
    print(
        "Initial checkpoint:"
    )

    print(
        f"Val MAE={initial['mae']:.6f} | "
        f"Pearson={initial['pearson']:.6f} | "
        f"AUROC={initial['auroc']:.6f} | "
        f"AUPRC={initial['auprc']:.6f} | "
        f"HardGroupIoU={initial['hard_group_iou']:.6f} | "
        f"Accept={initial['accept_rate']:.6f}"
    )


    # ========================================================
    # Epoch-0 checkpoint is a valid candidate.
    #
    # Refinement is only retained if it improves validation
    # objective over the original V2-1 baseline.
    # ========================================================

    best_val_loss = float(
        initial["total_loss"]
    )

    best_epoch = 0
    bad_epochs = 0

    torch.save(
        {
            "epoch":
                0,

            "model":
                model.state_dict(),

            "best_val_total_loss":
                best_val_loss,

            "args":
                vars(args),

            "num_parameters":
                n_params,

            "source":
                "initial_v2_1_baseline",
        },
        ckpt_path,
    )

    # Also preserve epoch 0 explicitly.
    torch.save(
        {
            "epoch":
                0,

            "model":
                model.state_dict(),

            "args":
                vars(args),

            "num_parameters":
                n_params,

            "source":
                "initial_v2_1_baseline",
        },
        os.path.join(
            epoch_ckpt_dir,
            "epoch_000.pth",
        ),
    )

    print(
        f"Epoch-0 candidate | "
        f"val_total={best_val_loss:.6f}"
    )


    # ========================================================
    # Refinement
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        train_total = 0.0
        n_seen = 0

        for batch in train_loader:

            x = batch["x"].to(
                device,
                non_blocking=True,
            )

            mag_t = batch["mag"].to(
                device,
                non_blocking=True,
            )

            sign_t = batch["sign"].to(
                device,
                non_blocking=True,
            )

            ba = batch["ba"].to(
                device,
                non_blocking=True,
            )

            ha = batch["ha"].to(
                device,
                non_blocking=True,
            )

            tp = batch["tp"].to(
                device,
                non_blocking=True,
            )

            fp = batch["fp"].to(
                device,
                non_blocking=True,
            )

            fn = batch["fn"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            mag_pred, sign_logit = model(
                x
            )

            mag_loss = criterion_mag(
                mag_pred,
                mag_t,
            )

            sign_loss = criterion_sign(
                sign_logit,
                sign_t,
            )

            (
                s_loss,
                _,
                _,
            ) = set_iou_loss(
                mag_pred,
                ba,
                ha,
                tp,
                fp,
                fn,
                args.temperature,
            )

            loss = (
                mag_loss
                + args.lambda_sign
                * sign_loss
                + args.lambda_set
                * s_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            bs = len(x)

            train_total += (
                loss.item()
                * bs
            )

            n_seen += bs


        val = evaluate(
            model,
            val_loader,
            device,
            criterion_mag,
            criterion_sign,
            args.lambda_sign,
            args.lambda_set,
            args.temperature,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_total="
            f"{train_total/n_seen:.6f} | "
            f"val_total="
            f"{val['total_loss']:.6f} | "
            f"MAE="
            f"{val['mae']:.6f} | "
            f"Pearson="
            f"{val['pearson']:.6f} | "
            f"AUPRC="
            f"{val['auprc']:.6f} | "
            f"HardGroupIoU="
            f"{val['hard_group_iou']:.6f} | "
            f"Accept="
            f"{val['accept_rate']:.6f}"
        )


        # Preserve this epoch for the exact full-validation
        # 30m / 100m downstream audit.
        torch.save(
            {
                "epoch":
                    epoch,

                "model":
                    model.state_dict(),

                "val_total_loss":
                    float(
                        val["total_loss"]
                    ),

                "hard_group_iou":
                    float(
                        val["hard_group_iou"]
                    ),

                "accept_rate":
                    float(
                        val["accept_rate"]
                    ),

                "args":
                    vars(args),

                "num_parameters":
                    n_params,
            },
            os.path.join(
                epoch_ckpt_dir,
                f"epoch_{epoch:03d}.pth",
            ),
        )


        # Within-run rule remains:
        # minimum validation total loss.
        if (
            val["total_loss"]
            < best_val_loss
        ):

            best_val_loss = float(
                val["total_loss"]
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

                    "args":
                        vars(args),

                    "num_parameters":
                        n_params,
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
                f"epoch {epoch}; "
                f"best epoch={best_epoch}"
            )

            break


    # ========================================================
    # Reload best
    # ========================================================

    best = torch.load(
        ckpt_path,
        map_location=device,
    )

    model.load_state_dict(
        best["model"]
    )

    final = evaluate(
        model,
        val_loader,
        device,
        criterion_mag,
        criterion_sign,
        args.lambda_sign,
        args.lambda_set,
        args.temperature,
    )


    print()
    print("=" * 110)
    print("FINAL VALIDATION")
    print("=" * 110)

    print(
        f"Best epoch       : "
        f"{best_epoch}"
    )

    print(
        f"Val total loss   : "
        f"{final['total_loss']:.6f}"
    )

    print(
        f"Val MAE          : "
        f"{final['mae']:.6f}"
    )

    print(
        f"Val RMSE         : "
        f"{final['rmse']:.6f}"
    )

    print(
        f"Val Pearson      : "
        f"{final['pearson']:.6f}"
    )

    print(
        f"Pred std         : "
        f"{final['pred_std']:.6f}"
    )

    print(
        f"Val AUROC        : "
        f"{final['auroc']:.6f}"
    )

    print(
        f"Val AUPRC        : "
        f"{final['auprc']:.6f}"
    )

    print(
        f"Hard group IoU   : "
        f"{final['hard_group_iou']:.6f}"
    )

    print(
        f"Accept rate      : "
        f"{final['accept_rate']:.6f}"
    )

    print(
        f"Soft gain pp     : "
        f"{final['soft_gain_pp']:.6f}"
    )


    # Save candidate-order predictions for downstream audit.
    np.savez_compressed(
        os.path.join(
            result_dir,
            "val_predictions.npz",
        ),

        magnitude_pred=
            final["mag_pred"],

        magnitude_target=
            final["mag_target"],

        sign_prob=
            final["sign_prob"],

        sign_target=
            final["sign_target"],
    )


    summary = {
        "lambda_set":
            args.lambda_set,

        "lambda_sign":
            args.lambda_sign,

        "temperature":
            args.temperature,

        "best_epoch":
            best_epoch,

        "validation": {
            k: v
            for k, v in final.items()
            if k not in [
                "mag_pred",
                "mag_target",
                "sign_prob",
                "sign_target",
            ]
        },

        "test":
            None,
    }

    with open(
        os.path.join(
            result_dir,
            "summary.json",
        ),
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )


    print()
    print(
        f"Saved checkpoint : "
        f"{ckpt_path}"
    )

    print(
        f"Saved result dir : "
        f"{result_dir}"
    )

    print()
    print(
        "TEST NOT EVALUATED."
    )

    print(
        "V2-1E-1 refinement finished."
    )


if __name__ == "__main__":
    main()
