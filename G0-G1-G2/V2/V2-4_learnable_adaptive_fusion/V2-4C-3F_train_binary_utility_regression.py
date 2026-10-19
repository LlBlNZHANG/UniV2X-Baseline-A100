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


# =============================================================================
# Paths / constants
# =============================================================================

ROOT = Path("/root/autodl-tmp/UniV2X")

BASE = (
    ROOT
    / "G0-G1-G2/V2/V2-4_learnable_adaptive_fusion"
)

TRAIN_DATA = (
    BASE
    / "V2-4C-3E2A_train_context_dataset.npz"
)

VAL_DATA = (
    BASE
    / "V2-4C-3E2A_val_context_dataset.npz"
)

VAL_FULL = (
    BASE
    / "V2-4B_val_learnable_fusion_dataset.npz"
)

E1_CKPT = (
    BASE
    / "checkpoints/V2-4C-3E1_binary_attenuation/best.pth"
)

E1_PRED = (
    BASE
    / "results/V2-4C-3E1_binary_attenuation/"
      "V2-4C-3E1_val_predictions.npz"
)

OLD_EVAL_SCRIPT = (
    BASE
    / "V2-4C_train_utility_conditioned_gate.py"
)

OUT_DIR = (
    BASE
    / "results/V2-4C-3F_binary_utility"
)

CKPT_DIR = (
    BASE
    / "checkpoints/V2-4C-3F_binary_utility"
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

SUMMARY_FILE = (
    OUT_DIR
    / "V2-4C-3F_summary.json"
)

PRED_FILE = (
    OUT_DIR
    / "V2-4C-3F_val_predictions.npz"
)


SEED = 2026

UTILITY_SCALE = 100.0

BATCH_SIZE = 512
EPOCHS = 100
PATIENCE = 15

LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.10

# Signed utility regression is primary.
SMOOTHL1_BETA = 0.10

# Sign classification is auxiliary only.
LAMBDA_SIGN = 0.50

EPS = 1e-12


# =============================================================================
# Exact evaluator
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "v24c_eval",
    OLD_EVAL_SCRIPT,
)

v24c = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    v24c
)


# =============================================================================
# Helpers
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


def binary_metrics_from_score(
    score,
    target,
):

    pred = (
        score > 0.0
    )

    target = target.astype(
        bool
    )

    tp = int(
        (
            pred
            &
            target
        ).sum()
    )

    fp = int(
        (
            pred
            &
            (~target)
        ).sum()
    )

    fn = int(
        (
            (~pred)
            &
            target
        ).sum()
    )

    tn = int(
        (
            (~pred)
            &
            (~target)
        ).sum()
    )

    precision = (
        tp
        /
        (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp
        /
        (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )
        if precision + recall > 0
        else 0.0
    )

    accuracy = (
        (tp + tn)
        /
        len(target)
    )

    return {
        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "tn":
            tn,

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                f1
            ),

        "accuracy":
            float(
                accuracy
            ),
    }


# =============================================================================
# Structured 129-D input
#
# Use exact E1 normalization so comparison is fair.
# =============================================================================

def build_input(
    data,
    extra_mean,
    extra_std,
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

    extra = (
        extra
        -
        extra_mean
    ) / extra_std

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

    return x


# =============================================================================
# Model
# =============================================================================

class BinaryUtilityPredictor(nn.Module):

    def __init__(
        self,
        dropout=0.1,
    ):

        super().__init__()

        self.input_proj = nn.Sequential(

            nn.Linear(
                129,
                512,
            ),

            nn.LayerNorm(
                512
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),
        )

        self.residual = nn.Sequential(

            nn.Linear(
                512,
                512,
            ),

            nn.LayerNorm(
                512
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),
        )

        self.down_256 = nn.Sequential(

            nn.Linear(
                512,
                256,
            ),

            nn.LayerNorm(
                256
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),
        )

        self.down_128 = nn.Sequential(

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
        )

        # Primary head:
        # signed relative utility
        self.magnitude_head = nn.Linear(
            128,
            1,
        )

        # Auxiliary head:
        # whether ΔU > 0
        self.sign_head = nn.Linear(
            128,
            1,
        )

    def forward(
        self,
        x,
    ):

        h = self.input_proj(
            x
        )

        h = (
            h
            +
            self.residual(
                h
            )
        )

        h = self.down_256(
            h
        )

        h = self.down_128(
            h
        )

        magnitude = (
            self.magnitude_head(
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
            magnitude,
            sign_logit,
        )


@torch.no_grad()
def predict(
    model,
    x,
    device,
):

    model.eval()

    mag_out = []
    sign_out = []

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

        (
            magnitude,
            sign_logit,
        ) = model(
            xb
        )

        mag_out.append(
            magnitude
            .cpu()
            .numpy()
        )

        sign_out.append(
            torch.sigmoid(
                sign_logit
            )
            .cpu()
            .numpy()
        )

    return (
        np.concatenate(
            mag_out
        ).astype(
            np.float32
        ),

        np.concatenate(
            sign_out
        ).astype(
            np.float32
        ),
    )


# =============================================================================
# Full validation
# =============================================================================

def get_full_val_ids(
    val_full,
):

    scene_tokens = set(
        np.unique(
            val_full[
                "scene_token"
            ].astype(str)
        ).tolist()
    )

    valid_file = (
        ROOT
        / "G0-G1-G2/G3/cache_valid_samples.txt"
    )

    ids = []

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
                scene_tokens
            ):

                ids.append(
                    parts[0]
                )

    ids = np.asarray(
        ids
    )

    assert len(
        np.unique(
            ids
        )
    ) == 96

    return ids


def evaluate_policy(
    val_full,
    accepted_idx,
    attenuation_mask,
    ego_counts,
    range30_mask,
):

    """
    Innovation 2:
      V2 rejected -> g=0

    Innovation 3:
      V2 accepted + predicted ΔU > 0 -> g=0.1
      V2 accepted + predicted ΔU <=0 -> g=1.0
    """

    gates = np.zeros(
        len(
            val_full[
                "features_127"
            ]
        ),
        dtype=np.float32,
    )

    gates[
        accepted_idx
    ] = 1.0

    gates[
        accepted_idx[
            attenuation_mask
        ]
    ] = 0.1

    result = (
        v24c.evaluate_gates(
            val_full,
            gates,
            ego_counts,
            range30_mask,
        )
    )

    return (
        result,
        gates,
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

    val_full = load_npz(
        VAL_FULL
    )

    e1_pred = load_npz(
        E1_PRED
    )

    # =========================================================================
    # Exact data audit
    # =========================================================================

    assert len(
        train[
            "features_127"
        ]
    ) == 19878

    assert len(
        val[
            "features_127"
        ]
    ) == 6156

    accepted_idx = val[
        "accepted_global_index"
    ].astype(
        np.int64
    )

    assert len(
        accepted_idx
    ) == 6156

    assert np.array_equal(
        accepted_idx,
        e1_pred[
            "accepted_indices"
        ].astype(
            np.int64
        ),
    )

    # =========================================================================
    # Exact E1 normalization
    # =========================================================================

    e1_ckpt = torch.load(
        E1_CKPT,
        map_location="cpu",
    )

    extra_mean = np.asarray(
        e1_ckpt[
            "extra_mean"
        ],
        dtype=np.float32,
    )

    extra_std = np.asarray(
        e1_ckpt[
            "extra_std"
        ],
        dtype=np.float32,
    )

    x_train = build_input(
        train,
        extra_mean,
        extra_std,
    )

    x_val = build_input(
        val,
        extra_mean,
        extra_std,
    )

    # =========================================================================
    # Signed utility target
    # =========================================================================

    y_train_raw = train[
        "true_binary_gain"
    ].astype(
        np.float32
    )

    y_val_raw = val[
        "true_binary_gain"
    ].astype(
        np.float32
    )

    y_train_mag = (
        UTILITY_SCALE
        *
        y_train_raw
    ).astype(
        np.float32
    )

    y_val_mag = (
        UTILITY_SCALE
        *
        y_val_raw
    ).astype(
        np.float32
    )

    y_train_sign = (
        y_train_raw
        >
        EPS
    ).astype(
        np.float32
    )

    y_val_sign = (
        y_val_raw
        >
        EPS
    ).astype(
        np.float32
    )

    assert int(
        y_train_sign.sum()
    ) == 8547

    assert int(
        y_val_sign.sum()
    ) == 3361

    train_pos = float(
        y_train_sign.sum()
    )

    train_neg = float(
        len(
            y_train_sign
        )
        -
        train_pos
    )

    pos_weight_value = (
        train_neg
        /
        train_pos
    )

    pos_weight = torch.tensor(
        pos_weight_value,
        dtype=torch.float32,
        device=device,
    )

    # =========================================================================
    # Full Val evaluator
    # =========================================================================

    full_val_ids = (
        get_full_val_ids(
            val_full
        )
    )

    ego_counts = (
        v24c.build_ego_baseline_counts(
            full_val_ids
        )
    )

    range30_mask = (
        v24c.build_range30_mask(
            val_full[
                "patch_r"
            ],
            val_full[
                "patch_c"
            ],
        )
    )

    v2_result = (
        v24c.evaluate_gates(
            val_full,
            val_full[
                "v2_accept"
            ].astype(
                np.float32
            ),
            ego_counts,
            range30_mask,
        )
    )

    binary_oracle_result, _ = (
        evaluate_policy(
            val_full,
            accepted_idx,
            y_val_sign.astype(
                bool
            ),
            ego_counts,
            range30_mask,
        )
    )

    e1_result, _ = (
        evaluate_policy(
            val_full,
            accepted_idx,
            e1_pred[
                "attenuation_mask"
            ].astype(
                bool
            ),
            ego_counts,
            range30_mask,
        )
    )

    assert abs(
        v2_result[
            "100m"
        ][
            "iou"
        ]
        -
        0.360369
    ) < 2e-6

    assert abs(
        binary_oracle_result[
            "100m"
        ][
            "iou"
        ]
        -
        0.379668
    ) < 2e-6

    assert abs(
        e1_result[
            "100m"
        ][
            "iou"
        ]
        -
        0.363508
    ) < 2e-6

    # =========================================================================
    # Print setup
    # =========================================================================

    print("=" * 116)
    print(
        "V2-4C-3F BINARY UTILITY REGRESSION"
    )
    print("=" * 116)

    print(
        "Train accepted          :",
        len(
            x_train
        ),
    )

    print(
        "Val accepted            :",
        len(
            x_val
        ),
    )

    print(
        "Train positive ΔU       :",
        int(
            y_train_sign.sum()
        ),
        f"({y_train_sign.mean():.6%})",
    )

    print(
        "Val positive ΔU         :",
        int(
            y_val_sign.sum()
        ),
        f"({y_val_sign.mean():.6%})",
    )

    print(
        "Target                  :",
        "100 * [U(g=0.1)-U(g=1.0)]",
    )

    print(
        "Input                   :",
        "129-D",
    )

    print(
        "Primary loss            :",
        f"SmoothL1(beta={SMOOTHL1_BETA})",
    )

    print(
        "Auxiliary sign loss     :",
        f"BCEWithLogits(pos_weight={pos_weight_value:.6f})",
    )

    print(
        "Lambda sign             :",
        LAMBDA_SIGN,
    )

    print(
        "Inference               :",
        "predicted magnitude > 0 -> g=0.1; else g=1",
    )

    print(
        "Test used               :",
        "NO",
    )

    print()
    print(
        "Train raw ΔU mean/std   :",
        f"{y_train_raw.mean():+.8f} / "
        f"{y_train_raw.std():.8f}",
    )

    print(
        "Train raw ΔU min/max    :",
        f"{y_train_raw.min():+.8f} / "
        f"{y_train_raw.max():+.8f}",
    )

    print(
        "Val raw ΔU mean/std     :",
        f"{y_val_raw.mean():+.8f} / "
        f"{y_val_raw.std():.8f}",
    )

    print(
        "Val raw ΔU min/max      :",
        f"{y_val_raw.min():+.8f} / "
        f"{y_val_raw.max():+.8f}",
    )

    print()
    print(
        "V2 Hard                 :",
        f"30m={v2_result['30m']['iou']:.6f} "
        f"100m={v2_result['100m']['iou']:.6f}",
    )

    print(
        "E1 BCE classifier       :",
        f"30m={e1_result['30m']['iou']:.6f} "
        f"100m={e1_result['100m']['iou']:.6f}",
    )

    print(
        "Binary Oracle           :",
        f"30m={binary_oracle_result['30m']['iou']:.6f} "
        f"100m={binary_oracle_result['100m']['iou']:.6f}",
    )

    print(
        "Reference reproduction  : PASS"
    )

    # =========================================================================
    # Dataset
    # =========================================================================

    dataset = TensorDataset(

        torch.from_numpy(
            x_train
        ),

        torch.from_numpy(
            y_train_mag
        ),

        torch.from_numpy(
            y_train_sign
        ),
    )

    generator = torch.Generator()

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

    # =========================================================================
    # Model
    # =========================================================================

    model = (
        BinaryUtilityPredictor(
            dropout=DROPOUT
        )
        .to(
            device
        )
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LR,

        weight_decay=WEIGHT_DECAY,
    )

    print(
        "Model parameters        :",
        n_params,
    )

    # =========================================================================
    # Train
    # =========================================================================

    best_iou100 = -1.0
    best_iou30 = -1.0
    best_epoch = -1

    bad_epochs = 0
    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        loss_sum = 0.0
        mag_sum = 0.0
        sign_sum = 0.0
        n_seen = 0

        for (
            xb,
            y_mag,
            y_sign,
        ) in loader:

            xb = xb.to(
                device,
                non_blocking=True,
            )

            y_mag = y_mag.to(
                device,
                non_blocking=True,
            )

            y_sign = y_sign.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                pred_mag,
                sign_logit,
            ) = model(
                xb
            )

            loss_mag = (
                F.smooth_l1_loss(
                    pred_mag,
                    y_mag,
                    beta=SMOOTHL1_BETA,
                )
            )

            loss_sign = (
                F.binary_cross_entropy_with_logits(
                    sign_logit,
                    y_sign,
                    pos_weight=pos_weight,
                )
            )

            loss = (
                loss_mag
                +
                LAMBDA_SIGN
                *
                loss_sign
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            bs = len(
                xb
            )

            loss_sum += (
                float(
                    loss.item()
                )
                *
                bs
            )

            mag_sum += (
                float(
                    loss_mag.item()
                )
                *
                bs
            )

            sign_sum += (
                float(
                    loss_sign.item()
                )
                *
                bs
            )

            n_seen += bs

        # =====================================================================
        # Validation
        # =====================================================================

        (
            pred_mag,
            pred_sign_prob,
        ) = predict(
            model,
            x_val,
            device,
        )

        # IMPORTANT:
        # primary magnitude head alone controls deployment.
        attenuation_mask = (
            pred_mag
            >
            0.0
        )

        stats = binary_metrics_from_score(
            pred_mag,
            y_val_sign,
        )

        (
            val_result,
            full_gates,
        ) = evaluate_policy(
            val_full,
            accepted_idx,
            attenuation_mask,
            ego_counts,
            range30_mask,
        )

        mae_scaled = float(
            np.mean(
                np.abs(
                    pred_mag
                    -
                    y_val_mag
                )
            )
        )

        rmse_scaled = float(
            np.sqrt(
                np.mean(
                    (
                        pred_mag
                        -
                        y_val_mag
                    )
                    **
                    2
                )
            )
        )

        selected_gain = (
            y_val_raw[
                attenuation_mask
            ]
        )

        harmful_rate = (
            float(
                (
                    selected_gain
                    <=
                    0.0
                ).mean()
            )
            if attenuation_mask.any()
            else 0.0
        )

        mean_true_gain = (
            float(
                selected_gain.mean()
            )
            if attenuation_mask.any()
            else 0.0
        )

        iou30 = float(
            val_result[
                "30m"
            ][
                "iou"
            ]
        )

        iou100 = float(
            val_result[
                "100m"
            ][
                "iou"
            ]
        )

        row = {
            "epoch":
                int(
                    epoch
                ),

            "train_loss":
                float(
                    loss_sum
                    /
                    n_seen
                ),

            "train_mag_loss":
                float(
                    mag_sum
                    /
                    n_seen
                ),

            "train_sign_loss":
                float(
                    sign_sum
                    /
                    n_seen
                ),

            "val_mae_scaled":
                mae_scaled,

            "val_rmse_scaled":
                rmse_scaled,

            "val_precision":
                stats[
                    "precision"
                ],

            "val_recall":
                stats[
                    "recall"
                ],

            "val_f1":
                stats[
                    "f1"
                ],

            "val_accuracy":
                stats[
                    "accuracy"
                ],

            "val_attenuation_rate":
                float(
                    attenuation_mask.mean()
                ),

            "val_harmful_rate":
                harmful_rate,

            "val_mean_true_gain":
                mean_true_gain,

            "val_iou30":
                iou30,

            "val_iou100":
                iou100,
        }

        history.append(
            row
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['train_loss']:.6f} "
            f"(mag={row['train_mag_loss']:.6f}, "
            f"sign={row['train_sign_loss']:.6f}) | "
            f"MAE={mae_scaled:.5f} "
            f"RMSE={rmse_scaled:.5f} | "
            f"P={stats['precision']:.4f} "
            f"R={stats['recall']:.4f} "
            f"F1={stats['f1']:.4f} | "
            f"atten={row['val_attenuation_rate']:.4f} "
            f"harm={harmful_rate:.4f} "
            f"gain={mean_true_gain:+.7f} | "
            f"IoU30={iou30:.6f} "
            f"IoU100={iou100:.6f}"
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

            best_iou100 = (
                iou100
            )

            best_iou30 = (
                iou30
            )

            best_epoch = (
                epoch
            )

            bad_epochs = 0

            torch.save(
                {
                    "epoch":
                        int(
                            epoch
                        ),

                    "model":
                        model.state_dict(),

                    "extra_mean":
                        extra_mean,

                    "extra_std":
                        extra_std,

                    "utility_scale":
                        UTILITY_SCALE,

                    "smoothl1_beta":
                        SMOOTHL1_BETA,

                    "lambda_sign":
                        LAMBDA_SIGN,

                    "pos_weight":
                        pos_weight_value,

                    "best_val_iou30":
                        best_iou30,

                    "best_val_iou100":
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
                f"Early stopping at epoch "
                f"{epoch}; "
                f"best epoch={best_epoch}"
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
        ],
        strict=True,
    )

    (
        pred_mag,
        pred_sign_prob,
    ) = predict(
        model,
        x_val,
        device,
    )

    attenuation_mask = (
        pred_mag
        >
        0.0
    )

    stats = binary_metrics_from_score(
        pred_mag,
        y_val_sign,
    )

    (
        final_result,
        full_gates,
    ) = evaluate_policy(
        val_full,
        accepted_idx,
        attenuation_mask,
        ego_counts,
        range30_mask,
    )

    selected_gain = (
        y_val_raw[
            attenuation_mask
        ]
    )

    harmful_rate = (
        float(
            (
                selected_gain
                <=
                0.0
            ).mean()
        )
        if attenuation_mask.any()
        else 0.0
    )

    mean_true_gain = (
        float(
            selected_gain.mean()
        )
        if attenuation_mask.any()
        else 0.0
    )

    mae_scaled = float(
        np.mean(
            np.abs(
                pred_mag
                -
                y_val_mag
            )
        )
    )

    rmse_scaled = float(
        np.sqrt(
            np.mean(
                (
                    pred_mag
                    -
                    y_val_mag
                )
                **
                2
            )
        )
    )

    # =========================================================================
    # Headroom recovery
    # =========================================================================

    delta30 = (
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
    )

    delta100 = (
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
    )

    oracle_delta30 = (
        binary_oracle_result[
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
    )

    oracle_delta100 = (
        binary_oracle_result[
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
    )

    recovery30 = (
        delta30
        /
        oracle_delta30
        if abs(
            oracle_delta30
        ) > EPS
        else 0.0
    )

    recovery100 = (
        delta100
        /
        oracle_delta100
        if abs(
            oracle_delta100
        ) > EPS
        else 0.0
    )

    gain_vs_e1_30 = (
        final_result[
            "30m"
        ][
            "iou"
        ]
        -
        e1_result[
            "30m"
        ][
            "iou"
        ]
    )

    gain_vs_e1_100 = (
        final_result[
            "100m"
        ][
            "iou"
        ]
        -
        e1_result[
            "100m"
        ][
            "iou"
        ]
    )

    # =========================================================================
    # Save
    # =========================================================================

    np.savez_compressed(
        PRED_FILE,

        accepted_indices=(
            accepted_idx.astype(
                np.int32
            )
        ),

        predicted_utility_scaled=(
            pred_mag.astype(
                np.float32
            )
        ),

        predicted_utility_raw=(
            (
                pred_mag
                /
                UTILITY_SCALE
            ).astype(
                np.float32
            )
        ),

        predicted_sign_probability=(
            pred_sign_prob.astype(
                np.float32
            )
        ),

        attenuation_mask=(
            attenuation_mask.astype(
                np.uint8
            )
        ),

        true_utility_scaled=(
            y_val_mag.astype(
                np.float32
            )
        ),

        true_utility_raw=(
            y_val_raw.astype(
                np.float32
            )
        ),

        true_sign=(
            y_val_sign.astype(
                np.uint8
            )
        ),

        full_candidate_gate=(
            full_gates.astype(
                np.float32
            )
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
            "V2-4C-3F Binary Utility Regression",

        "best_epoch":
            int(
                ckpt[
                    "epoch"
                ]
            ),

        "model_parameters":
            int(
                n_params
            ),

        "target":
            "100 * (U(g=0.1)-U(g=1.0))",

        "inference":
            "predicted magnitude > 0 => g=0.1; else g=1",

        "val_regression": {
            "mae_scaled":
                mae_scaled,

            "rmse_scaled":
                rmse_scaled,
        },

        "val_sign_metrics":
            stats,

        "attenuation_rate":
            float(
                attenuation_mask.mean()
            ),

        "harmful_attenuation_rate":
            harmful_rate,

        "mean_true_gain":
            mean_true_gain,

        "iou": {
            "v2_hard": {
                "30m":
                    float(
                        v2_result[
                            "30m"
                        ][
                            "iou"
                        ]
                    ),

                "100m":
                    float(
                        v2_result[
                            "100m"
                        ][
                            "iou"
                        ]
                    ),
            },

            "e1_bce": {
                "30m":
                    float(
                        e1_result[
                            "30m"
                        ][
                            "iou"
                        ]
                    ),

                "100m":
                    float(
                        e1_result[
                            "100m"
                        ][
                            "iou"
                        ]
                    ),
            },

            "binary_utility": {
                "30m":
                    float(
                        final_result[
                            "30m"
                        ][
                            "iou"
                        ]
                    ),

                "100m":
                    float(
                        final_result[
                            "100m"
                        ][
                            "iou"
                        ]
                    ),
            },

            "binary_oracle": {
                "30m":
                    float(
                        binary_oracle_result[
                            "30m"
                        ][
                            "iou"
                        ]
                    ),

                "100m":
                    float(
                        binary_oracle_result[
                            "100m"
                        ][
                            "iou"
                        ]
                    ),
            },
        },

        "delta_vs_v2": {
            "30m":
                float(
                    delta30
                ),

            "100m":
                float(
                    delta100
                ),
        },

        "delta_vs_e1": {
            "30m":
                float(
                    gain_vs_e1_30
                ),

            "100m":
                float(
                    gain_vs_e1_100
                ),
        },

        "binary_oracle_recovery": {
            "30m":
                float(
                    recovery30
                ),

            "100m":
                float(
                    recovery100
                ),
        },

        "test_used":
            False,

        "history":
            history,
    }

    with open(
        SUMMARY_FILE,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # =========================================================================
    # Final print
    # =========================================================================

    print()
    print("=" * 116)
    print(
        "V2-4C-3F FINAL VALIDATION"
    )
    print("=" * 116)

    print(
        "Best epoch              :",
        ckpt[
            "epoch"
        ],
    )

    print(
        "Model parameters        :",
        n_params,
    )

    print()
    print(
        "V2 Hard                :",
        f"30m={v2_result['30m']['iou']:.6f} "
        f"100m={v2_result['100m']['iou']:.6f}",
    )

    print(
        "E1 BCE classifier      :",
        f"30m={e1_result['30m']['iou']:.6f} "
        f"100m={e1_result['100m']['iou']:.6f}",
    )

    print(
        "Binary Utility         :",
        f"30m={final_result['30m']['iou']:.6f} "
        f"100m={final_result['100m']['iou']:.6f}",
    )

    print(
        "Binary Oracle          :",
        f"30m={binary_oracle_result['30m']['iou']:.6f} "
        f"100m={binary_oracle_result['100m']['iou']:.6f}",
    )

    print()
    print(
        "Delta vs V2 30m       :",
        f"{delta30:+.8f}",
    )

    print(
        "Delta vs V2 100m      :",
        f"{delta100:+.8f}",
    )

    print(
        "Delta vs E1 30m       :",
        f"{gain_vs_e1_30:+.8f}",
    )

    print(
        "Delta vs E1 100m      :",
        f"{gain_vs_e1_100:+.8f}",
    )

    print(
        "Oracle recovery 30m   :",
        f"{recovery30:.2%}",
    )

    print(
        "Oracle recovery 100m  :",
        f"{recovery100:.2%}",
    )

    print()
    print(
        "Utility MAE/RMSE      :",
        f"{mae_scaled:.6f} / "
        f"{rmse_scaled:.6f}",
    )

    print(
        "Precision              :",
        f"{stats['precision']:.6f}",
    )

    print(
        "Recall                 :",
        f"{stats['recall']:.6f}",
    )

    print(
        "F1                     :",
        f"{stats['f1']:.6f}",
    )

    print(
        "Accuracy               :",
        f"{stats['accuracy']:.6f}",
    )

    print(
        "Attenuation rate       :",
        f"{attenuation_mask.mean():.6%}",
    )

    print(
        "Harmful attenuation    :",
        f"{harmful_rate:.6%}",
    )

    print(
        "Mean true gain         :",
        f"{mean_true_gain:+.8f}",
    )

    print()
    print(
        "Checkpoint:",
        CKPT_FILE,
    )

    print(
        "Summary   :",
        SUMMARY_FILE,
    )

    print(
        "Predictions:",
        PRED_FILE,
    )

    print(
        "Test used : NO"
    )

    print()
    print(
        "[V2-4C-3F BINARY UTILITY REGRESSION] COMPLETE"
    )

    print("=" * 116)


if __name__ == "__main__":
    main()
