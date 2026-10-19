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

OLD_EVAL_SCRIPT = (
    BASE
    / "V2-4C_train_utility_conditioned_gate.py"
)

OUT_DIR = (
    BASE
    / "results/V2-4C-3E1_binary_attenuation"
)

CKPT_DIR = (
    BASE
    / "checkpoints/V2-4C-3E1_binary_attenuation"
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
    / "V2-4C-3E1_summary.json"
)

PRED_FILE = (
    OUT_DIR
    / "V2-4C-3E1_val_predictions.npz"
)


SEED = 2026

BATCH_SIZE = 512
EPOCHS = 100
PATIENCE = 15

LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.10

THRESHOLD = 0.50

EPS = 1e-12


# =============================================================================
# Reuse exact evaluator
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


# =============================================================================
# Model
# =============================================================================

class BinaryAttenuationPredictor(nn.Module):

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

        self.head = nn.Linear(
            128,
            1,
        )

    def forward(self, x):

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

        return (
            self.head(h)
            .squeeze(1)
        )


@torch.no_grad()
def predict(
    model,
    x,
    device,
):

    model.eval()

    probs = []

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

        logits = model(
            xb
        )

        probs.append(
            torch.sigmoid(
                logits
            )
            .cpu()
            .numpy()
        )

    return np.concatenate(
        probs
    ).astype(
        np.float32
    )


# =============================================================================
# Metrics
# =============================================================================

def binary_metrics(
    prob,
    target,
    threshold=0.5,
):

    pred = (
        prob
        >
        threshold
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

    acc = (
        tp + tn
    ) / len(
        target
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
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
                acc
            ),
    }


# =============================================================================
# Full Val evaluator
# =============================================================================

def get_full_val_ids(
    val,
):

    scene_tokens = set(
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
    val,
    accepted_idx,
    attenuation_mask,
    ego_counts,
    range30_mask,
):

    """
    V2 rejected candidate:
        g = 0

    V2 accepted:
        predicted attenuation -> g = 0.1
        otherwise             -> g = 1.0
    """

    gates = np.zeros(
        len(
            val[
                "features_127"
            ]
        ),
        dtype=np.float32,
    )

    gates[
        accepted_idx
    ] = 1.0

    attenuated_idx = accepted_idx[
        attenuation_mask
    ]

    gates[
        attenuated_idx
    ] = 0.1

    result = (
        v24c.evaluate_gates(
            val,
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

    train_curve = load_npz(
        TRAIN_CURVE
    )

    val_curve = load_npz(
        VAL_CURVE
    )

    # =========================================================================
    # V2 accepted subsets
    # =========================================================================

    train_accept = (
        train[
            "v2_accept"
        ].astype(bool)
    )

    val_accept = (
        val[
            "v2_accept"
        ].astype(bool)
    )

    train_idx = np.flatnonzero(
        train_accept
    )

    val_idx = np.flatnonzero(
        val_accept
    )

    assert len(
        train_idx
    ) == 19878

    assert len(
        val_idx
    ) == 6156

    # =========================================================================
    # Binary attenuation target
    #
    # y=1 iff g=0.1 is strictly better than original V2 g=1.
    # =========================================================================

    train_u = (
        train_curve[
            "utility_curve"
        ][
            train_idx
        ].astype(
            np.float32
        )
    )

    val_u = (
        val_curve[
            "utility_curve"
        ][
            val_idx
        ].astype(
            np.float32
        )
    )

    train_gain = (
        train_u[
            :,
            1
        ]
        -
        train_u[
            :,
            10
        ]
    )

    val_gain = (
        val_u[
            :,
            1
        ]
        -
        val_u[
            :,
            10
        ]
    )

    y_train = (
        train_gain
        >
        EPS
    ).astype(
        np.float32
    )

    y_val = (
        val_gain
        >
        EPS
    ).astype(
        np.float32
    )

    assert int(
        y_train.sum()
    ) == 8547

    assert int(
        y_val.sum()
    ) == 3361

    # =========================================================================
    # Input
    # =========================================================================

    (
        x_train_all,
        extra_mean,
        extra_std,
    ) = build_input(
        train
    )

    (
        x_val_all,
        _,
        _,
    ) = build_input(
        val,
        mean=extra_mean,
        std=extra_std,
    )

    x_train = (
        x_train_all[
            train_idx
        ]
    )

    x_val = (
        x_val_all[
            val_idx
        ]
    )

    # =========================================================================
    # Full-Val evaluator
    # =========================================================================

    full_val_ids = (
        get_full_val_ids(
            val
        )
    )

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

    v2_result = (
        v24c.evaluate_gates(
            val,
            val_accept.astype(
                np.float32
            ),
            ego_counts,
            range30_mask,
        )
    )

    # Binary oracle:
    # V2 accepted -> g=0.1 if y=1 else g=1.
    binary_oracle_result, _ = (
        evaluate_policy(
            val,
            val_idx,
            y_val.astype(
                bool
            ),
            ego_counts,
            range30_mask,
        )
    )

    print("=" * 112)
    print(
        "V2-4C-3E1 BINARY ATTENUATION PREDICTOR"
    )
    print("=" * 112)

    print(
        "Train V2 accepted       :",
        len(
            train_idx
        ),
    )

    print(
        "Val V2 accepted         :",
        len(
            val_idx
        ),
    )

    print(
        "Train attenuate target  :",
        int(
            y_train.sum()
        ),
        f"({y_train.mean():.6%})",
    )

    print(
        "Val attenuate target    :",
        int(
            y_val.sum()
        ),
        f"({y_val.mean():.6%})",
    )

    print(
        "Task                    :",
        "g=0.1 vs g=1.0",
    )

    print(
        "Input                   :",
        "129-D",
    )

    print(
        "Loss                    :",
        "BCEWithLogitsLoss",
    )

    print(
        "Decision threshold      :",
        THRESHOLD,
    )

    print(
        "Test used               :",
        "NO",
    )

    print()
    print(
        "V2 Hard                 :",
        f"30m={v2_result['30m']['iou']:.6f} "
        f"100m={v2_result['100m']['iou']:.6f}",
    )

    print(
        "Binary Oracle           :",
        f"30m={binary_oracle_result['30m']['iou']:.6f} "
        f"100m={binary_oracle_result['100m']['iou']:.6f}",
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
            y_train
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

    model = (
        BinaryAttenuationPredictor(
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

            logits = model(
                xb
            )

            loss = (
                F.binary_cross_entropy_with_logits(
                    logits,
                    yb,
                )
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

            n_seen += bs

        # =====================================================================
        # Val
        # =====================================================================

        prob = predict(
            model,
            x_val,
            device,
        )

        attenuation_mask = (
            prob
            >
            THRESHOLD
        )

        stats = binary_metrics(
            prob,
            y_val,
            threshold=THRESHOLD,
        )

        (
            val_result,
            full_gates,
        ) = evaluate_policy(
            val,
            val_idx,
            attenuation_mask,
            ego_counts,
            range30_mask,
        )

        # Actual utility effect of chosen binary action.
        chosen_gain = val_gain[
            attenuation_mask
        ]

        harmful_rate = (
            float(
                (
                    chosen_gain
                    <=
                    0.0
                ).mean()
            )
            if attenuation_mask.any()
            else 0.0
        )

        mean_true_gain = (
            float(
                chosen_gain.mean()
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

            "val_harmful_attenuation_rate":
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
            f"loss={row['train_loss']:.6f} | "
            f"P={stats['precision']:.4f} "
            f"R={stats['recall']:.4f} "
            f"F1={stats['f1']:.4f} "
            f"Acc={stats['accuracy']:.4f} | "
            f"atten={row['val_attenuation_rate']:.4f} "
            f"harm={harmful_rate:.4f} "
            f"gain={mean_true_gain:+.6f} | "
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

                    "threshold":
                        THRESHOLD,

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
                f"{epoch}; best epoch={best_epoch}"
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

    prob = predict(
        model,
        x_val,
        device,
    )

    attenuation_mask = (
        prob
        >
        THRESHOLD
    )

    stats = binary_metrics(
        prob,
        y_val,
        threshold=THRESHOLD,
    )

    (
        final_result,
        full_gates,
    ) = evaluate_policy(
        val,
        val_idx,
        attenuation_mask,
        ego_counts,
        range30_mask,
    )

    chosen_gain = (
        val_gain[
            attenuation_mask
        ]
    )

    harmful_rate = (
        float(
            (
                chosen_gain
                <=
                0.0
            ).mean()
        )
        if attenuation_mask.any()
        else 0.0
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

    # =========================================================================
    # Save
    # =========================================================================

    np.savez_compressed(
        PRED_FILE,

        accepted_indices=(
            val_idx.astype(
                np.int32
            )
        ),

        attenuation_probability=(
            prob
        ),

        attenuation_mask=(
            attenuation_mask.astype(
                np.uint8
            )
        ),

        true_label=(
            y_val.astype(
                np.uint8
            )
        ),

        true_gain=(
            val_gain.astype(
                np.float32
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
            "V2-4C-3E1 Binary Attenuation Predictor",

        "best_epoch":
            int(
                ckpt[
                    "epoch"
                ]
            ),

        "train_v2_accepted":
            int(
                len(
                    train_idx
                )
            ),

        "val_v2_accepted":
            int(
                len(
                    val_idx
                )
            ),

        "train_positive":
            int(
                y_train.sum()
            ),

        "val_positive":
            int(
                y_val.sum()
            ),

        "threshold":
            THRESHOLD,

        "val_metrics":
            stats,

        "attenuation_count":
            int(
                attenuation_mask.sum()
            ),

        "attenuation_rate":
            float(
                attenuation_mask.mean()
            ),

        "harmful_attenuation_rate":
            harmful_rate,

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

            "learned_binary_attenuation": {
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
    # Print final
    # =========================================================================

    print()
    print("=" * 112)
    print(
        "V2-4C-3E1 FINAL VALIDATION"
    )
    print("=" * 112)

    print(
        "Best epoch              :",
        ckpt[
            "epoch"
        ],
    )

    print()
    print(
        "V2 Hard                :",
        f"30m={v2_result['30m']['iou']:.6f} "
        f"100m={v2_result['100m']['iou']:.6f}",
    )

    print(
        "Binary Oracle           :",
        f"30m={binary_oracle_result['30m']['iou']:.6f} "
        f"100m={binary_oracle_result['100m']['iou']:.6f}",
    )

    print(
        "Learned Binary         :",
        f"30m={final_result['30m']['iou']:.6f} "
        f"100m={final_result['100m']['iou']:.6f}",
    )

    print()
    print(
        "Delta vs V2 30m        :",
        f"{delta30:+.8f}",
    )

    print(
        "Delta vs V2 100m       :",
        f"{delta100:+.8f}",
    )

    print(
        "Oracle recovery 30m    :",
        f"{recovery30:.2%}",
    )

    print(
        "Oracle recovery 100m   :",
        f"{recovery100:.2%}",
    )

    print()
    print(
        "Precision               :",
        f"{stats['precision']:.6f}",
    )

    print(
        "Recall                  :",
        f"{stats['recall']:.6f}",
    )

    print(
        "F1                      :",
        f"{stats['f1']:.6f}",
    )

    print(
        "Accuracy                :",
        f"{stats['accuracy']:.6f}",
    )

    print(
        "Attenuation rate        :",
        f"{attenuation_mask.mean():.6%}",
    )

    print(
        "Harmful attenuation rate:",
        f"{harmful_rate:.6%}",
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
        "[V2-4C-3E1 BINARY ATTENUATION] COMPLETE"
    )

    print("=" * 112)


if __name__ == "__main__":
    main()
