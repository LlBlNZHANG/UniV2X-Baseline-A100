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

TRAIN_CONTEXT = (
    BASE / "V2-4C-3E2A_train_context_dataset.npz"
)

VAL_CONTEXT = (
    BASE / "V2-4C-3E2A_val_context_dataset.npz"
)

VAL_FULL = (
    BASE / "V2-4B_val_learnable_fusion_dataset.npz"
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
    / "results/V2-4C-3E2B_context_binary"
)

CKPT_DIR = (
    BASE
    / "checkpoints/V2-4C-3E2B_context_binary"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CKPT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CKPT_FILE = CKPT_DIR / "best.pth"
SUMMARY_FILE = OUT_DIR / "V2-4C-3E2B_summary.json"
PRED_FILE = OUT_DIR / "V2-4C-3E2B_val_predictions.npz"


SEED = 2026

BATCH_SIZE = 256
EPOCHS = 100
PATIENCE = 15

LR = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.15

THRESHOLD = 0.50

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


def binary_metrics(
    prob,
    target,
    threshold=0.5,
):

    pred = prob > threshold
    target = target.astype(bool)

    tp = int((pred & target).sum())
    fp = int((pred & (~target)).sum())
    fn = int(((~pred) & target).sum())
    tn = int(((~pred) & (~target)).sum())

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    accuracy = (
        (tp + tn)
        /
        len(target)
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


# =============================================================================
# Exact 129-D structured input
#
# Reuse E1 normalization so E1 -> E2B comparison only changes context/model.
# =============================================================================

def build_structured_input(
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
# Context-enhanced model
# =============================================================================

class ContextEnhancedBinaryAttenuation(nn.Module):

    def __init__(
        self,
        dropout=0.15,
    ):

        super().__init__()

        # -------------------------------------------------------------
        # Raw spatial branch:
        # 5 x 12 x 12 -> 96 x 3 x 3 -> 256-D
        # -------------------------------------------------------------

        self.context_encoder = nn.Sequential(

            nn.Conv2d(
                5,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.GroupNorm(
                8,
                32,
            ),

            nn.GELU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.GroupNorm(
                8,
                64,
            ),

            nn.GELU(),

            nn.Conv2d(
                64,
                96,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.GroupNorm(
                8,
                96,
            ),

            nn.GELU(),
        )

        self.context_proj = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                96 * 3 * 3,
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

        # -------------------------------------------------------------
        # Existing STCV + V2 utility branch
        # -------------------------------------------------------------

        self.structured_proj = nn.Sequential(

            nn.Linear(
                129,
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

        # -------------------------------------------------------------
        # Cross-branch fusion
        # -------------------------------------------------------------

        self.fusion = nn.Sequential(

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

        self.down = nn.Sequential(

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

        self.head = nn.Linear(
            256,
            1,
        )

    def forward(
        self,
        context,
        structured,
    ):

        c = self.context_encoder(
            context
        )

        c = self.context_proj(
            c
        )

        s = self.structured_proj(
            structured
        )

        h = torch.cat(
            [
                c,
                s,
            ],
            dim=1,
        )

        h = self.fusion(
            h
        )

        h = (
            h
            +
            self.residual(
                h
            )
        )

        h = self.down(
            h
        )

        return (
            self.head(h)
            .squeeze(1)
        )


@torch.no_grad()
def predict(
    model,
    context,
    structured,
    device,
):

    model.eval()

    outputs = []

    for start in range(
        0,
        len(context),
        2048,
    ):

        cb = torch.from_numpy(
            context[
                start:
                start + 2048
            ]
        ).to(
            device
        )

        sb = torch.from_numpy(
            structured[
                start:
                start + 2048
            ]
        ).to(
            device
        )

        logits = model(
            cb,
            sb,
        )

        outputs.append(
            torch.sigmoid(
                logits
            )
            .cpu()
            .numpy()
        )

    return np.concatenate(
        outputs
    ).astype(
        np.float32
    )


# =============================================================================
# Full validation
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

            if parts[1] in scene_tokens:
                ids.append(
                    parts[0]
                )

    ids = np.asarray(
        ids
    )

    assert len(
        np.unique(ids)
    ) == 96

    return ids


def evaluate_policy(
    val_full,
    accepted_idx,
    attenuation_mask,
    ego_counts,
    range30_mask,
):

    gates = np.zeros(
        len(
            val_full[
                "features_127"
            ]
        ),
        dtype=np.float32,
    )

    # V2 accepted default.
    gates[
        accepted_idx
    ] = 1.0

    # Innovation 3 attenuation.
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
        TRAIN_CONTEXT
    )

    val = load_npz(
        VAL_CONTEXT
    )

    val_full = load_npz(
        VAL_FULL
    )

    e1_pred = load_npz(
        E1_PRED
    )

    # -------------------------------------------------------------------------
    # Dataset audit
    # -------------------------------------------------------------------------

    assert train[
        "context_5x12x12"
    ].shape == (
        19878,
        5,
        12,
        12,
    )

    assert val[
        "context_5x12x12"
    ].shape == (
        6156,
        5,
        12,
        12,
    )

    assert int(
        train[
            "attenuation_target"
        ].sum()
    ) == 8547

    assert int(
        val[
            "attenuation_target"
        ].sum()
    ) == 3361

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

    # -------------------------------------------------------------------------
    # Reuse exact E1 normalization.
    # -------------------------------------------------------------------------

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

    x_train = build_structured_input(
        train,
        extra_mean,
        extra_std,
    )

    x_val = build_structured_input(
        val,
        extra_mean,
        extra_std,
    )

    context_train = train[
        "context_5x12x12"
    ].astype(
        np.float32
    )

    context_val = val[
        "context_5x12x12"
    ].astype(
        np.float32
    )

    y_train = train[
        "attenuation_target"
    ].astype(
        np.float32
    )

    y_val = val[
        "attenuation_target"
    ].astype(
        np.float32
    )

    true_gain_val = val[
        "true_binary_gain"
    ].astype(
        np.float32
    )

    # -------------------------------------------------------------------------
    # Exact full-Val evaluator.
    # -------------------------------------------------------------------------

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

    oracle_result, _ = (
        evaluate_policy(
            val_full,
            accepted_idx,
            y_val.astype(bool),
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
            ].astype(bool),
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
        oracle_result[
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

    print("=" * 116)
    print(
        "V2-4C-3E2B CONTEXT-ENHANCED BINARY ATTENUATION"
    )
    print("=" * 116)

    print(
        "Train accepted          :",
        len(
            context_train
        ),
    )

    print(
        "Val accepted            :",
        len(
            context_val
        ),
    )

    print(
        "Context input           :",
        "5 x 12 x 12",
    )

    print(
        "Structured input        :",
        "129-D",
    )

    print(
        "Target                  :",
        "g=0.1 vs g=1.0",
    )

    print(
        "Loss                    :",
        "BCEWithLogitsLoss",
    )

    print(
        "Threshold               :",
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
        "E1 129-D MLP            :",
        f"30m={e1_result['30m']['iou']:.6f} "
        f"100m={e1_result['100m']['iou']:.6f}",
    )

    print(
        "Binary Oracle           :",
        f"30m={oracle_result['30m']['iou']:.6f} "
        f"100m={oracle_result['100m']['iou']:.6f}",
    )

    print(
        "Reference reproduction  : PASS"
    )

    # =========================================================================
    # Dataset
    # =========================================================================

    dataset = TensorDataset(
        torch.from_numpy(
            context_train
        ),

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

    # =========================================================================
    # Model
    # =========================================================================

    model = (
        ContextEnhancedBinaryAttenuation(
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
    # Training
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

        for (
            context_b,
            structured_b,
            yb,
        ) in loader:

            context_b = context_b.to(
                device,
                non_blocking=True,
            )

            structured_b = structured_b.to(
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
                context_b,
                structured_b,
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
                yb
            )

            loss_sum += (
                float(
                    loss.item()
                )
                *
                bs
            )

            n_seen += bs

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        prob = predict(
            model,
            context_val,
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
            THRESHOLD,
        )

        (
            result,
            full_gates,
        ) = evaluate_policy(
            val_full,
            accepted_idx,
            attenuation_mask,
            ego_counts,
            range30_mask,
        )

        selected_gain = (
            true_gain_val[
                attenuation_mask
            ]
        )

        harmful_rate = (
            float(
                (
                    selected_gain
                    <=
                    0
                ).mean()
            )
            if attenuation_mask.any()
            else 0.0
        )

        mean_gain = (
            float(
                selected_gain.mean()
            )
            if attenuation_mask.any()
            else 0.0
        )

        iou30 = float(
            result[
                "30m"
            ][
                "iou"
            ]
        )

        iou100 = float(
            result[
                "100m"
            ][
                "iou"
            ]
        )

        row = {
            "epoch":
                int(epoch),

            "loss":
                float(
                    loss_sum
                    /
                    n_seen
                ),

            "precision":
                stats[
                    "precision"
                ],

            "recall":
                stats[
                    "recall"
                ],

            "f1":
                stats[
                    "f1"
                ],

            "accuracy":
                stats[
                    "accuracy"
                ],

            "attenuation_rate":
                float(
                    attenuation_mask.mean()
                ),

            "harmful_rate":
                harmful_rate,

            "mean_true_gain":
                mean_gain,

            "iou30":
                iou30,

            "iou100":
                iou100,
        }

        history.append(
            row
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['loss']:.6f} | "
            f"P={stats['precision']:.4f} "
            f"R={stats['recall']:.4f} "
            f"F1={stats['f1']:.4f} "
            f"Acc={stats['accuracy']:.4f} | "
            f"atten={row['attenuation_rate']:.4f} "
            f"harm={harmful_rate:.4f} "
            f"gain={mean_gain:+.6f} | "
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

            best_iou100 = iou100
            best_iou30 = iou30
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
        context_val,
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
        THRESHOLD,
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
        true_gain_val[
            attenuation_mask
        ]
    )

    harmful_rate = (
        float(
            (
                selected_gain
                <=
                0
            ).mean()
        )
        if attenuation_mask.any()
        else 0.0
    )

    mean_gain = (
        float(
            selected_gain.mean()
        )
        if attenuation_mask.any()
        else 0.0
    )

    # =========================================================================
    # Headroom
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
        oracle_result[
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
        oracle_result[
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
            true_gain_val
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
            "V2-4C-3E2B Context-enhanced Binary Attenuation",

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

        "val_metrics":
            stats,

        "attenuation_rate":
            float(
                attenuation_mask.mean()
            ),

        "harmful_attenuation_rate":
            harmful_rate,

        "mean_true_gain":
            mean_gain,

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

            "e1_129d_mlp": {
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

            "context_enhanced": {
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
                        oracle_result[
                            "30m"
                        ][
                            "iou"
                        ]
                    ),

                "100m":
                    float(
                        oracle_result[
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
    # Final
    # =========================================================================

    print()
    print("=" * 116)
    print(
        "V2-4C-3E2B FINAL VALIDATION"
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
        "E1 129-D MLP           :",
        f"30m={e1_result['30m']['iou']:.6f} "
        f"100m={e1_result['100m']['iou']:.6f}",
    )

    print(
        "Context Enhanced       :",
        f"30m={final_result['30m']['iou']:.6f} "
        f"100m={final_result['100m']['iou']:.6f}",
    )

    print(
        "Binary Oracle          :",
        f"30m={oracle_result['30m']['iou']:.6f} "
        f"100m={oracle_result['100m']['iou']:.6f}",
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
        f"{mean_gain:+.8f}",
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
        "[V2-4C-3E2B CONTEXT BINARY] COMPLETE"
    )

    print("=" * 116)


if __name__ == "__main__":
    main()
