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

COND_ORACLE_VAL = (
    BASE
    / "V2-4C-3A_val_v2_conditioned_soft_oracle.npz"
)

OLD_EVAL_SCRIPT = (
    BASE
    / "V2-4C_train_utility_conditioned_gate.py"
)

OUT_DIR = (
    BASE
    / "results/V2-4C-3B_relative_refinement"
)

CKPT_DIR = (
    BASE
    / "checkpoints/V2-4C-3B_relative_refinement"
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
    / "V2-4C-3B_summary.json"
)

PRED_FILE = (
    OUT_DIR
    / "V2-4C-3B_val_predictions.npz"
)


SEED = 2026

UTILITY_SCALE = 100.0

BATCH_SIZE = 512
EPOCHS = 100
PATIENCE = 15

LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.10

# Relative utility regression + refinable classification.
LAMBDA_REFINE = 0.50

EPS = 1e-12

# First deployment policy:
# Only refine when both conditions are satisfied.
REFINE_PROB_THRESHOLD = 0.50
REFINE_UTILITY_THRESHOLD = 0.0


# =============================================================================
# Reuse exact full-split occupancy evaluator
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
# Input
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


# =============================================================================
# Model
# =============================================================================

class RelativeRefinementPredictor(nn.Module):

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

        # ΔU(g) relative to the original V2 g=1,
        # for g = 0.1 ... 0.9.
        self.relative_utility_head = nn.Linear(
            128,
            9,
        )

        # Whether lowering g below 1 can improve utility.
        self.refinable_head = nn.Linear(
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
            self.residual(h)
        )

        h = self.down_256(
            h
        )

        h = self.down_128(
            h
        )

        relative_utility = (
            self.relative_utility_head(
                h
            )
        )

        refinable_logit = (
            self.refinable_head(
                h
            )
            .squeeze(1)
        )

        return (
            relative_utility,
            refinable_logit,
        )


# =============================================================================
# Prediction
# =============================================================================

@torch.no_grad()
def predict(
    model,
    x,
    device,
):

    model.eval()

    relative_out = []
    prob_out = []

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
            relative,
            refinable_logit,
        ) = model(
            xb
        )

        relative_out.append(
            relative
            .cpu()
            .numpy()
        )

        prob_out.append(
            torch.sigmoid(
                refinable_logit
            )
            .cpu()
            .numpy()
        )

    return (
        np.concatenate(
            relative_out,
            axis=0,
        ).astype(
            np.float32
        ),

        np.concatenate(
            prob_out,
            axis=0,
        ).astype(
            np.float32
        ),
    )


# =============================================================================
# Policy
# =============================================================================

def make_refinement_policy(
    pred_relative_scaled,
    refinable_prob,
    prob_threshold,
    utility_threshold_scaled,
):

    """
    Accepted-candidate policy only.

    Default = g=1.

    Refine to one of 0.1...0.9 only if:
      p(refinable) > tau_p
      max predicted relative gain > tau_u
    """

    n = len(
        pred_relative_scaled
    )

    assert pred_relative_scaled.shape == (
        n,
        9,
    )

    gate_grid = np.arange(
        1,
        10,
        dtype=np.float32,
    ) / 10.0

    best_idx = np.argmax(
        pred_relative_scaled,
        axis=1,
    )

    best_gain = (
        pred_relative_scaled[
            np.arange(n),
            best_idx,
        ]
    )

    refine = (
        refinable_prob
        >
        prob_threshold
    ) & (
        best_gain
        >
        utility_threshold_scaled
    )

    gate = np.ones(
        n,
        dtype=np.float32,
    )

    gate[
        refine
    ] = gate_grid[
        best_idx[
            refine
        ]
    ]

    return (
        gate,
        refine,
        best_gain,
        best_idx,
    )


# =============================================================================
# Exact full-Val setup
# =============================================================================

def get_full_val_ids(
    val,
):

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
                val_scene_tokens
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


def evaluate_full_policy(
    val,
    accepted_indices,
    accepted_gates,
    ego_counts,
    range30_mask,
):

    # Innovation 2 remains untouched:
    # rejected by V2 -> g=0
    # accepted by V2 -> default g=1 or refined g.
    all_gates = np.zeros(
        len(
            val[
                "features_127"
            ]
        ),
        dtype=np.float32,
    )

    all_gates[
        accepted_indices
    ] = accepted_gates

    result = v24c.evaluate_gates(
        val,
        all_gates,
        ego_counts,
        range30_mask,
    )

    return (
        result,
        all_gates,
    )


# =============================================================================
# Classification diagnostics
# =============================================================================

def binary_stats(
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
        if (
            tp + fp
        ) > 0
        else 0.0
    )

    recall = (
        tp
        /
        (tp + fn)
        if (
            tp + fn
        ) > 0
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
        if (
            precision
            +
            recall
        ) > 0
        else 0.0
    )

    acc = (
        tp + tn
    ) / len(
        target
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
                acc
            ),
    }


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

    conditioned_oracle = load_npz(
        COND_ORACLE_VAL
    )

    # =========================================================================
    # Exact accepted-candidate subsets
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

    x_train = x_train_all[
        train_idx
    ]

    x_val = x_val_all[
        val_idx
    ]

    # =========================================================================
    # Relative refinement utility targets
    #
    # Raw curve:
    #   columns 0...10 -> g=0.0...1.0
    #
    # Innovation 3:
    #   compare g=0.1...0.9 against original V2 g=1.
    # =========================================================================

    train_u = train_curve[
        "utility_curve"
    ].astype(
        np.float32
    )[
        train_idx
    ]

    val_u = val_curve[
        "utility_curve"
    ].astype(
        np.float32
    )[
        val_idx
    ]

    train_relative_raw = (
        train_u[
            :,
            1:10
        ]
        -
        train_u[
            :,
            10:11
        ]
    )

    val_relative_raw = (
        val_u[
            :,
            1:10
        ]
        -
        val_u[
            :,
            10:11
        ]
    )

    train_refinable = (
        train_relative_raw.max(
            axis=1
        )
        >
        EPS
    )

    val_refinable = (
        val_relative_raw.max(
            axis=1
        )
        >
        EPS
    )

    # Exact reconstruction of V2-4C-3A.
    assert int(
        train_refinable.sum()
    ) == 10796

    assert int(
        val_refinable.sum()
    ) == 4067

    y_train_relative = (
        UTILITY_SCALE
        *
        train_relative_raw
    ).astype(
        np.float32
    )

    y_val_relative = (
        UTILITY_SCALE
        *
        val_relative_raw
    ).astype(
        np.float32
    )

    y_train_refinable = (
        train_refinable.astype(
            np.float32
        )
    )

    y_val_refinable = (
        val_refinable.astype(
            np.float32
        )
    )

    print("=" * 110)
    print(
        "V2-4C-3B ACCEPTED-CANDIDATE RELATIVE REFINEMENT UTILITY LEARNING"
    )
    print("=" * 110)

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
        "Train refinable         :",
        int(
            train_refinable.sum()
        ),
        f"({train_refinable.mean():.6%})",
    )

    print(
        "Val refinable           :",
        int(
            val_refinable.sum()
        ),
        f"({val_refinable.mean():.6%})",
    )

    print(
        "Input                   :",
        "129-D = X127 + V2 magnitude + V2 sign",
    )

    print(
        "Relative utility output :",
        "9-D for g=0.1...0.9 relative to V2 g=1",
    )

    print(
        "Auxiliary output        :",
        "refinable probability",
    )

    print(
        "Default policy          :",
        "keep V2 g=1",
    )

    print(
        "Oracle best gate label  :",
        "NOT directly used",
    )

    print(
        "Test used               :",
        "NO",
    )

    # =========================================================================
    # Full Val evaluator
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

    conditioned_oracle_result = (
        v24c.evaluate_gates(
            val,
            conditioned_oracle[
                "refined_gate"
            ].astype(
                np.float32
            ),
            ego_counts,
            range30_mask,
        )
    )

    print()
    print(
        "Reference V2 Hard       :",
        f"30m={v2_result['30m']['iou']:.6f} "
        f"100m={v2_result['100m']['iou']:.6f}",
    )

    print(
        "Conditioned Soft Oracle :",
        f"30m={conditioned_oracle_result['30m']['iou']:.6f} "
        f"100m={conditioned_oracle_result['100m']['iou']:.6f}",
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
        conditioned_oracle_result[
            "100m"
        ][
            "iou"
        ]
        -
        0.381884
    ) < 2e-6

    print(
        "Reference reproduction  : PASS"
    )

    # =========================================================================
    # Dataset / model
    # =========================================================================

    dataset = TensorDataset(
        torch.from_numpy(
            x_train
        ),

        torch.from_numpy(
            y_train_relative
        ),

        torch.from_numpy(
            y_train_refinable
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
        RelativeRefinementPredictor(
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

        total_sum = 0.0
        reg_sum = 0.0
        cls_sum = 0.0
        n_seen = 0

        for (
            xb,
            y_rel,
            y_ref,
        ) in loader:

            xb = xb.to(
                device,
                non_blocking=True,
            )

            y_rel = y_rel.to(
                device,
                non_blocking=True,
            )

            y_ref = y_ref.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                pred_rel,
                pred_ref_logit,
            ) = model(
                xb
            )

            loss_reg = (
                F.smooth_l1_loss(
                    pred_rel,
                    y_rel,
                    beta=0.10,
                )
            )

            loss_ref = (
                F.binary_cross_entropy_with_logits(
                    pred_ref_logit,
                    y_ref,
                )
            )

            loss = (
                loss_reg
                +
                LAMBDA_REFINE
                *
                loss_ref
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            bs = len(xb)

            total_sum += (
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

            cls_sum += (
                float(
                    loss_ref.item()
                )
                *
                bs
            )

            n_seen += bs

        # =====================================================================
        # Validation
        # =====================================================================

        (
            pred_relative,
            refinable_prob,
        ) = predict(
            model,
            x_val,
            device,
        )

        (
            accepted_gates,
            refine_mask,
            best_pred_gain,
            best_pred_idx,
        ) = make_refinement_policy(
            pred_relative,
            refinable_prob,
            REFINE_PROB_THRESHOLD,
            REFINE_UTILITY_THRESHOLD,
        )

        (
            val_result,
            all_val_gates,
        ) = evaluate_full_policy(
            val,
            val_idx,
            accepted_gates,
            ego_counts,
            range30_mask,
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

        rel_mae = float(
            np.mean(
                np.abs(
                    pred_relative
                    -
                    y_val_relative
                )
            )
        )

        stats = binary_stats(
            refinable_prob,
            val_refinable,
            threshold=REFINE_PROB_THRESHOLD,
        )

        true_best_gain = (
            y_val_relative.max(
                axis=1
            )
        )

        selected_true_gain = np.zeros(
            len(
                val_idx
            ),
            dtype=np.float32,
        )

        selected_true_gain[
            refine_mask
        ] = (
            y_val_relative[
                np.flatnonzero(
                    refine_mask
                ),
                best_pred_idx[
                    refine_mask
                ],
            ]
        )

        true_gain_mean = float(
            selected_true_gain[
                refine_mask
            ].mean()
        ) if refine_mask.any() else 0.0

        harmful_refine_rate = float(
            (
                selected_true_gain[
                    refine_mask
                ]
                <=
                0.0
            ).mean()
        ) if refine_mask.any() else 0.0

        row = {
            "epoch":
                int(epoch),

            "train_loss":
                float(
                    total_sum
                    /
                    n_seen
                ),

            "train_reg":
                float(
                    reg_sum
                    /
                    n_seen
                ),

            "train_cls":
                float(
                    cls_sum
                    /
                    n_seen
                ),

            "val_relative_mae_scaled":
                rel_mae,

            "val_refinable_precision":
                stats[
                    "precision"
                ],

            "val_refinable_recall":
                stats[
                    "recall"
                ],

            "val_refinable_f1":
                stats[
                    "f1"
                ],

            "val_refine_rate":
                float(
                    refine_mask.mean()
                ),

            "val_selected_true_gain_scaled":
                true_gain_mean,

            "val_harmful_refine_rate":
                harmful_refine_rate,

            "val_iou30":
                float(
                    iou30
                ),

            "val_iou100":
                float(
                    iou100
                ),
        }

        history.append(
            row
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['train_loss']:.6f} "
            f"reg={row['train_reg']:.6f} "
            f"cls={row['train_cls']:.6f} | "
            f"MAE={rel_mae:.5f} "
            f"F1={stats['f1']:.4f} "
            f"P={stats['precision']:.4f} "
            f"R={stats['recall']:.4f} | "
            f"refine={row['val_refine_rate']:.4f} "
            f"harm={harmful_refine_rate:.4f} | "
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

            best_iou100 = float(
                iou100
            )

            best_iou30 = float(
                iou30
            )

            best_epoch = int(
                epoch
            )

            bad_epochs = 0

            torch.save(
                {
                    "epoch":
                        best_epoch,

                    "model":
                        model.state_dict(),

                    "extra_mean":
                        extra_mean,

                    "extra_std":
                        extra_std,

                    "utility_scale":
                        UTILITY_SCALE,

                    "lambda_refine":
                        LAMBDA_REFINE,

                    "prob_threshold":
                        REFINE_PROB_THRESHOLD,

                    "utility_threshold":
                        REFINE_UTILITY_THRESHOLD,

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

    (
        pred_relative,
        refinable_prob,
    ) = predict(
        model,
        x_val,
        device,
    )

    (
        accepted_gates,
        refine_mask,
        best_pred_gain,
        best_pred_idx,
    ) = make_refinement_policy(
        pred_relative,
        refinable_prob,
        REFINE_PROB_THRESHOLD,
        REFINE_UTILITY_THRESHOLD,
    )

    (
        final_result,
        all_val_gates,
    ) = evaluate_full_policy(
        val,
        val_idx,
        accepted_gates,
        ego_counts,
        range30_mask,
    )

    stats = binary_stats(
        refinable_prob,
        val_refinable,
        threshold=REFINE_PROB_THRESHOLD,
    )

    # =========================================================================
    # Gate histogram
    # =========================================================================

    gate_values, gate_counts = np.unique(
        accepted_gates,
        return_counts=True,
    )

    gate_hist = {
        f"{float(g):.1f}":
            int(c)
        for g, c in
        zip(
            gate_values,
            gate_counts,
        )
    }

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
        conditioned_oracle_result[
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
        conditioned_oracle_result[
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

        predicted_relative_utility=(
            pred_relative
        ),

        refinable_probability=(
            refinable_prob
        ),

        predicted_refine_mask=(
            refine_mask.astype(
                np.uint8
            )
        ),

        predicted_gate_accepted=(
            accepted_gates
        ),

        full_candidate_gate=(
            all_val_gates
        ),

        true_relative_utility=(
            y_val_relative
        ),

        true_refinable=(
            val_refinable.astype(
                np.uint8
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
            "V2-4C-3B Accepted-candidate Relative Refinement Utility Learning",

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

        "train_refinable":
            int(
                train_refinable.sum()
            ),

        "val_refinable":
            int(
                val_refinable.sum()
            ),

        "val_refinable_metrics":
            stats,

        "val_refine_count":
            int(
                refine_mask.sum()
            ),

        "val_refine_rate":
            float(
                refine_mask.mean()
            ),

        "gate_histogram":
            gate_hist,

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

            "learned_refinement": {
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

            "conditioned_oracle": {
                "30m":
                    float(
                        conditioned_oracle_result[
                            "30m"
                        ][
                            "iou"
                        ]
                    ),

                "100m":
                    float(
                        conditioned_oracle_result[
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

        "conditioned_oracle_recovery": {
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
    # Print final result
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "V2-4C-3B FINAL VALIDATION"
    )
    print("=" * 110)

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
        "Conditioned Soft Oracle:",
        f"30m={conditioned_oracle_result['30m']['iou']:.6f} "
        f"100m={conditioned_oracle_result['100m']['iou']:.6f}",
    )

    print(
        "Learned Refinement     :",
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
        "Refinable precision    :",
        f"{stats['precision']:.6f}",
    )

    print(
        "Refinable recall       :",
        f"{stats['recall']:.6f}",
    )

    print(
        "Refinable F1           :",
        f"{stats['f1']:.6f}",
    )

    print(
        "Predicted refine rate  :",
        f"{refine_mask.mean():.6%}",
    )

    print()
    print(
        "Accepted gate histogram:"
    )

    for k in sorted(
        gate_hist.keys(),
        key=float,
    ):

        print(
            f"  g={k}: "
            f"{gate_hist[k]}"
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
        "[V2-4C-3B RELATIVE REFINEMENT] COMPLETE"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
