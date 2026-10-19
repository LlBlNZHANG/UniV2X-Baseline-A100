#!/usr/bin/env python3

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

TRAIN_FILE = (
    BASE
    / "V2-4B_train_learnable_fusion_dataset.npz"
)

VAL_FILE = (
    BASE
    / "V2-4B_val_learnable_fusion_dataset.npz"
)

CKPT_DIR = (
    BASE
    / "checkpoints/V2-4C_main"
)

RESULT_DIR = (
    BASE
    / "results/V2-4C_main"
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

SUMMARY_FILE = (
    RESULT_DIR
    / "V2-4C_training_summary.json"
)

VAL_PRED_FILE = (
    RESULT_DIR
    / "V2-4C_val_gate_predictions.npz"
)


SEED = 2026

SEG_THRESH = 0.1

# Smooth approximation of official hard threshold.
TEMPERATURE = 0.02

UTILITY_SCALE = 100.0

LAMBDA_BCE = 0.10

BATCH_SIZE = 1024
EPOCHS = 80
PATIENCE = 15

LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.10

EPS = 1e-6


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
# Load
# =============================================================================

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
# 129-D input
# =============================================================================

def build_input(
    data,
    extra_mean=None,
    extra_std=None,
):

    x127 = data[
        "features_127"
    ].astype(
        np.float32
    )

    assert x127.shape[1] == 127

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
# Gate model
# =============================================================================

class UtilityConditionedFusionGate(nn.Module):

    def __init__(
        self,
        dropout=0.1,
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                129,
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

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, x):

        gate_logit = (
            self.net(x)
            .squeeze(1)
        )

        gate = torch.sigmoid(
            gate_logit
        )

        return (
            gate,
            gate_logit,
        )


# =============================================================================
# Differentiable fusion
# =============================================================================

def fuse_probability(
    pv,
    pi,
    warp,
    gate,
):

    positive_delta = F.relu(
        pi - pv
    )

    positive_delta = (
        positive_delta
        *
        warp
    )

    pf = (
        pv
        +
        gate[
            :,
            None,
            None,
        ]
        *
        positive_delta
    )

    return pf


# =============================================================================
# Metric-aligned differentiable loss
# =============================================================================

def fusion_loss(
    gate,
    pv,
    pi,
    warp,
    gt,
    action_mask,
    ego_tp,
    ego_fp,
    ego_fn,
):

    pf = fuse_probability(
        pv,
        pi,
        warp,
        gate,
    )

    # -------------------------------------------------------------------------
    # Soft approximation of official threshold.
    # -------------------------------------------------------------------------

    soft_occ_f = torch.sigmoid(
        (
            pf
            -
            SEG_THRESH
        )
        /
        TEMPERATURE
    )

    soft_occ_v = torch.sigmoid(
        (
            pv
            -
            SEG_THRESH
        )
        /
        TEMPERATURE
    )

    # -------------------------------------------------------------------------
    # Soft "new occupancy" activation.
    #
    # Normalize relative to Ego so:
    #
    #   gate=0 => add_prob = 0
    #
    # and gate increases continuously toward the g=1 state.
    # -------------------------------------------------------------------------

    add_prob = (
        (
            soft_occ_f
            -
            soft_occ_v
        )
        /
        (
            1.0
            -
            soft_occ_v
            +
            EPS
        )
    )

    add_prob = torch.clamp(
        add_prob,
        min=0.0,
        max=1.0,
    )

    add_prob = (
        add_prob
        *
        action_mask
    )

    # -------------------------------------------------------------------------
    # Differentiable BA / HA.
    # -------------------------------------------------------------------------

    ba = (
        add_prob
        *
        gt
    ).sum(
        dim=(1, 2)
    )

    ha = (
        add_prob
        *
        (
            1.0 - gt
        )
    ).sum(
        dim=(1, 2)
    )

    d = (
        ego_tp
        +
        ego_fp
        +
        ego_fn
    )

    metric_valid = (
        d > 0.0
    )

    iou_before = (
        ego_tp
        /
        torch.clamp(
            d,
            min=1.0,
        )
    )

    iou_after = (
        ego_tp
        +
        ba
    ) / (
        d
        +
        ha
        +
        EPS
    )

    delta_iou = (
        iou_after
        -
        iou_before
    )

    if metric_valid.any():

        loss_utility = (
            -UTILITY_SCALE
            *
            delta_iou[
                metric_valid
            ].mean()
        )

    else:

        loss_utility = (
            gate.sum()
            *
            0.0
        )

    # -------------------------------------------------------------------------
    # Auxiliary threshold-aware BCE.
    #
    # Only cells that can actually become an infrastructure addition
    # under g=1 are supervised here.
    # -------------------------------------------------------------------------

    threshold_logits = (
        (
            pf
            -
            SEG_THRESH
        )
        /
        TEMPERATURE
    )

    mask = (
        action_mask
        >
        0.5
    )

    if mask.any():

        loss_bce = (
            F.binary_cross_entropy_with_logits(
                threshold_logits[
                    mask
                ],
                gt[
                    mask
                ],
            )
        )

    else:

        loss_bce = (
            gate.sum()
            *
            0.0
        )

    loss = (
        loss_utility
        +
        LAMBDA_BCE
        *
        loss_bce
    )

    return {
        "loss":
            loss,

        "utility":
            delta_iou,

        "loss_utility":
            loss_utility,

        "loss_bce":
            loss_bce,
    }


# =============================================================================
# Exact full-split IoU evaluator
# =============================================================================

def build_range30_mask(
    patch_r,
    patch_c,
):

    n = len(
        patch_r
    )

    mask = np.zeros(
        (
            n,
            4,
            4,
        ),
        dtype=bool,
    )

    for i, (
        r,
        c,
    ) in enumerate(
        zip(
            patch_r,
            patch_c,
        )
    ):

        y0 = (
            4 * int(r)
        )

        x0 = (
            4 * int(c)
        )

        for yy in range(4):

            for xx in range(4):

                gy = y0 + yy
                gx = x0 + xx

                if (
                    70
                    <= gy
                    < 130
                    and
                    70
                    <= gx
                    < 130
                ):

                    mask[
                        i,
                        yy,
                        xx,
                    ] = True

    return mask


def build_ego_baseline_counts(
    sample_ids,
):

    sample_ids = set(
        str(x)
        for x in sample_ids
    )

    cache_files = sorted(
        (
            ROOT
            /
            "G0-G1-G2/cache_full"
        ).glob(
            "sample_*.npz"
        )
    )

    counts = {
        "30m": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        },

        "100m": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        },
    }

    seen = set()

    for path in cache_files:

        with np.load(
            path,
            allow_pickle=False,
        ) as d:

            sid = str(
                d[
                    "sample_idx"
                ].item()
            )

            if sid not in sample_ids:
                continue

            ov = d[
                "Ov"
            ].astype(
                bool
            )

            gt = d[
                "GT"
            ].astype(
                bool
            )

            while (
                ov.ndim > 3
                and
                ov.shape[0] == 1
            ):
                ov = ov[0]

            while (
                gt.ndim > 3
                and
                gt.shape[0] == 1
            ):
                gt = gt[0]

            seen.add(sid)

        for name, (
            lo,
            hi,
        ) in {
            "30m": (
                70,
                130,
            ),

            "100m": (
                0,
                200,
            ),
        }.items():

            p = ov[
                :,
                lo:hi,
                lo:hi,
            ]

            g = gt[
                :,
                lo:hi,
                lo:hi,
            ]

            counts[
                name
            ][
                "tp"
            ] += int(
                (
                    p & g
                ).sum()
            )

            counts[
                name
            ][
                "fp"
            ] += int(
                (
                    p & (~g)
                ).sum()
            )

            counts[
                name
            ][
                "fn"
            ] += int(
                (
                    (~p) & g
                ).sum()
            )

    assert seen == sample_ids, (
        f"Missing samples: "
        f"{sample_ids - seen}"
    )

    return counts


def iou_from_counts(d):

    denom = (
        d["tp"]
        +
        d["fp"]
        +
        d["fn"]
    )

    return float(
        d["tp"]
        /
        denom
    )


def evaluate_gates(
    data,
    gates,
    ego_counts,
    range30_mask,
):

    gates = np.asarray(
        gates,
        dtype=np.float32,
    )

    assert gates.shape == (
        len(
            data[
                "Pv_patch"
            ]
        ),
    )

    pv = data[
        "Pv_patch"
    ].astype(
        np.float32
    )

    pi = data[
        "Pi_patch"
    ].astype(
        np.float32
    )

    ov = data[
        "Ov_patch"
    ].astype(
        bool
    )

    warp = data[
        "warp_patch"
    ].astype(
        bool
    )

    gt = data[
        "GT_patch"
    ].astype(
        bool
    )

    delta = np.maximum(
        pi
        -
        pv,
        0.0,
    )

    delta *= (
        warp.astype(
            np.float32
        )
    )

    pf = (
        pv
        +
        gates[
            :,
            None,
            None,
        ]
        *
        delta
    )

    pred = (
        pf
        >
        SEG_THRESH
    )

    add = (
        (~ov)
        &
        pred
        &
        warp
    )

    result = {}

    for name in [
        "30m",
        "100m",
    ]:

        if name == "30m":

            region = (
                range30_mask
            )

        else:

            region = np.ones_like(
                add,
                dtype=bool,
            )

        add_r = (
            add
            &
            region
        )

        ba = int(
            (
                add_r
                &
                gt
            ).sum()
        )

        ha = int(
            (
                add_r
                &
                (~gt)
            ).sum()
        )

        tp = (
            ego_counts[
                name
            ][
                "tp"
            ]
            +
            ba
        )

        fp = (
            ego_counts[
                name
            ][
                "fp"
            ]
            +
            ha
        )

        fn = (
            ego_counts[
                name
            ][
                "fn"
            ]
            -
            ba
        )

        result[
            name
        ] = {
            "tp":
                int(tp),

            "fp":
                int(fp),

            "fn":
                int(fn),

            "ba":
                int(ba),

            "ha":
                int(ha),

            "iou":
                float(
                    tp
                    /
                    (
                        tp
                        +
                        fp
                        +
                        fn
                    )
                ),
        }

    return result


# =============================================================================
# Prediction
# =============================================================================

@torch.no_grad()
def predict_gates(
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

        gate, _ = model(
            xb
        )

        out.append(
            gate
            .detach()
            .cpu()
            .numpy()
        )

    return np.concatenate(
        out
    ).astype(
        np.float32
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
        TRAIN_FILE
    )

    val = load_npz(
        VAL_FILE
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
        extra_mean=extra_mean,
        extra_std=extra_std,
    )

    # -------------------------------------------------------------------------
    # Exact Val evaluator preparation
    # -------------------------------------------------------------------------

    # Candidate-level Val dataset contains only samples that have at
    # least one candidate patch (historically 94/96). For full-split
    # occupancy evaluation we must restore ALL 96 valid Val samples,
    # including the two zero-candidate samples.
    val_candidate_sample_ids = np.unique(
        val[
            "sample_idx"
        ].astype(str)
    )

    val_scene_tokens = set(
        np.unique(
            val[
                "scene_token"
            ].astype(str)
        ).tolist()
    )

    assert len(
        val_scene_tokens
    ) == 4

    valid_file = (
        ROOT
        / "G0-G1-G2/G3/cache_valid_samples.txt"
    )

    val_sample_ids = []

    with open(
        valid_file,
        "r",
    ) as f:

        _ = next(f, None)

        for line in f:

            parts = (
                line.strip()
                .split("\t")
            )

            if len(parts) < 3:
                continue

            sample_idx = parts[0]
            scene_token = parts[1]

            if (
                scene_token
                in val_scene_tokens
            ):
                val_sample_ids.append(
                    sample_idx
                )

    val_sample_ids = np.asarray(
        val_sample_ids
    )

    assert len(
        np.unique(
            val_sample_ids
        )
    ) == 96, (
        f"Expected 96 full Val samples, "
        f"got {len(np.unique(val_sample_ids))}"
    )

    assert len(
        val_candidate_sample_ids
    ) == 94, (
        f"Expected 94 Val samples with candidates, "
        f"got {len(val_candidate_sample_ids)}"
    )

    print(
        "Val samples full        :",
        len(
            np.unique(
                val_sample_ids
            )
        ),
    )

    print(
        "Val samples w/candidate :",
        len(
            val_candidate_sample_ids
        ),
    )

    print(
        "Val zero-candidate      :",
        96
        -
        len(
            val_candidate_sample_ids
        ),
    )

    ego_counts = (
        build_ego_baseline_counts(
            val_sample_ids
        )
    )

    range30_mask = (
        build_range30_mask(
            val[
                "patch_r"
            ],
            val[
                "patch_c"
            ],
        )
    )

    ego30 = iou_from_counts(
        ego_counts[
            "30m"
        ]
    )

    ego100 = iou_from_counts(
        ego_counts[
            "100m"
        ]
    )

    print("=" * 110)
    print(
        "V2-4C UTILITY-CONDITIONED LEARNABLE FUSION GATE"
    )
    print("=" * 110)

    print(
        "Train candidates        :",
        len(x_train),
    )

    print(
        "Val candidates          :",
        len(x_val),
    )

    print(
        "Input dim               :",
        129,
    )

    print(
        "Fusion                  :",
        "Pf = Pv + g * relu(Pi-Pv)",
    )

    print(
        "Official threshold      :",
        SEG_THRESH,
    )

    print(
        "Smooth temperature      :",
        TEMPERATURE,
    )

    print(
        "Loss                    :",
        "-100*soft_delta_IoU100 + "
        "0.1*threshold_BCE",
    )

    print(
        "Checkpoint criterion    :",
        "Val 100m hard IoU",
    )

    print(
        "Oracle gate used in loss:",
        "NO",
    )

    print(
        "Test used               :",
        "NO",
    )

    print()
    print(
        "V2-extra mean           :",
        extra_mean.reshape(-1),
    )

    print(
        "V2-extra std            :",
        extra_std.reshape(-1),
    )

    print()
    print(
        "Val Ego 30m             :",
        f"{ego30:.6f}",
    )

    print(
        "Val Ego 100m            :",
        f"{ego100:.6f}",
    )

    # -------------------------------------------------------------------------
    # Frozen reference audit
    # -------------------------------------------------------------------------

    reference_gates = {
        "Official_g1":
            np.ones(
                len(x_val),
                dtype=np.float32,
            ),

        "Fixed_g025":
            np.full(
                len(x_val),
                0.25,
                dtype=np.float32,
            ),

        "Fixed_g050":
            np.full(
                len(x_val),
                0.50,
                dtype=np.float32,
            ),

        "Fixed_g075":
            np.full(
                len(x_val),
                0.75,
                dtype=np.float32,
            ),

        "V2_Hard":
            val[
                "v2_accept"
            ].astype(
                np.float32
            ),

        "HardOracle":
            val[
                "oracle_hard_best_gate"
            ].astype(
                np.float32
            ),

        "SoftOracle":
            val[
                "oracle_soft_best_gate"
            ].astype(
                np.float32
            ),
    }

    print()
    print("=" * 110)
    print(
        "VAL REFERENCE AUDIT"
    )
    print("=" * 110)

    reference_results = {}

    for name, gates in (
        reference_gates.items()
    ):

        result = evaluate_gates(
            val,
            gates,
            ego_counts,
            range30_mask,
        )

        reference_results[
            name
        ] = result

        print(
            f"{name:14s} "
            f"30m={result['30m']['iou']:.6f} "
            f"100m={result['100m']['iou']:.6f} "
            f"gate_mean={gates.mean():.4f}"
        )

    assert abs(
        reference_results[
            "V2_Hard"
        ][
            "100m"
        ][
            "iou"
        ]
        -
        0.360369
    ) < 2e-6

    assert abs(
        reference_results[
            "HardOracle"
        ][
            "100m"
        ][
            "iou"
        ]
        -
        0.403358
    ) < 2e-6

    assert abs(
        reference_results[
            "SoftOracle"
        ][
            "100m"
        ][
            "iou"
        ]
        -
        0.410608
    ) < 2e-6

    print(
        "Reference reproduction  : PASS"
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    dataset = TensorDataset(

        torch.from_numpy(
            x_train
        ),

        torch.from_numpy(
            train[
                "Pv_patch"
            ].astype(
                np.float32
            )
        ),

        torch.from_numpy(
            train[
                "Pi_patch"
            ].astype(
                np.float32
            )
        ),

        torch.from_numpy(
            train[
                "warp_patch"
            ].astype(
                np.float32
            )
        ),

        torch.from_numpy(
            train[
                "GT_patch"
            ].astype(
                np.float32
            )
        ),

        torch.from_numpy(
            train[
                "candidate_add_patch"
            ].astype(
                np.float32
            )
        ),

        torch.from_numpy(
            train[
                "ego_tp_100"
            ].astype(
                np.float32
            )
        ),

        torch.from_numpy(
            train[
                "ego_fp_100"
            ].astype(
                np.float32
            )
        ),

        torch.from_numpy(
            train[
                "ego_fn_100"
            ].astype(
                np.float32
            )
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

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = (
        UtilityConditionedFusionGate(
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

    print()
    print(
        "Gate parameters         :",
        n_params,
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

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
        util_sum = 0.0
        bce_sum = 0.0
        utility_mean_sum = 0.0

        gate_sum = 0.0
        n_seen = 0

        for batch in loader:

            (
                xb,
                pv,
                pi,
                warp,
                gt,
                action,
                tp,
                fp,
                fn,
            ) = batch

            xb = xb.to(
                device,
                non_blocking=True,
            )

            pv = pv.to(
                device,
                non_blocking=True,
            )

            pi = pi.to(
                device,
                non_blocking=True,
            )

            warp = warp.to(
                device,
                non_blocking=True,
            )

            gt = gt.to(
                device,
                non_blocking=True,
            )

            action = action.to(
                device,
                non_blocking=True,
            )

            tp = tp.to(
                device,
                non_blocking=True,
            )

            fp = fp.to(
                device,
                non_blocking=True,
            )

            fn = fn.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            gate, _ = model(
                xb
            )

            loss_out = fusion_loss(
                gate,
                pv,
                pi,
                warp,
                gt,
                action,
                tp,
                fp,
                fn,
            )

            loss = (
                loss_out[
                    "loss"
                ]
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

            util_sum += (
                float(
                    loss_out[
                        "loss_utility"
                    ].item()
                )
                *
                bs
            )

            bce_sum += (
                float(
                    loss_out[
                        "loss_bce"
                    ].item()
                )
                *
                bs
            )

            utility_mean_sum += (
                float(
                    loss_out[
                        "utility"
                    ]
                    .mean()
                    .item()
                )
                *
                bs
            )

            gate_sum += (
                float(
                    gate.sum()
                    .item()
                )
            )

            n_seen += bs

        # ---------------------------------------------------------------------
        # Exact full-Val hard IoU
        # ---------------------------------------------------------------------

        val_gates = predict_gates(
            model,
            x_val,
            device,
        )

        val_result = (
            evaluate_gates(
                val,
                val_gates,
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

        gate_mean = float(
            val_gates.mean()
        )

        gate_std = float(
            val_gates.std()
        )

        interior_rate = float(
            (
                (val_gates > 0.05)
                &
                (val_gates < 0.95)
            ).mean()
        )

        row = {
            "epoch":
                int(epoch),

            "train_loss":
                float(
                    total_sum
                    /
                    n_seen
                ),

            "train_loss_utility":
                float(
                    util_sum
                    /
                    n_seen
                ),

            "train_loss_bce":
                float(
                    bce_sum
                    /
                    n_seen
                ),

            "train_soft_delta_iou":
                float(
                    utility_mean_sum
                    /
                    n_seen
                ),

            "train_gate_mean":
                float(
                    gate_sum
                    /
                    n_seen
                ),

            "val_iou30":
                float(iou30),

            "val_iou100":
                float(iou100),

            "val_gate_mean":
                gate_mean,

            "val_gate_std":
                gate_std,

            "val_gate_interior_rate":
                interior_rate,
        }

        history.append(
            row
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['train_loss']:+.6f} "
            f"(U={row['train_loss_utility']:+.6f}, "
            f"BCE={row['train_loss_bce']:.6f}) | "
            f"soft_dIoU={row['train_soft_delta_iou']:+.7f} | "
            f"Val IoU30={iou30:.6f} "
            f"IoU100={iou100:.6f} | "
            f"g={gate_mean:.4f}±{gate_std:.4f} "
            f"interior={interior_rate:.4f}"
        )

        # Main criterion = official 100m IoU.
        improved = (
            iou100
            >
            best_iou100
            +
            1e-10
        )

        # Deterministic tie break = 30m IoU.
        tied = abs(
            iou100
            -
            best_iou100
        ) <= 1e-10

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

                    "input_dim":
                        129,

                    "architecture":
                        "129-128-64-1 sigmoid gate",

                    "seg_threshold":
                        SEG_THRESH,

                    "temperature":
                        TEMPERATURE,

                    "utility_scale":
                        UTILITY_SCALE,

                    "lambda_bce":
                        LAMBDA_BCE,

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
            >= PATIENCE
        ):

            print()
            print(
                f"Early stopping at "
                f"epoch {epoch}; "
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
        ]
    )

    best_gates = predict_gates(
        model,
        x_val,
        device,
    )

    best_result = evaluate_gates(
        val,
        best_gates,
        ego_counts,
        range30_mask,
    )

    # =========================================================================
    # Gate diagnostic
    # =========================================================================

    q = np.quantile(
        best_gates,
        [
            0.0,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            1.0,
        ],
    )

    interior = (
        (best_gates > 0.05)
        &
        (best_gates < 0.95)
    )

    # Oracle is ANALYSIS ONLY here.
    oracle_strict = (
        val[
            "oracle_strict_interior"
        ].astype(bool)
    )

    summary = {
        "stage":
            "V2-4C Utility-conditioned Learnable Fusion Gate",

        "best_epoch":
            int(
                ckpt["epoch"]
            ),

        "val": {
            "iou30":
                float(
                    best_result[
                        "30m"
                    ][
                        "iou"
                    ]
                ),

            "iou100":
                float(
                    best_result[
                        "100m"
                    ][
                        "iou"
                    ]
                ),

            "gate_mean":
                float(
                    best_gates.mean()
                ),

            "gate_std":
                float(
                    best_gates.std()
                ),

            "gate_interior_rate":
                float(
                    interior.mean()
                ),

            "gate_quantiles":
                q.tolist(),

            "strict_oracle_candidates":
                int(
                    oracle_strict.sum()
                ),

            "learned_interior_on_strict_oracle":
                int(
                    (
                        interior
                        &
                        oracle_strict
                    ).sum()
                ),
        },

        "references":
            {
                name: {
                    "iou30":
                        float(
                            d[
                                "30m"
                            ][
                                "iou"
                            ]
                        ),

                    "iou100":
                        float(
                            d[
                                "100m"
                            ][
                                "iou"
                            ]
                        ),
                }
                for name, d in
                reference_results.items()
            },

        "oracle_gate_used_for_training":
            False,

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

    np.savez_compressed(
        VAL_PRED_FILE,

        gate=best_gates,

        sample_idx=val[
            "sample_idx"
        ],

        horizon=val[
            "horizon"
        ],

        patch_r=val[
            "patch_r"
        ],

        patch_c=val[
            "patch_c"
        ],

        v2_mag=val[
            "v2_mag"
        ],

        v2_sign_prob=val[
            "v2_sign_prob"
        ],

        oracle_soft_best_gate=val[
            "oracle_soft_best_gate"
        ],

        oracle_strict_interior=val[
            "oracle_strict_interior"
        ],
    )

    # =========================================================================
    # Print
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "V2-4C FINAL VALIDATION"
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
        f"30m="
        f"{reference_results['V2_Hard']['30m']['iou']:.6f} "
        f"100m="
        f"{reference_results['V2_Hard']['100m']['iou']:.6f}",
    )

    print(
        "Hard Oracle            :",
        f"30m="
        f"{reference_results['HardOracle']['30m']['iou']:.6f} "
        f"100m="
        f"{reference_results['HardOracle']['100m']['iou']:.6f}",
    )

    print(
        "Soft Oracle            :",
        f"30m="
        f"{reference_results['SoftOracle']['30m']['iou']:.6f} "
        f"100m="
        f"{reference_results['SoftOracle']['100m']['iou']:.6f}",
    )

    print(
        "Learned Gate           :",
        f"30m="
        f"{best_result['30m']['iou']:.6f} "
        f"100m="
        f"{best_result['100m']['iou']:.6f}",
    )

    print()
    print(
        "Delta vs V2 Hard 30m   :",
        f"{best_result['30m']['iou'] - reference_results['V2_Hard']['30m']['iou']:+.8f}",
    )

    print(
        "Delta vs V2 Hard 100m  :",
        f"{best_result['100m']['iou'] - reference_results['V2_Hard']['100m']['iou']:+.8f}",
    )

    print()
    print(
        "Gate mean/std           :",
        f"{best_gates.mean():.6f} / "
        f"{best_gates.std():.6f}",
    )

    print(
        "Gate interior rate      :",
        f"{interior.mean():.6%}",
    )

    print(
        "Gate quantiles          :",
        q,
    )

    print(
        "Strict Oracle candidates:",
        int(
            oracle_strict.sum()
        ),
    )

    print(
        "Learned interior there :",
        int(
            (
                interior
                &
                oracle_strict
            ).sum()
        ),
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
        VAL_PRED_FILE,
    )

    print()
    print(
        "[V2-4C LEARNABLE FUSION GATE] COMPLETE"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
