#!/usr/bin/env python3

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# =============================================================================
# Paths / constants
# =============================================================================

ROOT = Path("/root/autodl-tmp/UniV2X")

CACHE_DIR = (
    ROOT
    / "G0-G1-G2/cache_full"
)

VALID_FILE = (
    ROOT
    / "G0-G1-G2/G3/cache_valid_samples.txt"
)

OUT_DIR = (
    ROOT
    / "V2-4_learnable_adaptive_fusion"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 2026

N_HORIZONS = 5
GRID_SIZE = 200
PATCH_SIZE = 4
N_PATCH = GRID_SIZE // PATCH_SIZE

SEG_THRESH = 0.1

# First feasibility grid.
GATES = np.asarray(
    [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
    dtype=np.float64,
)

EPS = 1e-12

# Exact current UniV2X evaluator ranges.
EVAL_RANGES = {
    "30m": (70, 130),
    "100m": (0, 200),
}


# =============================================================================
# Helpers
# =============================================================================

def norm(x, ndim):

    x = np.asarray(x)

    while (
        x.ndim > ndim
        and x.shape[0] == 1
    ):
        x = x[0]

    assert x.ndim == ndim, (
        f"Expected ndim={ndim}, got {x.shape}"
    )

    return x


def load_exact_split():

    scene_to_samples = defaultdict(list)
    valid_sample_ids = set()

    with open(
        VALID_FILE,
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

            valid_sample_ids.add(
                sample_idx
            )

            scene_to_samples[
                scene_token
            ].append(
                sample_idx
            )

    scenes = sorted(
        scene_to_samples.keys()
    )

    assert len(scenes) == 21

    np.random.seed(SEED)
    np.random.shuffle(scenes)

    train_scenes = set(
        scenes[:13]
    )

    val_scenes = set(
        scenes[13:17]
    )

    test_scenes = set(
        scenes[17:21]
    )

    return (
        valid_sample_ids,
        train_scenes,
        val_scenes,
        test_scenes,
    )


def binary_counts(
    pred,
    gt,
):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = int(
        np.logical_and(
            pred,
            gt,
        ).sum()
    )

    fp = int(
        np.logical_and(
            pred,
            np.logical_not(gt),
        ).sum()
    )

    fn = int(
        np.logical_and(
            np.logical_not(pred),
            gt,
        ).sum()
    )

    return (
        tp,
        fp,
        fn,
    )


def update_occ_acc(
    acc,
    pred,
    gt,
):

    for name, (
        start,
        end,
    ) in EVAL_RANGES.items():

        p = pred[
            :,
            start:end,
            start:end,
        ]

        g = gt[
            :,
            start:end,
            start:end,
        ]

        tp, fp, fn = binary_counts(
            p,
            g,
        )

        acc[name]["tp"] += tp
        acc[name]["fp"] += fp
        acc[name]["fn"] += fn


def make_occ_acc():

    return {
        name: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for name in EVAL_RANGES
    }


def iou_from_acc(
    d,
):

    denom = (
        d["tp"]
        +
        d["fp"]
        +
        d["fn"]
    )

    if denom <= 0:
        return 0.0

    return float(
        d["tp"]
        /
        denom
    )


def delta_iou_from_add(
    tp,
    fp,
    fn,
    ba,
    ha,
):
    """
    Exact V2 metric-aligned isolated marginal utility:

        IoU_before = TP / D
        IoU_after  = (TP + BA) / (D + HA)

    because ego-preserving fusion only adds occupied cells.

    For D=0 we return 0, matching the fact that this horizon
    cannot define a stable isolated occupied-class IoU delta.
    Val has zero such horizons in the frozen split.
    """

    d = (
        tp
        +
        fp
        +
        fn
    )

    if d <= 0:
        return 0.0

    before = (
        tp
        /
        d
    )

    after = (
        (tp + ba)
        /
        (d + ha)
    )

    return float(
        after
        -
        before
    )


def soft_patch_prediction(
    pv_patch,
    pi_patch,
    warp_patch,
    gate,
):
    """
    Ego-preserving continuous probability fusion:

        Pf = Pv + g * relu(Pi - Pv)

    Infra contribution is restricted to warp-valid cells.

    g=0:
        Pf = Pv

    g=1:
        Pf = max(Pv, Pi) on valid cells

    Final occupancy uses the official 0.1 threshold.
    """

    positive_infra_delta = np.maximum(
        pi_patch
        -
        pv_patch,
        0.0,
    )

    positive_infra_delta = (
        positive_infra_delta
        *
        warp_patch.astype(
            np.float32
        )
    )

    pf = (
        pv_patch
        +
        float(gate)
        *
        positive_infra_delta
    )

    return (
        pf
        >
        SEG_THRESH
    )


def candidate_gate_utilities(
    pv_patch,
    pi_patch,
    ov_patch,
    gt_patch,
    warp_patch,
    tp,
    fp,
    fn,
):

    utilities = []

    preds = []

    for gate in GATES:

        pred_patch = (
            soft_patch_prediction(
                pv_patch,
                pi_patch,
                warp_patch,
                gate,
            )
        )

        # Only newly added occupied cells matter.
        add = (
            np.logical_not(
                ov_patch
            )
            &
            pred_patch
            &
            warp_patch
        )

        ba = int(
            np.logical_and(
                add,
                gt_patch,
            ).sum()
        )

        ha = int(
            np.logical_and(
                add,
                np.logical_not(
                    gt_patch
                ),
            ).sum()
        )

        u = delta_iou_from_add(
            tp,
            fp,
            fn,
            ba,
            ha,
        )

        utilities.append(u)
        preds.append(pred_patch)

    return (
        np.asarray(
            utilities,
            dtype=np.float64,
        ),
        preds,
    )


def choose_soft_gate(
    utilities,
):
    """
    Conservative tie policy:

    An interior gate is selected ONLY if it strictly exceeds
    both hard endpoints.

    This prevents a tied g=0.4/g=1.0 binary result from being
    incorrectly counted as continuous-fusion headroom.
    """

    u0 = float(
        utilities[0]
    )

    u1 = float(
        utilities[-1]
    )

    hard_best = max(
        u0,
        u1,
    )

    interior = utilities[
        1:-1
    ]

    interior_idx_local = int(
        np.argmax(
            interior
        )
    )

    interior_best = float(
        interior[
            interior_idx_local
        ]
    )

    strict_interior = (
        interior_best
        >
        hard_best
        +
        EPS
    )

    if strict_interior:

        idx = (
            interior_idx_local
            +
            1
        )

        return (
            idx,
            True,
            interior_best
            -
            hard_best,
        )

    # Otherwise use the best hard endpoint.
    # Tie -> Ego / g=0 for conservative behavior.
    if (
        u1
        >
        u0
        +
        EPS
    ):

        return (
            len(GATES) - 1,
            False,
            0.0,
        )

    return (
        0,
        False,
        0.0,
    )


def process_split(
    split_name,
    split_scenes,
    valid_sample_ids,
):

    cache_files = sorted(
        CACHE_DIR.glob(
            "sample_*.npz"
        )
    )

    expected_samples = {
        "train": 361,
        "val": 96,
    }[
        split_name
    ]

    expected_candidates = {
        "train": 86143,
        "val": 18555,
    }[
        split_name
    ]

    strategy_acc = {
        "Ego":
            make_occ_acc(),

        "OfficialOR":
            make_occ_acc(),

        "HardOracle":
            make_occ_acc(),

        "SoftOracle":
            make_occ_acc(),
    }

    n_samples = 0
    n_horizons = 0
    n_zero_denom = 0
    n_candidates = 0

    strict_interior_count = 0

    strict_gain = []

    gate_hist = Counter()

    # Candidate-level saved arrays.
    out_sample_idx = []
    out_scene_token = []
    out_h = []
    out_r = []
    out_c = []

    out_u0 = []
    out_u1 = []

    out_best_gate = []
    out_best_utility = []

    out_hard_gate = []
    out_hard_utility = []

    out_strict_interior = []
    out_strict_gain = []

    print()
    print("=" * 110)
    print(
        f"{split_name.upper()} SPLIT"
    )
    print("=" * 110)

    for path in cache_files:

        with np.load(
            path,
            allow_pickle=False,
        ) as d:

            sample_idx = str(
                d[
                    "sample_idx"
                ].item()
            )

            scene_token = str(
                d[
                    "scene_token"
                ].item()
            )

            if (
                sample_idx
                not in
                valid_sample_ids
            ):
                continue

            if (
                scene_token
                not in
                split_scenes
            ):
                continue

            pv = norm(
                d["Pv"],
                3,
            ).astype(
                np.float32
            )

            pi = norm(
                d[
                    "Pi_aligned"
                ],
                3,
            ).astype(
                np.float32
            )

            ov = norm(
                d["Ov"],
                3,
            ).astype(
                bool
            )

            oi = norm(
                d["Oi"],
                3,
            ).astype(
                bool
            )

            official = norm(
                d["Oofficial"],
                3,
            ).astype(
                bool
            )

            gt = norm(
                d["GT"],
                3,
            ).astype(
                bool
            )

            warp2d = norm(
                d[
                    "warp_valid_mask"
                ],
                2,
            ).astype(
                bool
            )

        assert pv.shape == (
            5,
            200,
            200,
        )

        assert pi.shape == pv.shape
        assert ov.shape == pv.shape
        assert oi.shape == pv.shape
        assert gt.shape == pv.shape

        hard_oracle = (
            ov.copy()
        )

        soft_oracle = (
            ov.copy()
        )

        sample_candidates = 0

        for h in range(
            N_HORIZONS
        ):

            # Exact sample/horizon 100m Ego state,
            # matching the V2 metric-aligned target.
            tp, fp, fn = binary_counts(
                ov[h],
                gt[h],
            )

            if (
                tp + fp + fn
                ==
                0
            ):
                n_zero_denom += 1

            n_horizons += 1

            for r in range(
                N_PATCH
            ):

                y0 = (
                    PATCH_SIZE
                    *
                    r
                )

                y1 = (
                    y0
                    +
                    PATCH_SIZE
                )

                for c in range(
                    N_PATCH
                ):

                    x0 = (
                        PATCH_SIZE
                        *
                        c
                    )

                    x1 = (
                        x0
                        +
                        PATCH_SIZE
                    )

                    ov_patch = ov[
                        h,
                        y0:y1,
                        x0:x1,
                    ]

                    oi_patch = oi[
                        h,
                        y0:y1,
                        x0:x1,
                    ]

                    warp_patch = (
                        warp2d[
                            y0:y1,
                            x0:x1,
                        ]
                    )

                    # Exact V1/V2 P4 candidate definition.
                    add_candidate = (
                        np.logical_not(
                            ov_patch
                        )
                        &
                        oi_patch
                        &
                        warp_patch
                    )

                    if not bool(
                        add_candidate.any()
                    ):
                        continue

                    pv_patch = pv[
                        h,
                        y0:y1,
                        x0:x1,
                    ]

                    pi_patch = pi[
                        h,
                        y0:y1,
                        x0:x1,
                    ]

                    gt_patch = gt[
                        h,
                        y0:y1,
                        x0:x1,
                    ]

                    (
                        utilities,
                        preds,
                    ) = candidate_gate_utilities(
                        pv_patch,
                        pi_patch,
                        ov_patch,
                        gt_patch,
                        warp_patch,
                        tp,
                        fp,
                        fn,
                    )

                    # Hard oracle: g ∈ {0,1}
                    if (
                        utilities[-1]
                        >
                        utilities[0]
                        +
                        EPS
                    ):

                        hard_idx = (
                            len(GATES)
                            -
                            1
                        )

                    else:

                        hard_idx = 0

                    (
                        soft_idx,
                        strict_interior,
                        gain,
                    ) = choose_soft_gate(
                        utilities
                    )

                    hard_gate = float(
                        GATES[
                            hard_idx
                        ]
                    )

                    soft_gate = float(
                        GATES[
                            soft_idx
                        ]
                    )

                    # Apply complete patch prediction.
                    hard_oracle[
                        h,
                        y0:y1,
                        x0:x1,
                    ] = preds[
                        hard_idx
                    ]

                    soft_oracle[
                        h,
                        y0:y1,
                        x0:x1,
                    ] = preds[
                        soft_idx
                    ]

                    n_candidates += 1
                    sample_candidates += 1

                    gate_hist[
                        f"{soft_gate:.1f}"
                    ] += 1

                    if strict_interior:

                        strict_interior_count += 1

                        strict_gain.append(
                            float(gain)
                        )

                    out_sample_idx.append(
                        sample_idx
                    )

                    out_scene_token.append(
                        scene_token
                    )

                    out_h.append(h)
                    out_r.append(r)
                    out_c.append(c)

                    out_u0.append(
                        float(
                            utilities[0]
                        )
                    )

                    out_u1.append(
                        float(
                            utilities[-1]
                        )
                    )

                    out_best_gate.append(
                        soft_gate
                    )

                    out_best_utility.append(
                        float(
                            utilities[
                                soft_idx
                            ]
                        )
                    )

                    out_hard_gate.append(
                        hard_gate
                    )

                    out_hard_utility.append(
                        float(
                            utilities[
                                hard_idx
                            ]
                        )
                    )

                    out_strict_interior.append(
                        int(
                            strict_interior
                        )
                    )

                    out_strict_gain.append(
                        float(gain)
                    )

        update_occ_acc(
            strategy_acc[
                "Ego"
            ],
            ov,
            gt,
        )

        update_occ_acc(
            strategy_acc[
                "OfficialOR"
            ],
            official,
            gt,
        )

        update_occ_acc(
            strategy_acc[
                "HardOracle"
            ],
            hard_oracle,
            gt,
        )

        update_occ_acc(
            strategy_acc[
                "SoftOracle"
            ],
            soft_oracle,
            gt,
        )

        n_samples += 1

        if (
            n_samples <= 3
            or
            n_samples % 25 == 0
            or
            n_samples
            ==
            expected_samples
        ):

            print(
                f"[{n_samples:03d}/"
                f"{expected_samples:03d}] "
                f"{path.name} "
                f"sample={sample_idx} "
                f"cand={sample_candidates} "
                f"running="
                f"{n_candidates}"
            )

    # -------------------------------------------------------------------------
    # Exact reconstruction audits
    # -------------------------------------------------------------------------

    assert (
        n_samples
        ==
        expected_samples
    ), (
        f"{split_name}: "
        f"sample count mismatch "
        f"{n_samples} != "
        f"{expected_samples}"
    )

    assert (
        n_candidates
        ==
        expected_candidates
    ), (
        f"{split_name}: "
        f"candidate count mismatch "
        f"{n_candidates} != "
        f"{expected_candidates}"
    )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    strategy_result = {}

    for strategy, acc in (
        strategy_acc.items()
    ):

        strategy_result[
            strategy
        ] = {}

        for range_name in (
            EVAL_RANGES
        ):

            strategy_result[
                strategy
            ][
                range_name
            ] = (
                iou_from_acc(
                    acc[
                        range_name
                    ]
                )
            )

    strict_rate = (
        strict_interior_count
        /
        n_candidates
    )

    if len(
        strict_gain
    ) > 0:

        strict_gain_arr = np.asarray(
            strict_gain,
            dtype=np.float64,
        )

        gain_stats = {
            "mean":
                float(
                    strict_gain_arr.mean()
                ),

            "median":
                float(
                    np.median(
                        strict_gain_arr
                    )
                ),

            "p90":
                float(
                    np.quantile(
                        strict_gain_arr,
                        0.90,
                    )
                ),

            "max":
                float(
                    strict_gain_arr.max()
                ),
        }

    else:

        gain_stats = {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }

    hard100 = (
        strategy_result[
            "HardOracle"
        ][
            "100m"
        ]
    )

    soft100 = (
        strategy_result[
            "SoftOracle"
        ][
            "100m"
        ]
    )

    hard30 = (
        strategy_result[
            "HardOracle"
        ][
            "30m"
        ]
    )

    soft30 = (
        strategy_result[
            "SoftOracle"
        ][
            "30m"
        ]
    )

    print()
    print("-" * 110)
    print(
        "RECONSTRUCTION AUDIT"
    )
    print("-" * 110)

    print(
        "Samples                 :",
        n_samples,
    )

    print(
        "Horizons                :",
        n_horizons,
    )

    print(
        "Zero IoU denominators   :",
        n_zero_denom,
    )

    print(
        "Candidate patches       :",
        n_candidates,
    )

    print(
        "Candidate ordering/count: PASS"
    )

    print()
    print("-" * 110)
    print(
        "SOFT-GATE ORACLE DIAGNOSTIC"
    )
    print("-" * 110)

    print(
        "Strict interior better  :",
        strict_interior_count,
        f"({strict_rate:.6%})",
    )

    print(
        "Interior gain mean      :",
        f"{gain_stats['mean']:+.10f}",
    )

    print(
        "Interior gain median    :",
        f"{gain_stats['median']:+.10f}",
    )

    print(
        "Interior gain p90       :",
        f"{gain_stats['p90']:+.10f}",
    )

    print(
        "Interior gain max       :",
        f"{gain_stats['max']:+.10f}",
    )

    print()
    print(
        "Selected gate histogram:"
    )

    for gate in GATES:

        key = f"{float(gate):.1f}"

        count = int(
            gate_hist.get(
                key,
                0,
            )
        )

        print(
            f"  g={key}: "
            f"{count:7d} "
            f"({count/n_candidates:.6%})"
        )

    print()
    print("-" * 110)
    print(
        "FULL-SPLIT OCCUPANCY IOU"
    )
    print("-" * 110)

    print(
        f"{'Strategy':16s} "
        f"{'30m IoU':>12s} "
        f"{'100m IoU':>12s}"
    )

    print("-" * 44)

    for strategy in [
        "Ego",
        "OfficialOR",
        "HardOracle",
        "SoftOracle",
    ]:

        print(
            f"{strategy:16s} "
            f"{strategy_result[strategy]['30m']:12.6f} "
            f"{strategy_result[strategy]['100m']:12.6f}"
        )

    print()
    print(
        "Soft - Hard 30m        :",
        f"{soft30-hard30:+.8f}",
    )

    print(
        "Soft - Hard 100m       :",
        f"{soft100-hard100:+.8f}",
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    npz_path = (
        OUT_DIR
        /
        f"V3-0A_{split_name}_soft_oracle_candidates.npz"
    )

    np.savez_compressed(
        npz_path,

        sample_idx=np.asarray(
            out_sample_idx
        ),

        scene_token=np.asarray(
            out_scene_token
        ),

        horizon=np.asarray(
            out_h,
            dtype=np.int8,
        ),

        patch_r=np.asarray(
            out_r,
            dtype=np.int16,
        ),

        patch_c=np.asarray(
            out_c,
            dtype=np.int16,
        ),

        u_g0=np.asarray(
            out_u0,
            dtype=np.float32,
        ),

        u_g1=np.asarray(
            out_u1,
            dtype=np.float32,
        ),

        hard_best_gate=np.asarray(
            out_hard_gate,
            dtype=np.float32,
        ),

        hard_best_utility=np.asarray(
            out_hard_utility,
            dtype=np.float32,
        ),

        soft_best_gate=np.asarray(
            out_best_gate,
            dtype=np.float32,
        ),

        soft_best_utility=np.asarray(
            out_best_utility,
            dtype=np.float32,
        ),

        strict_interior=np.asarray(
            out_strict_interior,
            dtype=np.uint8,
        ),

        strict_gain=np.asarray(
            out_strict_gain,
            dtype=np.float32,
        ),

        gates=GATES.astype(
            np.float32
        ),
    )

    result = {
        "split":
            split_name,

        "samples":
            int(
                n_samples
            ),

        "horizons":
            int(
                n_horizons
            ),

        "zero_iou_denominators":
            int(
                n_zero_denom
            ),

        "candidates":
            int(
                n_candidates
            ),

        "seg_threshold":
            float(
                SEG_THRESH
            ),

        "gates":
            GATES.tolist(),

        "strict_interior_count":
            int(
                strict_interior_count
            ),

        "strict_interior_rate":
            float(
                strict_rate
            ),

        "strict_gain":
            gain_stats,

        "gate_histogram":
            {
                str(k):
                    int(v)
                for k, v in
                sorted(
                    gate_hist.items()
                )
            },

        "iou":
            strategy_result,

        "soft_minus_hard": {
            "30m":
                float(
                    soft30
                    -
                    hard30
                ),

            "100m":
                float(
                    soft100
                    -
                    hard100
                ),
        },

        "test_used":
            False,
    }

    json_path = (
        OUT_DIR
        /
        f"V3-0A_{split_name}_soft_oracle_summary.json"
    )

    with open(
        json_path,
        "w",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        "Saved NPZ :",
        npz_path,
    )

    print(
        "Saved JSON:",
        json_path,
    )

    return result


# =============================================================================
# Main
# =============================================================================

def main():

    (
        valid_sample_ids,
        train_scenes,
        val_scenes,
        test_scenes,
    ) = load_exact_split()

    print("=" * 110)
    print(
        "V3-0A SOFT FUSION ORACLE HEADROOM AUDIT"
    )
    print("=" * 110)

    print(
        "Valid samples          :",
        len(
            valid_sample_ids
        ),
    )

    print(
        "Train scenes           :",
        len(
            train_scenes
        ),
    )

    print(
        "Val scenes             :",
        len(
            val_scenes
        ),
    )

    print(
        "Test scenes            :",
        len(
            test_scenes
        ),
    )

    print(
        "Test used              : NO"
    )

    print(
        "Candidate patch        : P4 (4x4)"
    )

    print(
        "Official threshold     :",
        SEG_THRESH,
    )

    print(
        "Gate grid              :",
        GATES.tolist(),
    )

    print()
    print(
        "Fusion:"
    )

    print(
        "Pf = Pv + g * relu(Pi - Pv)"
    )

    print(
        "with infra contribution restricted "
        "to warp-valid cells."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Interior g is counted as useful ONLY "
        "when it strictly beats both g=0 and g=1."
    )

    train_result = process_split(
        "train",
        train_scenes,
        valid_sample_ids,
    )

    val_result = process_split(
        "val",
        val_scenes,
        valid_sample_ids,
    )

    # -------------------------------------------------------------------------
    # Frozen Val references from V2-1C.
    # This is an audit, not a hard assertion, because this script
    # uses the new continuous probability formulation.
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "VAL REFERENCE CHECK"
    )
    print("=" * 110)

    refs = {
        "Ego_30m":
            0.315315,

        "Ego_100m":
            0.336759,

        "OfficialOR_30m":
            0.229482,

        "OfficialOR_100m":
            0.330526,

        "V2HardOracle_30m":
            0.330727,

        "V2HardOracle_100m":
            0.403358,
    }

    checks = [
        (
            "Ego 30m",
            val_result[
                "iou"
            ][
                "Ego"
            ][
                "30m"
            ],
            refs[
                "Ego_30m"
            ],
        ),

        (
            "Ego 100m",
            val_result[
                "iou"
            ][
                "Ego"
            ][
                "100m"
            ],
            refs[
                "Ego_100m"
            ],
        ),

        (
            "OfficialOR 30m",
            val_result[
                "iou"
            ][
                "OfficialOR"
            ][
                "30m"
            ],
            refs[
                "OfficialOR_30m"
            ],
        ),

        (
            "OfficialOR 100m",
            val_result[
                "iou"
            ][
                "OfficialOR"
            ][
                "100m"
            ],
            refs[
                "OfficialOR_100m"
            ],
        ),

        (
            "HardOracle 30m",
            val_result[
                "iou"
            ][
                "HardOracle"
            ][
                "30m"
            ],
            refs[
                "V2HardOracle_30m"
            ],
        ),

        (
            "HardOracle 100m",
            val_result[
                "iou"
            ][
                "HardOracle"
            ][
                "100m"
            ],
            refs[
                "V2HardOracle_100m"
            ],
        ),
    ]

    for name, actual, expected in checks:

        print(
            f"{name:20s} "
            f"actual={actual:.6f} "
            f"reference={expected:.6f} "
            f"diff={actual-expected:+.8f}"
        )

    print()
    print("=" * 110)
    print(
        "V3-0A GO / NO-GO INFORMATION"
    )
    print("=" * 110)

    print(
        "Train strict-interior rate:",
        f"{train_result['strict_interior_rate']:.6%}",
    )

    print(
        "Val strict-interior rate  :",
        f"{val_result['strict_interior_rate']:.6%}",
    )

    print(
        "Val Soft-Hard 30m         :",
        f"{val_result['soft_minus_hard']['30m']:+.8f}",
    )

    print(
        "Val Soft-Hard 100m        :",
        f"{val_result['soft_minus_hard']['100m']:+.8f}",
    )

    print()
    print(
        "[V3-0A SOFT FUSION ORACLE AUDIT] COMPLETE"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
