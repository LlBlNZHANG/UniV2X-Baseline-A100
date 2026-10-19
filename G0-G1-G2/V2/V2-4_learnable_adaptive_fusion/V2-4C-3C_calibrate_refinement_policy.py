#!/usr/bin/env python3

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/UniV2X")

BASE = (
    ROOT
    / "G0-G1-G2/V2/V2-4_learnable_adaptive_fusion"
)

VAL_DATA = (
    BASE
    / "V2-4B_val_learnable_fusion_dataset.npz"
)

PRED_FILE = (
    BASE
    / "results/V2-4C-3B_relative_refinement/"
      "V2-4C-3B_val_predictions.npz"
)

OLD_EVAL_SCRIPT = (
    BASE
    / "V2-4C_train_utility_conditioned_gate.py"
)

OUT_DIR = (
    BASE
    / "results/V2-4C-3C_calibration"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_FILE = (
    OUT_DIR
    / "V2-4C-3C_threshold_sweep.csv"
)

JSON_FILE = (
    OUT_DIR
    / "V2-4C-3C_summary.json"
)

BEST_FILE = (
    OUT_DIR
    / "V2-4C-3C_best_policy.npz"
)


PROB_THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
]

UTILITY_THRESHOLDS = [
    0.00,
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
]

EPS = 1e-12


# =============================================================================
# Exact evaluator reuse
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


def get_full_val_ids(val):

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
        np.unique(ids)
    ) == 96

    return ids


def build_policy(
    pred_relative,
    refinable_prob,
    tau_p,
    tau_u,
):

    """
    Only V2-accepted candidates enter here.

    Default:
        g = 1.0

    Refine:
        if p_ref > tau_p
        AND max predicted ΔU > tau_u

    Selected gate:
        argmax predicted ΔU over g=0.1...0.9.
    """

    n = len(
        pred_relative
    )

    assert pred_relative.shape == (
        n,
        9,
    )

    gate_grid = (
        np.arange(
            1,
            10,
            dtype=np.float32,
        )
        /
        10.0
    )

    best_idx = np.argmax(
        pred_relative,
        axis=1,
    )

    best_gain = pred_relative[
        np.arange(n),
        best_idx,
    ]

    refine = (
        refinable_prob
        >
        tau_p
    ) & (
        best_gain
        >
        tau_u
    )

    gates = np.ones(
        n,
        dtype=np.float32,
    )

    gates[
        refine
    ] = gate_grid[
        best_idx[
            refine
        ]
    ]

    return (
        gates,
        refine,
        best_idx,
        best_gain,
    )


def evaluate_policy(
    val,
    accepted_indices,
    accepted_gates,
    ego_counts,
    range30_mask,
):

    full_gates = np.zeros(
        len(
            val[
                "features_127"
            ]
        ),
        dtype=np.float32,
    )

    full_gates[
        accepted_indices
    ] = accepted_gates

    result = (
        v24c.evaluate_gates(
            val,
            full_gates,
            ego_counts,
            range30_mask,
        )
    )

    return (
        result,
        full_gates,
    )


def main():

    val = load_npz(
        VAL_DATA
    )

    pred = load_npz(
        PRED_FILE
    )

    accepted_idx = pred[
        "accepted_indices"
    ].astype(
        np.int64
    )

    pred_relative = pred[
        "predicted_relative_utility"
    ].astype(
        np.float32
    )

    refinable_prob = pred[
        "refinable_probability"
    ].astype(
        np.float32
    )

    true_relative = pred[
        "true_relative_utility"
    ].astype(
        np.float32
    )

    true_refinable = pred[
        "true_refinable"
    ].astype(
        bool
    )

    assert len(
        val[
            "features_127"
        ]
    ) == 18555

    assert len(
        accepted_idx
    ) == 6156

    assert pred_relative.shape == (
        6156,
        9,
    )

    assert true_relative.shape == (
        6156,
        9,
    )

    assert refinable_prob.shape == (
        6156,
    )

    assert int(
        true_refinable.sum()
    ) == 4067

    # -------------------------------------------------------------------------
    # Full Val evaluator
    # -------------------------------------------------------------------------

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

    v2_gate = (
        val[
            "v2_accept"
        ].astype(
            np.float32
        )
    )

    v2_result = (
        v24c.evaluate_gates(
            val,
            v2_gate,
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

    print("=" * 118)
    print(
        "V2-4C-3C REFINEMENT DECISION CALIBRATION"
    )
    print("=" * 118)

    print(
        "Val candidates          :",
        18555,
    )

    print(
        "V2 accepted             :",
        len(
            accepted_idx
        ),
    )

    print(
        "True refinable          :",
        int(
            true_refinable.sum()
        ),
        f"({true_refinable.mean():.6%})",
    )

    print(
        "V2 Hard                :",
        f"30m={v2_result['30m']['iou']:.6f} "
        f"100m={v2_result['100m']['iou']:.6f}",
    )

    print(
        "Training                : NONE"
    )

    print(
        "Test used               : NO"
    )

    print()
    print(
        "tau_p grid              :",
        PROB_THRESHOLDS,
    )

    print(
        "tau_u grid              :",
        UTILITY_THRESHOLDS,
    )

    rows = []

    # =========================================================================
    # Sweep
    # =========================================================================

    for tau_p in PROB_THRESHOLDS:

        for tau_u in UTILITY_THRESHOLDS:

            (
                accepted_gates,
                refine_mask,
                best_idx,
                best_pred_gain,
            ) = build_policy(
                pred_relative,
                refinable_prob,
                tau_p,
                tau_u,
            )

            (
                result,
                full_gates,
            ) = evaluate_policy(
                val,
                accepted_idx,
                accepted_gates,
                ego_counts,
                range30_mask,
            )

            refine_indices = np.flatnonzero(
                refine_mask
            )

            selected_true_gain = np.zeros(
                len(
                    accepted_idx
                ),
                dtype=np.float32,
            )

            if len(
                refine_indices
            ) > 0:

                selected_true_gain[
                    refine_indices
                ] = true_relative[
                    refine_indices,
                    best_idx[
                        refine_indices
                    ],
                ]

                harmful_count = int(
                    (
                        selected_true_gain[
                            refine_indices
                        ]
                        <=
                        0.0
                    ).sum()
                )

                beneficial_count = int(
                    (
                        selected_true_gain[
                            refine_indices
                        ]
                        >
                        0.0
                    ).sum()
                )

                harmful_rate = (
                    harmful_count
                    /
                    len(
                        refine_indices
                    )
                )

                mean_true_gain = float(
                    selected_true_gain[
                        refine_indices
                    ].mean()
                )

                median_true_gain = float(
                    np.median(
                        selected_true_gain[
                            refine_indices
                        ]
                    )
                )

            else:

                harmful_count = 0
                beneficial_count = 0
                harmful_rate = 0.0
                mean_true_gain = 0.0
                median_true_gain = 0.0

            # Classification-style refinement precision:
            # whether candidate is theoretically refinable,
            # irrespective of exact chosen gate.
            if refine_mask.any():

                candidate_refine_precision = float(
                    true_refinable[
                        refine_mask
                    ].mean()
                )

            else:

                candidate_refine_precision = 0.0

            candidate_refine_recall = float(
                (
                    refine_mask
                    &
                    true_refinable
                ).sum()
                /
                true_refinable.sum()
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
                "tau_p":
                    float(
                        tau_p
                    ),

                "tau_u":
                    float(
                        tau_u
                    ),

                "refine_count":
                    int(
                        refine_mask.sum()
                    ),

                "refine_rate":
                    float(
                        refine_mask.mean()
                    ),

                "candidate_refine_precision":
                    candidate_refine_precision,

                "candidate_refine_recall":
                    candidate_refine_recall,

                "beneficial_selected_gate":
                    beneficial_count,

                "harmful_selected_gate":
                    harmful_count,

                "harmful_selected_gate_rate":
                    float(
                        harmful_rate
                    ),

                "mean_selected_true_gain_scaled":
                    mean_true_gain,

                "median_selected_true_gain_scaled":
                    median_true_gain,

                "iou30":
                    iou30,

                "iou100":
                    iou100,

                "delta30_vs_v2":
                    float(
                        iou30
                        -
                        v2_result[
                            "30m"
                        ][
                            "iou"
                        ]
                    ),

                "delta100_vs_v2":
                    float(
                        iou100
                        -
                        v2_result[
                            "100m"
                        ][
                            "iou"
                        ]
                    ),
            }

            rows.append(
                row
            )

    # =========================================================================
    # Reproduce old default policy
    # =========================================================================

    default_rows = [
        r
        for r in rows
        if (
            abs(
                r[
                    "tau_p"
                ]
                -
                0.50
            ) < EPS
            and
            abs(
                r[
                    "tau_u"
                ]
                -
                0.00
            ) < EPS
        )
    ]

    assert len(
        default_rows
    ) == 1

    default = default_rows[
        0
    ]

    print()
    print("-" * 118)
    print(
        "DEFAULT POLICY REPRODUCTION"
    )
    print("-" * 118)

    print(
        "tau_p=0.50 tau_u=0.00 :",
        f"IoU30={default['iou30']:.6f} "
        f"IoU100={default['iou100']:.6f} "
        f"refine={default['refine_rate']:.4f} "
        f"harm={default['harmful_selected_gate_rate']:.4f}",
    )

    assert abs(
        default[
            "iou100"
        ]
        -
        0.363149
    ) < 2e-6

    print(
        "V2-4C-3B reproduction  : PASS"
    )

    # =========================================================================
    # Select best calibration
    #
    # Primary: max 100m IoU
    # Tie 1  : max 30m IoU
    # Tie 2  : lower harmful rate
    # Tie 3  : more conservative / lower refine rate
    # =========================================================================

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -r[
                "iou100"
            ],
            -r[
                "iou30"
            ],
            r[
                "harmful_selected_gate_rate"
            ],
            r[
                "refine_rate"
            ],
        ),
    )

    best = rows_sorted[
        0
    ]

    # =========================================================================
    # Top configurations
    # =========================================================================

    print()
    print("-" * 118)
    print(
        "TOP-10 BY VAL 100m IoU"
    )
    print("-" * 118)

    print(
        f"{'rank':>4s} "
        f"{'tau_p':>6s} "
        f"{'tau_u':>6s} "
        f"{'refine':>8s} "
        f"{'harm':>8s} "
        f"{'30m':>10s} "
        f"{'100m':>10s} "
        f"{'d100':>10s}"
    )

    for rank, row in enumerate(
        rows_sorted[
            :10
        ],
        start=1,
    ):

        print(
            f"{rank:4d} "
            f"{row['tau_p']:6.2f} "
            f"{row['tau_u']:6.2f} "
            f"{row['refine_rate']:8.4f} "
            f"{row['harmful_selected_gate_rate']:8.4f} "
            f"{row['iou30']:10.6f} "
            f"{row['iou100']:10.6f} "
            f"{row['delta100_vs_v2']:+10.6f}"
        )

    # =========================================================================
    # Rebuild exact best policy for saving
    # =========================================================================

    (
        best_accepted_gates,
        best_refine_mask,
        best_idx,
        best_pred_gain,
    ) = build_policy(
        pred_relative,
        refinable_prob,
        best[
            "tau_p"
        ],
        best[
            "tau_u"
        ],
    )

    (
        best_result,
        best_full_gates,
    ) = evaluate_policy(
        val,
        accepted_idx,
        best_accepted_gates,
        ego_counts,
        range30_mask,
    )

    # =========================================================================
    # Save CSV
    # =========================================================================

    fieldnames = list(
        rows[
            0
        ].keys()
    )

    with open(
        CSV_FILE,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    np.savez_compressed(
        BEST_FILE,

        tau_p=np.asarray(
            best[
                "tau_p"
            ],
            dtype=np.float32,
        ),

        tau_u=np.asarray(
            best[
                "tau_u"
            ],
            dtype=np.float32,
        ),

        accepted_indices=(
            accepted_idx.astype(
                np.int32
            )
        ),

        accepted_gates=(
            best_accepted_gates.astype(
                np.float32
            )
        ),

        refine_mask=(
            best_refine_mask.astype(
                np.uint8
            )
        ),

        full_candidate_gate=(
            best_full_gates.astype(
                np.float32
            )
        ),

        predicted_best_gain=(
            best_pred_gain.astype(
                np.float32
            )
        ),

        predicted_best_index=(
            best_idx.astype(
                np.int8
            )
        ),
    )

    summary = {
        "stage":
            "V2-4C-3C Refinement Decision Calibration",

        "selection_rule":
            "max Val 100m IoU; tie -> 30m IoU -> lower harmful rate -> lower refine rate",

        "prob_thresholds":
            PROB_THRESHOLDS,

        "utility_thresholds":
            UTILITY_THRESHOLDS,

        "v2_reference": {
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

        "default_policy":
            default,

        "best_policy":
            best,

        "improvement_over_uncalibrated": {
            "30m":
                float(
                    best[
                        "iou30"
                    ]
                    -
                    default[
                        "iou30"
                    ]
                ),

            "100m":
                float(
                    best[
                        "iou100"
                    ]
                    -
                    default[
                        "iou100"
                    ]
                ),

            "harmful_rate_reduction":
                float(
                    default[
                        "harmful_selected_gate_rate"
                    ]
                    -
                    best[
                        "harmful_selected_gate_rate"
                    ]
                ),
        },

        "test_used":
            False,
    }

    with open(
        JSON_FILE,
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
    print("=" * 118)
    print(
        "V2-4C-3C BEST CALIBRATED POLICY"
    )
    print("=" * 118)

    print(
        "Best tau_p              :",
        best[
            "tau_p"
        ],
    )

    print(
        "Best tau_u              :",
        best[
            "tau_u"
        ],
    )

    print(
        "Refine count/rate       :",
        best[
            "refine_count"
        ],
        f"({best['refine_rate']:.6%})",
    )

    print(
        "Candidate refine P/R    :",
        f"{best['candidate_refine_precision']:.6f} / "
        f"{best['candidate_refine_recall']:.6f}",
    )

    print(
        "Harmful gate rate       :",
        f"{best['harmful_selected_gate_rate']:.6%}",
    )

    print(
        "Mean true gain          :",
        f"{best['mean_selected_true_gain_scaled']:+.6f}",
    )

    print()
    print(
        "V2 Hard                :",
        f"30m={v2_result['30m']['iou']:.6f} "
        f"100m={v2_result['100m']['iou']:.6f}",
    )

    print(
        "Uncalibrated 3B        :",
        f"30m={default['iou30']:.6f} "
        f"100m={default['iou100']:.6f}",
    )

    print(
        "Calibrated Refinement  :",
        f"30m={best['iou30']:.6f} "
        f"100m={best['iou100']:.6f}",
    )

    print()
    print(
        "Delta vs V2 30m        :",
        f"{best['delta30_vs_v2']:+.8f}",
    )

    print(
        "Delta vs V2 100m       :",
        f"{best['delta100_vs_v2']:+.8f}",
    )

    print(
        "Gain over uncalib 100m :",
        f"{best['iou100'] - default['iou100']:+.8f}",
    )

    print()
    print(
        "CSV       :",
        CSV_FILE,
    )

    print(
        "Summary   :",
        JSON_FILE,
    )

    print(
        "Best policy:",
        BEST_FILE,
    )

    print(
        "Test used : NO"
    )

    print()
    print(
        "[V2-4C-3C CALIBRATION SWEEP] COMPLETE"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
