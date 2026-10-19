#!/usr/bin/env python3

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/UniV2X")

BASE = (
    ROOT
    / "G0-G1-G2/V2/V2-4_learnable_adaptive_fusion"
)

OLD_C_SCRIPT = (
    BASE
    / "V2-4C_train_utility_conditioned_gate.py"
)

EPS = 1e-12


# =============================================================================
# Reuse exact V2-4C full-split evaluator
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "v24c",
    OLD_C_SCRIPT,
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


def get_full_split_sample_ids(
    data,
    expected_samples,
):

    scene_tokens = set(
        np.unique(
            data[
                "scene_token"
            ].astype(str)
        ).tolist()
    )

    valid_file = (
        ROOT
        / "G0-G1-G2/G3/cache_valid_samples.txt"
    )

    sample_ids = []

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

                sample_ids.append(
                    parts[0]
                )

    sample_ids = np.asarray(
        sample_ids
    )

    unique_ids = np.unique(
        sample_ids
    )

    assert len(
        unique_ids
    ) == expected_samples, (
        f"Expected {expected_samples} samples, "
        f"got {len(unique_ids)}"
    )

    return sample_ids


def conservative_refinement_oracle(
    utility_curve,
    gate_grid,
    v2_accept,
):

    """
    V2-conditioned refinement:

      rejected by V2 -> g = 0
      accepted by V2 -> baseline g = 1

    Only replace g=1 by an interior gate g=0.1...0.9
    when its true utility STRICTLY exceeds U(g=1).

    If several interior gates have the same best utility,
    choose the largest g (closest to original V2 g=1)
    for conservative tie-breaking.
    """

    utility_curve = np.asarray(
        utility_curve,
        dtype=np.float64,
    )

    gate_grid = np.asarray(
        gate_grid,
        dtype=np.float64,
    )

    v2_accept = np.asarray(
        v2_accept,
        dtype=bool,
    )

    n = len(
        utility_curve
    )

    assert utility_curve.shape == (
        n,
        11,
    )

    assert len(
        gate_grid
    ) == 11

    assert np.allclose(
        gate_grid,
        np.arange(11) / 10.0,
    )

    # V2 original policy:
    # reject -> 0
    # accept -> 1
    gates = np.zeros(
        n,
        dtype=np.float32,
    )

    gates[
        v2_accept
    ] = 1.0

    strict_refine = np.zeros(
        n,
        dtype=bool,
    )

    gain = np.zeros(
        n,
        dtype=np.float64,
    )

    selected_utility = np.zeros(
        n,
        dtype=np.float64,
    )

    selected_utility[
        v2_accept
    ] = utility_curve[
        v2_accept,
        10,
    ]

    accepted_indices = np.flatnonzero(
        v2_accept
    )

    for i in accepted_indices:

        u_g1 = float(
            utility_curve[
                i,
                10,
            ]
        )

        interior = utility_curve[
            i,
            1:10,
        ]

        best_u = float(
            interior.max()
        )

        # Only genuine improvement over original V2 g=1.
        if (
            best_u
            >
            u_g1
            +
            EPS
        ):

            # Conservative tie:
            # among equal-best interior choices,
            # take the largest gate closest to g=1.
            best_local = np.flatnonzero(
                np.abs(
                    interior
                    -
                    best_u
                )
                <=
                EPS
            )

            local_idx = int(
                best_local[-1]
            )

            global_idx = (
                local_idx
                +
                1
            )

            gates[i] = np.float32(
                gate_grid[
                    global_idx
                ]
            )

            strict_refine[i] = True

            gain[i] = (
                best_u
                -
                u_g1
            )

            selected_utility[i] = (
                best_u
            )

    return (
        gates,
        strict_refine,
        gain,
        selected_utility,
    )


def process_split(
    split,
    expected_samples,
    expected_candidates,
):

    data_file = (
        BASE
        /
        f"V2-4B_{split}_learnable_fusion_dataset.npz"
    )

    curve_file = (
        BASE
        /
        f"V2-4C-2A_{split}_gate_utility_curves.npz"
    )

    data = load_npz(
        data_file
    )

    curve = load_npz(
        curve_file
    )

    assert len(
        data[
            "features_127"
        ]
    ) == expected_candidates

    assert len(
        curve[
            "utility_curve"
        ]
    ) == expected_candidates

    assert np.array_equal(
        data[
            "sample_idx"
        ].astype(str),
        curve[
            "sample_idx"
        ].astype(str),
    )

    gate_grid = curve[
        "gate_grid"
    ].astype(
        np.float32
    )

    utility_curve = curve[
        "utility_curve"
    ].astype(
        np.float32
    )

    v2_accept = data[
        "v2_accept"
    ].astype(
        bool
    )

    (
        refined_gate,
        strict_refine,
        refinement_gain,
        selected_utility,
    ) = conservative_refinement_oracle(
        utility_curve,
        gate_grid,
        v2_accept,
    )

    # =========================================================================
    # Exact full-split evaluation
    # =========================================================================

    full_ids = (
        get_full_split_sample_ids(
            data,
            expected_samples,
        )
    )

    ego_counts = (
        v24c.build_ego_baseline_counts(
            full_ids
        )
    )

    range30_mask = (
        v24c.build_range30_mask(
            data[
                "patch_r"
            ],
            data[
                "patch_c"
            ],
        )
    )

    v2_gate = (
        v2_accept.astype(
            np.float32
        )
    )

    v2_result = (
        v24c.evaluate_gates(
            data,
            v2_gate,
            ego_counts,
            range30_mask,
        )
    )

    refined_result = (
        v24c.evaluate_gates(
            data,
            refined_gate,
            ego_counts,
            range30_mask,
        )
    )

    global_soft_result = (
        v24c.evaluate_gates(
            data,
            curve[
                "soft_best_gate"
            ].astype(
                np.float32
            ),
            ego_counts,
            range30_mask,
        )
    )

    global_hard_result = (
        v24c.evaluate_gates(
            data,
            curve[
                "hard_best_gate"
            ].astype(
                np.float32
            ),
            ego_counts,
            range30_mask,
        )
    )

    # =========================================================================
    # Diagnostics
    # =========================================================================

    accepted_count = int(
        v2_accept.sum()
    )

    refined_count = int(
        strict_refine.sum()
    )

    refined_among_accepted = (
        refined_count
        /
        accepted_count
        if accepted_count > 0
        else 0.0
    )

    gate_hist = Counter(
        f"{float(g):.1f}"
        for g in
        refined_gate[
            v2_accept
        ]
    )

    gains = refinement_gain[
        strict_refine
    ]

    if len(gains) > 0:

        gain_stats = {
            "mean":
                float(
                    gains.mean()
                ),

            "median":
                float(
                    np.median(
                        gains
                    )
                ),

            "p90":
                float(
                    np.quantile(
                        gains,
                        0.90,
                    )
                ),

            "max":
                float(
                    gains.max()
                ),
        }

    else:

        gain_stats = {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }

    # =========================================================================
    # Headroom recovery
    # =========================================================================

    delta30 = (
        refined_result[
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
        refined_result[
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

    global_headroom30 = (
        global_soft_result[
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

    global_headroom100 = (
        global_soft_result[
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
        global_headroom30
        if abs(
            global_headroom30
        ) > EPS
        else 0.0
    )

    recovery100 = (
        delta100
        /
        global_headroom100
        if abs(
            global_headroom100
        ) > EPS
        else 0.0
    )

    # =========================================================================
    # Print
    # =========================================================================

    print()
    print("=" * 110)
    print(
        f"V2-4C-3A {split.upper()} "
        "V2-CONDITIONED SOFT REFINEMENT ORACLE"
    )
    print("=" * 110)

    print(
        "Full samples            :",
        expected_samples,
    )

    print(
        "Candidate patches       :",
        expected_candidates,
    )

    print(
        "V2 accepted             :",
        accepted_count,
        f"({accepted_count/expected_candidates:.6%})",
    )

    print(
        "Strictly refinable      :",
        refined_count,
        f"({refined_count/expected_candidates:.6%} all, "
        f"{refined_among_accepted:.6%} of V2 accepted)",
    )

    print()
    print(
        "Refined gate histogram "
        "(V2-accepted candidates only):"
    )

    for g in gate_grid[
        1:
    ]:

        key = f"{float(g):.1f}"

        count = int(
            gate_hist.get(
                key,
                0,
            )
        )

        print(
            f"  g={key}: "
            f"{count:6d} "
            f"({count/accepted_count:.6%})"
        )

    print()
    print(
        "Strict refinement gain mean  :",
        f"{gain_stats['mean']:+.10f}",
    )

    print(
        "Strict refinement gain median:",
        f"{gain_stats['median']:+.10f}",
    )

    print(
        "Strict refinement gain p90   :",
        f"{gain_stats['p90']:+.10f}",
    )

    print(
        "Strict refinement gain max   :",
        f"{gain_stats['max']:+.10f}",
    )

    print()
    print("-" * 110)
    print(
        "FULL-SPLIT OCCUPANCY IOU"
    )
    print("-" * 110)

    print(
        f"{'Strategy':28s} "
        f"{'30m':>12s} "
        f"{'100m':>12s}"
    )

    print("-" * 56)

    print(
        f"{'V2 Hard':28s} "
        f"{v2_result['30m']['iou']:12.6f} "
        f"{v2_result['100m']['iou']:12.6f}"
    )

    print(
        f"{'V2-conditioned Soft Oracle':28s} "
        f"{refined_result['30m']['iou']:12.6f} "
        f"{refined_result['100m']['iou']:12.6f}"
    )

    print(
        f"{'Global Hard Oracle':28s} "
        f"{global_hard_result['30m']['iou']:12.6f} "
        f"{global_hard_result['100m']['iou']:12.6f}"
    )

    print(
        f"{'Global Soft Oracle':28s} "
        f"{global_soft_result['30m']['iou']:12.6f} "
        f"{global_soft_result['100m']['iou']:12.6f}"
    )

    print()
    print(
        "Refinement delta 30m   :",
        f"{delta30:+.8f}",
    )

    print(
        "Refinement delta 100m  :",
        f"{delta100:+.8f}",
    )

    print(
        "Global headroom 30m    :",
        f"{global_headroom30:+.8f}",
    )

    print(
        "Global headroom 100m   :",
        f"{global_headroom100:+.8f}",
    )

    print(
        "Recovered headroom 30m :",
        f"{recovery30:.2%}",
    )

    print(
        "Recovered headroom 100m:",
        f"{recovery100:.2%}",
    )

    # =========================================================================
    # Save
    # =========================================================================

    npz_path = (
        BASE
        /
        f"V2-4C-3A_{split}_v2_conditioned_soft_oracle.npz"
    )

    np.savez_compressed(
        npz_path,

        refined_gate=(
            refined_gate.astype(
                np.float32
            )
        ),

        v2_accept=(
            v2_accept.astype(
                np.uint8
            )
        ),

        strict_refine=(
            strict_refine.astype(
                np.uint8
            )
        ),

        refinement_gain=(
            refinement_gain.astype(
                np.float32
            )
        ),

        selected_utility=(
            selected_utility.astype(
                np.float32
            )
        ),

        sample_idx=(
            data[
                "sample_idx"
            ]
        ),

        horizon=(
            data[
                "horizon"
            ]
        ),

        patch_r=(
            data[
                "patch_r"
            ]
        ),

        patch_c=(
            data[
                "patch_c"
            ]
        ),
    )

    summary = {
        "stage":
            "V2-4C-3A V2-conditioned Soft Refinement Oracle",

        "split":
            split,

        "samples":
            int(
                expected_samples
            ),

        "candidates":
            int(
                expected_candidates
            ),

        "v2_accepted":
            accepted_count,

        "strictly_refinable":
            refined_count,

        "strictly_refinable_rate_all":
            float(
                refined_count
                /
                expected_candidates
            ),

        "strictly_refinable_rate_v2_accepted":
            float(
                refined_among_accepted
            ),

        "refinement_gain":
            gain_stats,

        "iou": {
            "v2_hard":
                {
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

            "v2_conditioned_soft_oracle":
                {
                    "30m":
                        float(
                            refined_result[
                                "30m"
                            ][
                                "iou"
                            ]
                        ),

                    "100m":
                        float(
                            refined_result[
                                "100m"
                            ][
                                "iou"
                            ]
                        ),
                },

            "global_soft_oracle":
                {
                    "30m":
                        float(
                            global_soft_result[
                                "30m"
                            ][
                                "iou"
                            ]
                        ),

                    "100m":
                        float(
                            global_soft_result[
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

        "recovered_global_soft_headroom": {
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
    }

    json_path = (
        BASE
        /
        f"V2-4C-3A_{split}_v2_conditioned_soft_oracle_summary.json"
    )

    with open(
        json_path,
        "w",
    ) as f:

        json.dump(
            summary,
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

    return summary


def main():

    print("=" * 110)
    print(
        "V2-4C-3A V2-CONDITIONED SOFT REFINEMENT ORACLE"
    )
    print("=" * 110)

    print(
        "Question:"
    )

    print(
        "Keep V2 accept/reject decisions fixed;"
    )

    print(
        "can adaptive fusion intensity alone provide meaningful headroom?"
    )

    print()
    print(
        "Rejected by V2 -> g=0"
    )

    print(
        "Accepted by V2 -> default g=1;"
    )

    print(
        "change to g=0.1...0.9 only when utility strictly improves."
    )

    print()
    print(
        "Oracle uses GT for diagnosis only."
    )

    print(
        "Test used: NO"
    )

    train = process_split(
        "train",
        expected_samples=361,
        expected_candidates=86143,
    )

    val = process_split(
        "val",
        expected_samples=96,
        expected_candidates=18555,
    )

    print()
    print("=" * 110)
    print(
        "V2-4C-3A FINAL GO / NO-GO"
    )
    print("=" * 110)

    print(
        "Train delta 100m       :",
        f"{train['delta_vs_v2']['100m']:+.8f}",
    )

    print(
        "Val delta 100m         :",
        f"{val['delta_vs_v2']['100m']:+.8f}",
    )

    print(
        "Val recovered headroom :",
        f"{val['recovered_global_soft_headroom']['100m']:.2%}",
    )

    print(
        "Val refinable/V2 accept:",
        f"{val['strictly_refinable_rate_v2_accepted']:.2%}",
    )

    print(
        "Test used              : NO"
    )

    print()
    print(
        "[V2-4C-3A V2-CONDITIONED SOFT ORACLE] COMPLETE"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
