#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/UniV2X")

BASE = (
    ROOT
    / "G0-G1-G2/V2/V2-4_learnable_adaptive_fusion"
)

OLD_EVAL_SCRIPT = (
    BASE
    / "V2-4C_train_utility_conditioned_gate.py"
)

EPS = 1e-12


# =============================================================================
# Candidate action sets
# =============================================================================

ACTION_SETS = {
    "V2_g1":
        [1.0],

    "Binary_01_10":
        [0.1, 1.0],

    "Ternary_01_03_10":
        [0.1, 0.3, 1.0],

    "FourLevel_01_02_05_10":
        [0.1, 0.2, 0.5, 1.0],

    "SixLevel_01_02_03_05_07_10":
        [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],

    "Full_01_to_10":
        [
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
        ],
}


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


def get_full_split_ids(
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
    ) == expected_samples

    return ids


def gate_to_index(g):

    idx = int(
        round(
            float(g)
            *
            10.0
        )
    )

    assert 0 <= idx <= 10

    return idx


def build_coarse_oracle(
    utility_curve,
    v2_accept,
    action_set,
):

    """
    V2-conditioned coarse-action oracle.

    V2 rejected:
        g = 0

    V2 accepted:
        default g = 1

    Only change from g=1 to another allowed g
    if its GT utility STRICTLY exceeds U(g=1).

    Conservative tie policy:
        keep the largest gate among equal-best alternatives,
        i.e. closest to original V2.
    """

    utility_curve = np.asarray(
        utility_curve,
        dtype=np.float64,
    )

    v2_accept = np.asarray(
        v2_accept,
        dtype=bool,
    )

    n = len(
        utility_curve
    )

    gates = np.zeros(
        n,
        dtype=np.float32,
    )

    gates[
        v2_accept
    ] = 1.0

    selected_indices = np.full(
        n,
        -1,
        dtype=np.int8,
    )

    selected_indices[
        v2_accept
    ] = 10

    refined = np.zeros(
        n,
        dtype=bool,
    )

    gain = np.zeros(
        n,
        dtype=np.float64,
    )

    action_indices = [
        gate_to_index(g)
        for g in action_set
    ]

    assert 10 in action_indices

    accepted_indices = np.flatnonzero(
        v2_accept
    )

    for i in accepted_indices:

        u1 = float(
            utility_curve[
                i,
                10,
            ]
        )

        best_u = u1
        best_idx = 10

        for idx in action_indices:

            if idx == 10:
                continue

            u = float(
                utility_curve[
                    i,
                    idx,
                ]
            )

            if (
                u
                >
                best_u
                +
                EPS
            ):
                best_u = u
                best_idx = idx

            elif (
                abs(
                    u
                    -
                    best_u
                )
                <=
                EPS
                and
                u
                >
                u1
                +
                EPS
                and
                idx
                >
                best_idx
            ):
                # Equal improvement:
                # choose larger g, closer to V2 g=1.
                best_idx = idx

        if best_idx != 10:

            gates[i] = np.float32(
                best_idx
                /
                10.0
            )

            selected_indices[i] = (
                best_idx
            )

            refined[i] = True

            gain[i] = (
                best_u
                -
                u1
            )

    return (
        gates,
        refined,
        gain,
        selected_indices,
    )


def process_split(
    split,
    expected_samples,
    expected_candidates,
):

    data = load_npz(
        BASE
        /
        f"V2-4B_{split}_learnable_fusion_dataset.npz"
    )

    curve = load_npz(
        BASE
        /
        f"V2-4C-2A_{split}_gate_utility_curves.npz"
    )

    assert len(
        data[
            "features_127"
        ]
    ) == expected_candidates

    assert curve[
        "utility_curve"
    ].shape == (
        expected_candidates,
        11,
    )

    assert np.array_equal(
        data[
            "sample_idx"
        ].astype(str),
        curve[
            "sample_idx"
        ].astype(str),
    )

    v2_accept = (
        data[
            "v2_accept"
        ].astype(bool)
    )

    utility_curve = (
        curve[
            "utility_curve"
        ].astype(
            np.float32
        )
    )

    full_ids = (
        get_full_split_ids(
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

    results = {}

    print()
    print("=" * 116)
    print(
        f"V2-4C-3D {split.upper()} COARSE ACTION-SET ORACLE"
    )
    print("=" * 116)

    print(
        "Full samples            :",
        expected_samples,
    )

    print(
        "Candidates              :",
        expected_candidates,
    )

    print(
        "V2 accepted             :",
        int(
            v2_accept.sum()
        ),
        f"({v2_accept.mean():.6%})",
    )

    print()

    print(
        f"{'Action set':32s} "
        f"{'#A':>4s} "
        f"{'Refine':>8s} "
        f"{'30m':>10s} "
        f"{'100m':>10s} "
        f"{'Δ100':>10s}"
    )

    print("-" * 82)

    v2_iou100 = None
    full_iou100 = None

    for name, actions in (
        ACTION_SETS.items()
    ):

        (
            gates,
            refined,
            gain,
            selected_indices,
        ) = build_coarse_oracle(
            utility_curve,
            v2_accept,
            actions,
        )

        result = (
            v24c.evaluate_gates(
                data,
                gates,
                ego_counts,
                range30_mask,
            )
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

        if name == "V2_g1":
            v2_iou100 = iou100

        if name == "Full_01_to_10":
            full_iou100 = iou100

        gains = gain[
            refined
        ]

        result_record = {
            "actions":
                actions,

            "num_actions":
                len(
                    actions
                ),

            "refine_count":
                int(
                    refined.sum()
                ),

            "refine_rate_v2_accepted":
                float(
                    refined.sum()
                    /
                    max(
                        int(
                            v2_accept.sum()
                        ),
                        1,
                    )
                ),

            "gain_mean":
                float(
                    gains.mean()
                )
                if len(
                    gains
                ) > 0
                else 0.0,

            "gain_median":
                float(
                    np.median(
                        gains
                    )
                )
                if len(
                    gains
                ) > 0
                else 0.0,

            "iou30":
                iou30,

            "iou100":
                iou100,

            "gate_histogram":
                {
                    f"{g:.1f}":
                        int(
                            (
                                gates[
                                    v2_accept
                                ]
                                ==
                                np.float32(g)
                            ).sum()
                        )
                    for g in actions
                },
        }

        results[
            name
        ] = result_record

        print(
            f"{name:32s} "
            f"{len(actions):4d} "
            f"{int(refined.sum()):8d} "
            f"{iou30:10.6f} "
            f"{iou100:10.6f} "
            f"{0.0 if v2_iou100 is None else iou100-v2_iou100:+10.6f}"
        )

    assert v2_iou100 is not None
    assert full_iou100 is not None

    full_headroom = (
        full_iou100
        -
        v2_iou100
    )

    print()
    print("-" * 116)
    print(
        "HEADROOM RECOVERY"
    )
    print("-" * 116)

    for name, record in (
        results.items()
    ):

        delta = (
            record[
                "iou100"
            ]
            -
            v2_iou100
        )

        recovery = (
            delta
            /
            full_headroom
            if abs(
                full_headroom
            ) > EPS
            else 0.0
        )

        record[
            "delta100_vs_v2"
        ] = float(
            delta
        )

        record[
            "full_oracle_recovery_100m"
        ] = float(
            recovery
        )

        print(
            f"{name:32s} "
            f"Δ100={delta:+.6f} "
            f"Recovery={recovery:7.2%}"
        )

    out_json = (
        BASE
        /
        f"V2-4C-3D_{split}_coarse_action_oracle_summary.json"
    )

    with open(
        out_json,
        "w",
    ) as f:

        json.dump(
            {
                "stage":
                    "V2-4C-3D Coarse Action-set Oracle Audit",

                "split":
                    split,

                "samples":
                    expected_samples,

                "candidates":
                    expected_candidates,

                "v2_accepted":
                    int(
                        v2_accept.sum()
                    ),

                "results":
                    results,

                "test_used":
                    False,
            },
            f,
            indent=2,
        )

    print()
    print(
        "Saved:",
        out_json,
    )

    return results


def main():

    print("=" * 116)
    print(
        "V2-4C-3D COARSE REFINEMENT ACTION-SET ORACLE AUDIT"
    )
    print("=" * 116)

    print(
        "Question:"
    )

    print(
        "How many fusion-intensity levels are actually needed"
    )

    print(
        "to retain the V2-conditioned soft-refinement oracle headroom?"
    )

    print()
    print(
        "Training : NONE"
    )

    print(
        "GT use   : Oracle diagnosis only"
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
    print("=" * 116)
    print(
        "V2-4C-3D VAL SUMMARY"
    )
    print("=" * 116)

    v2 = val[
        "V2_g1"
    ][
        "iou100"
    ]

    full = val[
        "Full_01_to_10"
    ][
        "iou100"
    ]

    print(
        "V2 100m                :",
        f"{v2:.6f}",
    )

    print(
        "Full conditioned oracle:",
        f"{full:.6f}",
    )

    print()

    for name in [
        "Binary_01_10",
        "Ternary_01_03_10",
        "FourLevel_01_02_05_10",
        "SixLevel_01_02_03_05_07_10",
        "Full_01_to_10",
    ]:

        r = val[
            name
        ]

        print(
            f"{name:32s} "
            f"100m={r['iou100']:.6f} "
            f"Δ={r['delta100_vs_v2']:+.6f} "
            f"Recovery={r['full_oracle_recovery_100m']:.2%}"
        )

    print()
    print(
        "Test used               : NO"
    )

    print(
        "[V2-4C-3D COARSE ACTION ORACLE] COMPLETE"
    )

    print("=" * 116)


if __name__ == "__main__":
    main()
