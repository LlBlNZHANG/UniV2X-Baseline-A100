#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/UniV2X")

BASE = (
    ROOT
    / "G0-G1-G2/V2/V2-4_learnable_adaptive_fusion"
)

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

SEG_THRESH = 0.1
EPS = 1e-12
CHUNK = 8192


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


def compute_curves(data):

    n = len(
        data["Pv_patch"]
    )

    utility_curve = np.zeros(
        (n, len(GATES)),
        dtype=np.float32,
    )

    ba_curve = np.zeros(
        (n, len(GATES)),
        dtype=np.uint8,
    )

    ha_curve = np.zeros(
        (n, len(GATES)),
        dtype=np.uint8,
    )

    add_count_curve = np.zeros(
        (n, len(GATES)),
        dtype=np.uint8,
    )

    for start in range(
        0,
        n,
        CHUNK,
    ):

        end = min(
            start + CHUNK,
            n,
        )

        pv = data[
            "Pv_patch"
        ][
            start:end
        ].astype(
            np.float64
        )

        pi = data[
            "Pi_patch"
        ][
            start:end
        ].astype(
            np.float64
        )

        ov = data[
            "Ov_patch"
        ][
            start:end
        ].astype(
            bool
        )

        warp = data[
            "warp_patch"
        ][
            start:end
        ].astype(
            bool
        )

        gt = data[
            "GT_patch"
        ][
            start:end
        ].astype(
            bool
        )

        tp = data[
            "ego_tp_100"
        ][
            start:end
        ].astype(
            np.float64
        )

        fp = data[
            "ego_fp_100"
        ][
            start:end
        ].astype(
            np.float64
        )

        fn = data[
            "ego_fn_100"
        ][
            start:end
        ].astype(
            np.float64
        )

        positive_delta = np.maximum(
            pi - pv,
            0.0,
        )

        positive_delta *= (
            warp.astype(
                np.float64
            )
        )

        # [B, 11, 4, 4]
        pf = (
            pv[
                :,
                None,
                :,
                :,
            ]
            +
            GATES[
                None,
                :,
                None,
                None,
            ]
            *
            positive_delta[
                :,
                None,
                :,
                :,
            ]
        )

        pred = (
            pf
            >
            SEG_THRESH
        )

        add = (
            (~ov[:, None, :, :])
            &
            pred
            &
            warp[:, None, :, :]
        )

        ba = (
            add
            &
            gt[:, None, :, :]
        ).sum(
            axis=(2, 3)
        )

        ha = (
            add
            &
            (~gt[:, None, :, :])
        ).sum(
            axis=(2, 3)
        )

        add_count = add.sum(
            axis=(2, 3)
        )

        d = (
            tp
            +
            fp
            +
            fn
        )

        valid = (
            d > 0
        )

        before = np.zeros(
            len(tp),
            dtype=np.float64,
        )

        before[
            valid
        ] = (
            tp[
                valid
            ]
            /
            d[
                valid
            ]
        )

        after = np.zeros(
            (
                len(tp),
                len(GATES),
            ),
            dtype=np.float64,
        )

        after[
            valid
        ] = (
            tp[
                valid,
                None,
            ]
            +
            ba[
                valid
            ]
        ) / (
            d[
                valid,
                None,
            ]
            +
            ha[
                valid
            ]
        )

        utility = (
            after
            -
            before[:, None]
        )

        utility[
            ~valid
        ] = 0.0

        utility_curve[
            start:end
        ] = utility.astype(
            np.float32
        )

        ba_curve[
            start:end
        ] = ba.astype(
            np.uint8
        )

        ha_curve[
            start:end
        ] = ha.astype(
            np.uint8
        )

        add_count_curve[
            start:end
        ] = add_count.astype(
            np.uint8
        )

        if (
            start == 0
            or
            end == n
            or
            end % 32768 == 0
        ):

            print(
                f"[{end:6d}/{n:6d}]"
            )

    return (
        utility_curve,
        ba_curve,
        ha_curve,
        add_count_curve,
    )


def derive_oracle(
    utility_curve,
):

    n = len(
        utility_curve
    )

    u0 = utility_curve[
        :,
        0
    ].astype(
        np.float64
    )

    u1 = utility_curve[
        :,
        -1
    ].astype(
        np.float64
    )

    hard_idx = np.zeros(
        n,
        dtype=np.int64,
    )

    hard_idx[
        u1
        >
        u0
        +
        EPS
    ] = (
        len(GATES)
        -
        1
    )

    hard_best_gate = GATES[
        hard_idx
    ].astype(
        np.float32
    )

    hard_best_utility = (
        utility_curve[
            np.arange(n),
            hard_idx,
        ]
    ).astype(
        np.float32
    )

    interior = utility_curve[
        :,
        1:-1
    ].astype(
        np.float64
    )

    interior_local = np.argmax(
        interior,
        axis=1,
    )

    interior_idx = (
        interior_local
        +
        1
    )

    interior_best = interior[
        np.arange(n),
        interior_local,
    ]

    hard_best = np.maximum(
        u0,
        u1,
    )

    strict = (
        interior_best
        >
        hard_best
        +
        EPS
    )

    soft_idx = hard_idx.copy()

    soft_idx[
        strict
    ] = interior_idx[
        strict
    ]

    soft_best_gate = GATES[
        soft_idx
    ].astype(
        np.float32
    )

    soft_best_utility = (
        utility_curve[
            np.arange(n),
            soft_idx,
        ]
    ).astype(
        np.float32
    )

    strict_gain = np.zeros(
        n,
        dtype=np.float32,
    )

    strict_gain[
        strict
    ] = (
        interior_best[
            strict
        ]
        -
        hard_best[
            strict
        ]
    ).astype(
        np.float32
    )

    return {
        "hard_best_gate":
            hard_best_gate,

        "hard_best_utility":
            hard_best_utility,

        "soft_best_gate":
            soft_best_gate,

        "soft_best_utility":
            soft_best_utility,

        "strict_interior":
            strict.astype(
                np.uint8
            ),

        "strict_gain":
            strict_gain,

        "soft_idx":
            soft_idx.astype(
                np.int8
            ),
    }


def process_split(split):

    src = (
        BASE
        /
        f"V2-4B_{split}_learnable_fusion_dataset.npz"
    )

    data = load_npz(
        src
    )

    expected = {
        "train": 86143,
        "val": 18555,
    }[
        split
    ]

    assert len(
        data[
            "Pv_patch"
        ]
    ) == expected

    print()
    print("=" * 110)
    print(
        f"V2-4C-2A {split.upper()} GATE UTILITY CURVES"
    )
    print("=" * 110)

    print(
        "Candidates              :",
        expected,
    )

    print(
        "Gate grid               :",
        GATES.tolist(),
    )

    print(
        "Test used               : NO"
    )

    (
        utility_curve,
        ba_curve,
        ha_curve,
        add_count_curve,
    ) = compute_curves(
        data
    )

    oracle = derive_oracle(
        utility_curve
    )

    # =========================================================================
    # Exact V2-4A reconstruction audit
    # =========================================================================

    old_soft_gate = data[
        "oracle_soft_best_gate"
    ].astype(
        np.float32
    )

    old_hard_gate = data[
        "oracle_hard_best_gate"
    ].astype(
        np.float32
    )

    old_strict = data[
        "oracle_strict_interior"
    ].astype(
        np.uint8
    )

    old_soft_utility = data[
        "oracle_soft_best_utility"
    ].astype(
        np.float32
    )

    old_hard_utility = data[
        "oracle_hard_best_utility"
    ].astype(
        np.float32
    )

    assert np.array_equal(
        oracle[
            "soft_best_gate"
        ],
        old_soft_gate,
    ), "Soft gate reconstruction mismatch"

    assert np.array_equal(
        oracle[
            "hard_best_gate"
        ],
        old_hard_gate,
    ), "Hard gate reconstruction mismatch"

    assert np.array_equal(
        oracle[
            "strict_interior"
        ],
        old_strict,
    ), "Strict-interior reconstruction mismatch"

    soft_u_diff = float(
        np.max(
            np.abs(
                oracle[
                    "soft_best_utility"
                ]
                -
                old_soft_utility
            )
        )
    )

    hard_u_diff = float(
        np.max(
            np.abs(
                oracle[
                    "hard_best_utility"
                ]
                -
                old_hard_utility
            )
        )
    )

    assert soft_u_diff < 1e-7, (
        soft_u_diff
    )

    assert hard_u_diff < 1e-7, (
        hard_u_diff
    )

    print()
    print(
        "V2-4A oracle reconstruction : PASS"
    )

    print(
        "Max soft utility diff        :",
        f"{soft_u_diff:.12g}",
    )

    print(
        "Max hard utility diff        :",
        f"{hard_u_diff:.12g}",
    )

    strict = (
        oracle[
            "strict_interior"
        ].astype(bool)
    )

    print(
        "Strict interior candidates   :",
        int(
            strict.sum()
        ),
        f"({strict.mean():.6%})",
    )

    # =========================================================================
    # Curve structure diagnostics
    # =========================================================================

    unique_count = np.asarray(
        [
            len(
                np.unique(
                    row
                )
            )
            for row in utility_curve
        ],
        dtype=np.int16,
    )

    nonflat = (
        unique_count > 1
    )

    rich_curve = (
        unique_count >= 3
    )

    print(
        "Non-flat utility curves      :",
        int(
            nonflat.sum()
        ),
        f"({nonflat.mean():.6%})",
    )

    print(
        ">=3 distinct utility levels  :",
        int(
            rich_curve.sum()
        ),
        f"({rich_curve.mean():.6%})",
    )

    print(
        "Mean distinct utility levels :",
        f"{unique_count.mean():.6f}",
    )

    gate_hist = {}

    for g in GATES:

        count = int(
            (
                oracle[
                    "soft_best_gate"
                ]
                ==
                np.float32(g)
            ).sum()
        )

        gate_hist[
            f"{g:.1f}"
        ] = count

    print()
    print(
        "Soft-oracle gate histogram:"
    )

    for k, v in (
        gate_hist.items()
    ):

        print(
            f"  g={k}: "
            f"{v:7d} "
            f"({v/expected:.6%})"
        )

    # =========================================================================
    # Save
    # =========================================================================

    out = (
        BASE
        /
        f"V2-4C-2A_{split}_gate_utility_curves.npz"
    )

    np.savez_compressed(
        out,

        gate_grid=GATES.astype(
            np.float32
        ),

        utility_curve=(
            utility_curve
        ),

        ba_curve=(
            ba_curve
        ),

        ha_curve=(
            ha_curve
        ),

        add_count_curve=(
            add_count_curve
        ),

        hard_best_gate=(
            oracle[
                "hard_best_gate"
            ]
        ),

        hard_best_utility=(
            oracle[
                "hard_best_utility"
            ]
        ),

        soft_best_gate=(
            oracle[
                "soft_best_gate"
            ]
        ),

        soft_best_utility=(
            oracle[
                "soft_best_utility"
            ]
        ),

        soft_best_index=(
            oracle[
                "soft_idx"
            ]
        ),

        strict_interior=(
            oracle[
                "strict_interior"
            ]
        ),

        strict_gain=(
            oracle[
                "strict_gain"
            ]
        ),

        distinct_utility_count=(
            unique_count
        ),

        sample_idx=(
            data[
                "sample_idx"
            ]
        ),

        scene_token=(
            data[
                "scene_token"
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
            "V2-4C-2A Gate Utility Curve Dataset",

        "split":
            split,

        "candidates":
            int(
                expected
            ),

        "gate_grid":
            GATES.tolist(),

        "utility_definition":
            "DeltaIoU100 metric-aligned marginal utility",

        "strict_interior_count":
            int(
                strict.sum()
            ),

        "strict_interior_rate":
            float(
                strict.mean()
            ),

        "nonflat_curve_count":
            int(
                nonflat.sum()
            ),

        "nonflat_curve_rate":
            float(
                nonflat.mean()
            ),

        "rich_curve_count":
            int(
                rich_curve.sum()
            ),

        "rich_curve_rate":
            float(
                rich_curve.mean()
            ),

        "mean_distinct_utility_levels":
            float(
                unique_count.mean()
            ),

        "soft_oracle_gate_histogram":
            gate_hist,

        "v2_4a_reconstruction":
            "PASS",

        "test_used":
            False,

        "output":
            str(out),
    }

    summary_path = (
        BASE
        /
        f"V2-4C-2A_{split}_gate_utility_curves_summary.json"
    )

    with open(
        summary_path,
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
        out,
    )

    print(
        "Saved JSON:",
        summary_path,
    )

    return summary


def main():

    print("=" * 110)
    print(
        "V2-4C-2A 11-POINT GATE UTILITY CURVE DATASET"
    )
    print("=" * 110)

    print(
        "Purpose:"
    )

    print(
        "Learn the complete gate-specific utility landscape,"
    )

    print(
        "not a single oracle best-gate label."
    )

    print(
        "Test used: NO"
    )

    train = process_split(
        "train"
    )

    val = process_split(
        "val"
    )

    print()
    print("=" * 110)
    print(
        "V2-4C-2A FINAL AUDIT"
    )
    print("=" * 110)

    print(
        "Train candidates        :",
        train[
            "candidates"
        ],
    )

    print(
        "Val candidates          :",
        val[
            "candidates"
        ],
    )

    print(
        "Train strict interior   :",
        f"{train['strict_interior_rate']:.6%}",
    )

    print(
        "Val strict interior     :",
        f"{val['strict_interior_rate']:.6%}",
    )

    print(
        "Train non-flat curves   :",
        f"{train['nonflat_curve_rate']:.6%}",
    )

    print(
        "Val non-flat curves     :",
        f"{val['nonflat_curve_rate']:.6%}",
    )

    print(
        "Test used               : NO"
    )

    print()
    print(
        "[V2-4C-2A GATE UTILITY CURVES] COMPLETE"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
