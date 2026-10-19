#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/UniV2X")

BASE = (
    ROOT
    / "G0-G1-G2/V2/V2-4_learnable_adaptive_fusion"
)

CACHE_DIR = (
    ROOT
    / "G0-G1-G2/cache_full"
)

CONTEXT_SIZE = 12
PATCH_SIZE = 4
PAD = 4

EPS = 1e-12


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


def build_cache_map():

    print(
        "Building sample_idx -> cache mapping..."
    )

    mapping = {}

    cache_files = sorted(
        CACHE_DIR.glob(
            "sample_*.npz"
        )
    )

    assert len(cache_files) == 675

    for i, path in enumerate(
        cache_files
    ):

        with np.load(
            path,
            allow_pickle=False,
        ) as d:

            sample_idx = str(
                d[
                    "sample_idx"
                ].item()
            )

        assert sample_idx not in mapping

        mapping[
            sample_idx
        ] = path

        if (
            i < 3
            or
            (i + 1) % 100 == 0
            or
            i + 1 == len(cache_files)
        ):

            print(
                f"[{i+1:03d}/675] "
                f"{sample_idx} -> {path.name}"
            )

    assert len(mapping) == 675

    return mapping


def build_full_feature(
    pv_h,
    pi_h,
    warp,
):

    return np.stack(
        [
            pv_h,
            pi_h,
            pi_h - pv_h,
            np.abs(
                pi_h - pv_h
            ),
            warp.astype(
                np.float32
            ),
        ],
        axis=0,
    ).astype(
        np.float32
    )


def extract_context(
    full_feat,
    r,
    c,
):

    """
    12x12 context = current P4 patch + one P4 ring.

    Original desired area:
        y = 4r-4 ... 4r+7
        x = 4c-4 ... 4c+7

    Zero padding is used outside the 200x200 grid.
    """

    assert full_feat.shape == (
        5,
        200,
        200,
    )

    padded = np.pad(
        full_feat,
        (
            (0, 0),
            (PAD, PAD),
            (PAD, PAD),
        ),
        mode="constant",
        constant_values=0.0,
    )

    y0 = (
        PATCH_SIZE
        *
        int(r)
    )

    x0 = (
        PATCH_SIZE
        *
        int(c)
    )

    context = padded[
        :,
        y0:
        y0 + CONTEXT_SIZE,
        x0:
        x0 + CONTEXT_SIZE,
    ]

    assert context.shape == (
        5,
        12,
        12,
    )

    return context.astype(
        np.float32
    )


def process_split(
    split,
    expected_candidates,
    expected_accepted,
    expected_positive,
    cache_map,
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

    assert curve[
        "utility_curve"
    ].shape == (
        expected_candidates,
        11,
    )

    # Exact ordering audit.
    for key in [
        "sample_idx",
        "horizon",
        "patch_r",
        "patch_c",
    ]:

        assert np.array_equal(
            data[key].astype(str),
            curve[key].astype(str),
        ), f"{key} ordering mismatch"

    accepted_mask = (
        data[
            "v2_accept"
        ].astype(bool)
    )

    accepted_idx = np.flatnonzero(
        accepted_mask
    )

    assert len(
        accepted_idx
    ) == expected_accepted

    # -------------------------------------------------------------------------
    # Binary attenuation target:
    #
    # y=1 iff g=0.1 is strictly better than original V2 g=1.
    # -------------------------------------------------------------------------

    utility = curve[
        "utility_curve"
    ].astype(
        np.float32
    )

    true_gain_all = (
        utility[:, 1]
        -
        utility[:, 10]
    )

    target_all = (
        true_gain_all
        >
        EPS
    )

    target = target_all[
        accepted_idx
    ].astype(
        np.uint8
    )

    true_gain = true_gain_all[
        accepted_idx
    ].astype(
        np.float32
    )

    assert int(
        target.sum()
    ) == expected_positive

    # -------------------------------------------------------------------------
    # Accepted candidate metadata
    # -------------------------------------------------------------------------

    sample_idx = data[
        "sample_idx"
    ][
        accepted_idx
    ].astype(str)

    scene_token = data[
        "scene_token"
    ][
        accepted_idx
    ].astype(str)

    horizon = data[
        "horizon"
    ][
        accepted_idx
    ].astype(
        np.int8
    )

    patch_r = data[
        "patch_r"
    ][
        accepted_idx
    ].astype(
        np.int16
    )

    patch_c = data[
        "patch_c"
    ][
        accepted_idx
    ].astype(
        np.int16
    )

    features_127 = data[
        "features_127"
    ][
        accepted_idx
    ].astype(
        np.float32
    )

    v2_mag = data[
        "v2_mag"
    ][
        accepted_idx
    ].astype(
        np.float32
    )

    v2_sign_prob = data[
        "v2_sign_prob"
    ][
        accepted_idx
    ].astype(
        np.float32
    )

    # -------------------------------------------------------------------------
    # Group candidate rows by sample to avoid repeatedly opening cache files.
    # -------------------------------------------------------------------------

    by_sample = defaultdict(list)

    for local_i, sid in enumerate(
        sample_idx
    ):

        by_sample[
            sid
        ].append(
            local_i
        )

    contexts = np.zeros(
        (
            expected_accepted,
            5,
            12,
            12,
        ),
        dtype=np.float32,
    )

    print()
    print("=" * 112)
    print(
        f"V2-4C-3E2A BUILD {split.upper()} CONTEXT DATASET"
    )
    print("=" * 112)

    print(
        "All candidates          :",
        expected_candidates,
    )

    print(
        "V2 accepted             :",
        expected_accepted,
    )

    print(
        "Attenuation positives   :",
        int(
            target.sum()
        ),
        f"({target.mean():.6%})",
    )

    print(
        "Context                 :",
        "5 x 12 x 12",
    )

    print(
        "Channels                :",
        "[Pv, Pi, Pi-Pv, |Pi-Pv|, Mwarp]",
    )

    print(
        "Test used               : NO"
    )

    ordered_samples = list(
        by_sample.keys()
    )

    audit_done = False

    for sample_no, sid in enumerate(
        ordered_samples,
        start=1,
    ):

        assert sid in cache_map, (
            f"Missing cache for sample_idx={sid}"
        )

        path = cache_map[
            sid
        ]

        with np.load(
            path,
            allow_pickle=False,
        ) as d:

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

            warp = norm(
                d[
                    "warp_valid_mask"
                ],
                2,
            ).astype(
                bool
            )

            cache_scene = str(
                d[
                    "scene_token"
                ].item()
            )

        assert pv.shape == (
            5,
            200,
            200,
        )

        assert pi.shape == pv.shape

        assert warp.shape == (
            200,
            200,
        )

        local_rows = by_sample[
            sid
        ]

        for local_i in local_rows:

            h = int(
                horizon[
                    local_i
                ]
            )

            r = int(
                patch_r[
                    local_i
                ]
            )

            c = int(
                patch_c[
                    local_i
                ]
            )

            assert (
                scene_token[
                    local_i
                ]
                ==
                cache_scene
            )

            full_feat = (
                build_full_feature(
                    pv[h],
                    pi[h],
                    warp,
                )
            )

            context = (
                extract_context(
                    full_feat,
                    r,
                    c,
                )
            )

            contexts[
                local_i
            ] = context

            # -------------------------------------------------------------
            # One-time exact audit:
            # center 4x4 of 12x12 context must equal the original local-80
            # feature contained in features_127.
            # -------------------------------------------------------------

            if not audit_done:

                center = context[
                    :,
                    4:8,
                    4:8,
                ]

                local80 = (
                    features_127[
                        local_i,
                        :80,
                    ]
                    .reshape(
                        5,
                        4,
                        4,
                    )
                )

                max_diff = float(
                    np.max(
                        np.abs(
                            center
                            -
                            local80
                        )
                    )
                )

                print()
                print(
                    "Center/local80 max diff:",
                    f"{max_diff:.12g}",
                )

                assert max_diff == 0.0

                audit_done = True

        if (
            sample_no <= 3
            or
            sample_no % 50 == 0
            or
            sample_no
            ==
            len(
                ordered_samples
            )
        ):

            done_candidates = sum(
                len(
                    by_sample[x]
                )
                for x in
                ordered_samples[
                    :sample_no
                ]
            )

            print(
                f"[{sample_no:03d}/"
                f"{len(ordered_samples):03d}] "
                f"sample={sid} "
                f"accepted={len(local_rows)} "
                f"running={done_candidates}"
            )

    assert np.isfinite(
        contexts
    ).all()

    assert contexts.shape == (
        expected_accepted,
        5,
        12,
        12,
    )

    # -------------------------------------------------------------------------
    # Context diagnostics
    # -------------------------------------------------------------------------

    center = contexts[
        :,
        :,
        4:8,
        4:8,
    ]

    ring_mask = np.ones(
        (
            12,
            12,
        ),
        dtype=bool,
    )

    ring_mask[
        4:8,
        4:8,
    ] = False

    ring_abs_delta = np.abs(
        contexts[
            :,
            2,
            :,
            :,
        ][
            :,
            ring_mask
        ]
    )

    print()
    print("-" * 112)
    print(
        "CONTEXT AUDIT"
    )
    print("-" * 112)

    print(
        "context shape           :",
        contexts.shape,
    )

    print(
        "Pv range                :",
        f"[{contexts[:,0].min():.6f}, "
        f"{contexts[:,0].max():.6f}]",
    )

    print(
        "Pi range                :",
        f"[{contexts[:,1].min():.6f}, "
        f"{contexts[:,1].max():.6f}]",
    )

    print(
        "Delta range             :",
        f"[{contexts[:,2].min():.6f}, "
        f"{contexts[:,2].max():.6f}]",
    )

    print(
        "Warp mean               :",
        f"{contexts[:,4].mean():.6f}",
    )

    print(
        "Mean |Pi-Pv| center     :",
        f"{np.abs(center[:,2]).mean():.6f}",
    )

    print(
        "Mean |Pi-Pv| outer ring :",
        f"{ring_abs_delta.mean():.6f}",
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    out_file = (
        BASE
        /
        f"V2-4C-3E2A_{split}_context_dataset.npz"
    )

    np.savez_compressed(
        out_file,

        context_5x12x12=(
            contexts
        ),

        features_127=(
            features_127
        ),

        v2_mag=(
            v2_mag
        ),

        v2_sign_prob=(
            v2_sign_prob
        ),

        attenuation_target=(
            target
        ),

        true_binary_gain=(
            true_gain
        ),

        accepted_global_index=(
            accepted_idx.astype(
                np.int32
            )
        ),

        sample_idx=(
            sample_idx
        ),

        scene_token=(
            scene_token
        ),

        horizon=(
            horizon
        ),

        patch_r=(
            patch_r
        ),

        patch_c=(
            patch_c
        ),
    )

    summary = {
        "stage":
            "V2-4C-3E2A Context Dataset",

        "split":
            split,

        "all_candidates":
            int(
                expected_candidates
            ),

        "v2_accepted":
            int(
                expected_accepted
            ),

        "attenuation_positive":
            int(
                target.sum()
            ),

        "attenuation_positive_rate":
            float(
                target.mean()
            ),

        "context_shape":
            [
                5,
                12,
                12,
            ],

        "channels":
            [
                "Pv",
                "Pi",
                "Pi-Pv",
                "|Pi-Pv|",
                "warp_valid",
            ],

        "center_local80_equivalence":
            "PASS",

        "test_used":
            False,

        "output":
            str(
                out_file
            ),
    }

    json_file = (
        BASE
        /
        f"V2-4C-3E2A_{split}_context_dataset_summary.json"
    )

    with open(
        json_file,
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
        out_file,
    )

    print(
        "Saved JSON:",
        json_file,
    )

    return summary


def main():

    print("=" * 112)
    print(
        "V2-4C-3E2A RAW SPATIAL CONTEXT DATASET"
    )
    print("=" * 112)

    print(
        "Purpose:"
    )

    print(
        "Preserve real 12x12 cell-level spatial structure"
    )

    print(
        "for the binary attenuation decision."
    )

    print()
    print(
        "Training : NONE"
    )

    print(
        "Test used: NO"
    )

    cache_map = (
        build_cache_map()
    )

    train = process_split(
        "train",
        expected_candidates=86143,
        expected_accepted=19878,
        expected_positive=8547,
        cache_map=cache_map,
    )

    val = process_split(
        "val",
        expected_candidates=18555,
        expected_accepted=6156,
        expected_positive=3361,
        cache_map=cache_map,
    )

    print()
    print("=" * 112)
    print(
        "V2-4C-3E2A FINAL AUDIT"
    )
    print("=" * 112)

    print(
        "Train accepted          :",
        train[
            "v2_accepted"
        ],
    )

    print(
        "Val accepted            :",
        val[
            "v2_accepted"
        ],
    )

    print(
        "Train positive          :",
        train[
            "attenuation_positive"
        ],
    )

    print(
        "Val positive            :",
        val[
            "attenuation_positive"
        ],
    )

    print(
        "Context                 :",
        "5 x 12 x 12",
    )

    print(
        "Center/local80 audit    : PASS"
    )

    print(
        "Test used               : NO"
    )

    print()
    print(
        "[V2-4C-3E2A CONTEXT DATASET] COMPLETE"
    )

    print("=" * 112)


if __name__ == "__main__":
    main()
