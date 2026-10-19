#!/usr/bin/env python3

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/root/autodl-tmp/UniV2X")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from projects.mmdet3d_plugin.univ2x.dense_heads.stcv_occ import (
    STCVFeatureExtractor,
)


# =============================================================================
# Paths
# =============================================================================

BASE = (
    ROOT
    / "G0-G1-G2/V2/V2-4_learnable_adaptive_fusion"
)

CACHE_DIR = (
    ROOT
    / "G0-G1-G2/cache_full"
)

VALID_FILE = (
    ROOT
    / "G0-G1-G2/G3/cache_valid_samples.txt"
)

V2_TRAIN_SCRIPT = (
    ROOT
    / "G0-G1-G2/V2/V2-1_magnitude_learning/"
      "train_v2_magnitude_utility.py"
)

V2_CKPT = (
    ROOT
    / "G0-G1-G2/V2/V2-1_magnitude_learning/"
      "checkpoints/main_lambda_0p50/best.pth"
)

SEED = 2026
PATCH_SIZE = 4
SEG_THRESH = 0.1


# =============================================================================
# Import exact frozen V2 model
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "v2_mag_module",
    V2_TRAIN_SCRIPT,
)

v2mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2mod)

V2UtilityPredictor = (
    v2mod.V2UtilityPredictor
)


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
        f"Expected ndim={ndim}, "
        f"got {x.shape}"
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


def find_oracle_file(split):

    candidates = [
        BASE
        / f"V2-4-0A_{split}_soft_oracle_candidates.npz",

        BASE
        / f"V2-4A_{split}_soft_oracle_candidates.npz",

        ROOT
        / "G0-G1-G2/V3/V3-0_soft_fusion_oracle"
        / f"V3-0A_{split}_soft_oracle_candidates.npz",
    ]

    for p in candidates:

        if p.exists():

            print(
                f"[Oracle] {split}: {p}"
            )

            return p

    raise FileNotFoundError(
        "Cannot find V2-4A oracle NPZ. "
        f"Tried: {candidates}"
    )


def binary_counts(pred, gt):

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

    return tp, fp, fn


def range30_fraction(r, c):

    y0 = PATCH_SIZE * r
    y1 = y0 + PATCH_SIZE

    x0 = PATCH_SIZE * c
    x1 = x0 + PATCH_SIZE

    iy0 = max(
        y0,
        70,
    )

    iy1 = min(
        y1,
        130,
    )

    ix0 = max(
        x0,
        70,
    )

    ix1 = min(
        x1,
        130,
    )

    h = max(
        0,
        iy1 - iy0,
    )

    w = max(
        0,
        ix1 - ix0,
    )

    return float(
        h * w
        /
        16.0
    )


def exact_feature_fast(
    extractor,
    full_feat,
    mean_grid,
    h,
    r,
    c,
):
    """
    Same 127-D definition as STCVFeatureExtractor.extract_127_features,
    but reuses per-horizon full_feat/mean_grid for efficiency.
    """

    local_80 = (
        extractor.extract_local_80(
            full_feat,
            r,
            c,
        )
    )

    horizon_5 = (
        extractor.extract_horizon_5(
            h
        ).to(
            device=full_feat.device,
            dtype=full_feat.dtype,
        )
    )

    spatial_2 = (
        extractor.extract_spatial_2(
            r,
            c,
        ).to(
            device=full_feat.device,
            dtype=full_feat.dtype,
        )
    )

    neighborhood_40 = (
        extractor.extract_neighborhood_40(
            mean_grid,
            r,
            c,
            full_feat.device,
            full_feat.dtype,
        )
    )

    feat = torch.cat(
        [
            local_80,
            horizon_5,
            spatial_2,
            neighborhood_40,
        ],
        dim=0,
    )

    assert feat.shape == (127,)

    return feat


# =============================================================================
# Frozen V2 inference
# =============================================================================

def infer_v2(
    features,
):

    if not torch.cuda.is_available():

        raise RuntimeError(
            "Canonical V2 inference requires CUDA."
        )

    device = torch.device(
        "cuda"
    )

    model = V2UtilityPredictor(
        dropout=0.1
    ).to(
        device
    )

    ckpt = torch.load(
        V2_CKPT,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    model.eval()

    mag_all = []
    sign_all = []

    with torch.no_grad():

        for start in range(
            0,
            len(features),
            2048,
        ):

            x = torch.from_numpy(
                features[
                    start:
                    start + 2048
                ]
            ).to(
                device,
                non_blocking=True,
            )

            (
                mag,
                sign_logit,
            ) = model(x)

            mag_all.append(
                mag.detach()
                .cpu()
                .numpy()
            )

            sign_all.append(
                torch.sigmoid(
                    sign_logit
                )
                .detach()
                .cpu()
                .numpy()
            )

    mag = np.concatenate(
        mag_all
    ).astype(
        np.float32
    )

    sign_prob = np.concatenate(
        sign_all
    ).astype(
        np.float32
    )

    return (
        mag,
        sign_prob,
    )


# =============================================================================
# Build split dataset
# =============================================================================

def build_split(
    split_name,
    split_scenes,
    valid_sample_ids,
):

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

    oracle_file = find_oracle_file(
        split_name
    )

    with np.load(
        oracle_file,
        allow_pickle=False,
    ) as d:

        oracle = {
            k: d[k].copy()
            for k in d.files
        }

    assert len(
        oracle["sample_idx"]
    ) == expected_candidates


    extractor = STCVFeatureExtractor()

    features_127 = []

    pv_patch_all = []
    pi_patch_all = []

    ov_patch_all = []
    oi_patch_all = []

    warp_patch_all = []
    gt_patch_all = []

    candidate_add_patch_all = []

    sample_idx_all = []
    scene_token_all = []

    horizon_all = []
    patch_r_all = []
    patch_c_all = []

    range30_fraction_all = []

    ego_tp_100_all = []
    ego_fp_100_all = []
    ego_fn_100_all = []

    n_samples = 0
    n_candidates = 0

    feature_audit_done = False


    cache_files = sorted(
        CACHE_DIR.glob(
            "sample_*.npz"
        )
    )


    print()
    print("=" * 110)
    print(
        f"V2-4B BUILD {split_name.upper()} DATASET"
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

            pv_np = norm(
                d["Pv"],
                3,
            ).astype(
                np.float32
            )

            pi_np = norm(
                d["Pi_aligned"],
                3,
            ).astype(
                np.float32
            )

            ov_np = norm(
                d["Ov"],
                3,
            ).astype(
                bool
            )

            oi_np = norm(
                d["Oi"],
                3,
            ).astype(
                bool
            )

            gt_np = norm(
                d["GT"],
                3,
            ).astype(
                bool
            )

            warp_np = norm(
                d[
                    "warp_valid_mask"
                ],
                2,
            ).astype(
                bool
            )


        assert pv_np.shape == (
            5,
            200,
            200,
        )

        assert pi_np.shape == pv_np.shape
        assert ov_np.shape == pv_np.shape
        assert oi_np.shape == pv_np.shape
        assert gt_np.shape == pv_np.shape

        assert warp_np.shape == (
            200,
            200,
        )


        pv = torch.from_numpy(
            pv_np
        ).float()

        pi = torch.from_numpy(
            pi_np
        ).float()

        warp = torch.from_numpy(
            warp_np
        ).bool()


        sample_candidates = 0


        for h in range(5):

            # -------------------------------------------------------------
            # Full 100m Ego statistics for the exact V2 metric-aligned
            # horizon context. Kept for later loss/diagnostic design.
            # -------------------------------------------------------------

            tp100, fp100, fn100 = (
                binary_counts(
                    ov_np[h],
                    gt_np[h],
                )
            )


            # -------------------------------------------------------------
            # Reuse feature maps once per horizon.
            # -------------------------------------------------------------

            full_feat = (
                extractor.build_full_feature(
                    pv[h],
                    pi[h],
                    warp,
                )
            )

            mean_grid = (
                extractor.build_patch_mean_grid(
                    full_feat
                )
            )


            for r in range(50):

                y0 = 4 * r
                y1 = y0 + 4

                for c in range(50):

                    x0 = 4 * c
                    x1 = x0 + 4


                    ov_patch = (
                        ov_np[
                            h,
                            y0:y1,
                            x0:x1,
                        ]
                    )

                    oi_patch = (
                        oi_np[
                            h,
                            y0:y1,
                            x0:x1,
                        ]
                    )

                    warp_patch = (
                        warp_np[
                            y0:y1,
                            x0:x1,
                        ]
                    )


                    # Exact frozen candidate definition.
                    add_patch = (
                        np.logical_not(
                            ov_patch
                        )
                        &
                        oi_patch
                        &
                        warp_patch
                    )


                    if not bool(
                        add_patch.any()
                    ):
                        continue


                    feat = (
                        exact_feature_fast(
                            extractor,
                            full_feat,
                            mean_grid,
                            h,
                            r,
                            c,
                        )
                    )


                    # -----------------------------------------------------
                    # One-time feature equivalence audit against official
                    # extractor path.
                    # -----------------------------------------------------

                    if not feature_audit_done:

                        feat_ref = (
                            extractor.extract_127_features(
                                pv,
                                pi,
                                warp,
                                h,
                                r,
                                c,
                            )
                        )

                        max_diff = float(
                            (
                                feat
                                -
                                feat_ref
                            )
                            .abs()
                            .max()
                            .item()
                        )

                        print(
                            "Feature equivalence "
                            f"max diff: {max_diff:.12g}"
                        )

                        assert max_diff == 0.0

                        feature_audit_done = True


                    features_127.append(
                        feat.numpy()
                        .astype(
                            np.float32
                        )
                    )


                    pv_patch_all.append(
                        pv_np[
                            h,
                            y0:y1,
                            x0:x1,
                        ].copy()
                    )

                    pi_patch_all.append(
                        pi_np[
                            h,
                            y0:y1,
                            x0:x1,
                        ].copy()
                    )


                    ov_patch_all.append(
                        ov_patch.astype(
                            np.uint8
                        )
                    )

                    oi_patch_all.append(
                        oi_patch.astype(
                            np.uint8
                        )
                    )

                    warp_patch_all.append(
                        warp_patch.astype(
                            np.uint8
                        )
                    )

                    gt_patch_all.append(
                        gt_np[
                            h,
                            y0:y1,
                            x0:x1,
                        ].astype(
                            np.uint8
                        )
                    )

                    candidate_add_patch_all.append(
                        add_patch.astype(
                            np.uint8
                        )
                    )


                    sample_idx_all.append(
                        sample_idx
                    )

                    scene_token_all.append(
                        scene_token
                    )

                    horizon_all.append(h)
                    patch_r_all.append(r)
                    patch_c_all.append(c)

                    range30_fraction_all.append(
                        range30_fraction(
                            r,
                            c,
                        )
                    )

                    ego_tp_100_all.append(
                        tp100
                    )

                    ego_fp_100_all.append(
                        fp100
                    )

                    ego_fn_100_all.append(
                        fn100
                    )


                    n_candidates += 1
                    sample_candidates += 1


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
                f"running={n_candidates}"
            )


    # =========================================================================
    # Convert arrays
    # =========================================================================

    features_127 = np.asarray(
        features_127,
        dtype=np.float32,
    )

    pv_patch_all = np.asarray(
        pv_patch_all,
        dtype=np.float32,
    )

    pi_patch_all = np.asarray(
        pi_patch_all,
        dtype=np.float32,
    )

    ov_patch_all = np.asarray(
        ov_patch_all,
        dtype=np.uint8,
    )

    oi_patch_all = np.asarray(
        oi_patch_all,
        dtype=np.uint8,
    )

    warp_patch_all = np.asarray(
        warp_patch_all,
        dtype=np.uint8,
    )

    gt_patch_all = np.asarray(
        gt_patch_all,
        dtype=np.uint8,
    )

    candidate_add_patch_all = np.asarray(
        candidate_add_patch_all,
        dtype=np.uint8,
    )


    assert n_samples == expected_samples

    assert n_candidates == (
        expected_candidates
    )

    assert features_127.shape == (
        expected_candidates,
        127,
    )

    for arr in [
        pv_patch_all,
        pi_patch_all,
        ov_patch_all,
        oi_patch_all,
        warp_patch_all,
        gt_patch_all,
        candidate_add_patch_all,
    ]:

        assert arr.shape == (
            expected_candidates,
            4,
            4,
        )


    # =========================================================================
    # Oracle ordering alignment
    # =========================================================================

    sample_idx_np = np.asarray(
        sample_idx_all
    )

    scene_token_np = np.asarray(
        scene_token_all
    )

    horizon_np = np.asarray(
        horizon_all,
        dtype=np.int8,
    )

    patch_r_np = np.asarray(
        patch_r_all,
        dtype=np.int16,
    )

    patch_c_np = np.asarray(
        patch_c_all,
        dtype=np.int16,
    )


    assert np.array_equal(
        sample_idx_np.astype(str),
        oracle[
            "sample_idx"
        ].astype(str),
    ), "Oracle sample_idx ordering mismatch"


    assert np.array_equal(
        horizon_np.astype(np.int64),
        oracle[
            "horizon"
        ].astype(np.int64),
    ), "Oracle horizon ordering mismatch"


    assert np.array_equal(
        patch_r_np.astype(np.int64),
        oracle[
            "patch_r"
        ].astype(np.int64),
    ), "Oracle patch_r ordering mismatch"


    assert np.array_equal(
        patch_c_np.astype(np.int64),
        oracle[
            "patch_c"
        ].astype(np.int64),
    ), "Oracle patch_c ordering mismatch"


    print()
    print(
        "Oracle candidate alignment : PASS"
    )


    # =========================================================================
    # Exact frozen V2 predictor
    # =========================================================================

    print()
    print(
        "Running frozen V2 predictor "
        "(CUDA, batch=2048)..."
    )


    (
        v2_mag,
        v2_sign_prob,
    ) = infer_v2(
        features_127
    )


    assert v2_mag.shape == (
        expected_candidates,
    )

    assert v2_sign_prob.shape == (
        expected_candidates,
    )


    v2_accept = (
        v2_mag > 0.0
    )


    if split_name == "val":

        assert int(
            v2_accept.sum()
        ) == 6156, (
            "Frozen Val V2 decision "
            f"mismatch: {int(v2_accept.sum())} "
            "!= 6156"
        )


    # =========================================================================
    # Data diagnostics
    # =========================================================================

    strict_interior = (
        oracle[
            "strict_interior"
        ].astype(bool)
    )

    soft_best_gate = (
        oracle[
            "soft_best_gate"
        ].astype(
            np.float32
        )
    )

    hard_best_gate = (
        oracle[
            "hard_best_gate"
        ].astype(
            np.float32
        )
    )


    print()
    print("=" * 110)
    print(
        f"{split_name.upper()} DATASET AUDIT"
    )
    print("=" * 110)

    print(
        "Samples                 :",
        n_samples,
    )

    print(
        "Candidates              :",
        n_candidates,
    )

    print(
        "features_127            :",
        features_127.shape,
    )

    print(
        "Pv/Pi patches           :",
        pv_patch_all.shape,
    )

    print(
        "GT patches              :",
        gt_patch_all.shape,
    )

    print(
        "V2 accepted             :",
        int(
            v2_accept.sum()
        ),
        f"({v2_accept.mean():.6%})",
    )

    print(
        "Strict soft-oracle      :",
        int(
            strict_interior.sum()
        ),
        f"({strict_interior.mean():.6%})",
    )

    print(
        "Pv range                :",
        f"[{pv_patch_all.min():.6f}, "
        f"{pv_patch_all.max():.6f}]",
    )

    print(
        "Pi range                :",
        f"[{pi_patch_all.min():.6f}, "
        f"{pi_patch_all.max():.6f}]",
    )

    print(
        "GT positive rate        :",
        f"{gt_patch_all.mean():.6%}",
    )

    print(
        "Candidate-add cell rate :",
        f"{candidate_add_patch_all.mean():.6%}",
    )


    # =========================================================================
    # Save
    # =========================================================================

    out_file = (
        BASE
        /
        f"V2-4B_{split_name}_learnable_fusion_dataset.npz"
    )


    np.savez_compressed(

        out_file,

        # -------------------------------------------------------------
        # Learnable gate input
        # -------------------------------------------------------------

        features_127=features_127,

        v2_mag=v2_mag,

        v2_sign_prob=v2_sign_prob,

        # -------------------------------------------------------------
        # Differentiable fusion tensors
        # -------------------------------------------------------------

        Pv_patch=pv_patch_all,

        Pi_patch=pi_patch_all,

        Ov_patch=ov_patch_all,

        Oi_patch=oi_patch_all,

        warp_patch=warp_patch_all,

        GT_patch=gt_patch_all,

        candidate_add_patch=(
            candidate_add_patch_all
        ),

        # -------------------------------------------------------------
        # Metric-aligned context
        # -------------------------------------------------------------

        ego_tp_100=np.asarray(
            ego_tp_100_all,
            dtype=np.int32,
        ),

        ego_fp_100=np.asarray(
            ego_fp_100_all,
            dtype=np.int32,
        ),

        ego_fn_100=np.asarray(
            ego_fn_100_all,
            dtype=np.int32,
        ),

        range30_fraction=np.asarray(
            range30_fraction_all,
            dtype=np.float32,
        ),

        # -------------------------------------------------------------
        # Metadata
        # -------------------------------------------------------------

        sample_idx=sample_idx_np,

        scene_token=scene_token_np,

        horizon=horizon_np,

        patch_r=patch_r_np,

        patch_c=patch_c_np,

        # -------------------------------------------------------------
        # Frozen V2 decision diagnostic
        # -------------------------------------------------------------

        v2_accept=(
            v2_accept.astype(
                np.uint8
            )
        ),

        # -------------------------------------------------------------
        # Oracle fields:
        # ANALYSIS ONLY, NOT direct training labels.
        # -------------------------------------------------------------

        oracle_soft_best_gate=(
            soft_best_gate
        ),

        oracle_hard_best_gate=(
            hard_best_gate
        ),

        oracle_soft_best_utility=(
            oracle[
                "soft_best_utility"
            ].astype(
                np.float32
            )
        ),

        oracle_hard_best_utility=(
            oracle[
                "hard_best_utility"
            ].astype(
                np.float32
            )
        ),

        oracle_strict_interior=(
            oracle[
                "strict_interior"
            ].astype(
                np.uint8
            )
        ),

        oracle_strict_gain=(
            oracle[
                "strict_gain"
            ].astype(
                np.float32
            )
        ),

        seg_threshold=np.asarray(
            SEG_THRESH,
            dtype=np.float32,
        ),
    )


    summary = {
        "stage":
            "V2-4B Learnable Fusion Dataset",

        "split":
            split_name,

        "samples":
            int(
                n_samples
            ),

        "candidates":
            int(
                n_candidates
            ),

        "feature_dim":
            127,

        "gate_condition_dim_future":
            129,

        "patch_shape":
            [4, 4],

        "v2_accept_count":
            int(
                v2_accept.sum()
            ),

        "v2_accept_rate":
            float(
                v2_accept.mean()
            ),

        "strict_soft_oracle_count":
            int(
                strict_interior.sum()
            ),

        "strict_soft_oracle_rate":
            float(
                strict_interior.mean()
            ),

        "training_fields": [
            "features_127",
            "v2_mag",
            "v2_sign_prob",
            "Pv_patch",
            "Pi_patch",
            "warp_patch",
            "GT_patch",
            "candidate_add_patch",
        ],

        "oracle_fields_are_analysis_only":
            True,

        "test_used":
            False,

        "output":
            str(
                out_file
            ),
    }


    summary_file = (
        BASE
        /
        f"V2-4B_{split_name}_learnable_fusion_dataset_summary.json"
    )


    with open(
        summary_file,
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
        summary_file,
    )


    return summary


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
        "V2-4B LEARNABLE ADAPTIVE FUSION DATASET"
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

    print()
    print(
        "Training principle:"
    )

    print(
        "soft_best_gate is saved for "
        "analysis only; it is NOT a direct label."
    )

    print(
        "Future gate training will optimize "
        "the fused occupancy prediction itself."
    )


    train_summary = build_split(
        "train",
        train_scenes,
        valid_sample_ids,
    )


    val_summary = build_split(
        "val",
        val_scenes,
        valid_sample_ids,
    )


    print()
    print("=" * 110)
    print(
        "FINAL DATASET AUDIT"
    )
    print("=" * 110)

    print(
        "Train candidates       :",
        train_summary[
            "candidates"
        ],
    )

    print(
        "Val candidates         :",
        val_summary[
            "candidates"
        ],
    )

    print(
        "Train gate input       :",
        "127 + V2 mag + V2 sign = 129-D",
    )

    print(
        "Train fusion tensors   :",
        "Pv/Pi/warp/GT, each P4",
    )

    print(
        "Test used              : NO"
    )

    print()
    print(
        "[V2-4B LEARNABLE FUSION DATASET] COMPLETE"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
