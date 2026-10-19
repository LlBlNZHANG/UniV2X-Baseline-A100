#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Paths
# ============================================================

ROOT = Path("/root/autodl-tmp/UniV2X")
sys.path.insert(0, str(ROOT))

G0_CACHE = ROOT / "G0-G1-G2/cache_full"

PLAN_CACHE = (
    ROOT
    / "G0-G1-G2/V2/V2-2_planning_task_consistency/cache_full"
)

VALID_FILE = (
    ROOT
    / "G0-G1-G2/G3/cache_valid_samples.txt"
)

V1_CKPT = (
    ROOT
    / "G0-G1-G2/G3/checkpoints_soft/g3_neighborhood_best.pth"
)

V2_CKPT = (
    ROOT
    / "G0-G1-G2/V2/V2-1_magnitude_learning/"
      "checkpoints/main_lambda_0p50/best.pth"
)

OUT_DIR = (
    ROOT
    / "G0-G1-G2/V2/V2-2_planning_task_consistency"
)

SEED = 2026
PLANNING_STEPS = 10

STRATEGIES = [
    "Ego",
    "OfficialOR",
    "V1",
    "V2",
]


# ============================================================
# Existing UniV2X / STCV components
# ============================================================

from projects.mmdet3d_plugin.univ2x.dense_heads.stcv_occ import (
    STCVOcc,
)

from projects.mmdet3d_plugin.univ2x.dense_heads.planning_head import (
    PlanningHeadSingleMode,
)

from projects.mmdet3d_plugin.univ2x.dense_heads.planning_head_plugin import (
    PlanningMetric,
)


# ============================================================
# V2 predictor
#
# Exact frozen V2-1 architecture:
#   127 -> 512
#   residual 512 -> 512
#   512 -> 256 -> 128
#   magnitude head + sign head
#
# Parameter count = 495,490
# ============================================================

class V2UtilityPredictor(nn.Module):

    def __init__(self, dropout=0.1):

        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(127, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.residual_block = nn.Sequential(
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

        self.magnitude_head = nn.Linear(128, 1)
        self.sign_head = nn.Linear(128, 1)

    def forward(self, x):

        x = self.input_proj(x)

        x = x + self.residual_block(x)

        x = self.down_256(x)
        x = self.down_128(x)

        mag_pred = self.magnitude_head(x).squeeze(-1)
        sign_logit = self.sign_head(x).squeeze(-1)

        return mag_pred, sign_logit


# ============================================================
# Helpers
# ============================================================

def normalize_array(x, ndim):

    x = np.asarray(x)

    while x.ndim > ndim and x.shape[0] == 1:
        x = x[0]

    return x


def build_exact_val_split():

    scene_to_samples = defaultdict(list)
    valid_sample_ids = set()

    with open(VALID_FILE, "r") as f:

        _ = next(f, None)

        for line in f:

            parts = line.strip().split("\t")

            if len(parts) < 3:
                continue

            sample_idx = parts[0]
            scene_token = parts[1]

            valid_sample_ids.add(sample_idx)
            scene_to_samples[scene_token].append(sample_idx)

    scenes = sorted(scene_to_samples.keys())

    assert len(scenes) == 21, (
        f"Expected 21 scenes, got {len(scenes)}"
    )

    np.random.seed(SEED)
    np.random.shuffle(scenes)

    train_scenes = set(scenes[:13])
    val_scenes = set(scenes[13:17])
    test_scenes = set(scenes[17:21])

    assert len(train_scenes) == 13
    assert len(val_scenes) == 4
    assert len(test_scenes) == 4

    return (
        valid_sample_ids,
        train_scenes,
        val_scenes,
        test_scenes,
    )


def build_replay_planner():
    """
    Do NOT reimplement UniV2X planning post-processing.

    Construct a minimal PlanningHead object and directly reuse:
        PlanningHead.collision_optimization()
        PlanningHead.drivable_optimization()

    Runtime parameters were audited in V2-2A.
    """

    planner = PlanningHeadSingleMode.__new__(
        PlanningHeadSingleMode
    )

    # Required because PlanningHead inherits nn.Module.
    nn.Module.__init__(planner)

    planner.planning_steps = 10

    # Occupancy output has current + 4 future = 5 horizons.
    planner.occ_n_future_only_occ = 4

    planner.bev_h = 200
    planner.bev_w = 200

    planner.occ_filter_range = 5.0
    planner.sigma = 1.0
    planner.alpha_collision = 5.0

    planner.eval()

    return planner


def replay_planning(
    planner,
    raw_traj,
    occ,
    drivable_pred,
):
    """
    Exact official post-processing sequence:

        T_raw
          -> collision_optimization(occupancy)
          -> drivable_optimization(drivable_pred)
    """

    assert tuple(raw_traj.shape) == (1, 10, 2)
    assert tuple(occ.shape) == (5, 200, 200)
    assert tuple(drivable_pred.shape) == (200, 200)

    occ_input = occ.unsqueeze(0)

    traj_col = planner.collision_optimization(
        raw_traj.clone(),
        occ_input,
    )

    traj_final = planner.drivable_optimization(
        traj_col.clone(),
        drivable_pred.clone(),
    )

    return traj_col, traj_final


def changed(a, b, atol=1e-6):

    return bool(
        torch.max(
            torch.abs(
                a.detach().cpu()
                - b.detach().cpu()
            )
        ).item()
        > atol
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test only the first N Val samples.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Path audit
    # --------------------------------------------------------

    for p in [
        G0_CACHE,
        PLAN_CACHE,
        VALID_FILE,
        V1_CKPT,
        V2_CKPT,
    ]:
        assert p.exists(), f"Missing required path: {p}"

    # --------------------------------------------------------
    # Exact V2-1 split
    # --------------------------------------------------------

    (
        valid_sample_ids,
        train_scenes,
        val_scenes,
        test_scenes,
    ) = build_exact_val_split()

    print("=" * 100)
    print("V2-2C COUNTERFACTUAL PLANNING EVALUATION")
    print("=" * 100)

    print("Seed               :", SEED)
    print("Valid sample IDs   :", len(valid_sample_ids))
    print("Train scenes       :", len(train_scenes))
    print("Val scenes         :", len(val_scenes))
    print("Test scenes        :", len(test_scenes))
    print("Val scene tokens   :", sorted(val_scenes))

    # --------------------------------------------------------
    # Resolve exact Val planning files
    # --------------------------------------------------------

    all_plan_files = sorted(
        PLAN_CACHE.glob("sample_*.npz")
    )

    assert len(all_plan_files) == 675, (
        f"Expected 675 planning caches, got "
        f"{len(all_plan_files)}"
    )

    val_files = []

    for p in all_plan_files:

        with np.load(
            p,
            allow_pickle=False,
        ) as d:

            sample_idx = str(
                d["sample_idx"].item()
            )

            scene_token = str(
                d["scene_token"].item()
            )

        if (
            sample_idx in valid_sample_ids
            and scene_token in val_scenes
        ):
            val_files.append(p)

    print("Val samples        :", len(val_files))

    assert len(val_files) == 96, (
        f"Expected 96 V2-1 Val samples, "
        f"got {len(val_files)}"
    )

    if args.limit is not None:

        assert args.limit > 0

        val_files = val_files[:args.limit]

        print(
            "SMOKE LIMIT        :",
            len(val_files),
        )

    # --------------------------------------------------------
    # V1
    # --------------------------------------------------------

    v1 = STCVOcc(
        checkpoint_path=str(V1_CKPT),
        threshold=0.70,
    )

    v1.eval()

    # --------------------------------------------------------
    # V2
    # --------------------------------------------------------

    v2 = V2UtilityPredictor(
        dropout=0.1
    )

    ckpt = torch.load(
        V2_CKPT,
        map_location="cpu",
    )

    assert "model" in ckpt, (
        "V2 checkpoint missing key: model"
    )

    v2.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    v2.eval()

    # --------------------------------------------------------
    # IMPORTANT:
    # Frozen V2-1C used CUDA batched inference.
    # Numerical audit proved that the hard zero threshold has
    # four CPU/CUDA boundary flips on the 18555 Val candidates.
    #
    # Therefore V2-2C must reproduce the frozen V2-1 execution
    # path exactly: CUDA + batch_size=2048.
    # --------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError(
            "V2-2C exact reproduction requires CUDA."
        )

    v2_device = torch.device("cuda")
    v2 = v2.to(v2_device)

    n_v2_params = sum(
        p.numel()
        for p in v2.parameters()
    )

    assert n_v2_params == 495490, (
        f"Expected 495490 V2 params, "
        f"got {n_v2_params}"
    )

    print("V2 parameters      :", n_v2_params)

    # --------------------------------------------------------
    # Exact UniV2X planning postprocessor
    # --------------------------------------------------------

    planner = build_replay_planner()

    print(
        "Planning runtime   :",
        "steps=10, occ_horizon=5, "
        "range=5.0m, sigma=1.0, "
        "alpha_collision=5.0",
    )

    # ========================================================
    # Frozen V2-1 exact prediction path
    #
    # Collect candidates/features in the exact Val file order,
    # then run one CUDA batched prediction stream.
    #
    # This reproduces the frozen V2-1C numerical path:
    #     device = CUDA
    #     batch_size = 2048
    #     decision = magnitude_pred > 0
    # ========================================================

    v2_feature_chunks = []
    v2_candidate_counts = []

    for p_plan in val_files:

        p_g0 = G0_CACHE / p_plan.name

        assert p_g0.exists(), (
            f"Missing G0 pair during V2 precompute: {p_g0}"
        )

        with np.load(
            p_g0,
            allow_pickle=False,
        ) as g0:

            pv = torch.from_numpy(
                normalize_array(
                    g0["Pv"], 3
                )
            ).float()

            pi = torch.from_numpy(
                normalize_array(
                    g0["Pi_aligned"], 3
                )
            ).float()

            ov = torch.from_numpy(
                normalize_array(
                    g0["Ov"], 3
                )
            ).bool()

            oi = torch.from_numpy(
                normalize_array(
                    g0["Oi"], 3
                )
            ).bool()

            warp = torch.from_numpy(
                normalize_array(
                    g0["warp_valid_mask"], 2
                )
            ).bool()

            (
                pre_candidate_indices,
                pre_features,
            ) = v1.extract_candidates_and_features(
                pv,
                pi,
                ov,
                oi,
                warp,
            )

        v2_candidate_counts.append(
            len(pre_candidate_indices)
        )

        if len(pre_candidate_indices) > 0:
            v2_feature_chunks.append(
                pre_features.cpu()
            )

    if len(v2_feature_chunks) > 0:
        v2_features_all = torch.cat(
            v2_feature_chunks,
            dim=0,
        )
    else:
        v2_features_all = torch.empty(
            0,
            127,
            dtype=torch.float32,
        )

    v2_mag_chunks = []

    with torch.no_grad():

        for i in range(
            0,
            len(v2_features_all),
            2048,
        ):

            x = v2_features_all[
                i:i + 2048
            ].to(
                v2_device,
                non_blocking=True,
            )

            mag_pred, _ = v2(x)

            v2_mag_chunks.append(
                mag_pred.detach().cpu()
            )

    if len(v2_mag_chunks) > 0:
        v2_mag_all = torch.cat(
            v2_mag_chunks,
            dim=0,
        )
    else:
        v2_mag_all = torch.empty(
            0,
            dtype=torch.float32,
        )

    v2_precomputed_accept = int(
        (v2_mag_all > 0.0).sum().item()
    )

    print(
        "V2 exact execution : CUDA, batch=2048"
    )

    print(
        "V2 precomputed cand:",
        len(v2_mag_all),
    )

    print(
        "V2 precomputed acc :",
        v2_precomputed_accept,
    )

    if args.limit is None:

        assert len(v2_mag_all) == 18555, (
            f"Expected 18555 Val V2 candidates, "
            f"got {len(v2_mag_all)}"
        )

        assert v2_precomputed_accept == 6156, (
            f"Frozen V2-1C decision mismatch: "
            f"{v2_precomputed_accept} != 6156"
        )

        print(
            "V2 frozen decision audit: PASS (6156)"
        )

    v2_cursor = 0

    # --------------------------------------------------------
    # Official PlanningMetric, one accumulator per strategy
    # --------------------------------------------------------

    metrics = {
        name: PlanningMetric(
            n_future=PLANNING_STEPS
        ).cpu()
        for name in STRATEGIES
    }

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    candidate_total = 0
    v1_accept_total = 0
    v2_accept_total = 0

    mechanism = {
        name: {
            "collision_changed_samples": 0,
            "drivable_changed_samples": 0,
            "final_changed_samples": 0,
        }
        for name in STRATEGIES
    }

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    for n, p_plan in enumerate(
        val_files,
        start=1,
    ):

        p_g0 = G0_CACHE / p_plan.name

        assert p_g0.exists(), (
            f"Missing G0 pair: {p_g0}"
        )

        with np.load(
            p_plan,
            allow_pickle=False,
        ) as plan, np.load(
            p_g0,
            allow_pickle=False,
        ) as g0:

            # -----------------------------------------------
            # Alignment audit
            # -----------------------------------------------

            ps = str(
                plan["sample_idx"].item()
            )
            gs = str(
                g0["sample_idx"].item()
            )

            pscene = str(
                plan["scene_token"].item()
            )
            gscene = str(
                g0["scene_token"].item()
            )

            pts = float(
                plan["timestamp"].item()
            )
            gts = float(
                g0["timestamp"].item()
            )

            assert ps == gs
            assert pscene == gscene

            assert np.isclose(
                pts,
                gts,
                atol=1e-6,
                rtol=0.0,
            )

            # -----------------------------------------------
            # G0 occupancy inputs
            # -----------------------------------------------

            pv_np = normalize_array(
                g0["Pv"], 3
            )

            pi_np = normalize_array(
                g0["Pi_aligned"], 3
            )

            ov_np = normalize_array(
                g0["Ov"], 3
            )

            oi_np = normalize_array(
                g0["Oi"], 3
            )

            official_np = normalize_array(
                g0["Oofficial"], 3
            )

            warp_np = normalize_array(
                g0["warp_valid_mask"], 2
            )

            assert pv_np.shape == (
                5, 200, 200
            )

            assert pi_np.shape == (
                5, 200, 200
            )

            assert ov_np.shape == (
                5, 200, 200
            )

            assert oi_np.shape == (
                5, 200, 200
            )

            assert official_np.shape == (
                5, 200, 200
            )

            assert warp_np.shape == (
                200, 200
            )

            pv = torch.from_numpy(
                pv_np
            ).float()

            pi = torch.from_numpy(
                pi_np
            ).float()

            ov = torch.from_numpy(
                ov_np
            ).bool()

            oi = torch.from_numpy(
                oi_np
            ).bool()

            official = torch.from_numpy(
                official_np
            ).bool()

            warp = torch.from_numpy(
                warp_np
            ).bool()

            # Official OR exact audit.
            assert torch.equal(
                official,
                ov | oi,
            ), (
                f"{p_plan.name}: "
                f"Oofficial != Ov | Oi"
            )

            # -----------------------------------------------
            # Exact shared candidate + 127-D feature
            # -----------------------------------------------

            (
                candidate_indices,
                features,
            ) = v1.extract_candidates_and_features(
                pv,
                pi,
                ov,
                oi,
                warp,
            )

            n_candidates = len(
                candidate_indices
            )

            candidate_total += n_candidates

            assert tuple(features.shape) == (
                n_candidates,
                127,
            )

            # -----------------------------------------------
            # V1:
            # sigmoid(logit) > 0.70
            # -----------------------------------------------

            if n_candidates > 0:

                with torch.no_grad():

                    v1_logits = v1.predictor(
                        features
                    )

                    v1_scores = torch.sigmoid(
                        v1_logits
                    )

                    v1_accept = (
                        v1_scores
                        > 0.70
                    )

                v1_accept_total += int(
                    v1_accept.sum().item()
                )

            else:

                v1_accept = torch.empty(
                    0,
                    dtype=torch.bool,
                )

            occ_v1 = v1.selective_fusion(
                ov,
                oi,
                candidate_indices,
                v1_accept,
            )

            # -----------------------------------------------
            # V2:
            # frozen V2-1 main decision
            # magnitude_pred > 0
            # -----------------------------------------------

            # Exact candidate-count/order consistency with the
            # frozen CUDA precomputation pass.
            assert n_candidates == v2_candidate_counts[
                n - 1
            ], (
                f"{p_plan.name}: candidate count changed "
                f"between V2 precompute and planning replay"
            )

            if n_candidates > 0:

                mag_pred = v2_mag_all[
                    v2_cursor:
                    v2_cursor + n_candidates
                ]

                assert len(mag_pred) == n_candidates

                v2_accept = (
                    mag_pred > 0.0
                )

                v2_cursor += n_candidates

                v2_accept_total += int(
                    v2_accept.sum().item()
                )

            else:

                v2_accept = torch.empty(
                    0,
                    dtype=torch.bool,
                )

            occ_v2 = v1.selective_fusion(
                ov,
                oi,
                candidate_indices,
                v2_accept,
            )

            # Ego-preserving audit.
            assert torch.all(
                occ_v1 | (~ov)
            )

            assert torch.all(
                occ_v2 | (~ov)
            )

            # -----------------------------------------------
            # Fixed raw planning information
            # -----------------------------------------------

            raw_traj = torch.from_numpy(
                normalize_array(
                    plan["T_raw"],
                    3,
                )
            ).float()

            drivable_pred = torch.from_numpy(
                normalize_array(
                    plan["drivable_pred"],
                    2,
                )
            )

            gt_traj = torch.from_numpy(
                normalize_array(
                    plan[
                        "sdc_planning_eval"
                    ],
                    3,
                )
            ).float()

            gt_mask = torch.from_numpy(
                normalize_array(
                    plan[
                        "sdc_planning_mask_eval"
                    ],
                    3,
                )
            ).float()

            segmentation = torch.from_numpy(
                normalize_array(
                    plan[
                        "gt_segmentation_eval"
                    ],
                    4,
                )
            ).long()

            drivable_gt = torch.from_numpy(
                normalize_array(
                    plan["drivable_gt"],
                    2,
                )
            ).long()

            assert tuple(raw_traj.shape) == (
                1, 10, 2
            )

            assert tuple(gt_traj.shape) == (
                1, 10, 2
            )

            assert tuple(gt_mask.shape) == (
                1, 10, 2
            )

            assert tuple(
                segmentation.shape
            ) == (
                1, 10, 200, 200
            )

            assert tuple(
                drivable_pred.shape
            ) == (
                200, 200
            )

            assert tuple(
                drivable_gt.shape
            ) == (
                200, 200
            )

            # -----------------------------------------------
            # Four occupancy strategies
            # -----------------------------------------------

            occ_dict = {
                "Ego": ov,
                "OfficialOR": official,
                "V1": occ_v1,
                "V2": occ_v2,
            }

            for name in STRATEGIES:

                try:

                    (
                        traj_col,
                        traj_final,
                    ) = replay_planning(
                        planner,
                        raw_traj,
                        occ_dict[name],
                        drivable_pred,
                    )

                except Exception as e:

                    raise RuntimeError(
                        f"Planning replay failed: "
                        f"sample={ps}, "
                        f"strategy={name}"
                    ) from e

                if changed(
                    raw_traj,
                    traj_col,
                ):
                    mechanism[
                        name
                    ][
                        "collision_changed_samples"
                    ] += 1

                if changed(
                    traj_col,
                    traj_final,
                ):
                    mechanism[
                        name
                    ][
                        "drivable_changed_samples"
                    ] += 1

                if changed(
                    raw_traj,
                    traj_final,
                ):
                    mechanism[
                        name
                    ][
                        "final_changed_samples"
                    ] += 1

                # PlanningMetric.update() mutates x coordinates,
                # therefore every tensor must be cloned.
                metrics[name].update(
                    traj_final.clone(),
                    gt_traj.clone(),
                    gt_mask.clone(),
                    segmentation.clone(),
                    drivable_gt.clone(),
                )

        if (
            n <= 3
            or n % 10 == 0
            or n == len(val_files)
        ):

            print(
                f"[{n:03d}/{len(val_files):03d}] "
                f"{p_plan.name} "
                f"sample={ps} "
                f"cand={n_candidates} "
                f"V1_accept="
                f"{int(v1_accept.sum())} "
                f"V2_accept="
                f"{int(v2_accept.sum())}"
            )

    # --------------------------------------------------------
    # Reconstruction audits
    # --------------------------------------------------------

    assert v2_cursor == len(v2_mag_all), (
        f"V2 prediction cursor mismatch: "
        f"{v2_cursor} != {len(v2_mag_all)}"
    )

    print()
    print("=" * 100)
    print("OCCUPANCY RECONSTRUCTION AUDIT")
    print("=" * 100)

    print(
        "Evaluated samples :",
        len(val_files),
    )

    print(
        "Candidates        :",
        candidate_total,
    )

    print(
        "V1 accepted       :",
        v1_accept_total,
    )

    print(
        "V2 accepted       :",
        v2_accept_total,
    )

    if args.limit is None:

        # Exact candidate count from frozen V2-1C Val audit.
        assert candidate_total == 18555, (
            f"Val candidate mismatch: "
            f"{candidate_total} != 18555"
        )

        print(
            "Candidate audit   : PASS "
            "(18555)"
        )

        # Known frozen Val operating points.
        print(
            "Expected V1       : 5691 "
            "(30.670978%)"
        )

        print(
            "Expected V2       : 6156 "
            "(33.177041%)"
        )

        if v1_accept_total == 5691:
            print(
                "V1 decision audit : PASS"
            )
        else:
            print(
                "V1 decision audit : WARNING "
                f"({v1_accept_total} != 5691)"
            )

        if v2_accept_total == 6156:
            print(
                "V2 decision audit : PASS"
            )
        else:
            print(
                "V2 decision audit : WARNING "
                f"({v2_accept_total} != 6156)"
            )

    # --------------------------------------------------------
    # Official PlanningMetric results
    # --------------------------------------------------------

    result_json = {
        "protocol": {
            "split": "V2-1 exact Validation split",
            "seed": SEED,
            "n_samples": len(val_files),
            "planning_steps": 10,
            "dt_seconds": 0.5,
            "candidate_total": candidate_total,
            "v1_accept_total": v1_accept_total,
            "v2_accept_total": v2_accept_total,
        },
        "mechanism": mechanism,
        "metrics": {},
    }

    print()
    print("=" * 100)
    print(
        "OFFICIAL PLANNING METRIC "
        "(LOWER IS BETTER)"
    )
    print("=" * 100)

    # 0.5 s per planning step:
    # indices 1,3,5,7,9 -> 1,2,3,4,5 s
    time_indices = {
        "1s": 1,
        "2s": 3,
        "3s": 5,
        "4s": 7,
        "5s": 9,
    }

    for name in STRATEGIES:

        result = metrics[name].compute()

        result_json["metrics"][name] = {}

        print()
        print(f"[{name}]")

        for metric_name in [
            "L2",
            "obj_col",
            "obj_box_col",
            "obj_out",
            "obj_box_out",
        ]:

            arr = (
                result[metric_name]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )

            assert arr.shape == (
                PLANNING_STEPS,
            )

            record = {
                "per_step": arr.tolist(),
                "mean_10_steps": float(
                    arr.mean()
                ),
            }

            for label, idx in (
                time_indices.items()
            ):
                record[label] = float(
                    arr[idx]
                )

            result_json[
                "metrics"
            ][name][metric_name] = record

            print(
                f"{metric_name:12s} "
                f"mean={arr.mean():.6f} | "
                f"1s={arr[1]:.6f} "
                f"2s={arr[3]:.6f} "
                f"3s={arr[5]:.6f} "
                f"4s={arr[7]:.6f} "
                f"5s={arr[9]:.6f}"
            )

        print(
            "postprocess    ",
            mechanism[name],
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if args.limit is None:

        out_path = (
            OUT_DIR
            / "V2-2C_val_counterfactual_planning.json"
        )

    else:

        out_path = (
            OUT_DIR
            / f"V2-2C_smoke_limit_{args.limit}.json"
        )

    with open(
        out_path,
        "w",
    ) as f:

        json.dump(
            result_json,
            f,
            indent=2,
        )

    print()
    print("=" * 100)
    print("Saved:", out_path)

    if args.limit is None:
        print(
            "[V2-2C VAL COUNTERFACTUAL] COMPLETE"
        )
    else:
        print(
            "[V2-2C SMOKE] COMPLETE"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
