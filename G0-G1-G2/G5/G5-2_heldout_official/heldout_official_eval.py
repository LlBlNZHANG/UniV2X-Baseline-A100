#!/usr/bin/env python3
"""
Held-out Scene Official Occupancy Evaluation

Purpose
-------
Evaluate Official OR or STCV-Occ on the four G3 held-out scenes using
the exact UniV2X Occupancy metric implementation:

    IoU / PQ / SQ / RQ
    30x30 and 100x100 evaluation ranges

Held-out scenes:
    0007, 0021, 0071, 0088

Important:
- Forward ALL frames of these four scenes to preserve temporal state.
- Only frames satisfying official gt_occ_has_invalid_frame=False
  are accumulated into Occupancy metrics.
- Expected:
      full forwarded frames = 116
      official-valid frames = 92
- The 86 G3 Test samples are NOT used as the evaluation filter.
  Six additional official-valid frames without STCV candidates are retained.
"""

import os
import sys
import json
import argparse
import importlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path("/root/autodl-tmp/UniV2X")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.utils import import_modules_from_strings

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent
from projects.mmdet3d_plugin.univ2x.dense_heads.occ_head_plugin import (
    IntersectionOverUnion,
    PanopticMetric,
)


# ============================================================
# Fixed experiment configuration
# ============================================================

BASELINE_CONFIG = (
    ROOT / "projects/configs_e2e_univ2x/univ2x_coop_e2e.py"
)

STCV_CONFIG = (
    ROOT / "projects/configs_e2e_univ2x/univ2x_coop_e2e_stcv.py"
)

OFFICIAL_CKPT = (
    ROOT / "ckpts/univ2x_coop_e2e_stg2.pth"
)

STCV_CKPT = (
    ROOT / "G0-G1-G2/G3/checkpoints_soft/g3_neighborhood_best.pth"
)

CACHE_DIR = ROOT / "G0-G1-G2/cache_full"

OUTPUT_DIR = (
    ROOT / "G0-G1-G2/G5/G5-2_heldout_official"
)

TARGET_SCENES = {"0007", "0021", "0071", "0088"}

EXPECTED_TOTAL_FORWARD = 116
EXPECTED_VALID = 92
TAU = 0.70

EVALUATION_RANGES = {
    "30x30": (70, 130),
    "100x100": (0, 200),
}

N_CLASSES = 2


# ============================================================
# Utilities
# ============================================================

def canon(x):
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode()
    return str(x)


def load_heldout_indices():
    """
    Derive the complete four-scene frame ranges from the trusted
    675-sample G0 cache.

    Also recover the expected official Occupancy-valid mask from
    future_valid_mask, previously audited against UniV2X evaluator logic.
    """

    files = sorted(CACHE_DIR.glob("sample_*.npz"))

    assert len(files) == 675, (
        f"Expected 675 G0 caches, found {len(files)}"
    )

    target_indices = []
    expected_valid = {}
    idx_to_scene = {}
    scene_indices = defaultdict(list)

    for p in files:
        idx = int(p.stem.split("_")[-1])

        with np.load(p, allow_pickle=True) as d:
            scene = canon(d["scene_token"])

            if scene not in TARGET_SCENES:
                continue

            valid = bool(
                np.asarray(
                    d["future_valid_mask"]
                ).astype(bool).all()
            )

            target_indices.append(idx)
            expected_valid[idx] = valid
            idx_to_scene[idx] = scene
            scene_indices[scene].append(idx)

    target_indices = sorted(target_indices)

    assert len(target_indices) == EXPECTED_TOTAL_FORWARD, (
        f"Expected {EXPECTED_TOTAL_FORWARD} held-out frames, "
        f"found {len(target_indices)}"
    )

    n_expected_valid = sum(expected_valid.values())

    assert n_expected_valid == EXPECTED_VALID, (
        f"Expected {EXPECTED_VALID} official-valid frames, "
        f"found {n_expected_valid}"
    )

    print("=" * 90)
    print("HELD-OUT SCENE PROTOCOL")
    print("=" * 90)

    for scene in sorted(TARGET_SCENES):
        ids = sorted(scene_indices[scene])

        scene_valid = sum(
            expected_valid[i] for i in ids
        )

        contiguous = (
            ids
            == list(range(min(ids), max(ids) + 1))
        )

        print(
            f"Scene {scene}: "
            f"idx {min(ids)}->{max(ids)}, "
            f"total={len(ids)}, "
            f"valid={scene_valid}, "
            f"contiguous={contiguous}"
        )

        assert contiguous

    print("-" * 90)
    print(
        f"Full frames to forward : {len(target_indices)}"
    )
    print(
        f"Official-valid frames  : {n_expected_valid}"
    )
    print("=" * 90)
    print()

    return (
        set(target_indices),
        expected_valid,
        idx_to_scene,
    )


def build_univ2x(mode):
    if mode == "official":
        config_path = BASELINE_CONFIG
    elif mode == "stcv":
        config_path = STCV_CONFIG
    else:
        raise ValueError(mode)

    print(f"Mode                  : {mode}")
    print(f"Config                : {config_path}")
    print(f"Official checkpoint   : {OFFICIAL_CKPT}")

    if mode == "stcv":
        print(f"STCV checkpoint       : {STCV_CKPT}")
        print(f"STCV threshold        : {TAU}")

    print()

    cfg = Config.fromfile(str(config_path))

    if cfg.get("custom_imports", None):
        import_modules_from_strings(
            **cfg["custom_imports"]
        )

    importlib.import_module(
        "projects.mmdet3d_plugin"
    )

    if hasattr(cfg, "plugin") and cfg.plugin:
        if hasattr(cfg, "plugin_dir"):
            plugin_dir = cfg.plugin_dir.rstrip("/")
            module_path = plugin_dir.replace("/", ".")
            importlib.import_module(module_path)

    # Freeze config to inference mode
    if hasattr(cfg.model_ego_agent, "pretrained"):
        cfg.model_ego_agent.pretrained = None

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # Explicitly lock STCV configuration before model construction
    if mode == "stcv":
        cfg.model_ego_agent.occ_head.use_stcv_occ = True
        cfg.model_ego_agent.occ_head.stcv_checkpoint_path = str(
            STCV_CKPT
        )
        cfg.model_ego_agent.occ_head.stcv_threshold = TAU

    dataset = build_dataset(cfg.data.test)

    print(f"Dataset size          : {len(dataset)}")

    assert len(dataset) == 675, (
        f"Expected validation dataset size 675, "
        f"got {len(dataset)}"
    )

    nonshuffler_sampler = cfg.data.get(
        "nonshuffler_sampler",
        None,
    )

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=nonshuffler_sampler,
    )

    print("\nBuilding model...")

    # --------------------------------------------------------
    # Infrastructure agents
    # --------------------------------------------------------

    other_agent_names = [
        k for k in cfg.keys()
        if "model_other_agent" in k
    ]

    model_other_agents = {}

    for other_agent_name in other_agent_names:
        agent_cfg = cfg.get(other_agent_name)
        agent_cfg.train_cfg = None

        model_other_agent = build_model(
            agent_cfg,
            test_cfg=cfg.get("test_cfg"),
        )

        load_from = agent_cfg.load_from

        if load_from:
            load_checkpoint(
                model_other_agent,
                load_from,
                map_location="cpu",
                revise_keys=[
                    (r"^model_ego_agent\.", "")
                ],
            )

        model_other_agents[
            other_agent_name
        ] = model_other_agent

    # --------------------------------------------------------
    # Ego agent
    # --------------------------------------------------------

    cfg.model_ego_agent.train_cfg = None

    model_ego_agent = build_model(
        cfg.model_ego_agent,
        test_cfg=cfg.get("test_cfg"),
    )

    load_from = cfg.model_ego_agent.load_from

    if load_from:
        load_checkpoint(
            model_ego_agent,
            load_from,
            map_location="cpu",
            revise_keys=[
                (r"^model_ego_agent\.", "")
            ],
        )

    # --------------------------------------------------------
    # MultiAgent
    # --------------------------------------------------------

    model_multi_agents = MultiAgent(
        model_ego_agent,
        model_other_agents,
    )

    fp16_cfg = cfg.get("fp16", None)

    if fp16_cfg is not None:
        wrap_fp16_model(model_multi_agents)

    load_checkpoint(
        model_multi_agents,
        str(OFFICIAL_CKPT),
        map_location="cpu",
        strict=False,
    )

    model = MMDataParallel(
        model_multi_agents.cuda(),
        device_ids=[0],
    )

    model.eval()

    occ_head = (
        model.module
        .model_ego_agent
        .occ_head
    )

    if mode == "official":
        # Explicit safety lock
        if hasattr(occ_head, "use_stcv_occ"):
            occ_head.use_stcv_occ = False

        print(
            "Runtime STCV         : "
            f"{getattr(occ_head, 'use_stcv_occ', False)}"
        )

        assert (
            getattr(
                occ_head,
                "use_stcv_occ",
                False,
            )
            is False
        )

    else:
        occ_head.use_stcv_occ = True

        print(
            "Runtime STCV         : "
            f"{occ_head.use_stcv_occ}"
        )

        assert occ_head.use_stcv_occ is True

        if hasattr(occ_head, "stcv_threshold"):
            print(
                "Runtime threshold    : "
                f"{occ_head.stcv_threshold}"
            )

    print("Model loaded successfully\n")

    return model, data_loader


def create_metrics():
    iou_metrics = {}
    panoptic_metrics = {}

    for key in EVALUATION_RANGES:
        iou_metrics[key] = (
            IntersectionOverUnion(
                N_CLASSES
            ).cuda()
        )

        panoptic_metrics[key] = (
            PanopticMetric(
                n_classes=N_CLASSES,
                temporally_consistent=True,
            ).cuda()
        )

    return iou_metrics, panoptic_metrics


def compute_results(
    iou_metrics,
    panoptic_metrics,
    num_occ,
    total_forward,
):
    occ_results = {}

    for key in EVALUATION_RANGES:
        panoptic_scores = (
            panoptic_metrics[key].compute()
        )

        for panoptic_key, value in (
            panoptic_scores.items()
        ):
            occ_results.setdefault(
                panoptic_key,
                [],
            ).append(
                100 * value[1].item()
            )

        iou_scores = iou_metrics[key].compute()

        occ_results.setdefault(
            "iou",
            [],
        ).append(
            100 * iou_scores[1].item()
        )

    occ_results["num_occ"] = num_occ
    occ_results["total_forward"] = total_forward
    occ_results["ratio_occ"] = (
        num_occ / total_forward
    )

    return occ_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["official", "stcv"],
        required=True,
    )

    args = parser.parse_args()

    # Prevent old G0/G1/G2 exporter settings from contaminating run
    for k in [
        "UNIV2X_G012_EXPORT_DIR",
        "UNIV2X_G012_EXPORT_LIMIT",
    ]:
        os.environ.pop(k, None)

    os.environ.pop(
        "ENABLE_PLOT_MODE",
        None,
    )

    (
        target_idx_set,
        expected_valid,
        idx_to_scene,
    ) = load_heldout_indices()

    model, data_loader = build_univ2x(
        args.mode
    )

    (
        iou_metrics,
        panoptic_metrics,
    ) = create_metrics()

    num_occ = 0
    total_forward = 0

    scene_forward_counts = defaultdict(int)
    scene_valid_counts = defaultdict(int)

    print("=" * 90)
    print("START HELD-OUT SEQUENTIAL INFERENCE")
    print("=" * 90)

    with torch.no_grad():

        for idx, data in enumerate(
            tqdm(
                data_loader,
                total=len(data_loader),
                desc=args.mode,
                unit="sample",
            )
        ):
            # Skip all samples outside the four complete held-out scenes.
            #
            # This is safe because each selected scene begins at its real
            # first frame. When the next selected scene begins, UniV2X sees
            # a new scene_token and resets temporal state.
            if idx not in target_idx_set:
                continue

            scene = idx_to_scene[idx]

            result = model(
                return_loss=False,
                rescale=True,
                **data,
            )

            total_forward += 1
            scene_forward_counts[scene] += 1

            # Exact official validity rule
            occ_has_invalid_frame = (
                data[
                    "ego_agent_data"
                ][
                    "gt_occ_has_invalid_frame"
                ][0]
            )

            occ_to_eval = not (
                occ_has_invalid_frame.item()
            )

            # Cross-check runtime evaluator validity against
            # previously audited G0 future_valid_mask.
            assert (
                occ_to_eval
                == expected_valid[idx]
            ), (
                f"Validity mismatch at idx={idx}: "
                f"runtime={occ_to_eval}, "
                f"cache={expected_valid[idx]}"
            )

            if not occ_to_eval:
                continue

            if "occ" not in result[0]:
                raise RuntimeError(
                    f"No occ output at idx={idx}"
                )

            num_occ += 1
            scene_valid_counts[scene] += 1

            occ = result[0]["occ"]

            for key, grid in (
                EVALUATION_RANGES.items()
            ):
                limits = slice(
                    grid[0],
                    grid[1],
                )

                # Exact official IoU call
                iou_metrics[key](
                    occ[
                        "seg_out"
                    ][
                        ...,
                        limits,
                        limits,
                    ].contiguous(),
                    occ[
                        "seg_gt"
                    ][
                        ...,
                        limits,
                        limits,
                    ].contiguous(),
                )

                # Exact official panoptic call
                panoptic_metrics[key](
                    occ[
                        "ins_seg_out"
                    ][
                        ...,
                        limits,
                        limits,
                    ].contiguous().detach(),
                    occ[
                        "ins_seg_gt"
                    ][
                        ...,
                        limits,
                        limits,
                    ].contiguous(),
                )

    print("\n" + "=" * 90)
    print("FORWARD AUDIT")
    print("=" * 90)

    for scene in sorted(TARGET_SCENES):
        print(
            f"Scene {scene}: "
            f"forward={scene_forward_counts[scene]}, "
            f"official_valid={scene_valid_counts[scene]}"
        )

    print("-" * 90)
    print(
        f"Total forwarded       : {total_forward}"
    )
    print(
        f"Official-valid        : {num_occ}"
    )

    assert total_forward == EXPECTED_TOTAL_FORWARD, (
        f"Expected {EXPECTED_TOTAL_FORWARD} forwarded frames, "
        f"got {total_forward}"
    )

    assert num_occ == EXPECTED_VALID, (
        f"Expected {EXPECTED_VALID} valid frames, "
        f"got {num_occ}"
    )

    results = compute_results(
        iou_metrics,
        panoptic_metrics,
        num_occ,
        total_forward,
    )

    print("\n" + "=" * 90)
    print("HELD-OUT OFFICIAL OCCUPANCY RESULTS")
    print("=" * 90)
    print(f"Mode: {args.mode}")
    print()

    for key in [
        "iou",
        "pq",
        "sq",
        "rq",
    ]:
        if key in results:
            print(key)
            print(
                " & ".join(
                    f"{x:.1f}"
                    for x in results[key]
                )
            )

    print(
        f"\nnum occ evaluated: "
        f"{results['num_occ']}"
    )

    print(
        "ratio occ evaluated: "
        f"{100 * results['ratio_occ']:.1f}%"
    )

    print("=" * 90)

    out = {
        "mode": args.mode,
        "heldout_scenes": sorted(
            TARGET_SCENES
        ),
        "total_forward": total_forward,
        "num_occ": num_occ,
        "metrics": results,
        "scene_forward_counts": dict(
            scene_forward_counts
        ),
        "scene_valid_counts": dict(
            scene_valid_counts
        ),
        "official_checkpoint": str(
            OFFICIAL_CKPT
        ),
        "stcv_checkpoint": (
            str(STCV_CKPT)
            if args.mode == "stcv"
            else None
        ),
        "tau": (
            TAU
            if args.mode == "stcv"
            else None
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_file = (
        OUTPUT_DIR
        / f"heldout_{args.mode}_results.json"
    )

    with open(
        out_file,
        "w",
    ) as f:
        json.dump(
            out,
            f,
            indent=2,
        )

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
