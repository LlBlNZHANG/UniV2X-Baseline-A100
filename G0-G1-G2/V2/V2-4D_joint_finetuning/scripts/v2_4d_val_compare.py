"""
V2-4D validation comparison.

Modes:
  baseline : Stage-2 + frozen V2, g=1 if accepted
  step10   : Stage-2 + trained OccHead + trained Inn3
  step20   : Stage-2 + trained OccHead + trained Inn3

Evaluation:
  Official foreground IoU, global TP/FP/FN
  30m: [70:130, 70:130]
  100m: [0:200, 0:200]
  Invalid frames excluded
  Test split NEVER used
"""

import os
import sys
import argparse
import importlib.util
import types

import torch
import mmcv
from mmcv.parallel import scatter
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

ROOT = "/root/autodl-tmp/UniV2X"
BASE = os.path.join(
    ROOT, "G0-G1-G2/V2/V2-4D_joint_finetuning"
)

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "projects"))
sys.path.insert(0, BASE)

AUDIT_FILE = os.path.join(
    BASE, "scripts/v2_4d_exact_label_audit.py"
)

spec = importlib.util.spec_from_file_location(
    "v2_4d_audit_reference", AUDIT_FILE
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

MultiAgent = audit.MultiAgent
V2UtilityPredictor = audit.V2UtilityPredictor
Innovation3LearnableFusion = audit.Innovation3LearnableFusion
STCVFeatureExtractor = audit.STCVFeatureExtractor

from projects.mmdet3d_plugin.univ2x.dense_heads.occ_head_plugin import (
    IntersectionOverUnion
)

RANGES = {
    "30m": (70, 130),
    "100m": (0, 200),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--max-samples", type=int, default=30,
        help="0 means full validation dataset"
    )
    return p.parse_args()


@torch.no_grad()
def baseline_v2_fusion(
    self, dense_occ_feat, pv, pi_aligned, warp_valid
):
    """
    Exact V2 hard decision:
      reject -> g=0
      accept -> g=1
    No Innovation-3 gate prediction.
    """
    cand = self._extract_candidates(
        pv, pi_aligned, warp_valid, pv.device
    )

    accepted = cand["v2_accept"].bool()

    gates = torch.where(
        accepted,
        torch.ones_like(accepted, dtype=pv.dtype),
        torch.zeros_like(accepted, dtype=pv.dtype),
    )

    pf = self.innovation3_module.apply_fusion(
        pv=pv,
        pi_aligned=pi_aligned,
        gates=gates,
        horizons=cand["horizons"],
        patch_rs=cand["patch_rs"],
        patch_cs=cand["patch_cs"],
    )

    self._v2_4d_debug = {
        "candidate_count": int(gates.numel()),
        "v2_accept_count": int(accepted.sum().item()),
        "g0_count": int((gates == 0).sum().item()),
        "g01_count": 0,
        "g1_count": int((gates == 1).sum().item()),
        "mode": "inference",
    }

    return pf


def make_model(device):
    cfg = mmcv.Config.fromfile(
        os.path.join(
            ROOT,
            "projects/configs_e2e_univ2x/univ2x_coop_e2e.py",
        )
    )

    cfg.model_ego_agent.occ_head.use_stcv_occ = False

    ego = build_model(
        cfg.model_ego_agent,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    ego.init_weights()

    others = {}
    for key in cfg.keys():
        if "model_other_agent" in key:
            m = build_model(
                cfg.get(key),
                train_cfg=cfg.get("train_cfg"),
                test_cfg=cfg.get("test_cfg"),
            )
            m.init_weights()
            others[key] = m

    model = MultiAgent(ego, others).to(device)

    stage2 = torch.load(
        audit.STAGE2_CKPT, map_location="cpu"
    )
    state = stage2.get("state_dict", stage2)

    missing, unexpected = model.load_state_dict(
        state, strict=False
    )

    if missing:
        raise RuntimeError(
            f"Stage-2 missing keys: {missing[:20]}"
        )

    print(
        f"[STAGE2] missing={len(missing)} "
        f"unexpected={len(unexpected)}",
        flush=True,
    )

    v2 = V2UtilityPredictor().to(device)
    v2_state = torch.load(
        audit.V2_CKPT, map_location="cpu"
    )
    v2.load_state_dict(
        v2_state["model"], strict=True
    )

    inn3 = Innovation3LearnableFusion(
        bev_feat_dim=256,
        offline_feat_dim=129,
        dropout=0.1,
    ).to(device)

    model.attach_innovation3(inn3, v2)
    model.attach_stcv_extractor(
        STCVFeatureExtractor()
    )

    model.eval()

    val_cfg = cfg.data.val.copy()
    val_cfg.pop("samples_per_gpu", None)
    val_cfg.test_mode = True

    dataset = build_dataset(val_cfg)

    return model, dataset


def load_joint_checkpoint(model, step):
    path = os.path.join(
        BASE,
        "checkpoints/V2-4D_pilot20",
        f"step_{step:06d}.pth",
    )

    ckpt = torch.load(path, map_location="cpu")

    if int(ckpt["step"]) != step:
        raise RuntimeError(
            f"Checkpoint step mismatch: {path}"
        )

    ego = model.model_ego_agent

    model.innovation3_module.load_state_dict(
        ckpt["innovation3"], strict=True
    )
    ego.occ_head.dense_decoder.load_state_dict(
        ckpt["dense_decoder"], strict=True
    )
    ego.occ_head.pred_prob.load_state_dict(
        ckpt["pred_prob"], strict=True
    )

    print(f"[LOAD] Step {step}: {path}", flush=True)


def run_evaluation(
    model, dataset, name, max_samples, device
):
    model.eval()

    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=42,
    )

    metrics = {
        name: IntersectionOverUnion(2).to(device)
        for name in RANGES
    }

    counts = {
        "candidates": 0,
        "accepted": 0,
        "g0": 0,
        "g01": 0,
        "g1": 0,
    }

    evaluated = 0
    skipped = 0

    limit = (
        len(dataset)
        if max_samples == 0
        else min(max_samples, len(dataset))
    )

    for idx, data in enumerate(loader):
        if idx >= limit:
            break

        data_gpu = scatter(
            data, [torch.cuda.current_device()]
        )[0]

        with torch.no_grad():
            result = model(
                return_loss=False,
                rescale=True,
                **data_gpu
            )

        dbg = getattr(
            model, "_v2_4d_debug", {}
        )

        if dbg.get("mode") != "inference":
            raise RuntimeError(
                f"{name}: V2 fusion callback missing"
            )

        n = int(dbg["candidate_count"])
        a = int(dbg["v2_accept_count"])
        g0 = int(dbg["g0_count"])
        g01 = int(dbg["g01_count"])
        g1 = int(dbg["g1_count"])

        assert 0 <= a <= n
        assert g0 == n - a
        assert g01 + g1 == a
        assert g0 + g01 + g1 == n

        counts["candidates"] += n
        counts["accepted"] += a
        counts["g0"] += g0
        counts["g01"] += g01
        counts["g1"] += g1

        invalid = data[
            "ego_agent_data"
        ]["gt_occ_has_invalid_frame"][0]

        if bool(invalid.item()):
            skipped += 1
            continue

        occ = result[0]["occ"]
        pred = occ["seg_out"]
        gt = occ["seg_gt"]

        assert pred.shape == gt.shape, (
            f"Shape mismatch: {pred.shape}, {gt.shape}"
        )

        for range_name, (lo, hi) in RANGES.items():
            metrics[range_name].update(
                pred[..., lo:hi, lo:hi].contiguous(),
                gt[..., lo:hi, lo:hi].contiguous(),
            )

        evaluated += 1

        if (idx + 1) % 10 == 0:
            print(
                f"[{name}] processed={idx + 1}/{limit} "
                f"valid={evaluated} "
                f"accepted={counts['accepted']}",
                flush=True,
            )

    if evaluated == 0:
        raise RuntimeError(
            f"{name}: no valid evaluation frames"
        )

    scores = {
        r: float(metric.compute()[1].item())
        for r, metric in metrics.items()
    }

    print(
        f"[RESULT] {name} "
        f"30m={scores['30m']:.6f} "
        f"100m={scores['100m']:.6f} "
        f"valid={evaluated} "
        f"skipped={skipped} "
        f"candidates={counts['candidates']} "
        f"accepted={counts['accepted']} "
        f"g0={counts['g0']} "
        f"g01={counts['g01']} "
        f"g1={counts['g1']}",
        flush=True,
    )

    return scores, evaluated, skipped


def main():
    args = parse_args()

    assert args.max_samples >= 0
    assert torch.cuda.is_available()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda:0")

    print("V2-4D ONLINE VAL COMPARISON")
    print("Test used: NO")
    print("Max samples:", args.max_samples)

    results = {}

    for name, step, gate_mode in (
        ("V2_Baseline", None, "hard"),
        ("Step10_Hard", 10, "hard"),
        ("Step10_Learned", 10, "learned"),
        ("Step20_Hard", 20, "hard"),
        ("Step20_Learned", 20, "learned"),
    ):
        # Create a fresh model for every experiment.
        # This isolates all runtime temporal/tracking state.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        model, dataset = make_model(device)
        print("Val dataset:", len(dataset), flush=True)

        if step is not None:
            load_joint_checkpoint(model, step)

        if gate_mode == "hard":
            model._v2_4d_infer_fusion = types.MethodType(
                baseline_v2_fusion, model
            )

        model.eval()

        print(f"\n=== {name} ===", flush=True)

        results[name] = run_evaluation(
            model, dataset, name,
            args.max_samples, device
        )

    # All comparisons must use the same evaluation set.
    frame_counts = [
        (value[1], value[2])
        for value in results.values()
    ]
    assert len(set(frame_counts)) == 1, (
        "Evaluated frame count differs between models"
    )

    print("\n=== FINAL COMPARISON ===")
    print(
        f"{'Model':18s} "
        f"{'30m IoU':>12s} "
        f"{'100m IoU':>12s}"
    )

    for name, (scores, _, _) in results.items():
        print(
            f"{name:18s} "
            f"{scores['30m']:12.6f} "
            f"{scores['100m']:12.6f}"
        )

    print(
        f"Valid frames: {frame_counts[0][0]}"
    )
    print("Test used: NO")


if __name__ == "__main__":
    main()
