#!/usr/bin/env python3
"""
G5-1B: 86-Sample Sequential Consistency Sweep

Goal: Validate offline-online consistency across all 86 held-out Test samples
      while maintaining UniV2X temporal state through sequential inference.

Protocol:
  - Sequential inference: 0 → 1 → 2 → ... → 674 (675 validation samples)
  - Detailed consistency check on 86 Test samples
  - Save outputs for G5-2 official evaluation

Verification:
  - 14,750/14,750 candidates exact
  - Features/logits/scores: soft consistency (< 1e-5)
  - Accept decisions: exact
  - Ofused: exact match with G4 offline
  - Candidate-domain IoU ≈ 0.2732
"""

import os
import sys
import json
import shutil
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


# ============================================================
# Configuration
# ============================================================

STCV_CONFIG = ROOT / "projects/configs_e2e_univ2x/univ2x_coop_e2e_stcv.py"
OFFICIAL_CKPT = ROOT / "ckpts/univ2x_coop_e2e_stg2.pth"

G3_TEST_NPZ = ROOT / "G0-G1-G2/G3/soft/g3_neighborhood_test.npz"
G4_RESULTS = ROOT / "G0-G1-G2/G4/results/g4_final_results.json"
SAMPLE_IDX_MAPPING = ROOT / "G0-G1-G2/G4/sample_idx_to_export_idx.json"

RUNTIME_EXPORT_DIR = ROOT / "G0-G1-G2/G5/G5-1_consistency_check/runtime_sweep_675"
OUTPUT_DIR = ROOT / "G0-G1-G2/G5/G5-1_consistency_check"

TAU = 0.70
SOFT_TOL = 1e-5

# G3 Test: 86 samples, 14750 candidates
EXPECTED_TEST_SAMPLES = 86
EXPECTED_TOTAL_CANDIDATES = 14750


# ============================================================
# Utilities
# ============================================================

def canon(x):
    """Canonicalize sample_idx to 6-digit string"""
    if isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode()

    s = str(x)

    if s.isdigit() and len(s) < 6:
        s = s.zfill(6)

    return s


def soft_stats(a, b):
    """Compute max and mean absolute difference"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.shape != b.shape:
        return float("inf"), float("inf")

    diff = np.abs(a - b)
    return float(diff.max()), float(diff.mean())


def exact_np(a, b):
    """Check exact numpy array equality"""
    a = np.asarray(a)
    b = np.asarray(b)

    return a.shape == b.shape and np.array_equal(a, b)


def compute_patch_iou(pred, target):
    """
    Compute IoU for binary occupancy prediction.

    Args:
        pred: [H, 200, 200] uint8 binary
        target: [H, 200, 200] uint8 binary

    Returns:
        iou: float
    """
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:
        return 1.0

    return float(intersection) / float(union)


# ============================================================
# 0. Setup
# ============================================================

print("=" * 90)
print("G5-1B: 86-Sample Sequential Consistency Sweep")
print("=" * 90)
print()

# Clean and prepare runtime export directory
if RUNTIME_EXPORT_DIR.exists():
    shutil.rmtree(RUNTIME_EXPORT_DIR)

RUNTIME_EXPORT_DIR.mkdir(parents=True)

# Configure G0-style export (no limit, capture all)
os.environ["UNIV2X_G012_EXPORT_DIR"] = str(RUNTIME_EXPORT_DIR)
os.environ["UNIV2X_G012_EXPORT_LIMIT"] = "-1"  # No limit


# ============================================================
# 1. Load G3 Test metadata
# ============================================================

print("Loading G3 Test metadata...")

g3_test = np.load(G3_TEST_NPZ, allow_pickle=True)
g3_metadata = g3_test["metadata"]
g3_features = g3_test["features"].astype(np.float32)

# Build sample_idx → rows mapping
g3_sample_to_rows = defaultdict(list)

for i, meta in enumerate(g3_metadata):
    sid = canon(meta["sample_idx"])
    g3_sample_to_rows[sid].append(i)

test_sample_ids = sorted(g3_sample_to_rows.keys())

print(f"  G3 Test samples       : {len(test_sample_ids)}")
print(f"  G3 Total candidates   : {len(g3_metadata)}")

assert len(test_sample_ids) == EXPECTED_TEST_SAMPLES
assert len(g3_metadata) == EXPECTED_TOTAL_CANDIDATES


# ============================================================
# 2. Load sample_idx to export_idx mapping
# ============================================================

with open(SAMPLE_IDX_MAPPING) as f:
    sample_idx_to_export_idx = json.load(f)

# Canonicalize keys
sample_idx_to_export_idx = {
    canon(k): int(v)
    for k, v in sample_idx_to_export_idx.items()
}


# ============================================================
# 3. Load config and build dataset
# ============================================================

print("\nLoading config and building dataset...")

cfg = Config.fromfile(str(STCV_CONFIG))

if cfg.get("custom_imports", None):
    import_modules_from_strings(**cfg["custom_imports"])

importlib.import_module("projects.mmdet3d_plugin")

if hasattr(cfg, "plugin") and cfg.plugin:
    if hasattr(cfg, "plugin_dir"):
        plugin_dir = cfg.plugin_dir.rstrip("/")
        module_path = plugin_dir.replace("/", ".")
        importlib.import_module(module_path)

cfg.model_ego_agent.pretrained = None

if isinstance(cfg.data.test, dict):
    cfg.data.test.test_mode = True
elif isinstance(cfg.data.test, list):
    for ds_cfg in cfg.data.test:
        ds_cfg.test_mode = True

dataset = build_dataset(cfg.data.test)

print(f"  Dataset size          : {len(dataset)}")


# ============================================================
# 4. Build dataloader (sequential, no shuffle)
# ============================================================

nonshuffler_sampler = cfg.data.get("nonshuffler_sampler", None)

data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=2,
    dist=False,
    shuffle=False,
    nonshuffler_sampler=nonshuffler_sampler,
)


# ============================================================
# 5. Build model
# ============================================================

print("\nBuilding model...")

# Infrastructure agents
other_agent_names = [k for k in cfg.keys() if "model_other_agent" in k]
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
            revise_keys=[(r"^model_ego_agent\.", "")],
        )

    model_other_agents[other_agent_name] = model_other_agent

# Ego agent
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
        revise_keys=[(r"^model_ego_agent\.", "")],
    )

# MultiAgent
model_multi_agents = MultiAgent(model_ego_agent, model_other_agents)

fp16_cfg = cfg.get("fp16", None)
if fp16_cfg is not None:
    wrap_fp16_model(model_multi_agents)

load_checkpoint(
    model_multi_agents,
    str(OFFICIAL_CKPT),
    map_location="cpu",
    strict=False,
)

model = MMDataParallel(model_multi_agents.cuda(), device_ids=[0])
model.eval()

print("  Model loaded successfully")


# ============================================================
# 6. Determine required inference range
# ============================================================

# Find maximum dataset index needed to cover all 86 Test samples
test_export_indices = [
    sample_idx_to_export_idx[sid]
    for sid in test_sample_ids
    if sid in sample_idx_to_export_idx
]

max_test_export_idx = max(test_export_indices)
required_samples = max_test_export_idx + 1

print(f"\nTest sample dataset indices: {min(test_export_indices)} → {max_test_export_idx}")
print(f"Required sequential range: 0 → {max_test_export_idx} ({required_samples} samples)")
print(f"Estimated time: {required_samples * 5.5 / 60:.1f} minutes\n")

# ============================================================
# 7. Sequential inference: 0 → max_test_export_idx
# ============================================================

print("=" * 90)
print(f"SEQUENTIAL INFERENCE: 0 → {max_test_export_idx} ({required_samples} samples)")
print("=" * 90)
print()

processed_count = 0

# G5-1B protocol:
# preserve official OR on non-Test frames; enable STCV only on audited Test frames.
test_idx_set = set(test_export_indices)
occ_head = model_ego_agent.occ_head
occ_head.use_stcv_occ = False

print(f"STCV audit protocol: ON for {len(test_idx_set)} Test frames, OFF otherwise")

with torch.no_grad():
    for idx, data in enumerate(tqdm(data_loader, desc="Forward", unit="sample")):
        is_test_frame = idx in test_idx_set
        occ_head.use_stcv_occ = is_test_frame

        if is_test_frame:
            print(f"[G5-1B] STCV ON at dataset idx={idx}")

        _ = model(return_loss=False, rescale=True, **data)
        processed_count += 1

        # Restore official OR for the next warm-up/non-Test frame.
        occ_head.use_stcv_occ = False

        # Stop after covering all test samples
        if idx >= max_test_export_idx:
            break

print(f"\n✓ Sequential inference complete: {processed_count} samples processed")
print(f"  Runtime exports saved to: {RUNTIME_EXPORT_DIR}")


# ============================================================
# 8. Consistency check on 86 Test samples
# ============================================================

print("\n" + "=" * 90)
print("CONSISTENCY CHECK: 86 Test Samples")
print("=" * 90)
print()

# Get STCV module
occ_head = model.module.model_ego_agent.occ_head
stcv = occ_head.stcv_occ
device = next(stcv.predictor.parameters()).device

# Storage for results
consistency_results = []
failed_samples = []

total_candidates_offline = 0
total_candidates_runtime = 0
total_accepted_offline = 0
total_accepted_runtime = 0

all_ious = []

for sid in tqdm(test_sample_ids, desc="Consistency", unit="sample"):

    # Load runtime export
    export_idx = sample_idx_to_export_idx.get(sid)

    if export_idx is None:
        print(f"  WARNING: {sid} not in export mapping, skipping")
        continue

    runtime_file = RUNTIME_EXPORT_DIR / f"sample_{export_idx:05d}.npz"

    if not runtime_file.exists():
        print(f"  WARNING: {runtime_file.name} not found, skipping")
        failed_samples.append({"sample_idx": sid, "reason": "runtime_file_missing"})
        continue

    runtime = np.load(runtime_file, allow_pickle=True)
    runtime_sid = canon(runtime["sample_idx"])

    if runtime_sid != sid:
        print(f"  ERROR: {sid} → {runtime_file.name} but contains {runtime_sid}")
        failed_samples.append({"sample_idx": sid, "reason": "sample_idx_mismatch"})
        continue

    # Get offline reference from G3
    offline_rows = g3_sample_to_rows[sid]
    offline_features = g3_features[offline_rows]
    offline_indices = [
        (
            int(g3_metadata[i]["horizon"]),
            int(g3_metadata[i]["patch_row"]),
            int(g3_metadata[i]["patch_col"]),
        )
        for i in offline_rows
    ]

    # Reconstruct runtime candidates/features
    rt_pv = torch.from_numpy(runtime["Pv"].astype(np.float32)).to(device)
    rt_pi = torch.from_numpy(runtime["Pi_aligned"].astype(np.float32)).to(device)
    rt_ov = torch.from_numpy(runtime["Ov"].astype(np.int64)).to(device)
    rt_oi = torch.from_numpy(runtime["Oi"].astype(np.int64)).to(device)
    rt_warp = torch.from_numpy(runtime["warp_valid_mask"].astype(np.uint8)).to(device)

    runtime_indices, runtime_features = stcv.extract_candidates_and_features(
        pv=rt_pv, pi=rt_pi, ov=rt_ov, oi=rt_oi, warp=rt_warp
    )

    with torch.no_grad():
        runtime_logits = stcv.predictor(runtime_features)
        runtime_scores = torch.sigmoid(runtime_logits)
        runtime_accept = runtime_scores > TAU

    # Offline predictor
    offline_features_t = torch.from_numpy(offline_features).to(device)

    with torch.no_grad():
        offline_logits = stcv.predictor(offline_features_t)
        offline_scores = torch.sigmoid(offline_logits)
        offline_accept = offline_scores > TAU

    # Convert to numpy
    runtime_features_np = runtime_features.detach().cpu().numpy()
    runtime_logits_np = runtime_logits.detach().cpu().numpy()
    runtime_scores_np = runtime_scores.detach().cpu().numpy()
    runtime_accept_np = runtime_accept.detach().cpu().numpy()

    offline_logits_np = offline_logits.detach().cpu().numpy()
    offline_scores_np = offline_scores.detach().cpu().numpy()
    offline_accept_np = offline_accept.detach().cpu().numpy()

    # Consistency checks
    candidate_exact = (runtime_indices == offline_indices)

    feat_max, feat_mean = soft_stats(runtime_features_np, offline_features)
    logit_max, logit_mean = soft_stats(runtime_logits_np, offline_logits_np)
    score_max, score_mean = soft_stats(runtime_scores_np, offline_scores_np)

    accept_exact = (
        runtime_accept_np.shape == offline_accept_np.shape
        and np.array_equal(runtime_accept_np, offline_accept_np)
    )

    # Compute IoU
    runtime_ofused = runtime["Ofused"].astype(np.uint8)

    # Reconstruct offline learned from G3 accept decisions
    offline_learned = runtime["Ov"].copy().astype(np.uint8)

    for i, (h, r, c) in enumerate(offline_indices):
        if not bool(offline_accept_np[i]):
            continue

        rs = slice(4 * r, 4 * r + 4)
        cs = slice(4 * c, 4 * c + 4)

        offline_learned[h, rs, cs] = np.logical_or(
            offline_learned[h, rs, cs],
            runtime["Oi"][h, rs, cs],
        ).astype(np.uint8)

    ofused_exact = exact_np(runtime_ofused, offline_learned)
    ofused_diff = int((runtime_ofused != offline_learned).sum())

    iou = compute_patch_iou(runtime_ofused, offline_learned)
    all_ious.append(iou)

    # Accumulate stats
    n_candidates_offline = len(offline_indices)
    n_candidates_runtime = len(runtime_indices)
    n_accepted_offline = int(offline_accept_np.sum())
    n_accepted_runtime = int(runtime_accept_np.sum())

    total_candidates_offline += n_candidates_offline
    total_candidates_runtime += n_candidates_runtime
    total_accepted_offline += n_accepted_offline
    total_accepted_runtime += n_accepted_runtime

    # Soft/hard pass
    soft_pass = (
        feat_max < SOFT_TOL
        and logit_max < SOFT_TOL
        and score_max < SOFT_TOL
    )

    hard_pass = (
        candidate_exact
        and accept_exact
        and ofused_exact
    )

    sample_pass = soft_pass and hard_pass

    # Record
    consistency_results.append({
        "sample_idx": sid,
        "export_idx": export_idx,
        "candidates_offline": n_candidates_offline,
        "candidates_runtime": n_candidates_runtime,
        "candidate_exact": candidate_exact,
        "feature_max_diff": float(feat_max),
        "feature_mean_diff": float(feat_mean),
        "logit_max_diff": float(logit_max),
        "logit_mean_diff": float(logit_mean),
        "score_max_diff": float(score_max),
        "score_mean_diff": float(score_mean),
        "accepted_offline": n_accepted_offline,
        "accepted_runtime": n_accepted_runtime,
        "accept_exact": accept_exact,
        "ofused_diff_cells": ofused_diff,
        "ofused_exact": ofused_exact,
        "iou": float(iou),
        "soft_pass": soft_pass,
        "hard_pass": hard_pass,
        "final_pass": sample_pass,
    })

    if not sample_pass:
        failed_samples.append({
            "sample_idx": sid,
            "reason": "consistency_fail",
            "details": {
                "candidate_exact": candidate_exact,
                "feature_max_diff": float(feat_max),
                "accept_exact": accept_exact,
                "ofused_exact": ofused_exact,
            }
        })


# ============================================================
# 9. Aggregate results
# ============================================================

print("\n" + "=" * 90)
print("G5-1B FINAL RESULTS")
print("=" * 90)
print()

n_checked = len(consistency_results)
n_passed = sum(r["final_pass"] for r in consistency_results)
n_failed = len(failed_samples)

mean_iou = np.mean(all_ious) if all_ious else 0.0

print(f"Sequential samples processed  : {processed_count}")
print(f"Samples checked               : {n_checked} / {EXPECTED_TEST_SAMPLES}")
print(f"Samples PASS                  : {n_passed} / {n_checked}")
print(f"Samples FAIL                  : {n_failed}")
print()

print(f"Total candidates offline  : {total_candidates_offline}")
print(f"Total candidates runtime  : {total_candidates_runtime}")
print(f"Candidate exact           : {'PASS' if total_candidates_offline == total_candidates_runtime == EXPECTED_TOTAL_CANDIDATES else 'FAIL'}")
print()

print(f"Total accepted offline    : {total_accepted_offline}")
print(f"Total accepted runtime    : {total_accepted_runtime}")
print(f"Accept rate offline       : {100.0 * total_accepted_offline / total_candidates_offline:.2f}%")
print(f"Accept rate runtime       : {100.0 * total_accepted_runtime / total_candidates_runtime:.2f}%")
print()

print(f"Mean IoU (runtime vs offline): {mean_iou:.6f}")
print(f"  (Sanity check: expect ≈ G4 Test reference 0.2732)")
print()

final_pass = (
    n_checked == EXPECTED_TEST_SAMPLES
    and n_passed == EXPECTED_TEST_SAMPLES
    and total_candidates_offline == total_candidates_runtime == EXPECTED_TOTAL_CANDIDATES
)

print(f"FINAL RESULT              : {'PASS' if final_pass else 'FAIL'}")
print("=" * 90)


# ============================================================
# 10. Save results
# ============================================================

output_json = OUTPUT_DIR / "g5_1b_consistency_results.json"

results_summary = {
    "status": "PASS" if final_pass else "FAIL",
    "sequential_samples_processed": processed_count,
    "required_range": f"0 → {max_test_export_idx}",
    "samples_checked": n_checked,
    "samples_passed": n_passed,
    "samples_failed": n_failed,
    "total_candidates_offline": total_candidates_offline,
    "total_candidates_runtime": total_candidates_runtime,
    "total_accepted_offline": total_accepted_offline,
    "total_accepted_runtime": total_accepted_runtime,
    "mean_iou": float(mean_iou),
    "per_sample_results": consistency_results,
    "failed_samples": failed_samples,
}

with open(output_json, "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✓ Results saved to: {output_json}")

if not final_pass:
    sys.exit(1)
