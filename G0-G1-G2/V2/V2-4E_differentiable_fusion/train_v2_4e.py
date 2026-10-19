import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 2026
THRESHOLD = 0.1
REFERENCE_V2_100 = 0.36036945366030626
TOLERANCE = 0.0001
EPOCHS = 8
BATCH_SIZE = 2048

ROOT = Path("G0-G1-G2")
DATA = ROOT / "V2/V2-4_learnable_adaptive_fusion"
OUT = ROOT / "V2/V2-4E_differentiable_fusion"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)


def load_dataset(split):
    path = DATA / f"V2-4B_{split}_learnable_fusion_dataset.npz"
    with np.load(path, allow_pickle=False) as d:
        x = np.column_stack((
            d["features_127"],
            d["v2_mag"],
            d["v2_sign_prob"]
        )).astype(np.float32)

        result = {
            "x": x,
            "pv": torch.from_numpy(d["Pv_patch"].copy()),
            "pi": torch.from_numpy(d["Pi_patch"].copy()),
            "warp": torch.from_numpy(
                d["warp_patch"].astype(np.float32)
            ),
            "gt": torch.from_numpy(
                d["GT_patch"].astype(np.float32)
            ),
            "accept": d["v2_accept"].copy(),
            "sample": d["sample_idx"].copy(),
            "scene": d["scene_token"].copy(),
            "h": d["horizon"].copy(),
            "r": d["patch_r"].copy(),
            "c": d["patch_c"].copy(),
        }
    assert np.isfinite(result["x"]).all()
    assert np.isin(result["gt"].numpy(), [0, 1]).all()
    return result


train = load_dataset("train")
val = load_dataset("val")

assert not (
    set(train["scene"].tolist()) &
    set(val["scene"].tolist())
), "TRAIN/VAL SCENE LEAKAGE"

mean = train["x"].mean(axis=0)
std = train["x"].std(axis=0)
std = np.maximum(std, 1e-4)

for d in (train, val):
    d["x"] = torch.from_numpy(
        ((d["x"] - mean) / std).astype(np.float32)
    )

np.savez(
    OUT / "normalization.npz",
    mean=mean,
    std=std
)


class GateMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(129, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.layers(x)).reshape(-1, 1, 1)


model = GateMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Every validation patch maps to one unique cache location.
groups = defaultdict(list)
for i in range(len(val["sample"])):
    key = (str(val["sample"][i]), str(val["scene"][i]))
    groups[key].append(i)

assert len(groups) == 94, (
    f"Unexpected validation candidate samples: {len(groups)}"
)

# Preserve samples with zero candidates.
val_scenes = set(val["scene"].tolist())

import csv
with open(ROOT / "G3/cache_valid_samples.txt", newline="") as f:
    valid_keys = {
        (row["sample_idx"], row["scene_token"])
        for row in csv.DictReader(f, delimiter="\t")
    }

cache_paths = sorted((ROOT / "cache_full").glob("sample_*.npz"))

frames = {}
for path in cache_paths:
    with np.load(path, allow_pickle=False) as d:
        sid = str(d["sample_idx"].item())
        scene = str(d["scene_token"].item())
        key = (sid, scene)

        if scene not in val_scenes or key not in valid_keys:
            continue

        frames[key] = {
            "pv": d["Pv"].copy(),
            "pi": d["Pi_aligned"].copy(),
            "gt": d["GT"].copy(),
            "cell_valid": d["gt_cell_valid_mask"].astype(bool),
            "future_valid": d["future_valid_mask"].astype(bool),
            "warp": d["warp_valid_mask"].astype(bool),
            "threshold": float(d["test_seg_thresh"].item()),
        }

assert all(k in frames for k in groups), "Missing validation cache"
print("VALIDATION CACHE FRAMES:", len(frames))
print("FRAMES WITH CANDIDATES:", len(groups))

assert all(
    abs(f["threshold"] - THRESHOLD) < 1e-8
    for f in frames.values()
), "Unexpected occupancy threshold"


@torch.no_grad()
def predict_weights(dataset):
    model.eval()
    outputs = []

    for start in range(0, len(dataset["x"]), 4096):
        x = dataset["x"][start:start + 4096]
        outputs.append(
            model(x).reshape(-1).numpy()
        )

    return np.concatenate(outputs)


def evaluate(weights=None, hard=False):
    tp = fp = fn = 0
    gate_values = []

    for key, frame in frames.items():
        fused = frame["pv"].copy()

        for i in groups.get(key, []):
            h = int(val["h"][i])
            r = int(val["r"][i])
            c = int(val["c"][i])

            rs = slice(r * 4, (r + 1) * 4)
            cs = slice(c * 4, (c + 1) * 4)

            if hard:
                w = float(val["accept"][i])
            else:
                w = float(weights[i])

            gate_values.append(w)

            pv_patch = frame["pv"][h, rs, cs]
            pi_patch = frame["pi"][h, rs, cs]
            warp_patch = frame["warp"][rs, cs]

            addition = (
                np.maximum(pi_patch - pv_patch, 0)
                * warp_patch
            )

            fused[h, rs, cs] = pv_patch + w * addition

        valid = (
            frame["cell_valid"] &
            frame["future_valid"][:, None, None] &
            (frame["gt"] != 255)
        )

        pred = fused > THRESHOLD
        gt = frame["gt"] == 1

        tp += int(np.count_nonzero(pred & gt & valid))
        fp += int(np.count_nonzero(pred & ~gt & valid))
        fn += int(np.count_nonzero(~pred & gt & valid))

    iou = tp / max(tp + fp + fn, 1)
    return {
        "iou100": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_gate": float(np.mean(gate_values)),
    }


# Safety gate: never train with an unverified evaluator.
baseline = evaluate(hard=True)
print("\n=== HISTORICAL BASELINE PREFLIGHT ===")
print("V2 Hard calculated:", baseline["iou100"])
print("V2 Hard expected:  ", REFERENCE_V2_100)
print("Absolute difference:",
      abs(baseline["iou100"] - REFERENCE_V2_100))

if abs(baseline["iou100"] - REFERENCE_V2_100) > TOLERANCE:
    print("BASELINE MISMATCH: TRAINING BLOCKED")
    print("Check frame selection, GT masking and IoU protocol.")
    raise SystemExit(2)

print("BASELINE MATCH: PASS")
print("\n=== V2-4E BCE TRAINING ===")

best = -1.0
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    order = torch.randperm(len(train["x"]))
    running_loss = 0.0
    batches = 0

    for indices in order.split(BATCH_SIZE):
        x = train["x"][indices]
        pv = train["pv"][indices]
        pi = train["pi"][indices]
        warp = train["warp"][indices]
        gt = train["gt"][indices]

        weights = model(x)
        addition = torch.clamp(pi - pv, min=0) * warp
        fused = pv + weights * addition

        loss = F.binary_cross_entropy(
            fused.clamp(1e-6, 1 - 1e-6),
            gt
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        batches += 1

    val_weights = predict_weights(val)
    metrics = evaluate(weights=val_weights)

    record = {
        "epoch": epoch,
        "train_bce": running_loss / batches,
        **metrics
    }
    history.append(record)

    print(
        f"Epoch {epoch:02d} | "
        f"BCE={record['train_bce']:.6f} | "
        f"Val100={record['iou100']:.6f} | "
        f"MeanGate={record['mean_gate']:.4f}"
    )

    if metrics["iou100"] > best:
        best = metrics["iou100"]
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_iou100": best,
            },
            OUT / "best.pth"
        )

summary = {
    "experiment": "V2-4E Direct Differentiable Fusion BCE",
    "seed": SEED,
    "stage2_frozen": True,
    "v2_frozen": True,
    "train_candidates": len(train["x"]),
    "val_candidates": len(val["x"]),
    "val_cache_frames": len(frames),
    "v2_hard_reference": REFERENCE_V2_100,
    "v2_hard_reproduced": baseline["iou100"],
    "best_val_iou100": best,
    "historical_e1_iou100": 0.3635077623206901,
    "history": history,
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== FINAL RESULT ===")
print("V2 Hard:", baseline["iou100"])
print("Historical E1:", summary["historical_e1_iou100"])
print("V2-4E Best:", best)
print("OUTPUT:", OUT / "summary.json")
print("V2-4E TRAINING COMPLETE")
