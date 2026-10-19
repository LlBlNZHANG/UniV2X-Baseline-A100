import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path("G0-G1-G2")
BASE = ROOT / "V2/V2-4_learnable_adaptive_fusion"
F4 = ROOT / "V2/V2-4F-4_combined_ablation"
OUT = ROOT / "V2/V2-4F-5_scene_robustness"

with open(F4 / "summary.json") as f:
    reference = json.load(f)

# Reuse the F4 script only up to the evaluation loop.
source = (F4 / "run.py").read_text()
tree = ast.parse(source)

marker = None
for node in tree.body:
    if isinstance(node, ast.Assign):
        names = [
            t.id for t in node.targets
            if isinstance(t, ast.Name)
        ]
        if "counts" in names:
            marker = node.lineno
            break

assert marker is not None, "F4 evaluation marker missing"

prefix = "\n".join(source.splitlines()[:marker-1])

# The trusted F4 setup reconstructs its Train-fitted classifier
# and frozen four-policy gate arrays.
exec(compile(prefix, str(F4 / "run.py"), "exec"), globals())

assert len(selected) == 2479
assert len(e1_gate) == 18555

ranges = {
    "30m": (70, 130),
    "100m": (0, 200),
}

totals = {
    method: {
        region: np.zeros(3, dtype=np.int64)
        for region in ranges
    }
    for method in policies
}

by_scene = defaultdict(
    lambda: {
        method: {
            region: np.zeros(3, dtype=np.int64)
            for region in ranges
        }
        for method in policies
    }
)

frame_counts = defaultdict(int)

for path in sorted((ROOT / "cache_full").glob("sample_*.npz")):
    with np.load(path, allow_pickle=False) as d:
        key = (
            str(d["sample_idx"].item()),
            str(d["scene_token"].item())
        )

        if key not in valid_keys or key[1] not in scenes:
            continue

        scene = key[1]
        frame_counts[scene] += 1

        pv = d["Pv"].copy()
        pi = d["Pi_aligned"]
        gt = d["GT"]
        warp = d["warp_valid_mask"].astype(bool)
        threshold = float(d["test_seg_thresh"].item())

        assert abs(threshold - 0.1) < 1e-8

        valid = (
            d["gt_cell_valid_mask"].astype(bool)
            & d["future_valid_mask"].astype(bool)[:, None, None]
            & (gt != 255)
        )

        target = gt == 1

        for method, gates in policies.items():
            fused = pv.copy()

            for i in groups.get(key, []):
                g = float(gates[i])
                if g == 0:
                    continue

                h = int(val["h"][i])
                r = int(val["r"][i])
                c = int(val["c"][i])

                rs = slice(4*r, 4*r+4)
                cs = slice(4*c, 4*c+4)

                local_pv = pv[h, rs, cs]

                delta = (
                    np.maximum(
                        pi[h, rs, cs] - local_pv, 0
                    ) * warp[rs, cs]
                )

                fused[h, rs, cs] = local_pv + g * delta

            pred = fused > threshold

            for region, (start, end) in ranges.items():
                s = slice(start, end)

                p = pred[:, s, s]
                t = target[:, s, s]
                v = valid[:, s, s]

                counts = np.array([
                    np.count_nonzero(p & t & v),
                    np.count_nonzero(p & ~t & v),
                    np.count_nonzero(~p & t & v),
                ], dtype=np.int64)

                totals[method][region] += counts
                by_scene[scene][method][region] += counts

assert sum(frame_counts.values()) == 96

def iou(x):
    return float(x[0] / x.sum()) if x.sum() else 0.0

print("=== V2-4F-5 GLOBAL METRICS ===")

results = {}

for method in policies:
    results[method] = {}

    for region in ranges:
        acc = totals[method][region]
        score = iou(acc)

        results[method][region] = {
            "iou": score,
            "tp": int(acc[0]),
            "fp": int(acc[1]),
            "fn": int(acc[2]),
        }

        print(
            f"{method:14s} {region:4s} "
            f"IoU={score:.9f} "
            f"TP={acc[0]} FP={acc[1]} FN={acc[2]}"
        )

# Exact reproduction of the verified F4 global 100m results.
for method in policies:
    actual = results[method]["100m"]["iou"]
    expected = reference["results"][method]["iou100"]

    assert abs(actual - expected) < 1e-9, (
        method, "100m baseline mismatch"
    )

print("\n100m BASELINE REPRODUCTION: PASS")

print("\n=== SCENE-LEVEL RESULTS ===")

scene_results = {}

for scene in sorted(by_scene):
    scene_results[scene] = {}
    print(f"\nSCENE {scene} | FRAMES={frame_counts[scene]}")

    for method in policies:
        scene_results[scene][method] = {}

        for region in ranges:
            acc = by_scene[scene][method][region]
            score = iou(acc)
            scene_results[scene][method][region] = score

            print(
                f"{method:14s} {region:4s} "
                f"IoU={score:.6f}"
            )

    for region in ranges:
        f4 = scene_results[scene]["F4_COMBINED"][region]
        e1 = scene_results[scene]["E1"][region]

        print(
            f"F4-E1 {region}: "
            f"{(f4-e1)*100:+.5f} pp"
        )

print("\n=== GLOBAL DELTAS ===")

for region in ranges:
    f4 = results["F4_COMBINED"][region]["iou"]
    e1 = results["E1"][region]["iou"]
    v2 = results["V2_HARD"][region]["iou"]

    print(
        region,
        "F4-V2:", round((f4-v2)*100, 6), "pp",
        "F4-E1:", round((f4-e1)*100, 6), "pp"
    )

report = {
    "experiment": "V2-4F-5 30m and Scene Robustness",
    "ranges": ranges,
    "frame_counts": dict(frame_counts),
    "global_results": results,
    "scene_results": scene_results,
    "test_used": False,
}

with open(OUT / "summary.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nREPORT:", OUT / "summary.json")
print("V2-4F-5 COMPLETE")
