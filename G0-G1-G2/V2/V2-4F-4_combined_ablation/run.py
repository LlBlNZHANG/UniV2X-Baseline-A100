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
F3 = ROOT / "V2/V2-4F-3_risk_constrained_readmission"
OUT = ROOT / "V2/V2-4F-4_combined_ablation"

REFERENCES = {
    "V2_HARD": 0.36036945366030626,
    "E1": 0.3635077623206901,
    "F3": 0.361133937,
}

with open(F3 / "summary.json", encoding="utf-8") as f:
    f3 = json.load(f)

policy = f3["train_selected_policy"]
assert abs(float(policy["gate"]) - 0.1) < 1e-7
assert abs(float(policy["fraction"]) - 0.2) < 1e-7

def load(split):
    with np.load(
        BASE / f"V2-4B_{split}_learnable_fusion_dataset.npz",
        allow_pickle=False
    ) as d:
        return {
            "x": np.column_stack([
                d["features_127"],
                d["v2_mag"],
                d["v2_sign_prob"],
            ]).astype(np.float32),
            "accept": d["v2_accept"].astype(bool),
            "sample": d["sample_idx"].copy(),
            "scene": d["scene_token"].copy(),
            "h": d["horizon"].copy(),
            "r": d["patch_r"].copy(),
            "c": d["patch_c"].copy(),
        }

train = load("train")
val = load("val")

assert not set(train["scene"]).intersection(set(val["scene"]))

def benefit_labels(split, data):
    with np.load(
        BASE / f"V2-4C-2A_{split}_gate_utility_curves.npz",
        allow_pickle=False
    ) as d:
        for source, target in [
            ("sample_idx", "sample"),
            ("scene_token", "scene"),
            ("horizon", "h"),
            ("patch_r", "r"),
            ("patch_c", "c"),
        ]:
            assert np.array_equal(d[source], data[target])

        gates = d["gate_grid"]
        curve = d["utility_curve"]

        def idx(g):
            found = np.where(np.isclose(gates, g))[0]
            assert len(found) == 1
            return int(found[0])

        gain = (
            np.maximum(curve[:, idx(0.1)], curve[:, idx(1)])
            - curve[:, idx(0)]
        )
        return (gain > 1e-12).astype(int)

y_train = benefit_labels("train", train)

with np.load(
    BASE / "results/V2-4C-3E1_binary_attenuation/"
    "V2-4C-3E1_val_predictions.npz",
    allow_pickle=False
) as d:
    for source, target in [
        ("sample_idx", "sample"),
        ("horizon", "h"),
        ("patch_r", "r"),
        ("patch_c", "c"),
    ]:
        assert np.array_equal(d[source], val[target])

    e1_gate = d["full_candidate_gate"].astype(np.float32)

assert len(e1_gate) == len(val["x"])
assert np.all(e1_gate[~val["accept"]] == 0)

# Refit the same F3 predictor using Train only.
rejected_train = ~train["accept"]
rejected_val = ~val["accept"]

classifier = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=1000,
        solver="lbfgs",
        random_state=2026,
    )
)

classifier.fit(
    train["x"][rejected_train],
    y_train[rejected_train]
)

scores = classifier.predict_proba(
    val["x"][rejected_val]
)[:, 1]

reject_indices = np.flatnonzero(rejected_val)
count = int(np.floor(len(reject_indices) * 0.2))

ranking = np.lexsort((
    np.arange(len(scores)),
    -scores
))

selected = reject_indices[ranking[:count]]

# Construct four policies.
v2_gate = val["accept"].astype(np.float32)

f3_gate = v2_gate.copy()
f3_gate[selected] = 0.1

combined_gate = e1_gate.copy()
combined_gate[selected] = 0.1

policies = {
    "V2_HARD": v2_gate,
    "E1": e1_gate,
    "F3": f3_gate,
    "F4_COMBINED": combined_gate,
}

# Exact historical valid frames.
with open(
    ROOT / "G3/cache_valid_samples.txt",
    newline=""
) as f:
    valid_keys = {
        (row["sample_idx"], row["scene_token"])
        for row in csv.DictReader(f, delimiter="\t")
    }

groups = defaultdict(list)

for i, key in enumerate(zip(val["sample"], val["scene"])):
    groups[tuple(map(str, key))].append(i)

scenes = set(val["scene"].tolist())

counts = {
    name: np.zeros(3, dtype=np.int64)
    for name in policies
}

n_frames = 0

for path in sorted((ROOT / "cache_full").glob("sample_*.npz")):
    with np.load(path, allow_pickle=False) as d:
        key = (
            str(d["sample_idx"].item()),
            str(d["scene_token"].item())
        )

        if key not in valid_keys or key[1] not in scenes:
            continue

        n_frames += 1

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

        sample_indices = groups.get(key, [])

        for name, gates in policies.items():
            fused = pv.copy()

            for i in sample_indices:
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

            counts[name][0] += np.count_nonzero(
                pred & target & valid
            )
            counts[name][1] += np.count_nonzero(
                pred & ~target & valid
            )
            counts[name][2] += np.count_nonzero(
                ~pred & target & valid
            )

assert n_frames == 96, f"Expected 96 frames, got {n_frames}"

def get_iou(x):
    return float(x[0] / x.sum())

print("=== V2-4F-4 COMBINED ABLATION ===")
print("VAL FRAMES:", n_frames)
print("VAL CANDIDATES:", len(e1_gate))
print("E1 ATTENUATED:", int(np.sum(
    val["accept"] & (e1_gate < 1)
)))
print("F3 READMITTED:", len(selected))

results = {}

for name, x in counts.items():
    score = get_iou(x)
    results[name] = {
        "iou100": score,
        "tp": int(x[0]),
        "fp": int(x[1]),
        "fn": int(x[2]),
    }

    print(
        f"{name:14s} IoU100={score:.9f} "
        f"TP={x[0]} FP={x[1]} FN={x[2]}"
    )

print("\n=== HISTORICAL REPRODUCTION ===")

for name, expected in REFERENCES.items():
    actual = results[name]["iou100"]
    error = abs(actual - expected)

    print(
        f"{name}: absolute_error={error:.9f}"
    )

    assert error < 1e-6, (
        f"{name} baseline mismatch; "
        "combined result must not be interpreted"
    )

print("ALL THREE BASELINES: PASS")

combined = results["F4_COMBINED"]["iou100"]
e1 = results["E1"]["iou100"]
v2 = results["V2_HARD"]["iou100"]

print("\n=== COMBINED RESULT ===")
print("DELTA VS V2 (pp):", round((combined-v2)*100, 6))
print("DELTA VS E1 (pp):", round((combined-e1)*100, 6))
print("EXCEEDS E1:", combined > e1)

report = {
    "experiment": "V2-4F-4 E1 + F3 Combined Ablation",
    "val_frames": n_frames,
    "readmission_fraction": 0.2,
    "readmission_gate": 0.1,
    "readmitted_count": int(len(selected)),
    "results": results,
    "delta_vs_v2_pp": (combined-v2)*100,
    "delta_vs_e1_pp": (combined-e1)*100,
    "test_used": False,
}

with open(OUT / "summary.json", "w") as f:
    json.dump(report, f, indent=2)

print("REPORT:", OUT / "summary.json")
print("V2-4F-4 COMPLETE")
