import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("G0-G1-G2")
BASE = ROOT / "V2/V2-4_learnable_adaptive_fusion"
OUT = ROOT / "V2/V2-4F-3_risk_constrained_readmission"
REF = 0.36036945366030626

# Predetermined policy search space.
FRACTIONS = [0, 0.001, 0.0025, 0.005, 0.01,
             0.02, 0.05, 0.10, 0.20, 0.40]
ACTIONS = [0.1, 1.0]


def load(split):
    with np.load(
        BASE / f"V2-4B_{split}_learnable_fusion_dataset.npz",
        allow_pickle=False
    ) as d:
        keys = ("sample_idx", "scene_token", "horizon",
                "patch_r", "patch_c")
        identity = {k: d[k].copy() for k in keys}
        x = np.column_stack([
            d["features_127"],
            d["v2_mag"],
            d["v2_sign_prob"],
        ]).astype(np.float32)
        accept = d["v2_accept"].astype(bool)
        scenes = d["scene_token"].copy()

    with np.load(
        BASE / f"V2-4C-2A_{split}_gate_utility_curves.npz",
        allow_pickle=False
    ) as d:
        for k in keys:
            assert np.array_equal(identity[k], d[k]), (
                split, k, "alignment mismatch"
            )
        gates = d["gate_grid"].copy()
        curves = d["utility_curve"].copy()
        ba = d["ba_curve"].copy()
        ha = d["ha_curve"].copy()

    def index(g):
        found = np.where(np.isclose(gates, g, atol=1e-6))[0]
        assert len(found) == 1
        return int(found[0])

    g0, g01, g1 = index(0), index(0.1), index(1)

    gain = np.maximum(curves[:, g01], curves[:, g1]) - curves[:, g0]
    labels = gain > 1e-12

    assert np.isfinite(x).all()
    assert np.isfinite(curves).all()
    assert np.all(ba[:, g0] == 0)
    assert np.all(ha[:, g0] == 0)

    return dict(
        x=x, accept=accept, scenes=scenes,
        identity=identity, labels=labels,
        ba=ba, ha=ha, gate_index={0.1: g01, 1.0: g1}
    )


def model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, max_iter=1000,
            solver="lbfgs", random_state=2026
        )
    )


train = load("train")
val = load("val")

assert not set(train["scenes"]).intersection(set(val["scenes"]))
assert len(val["x"]) == 18555
assert np.count_nonzero(~val["accept"]) == 12399
assert np.count_nonzero(
    val["labels"] & ~val["accept"]
) == 2227

print("=== SCENE-SPLIT DATA CHECK: PASS ===")

# Build out-of-fold probabilities only for V2-rejected candidates.
tr_rej = np.where(~train["accept"])[0]
va_rej = np.where(~val["accept"])[0]

xtr = train["x"][tr_rej]
ytr = train["labels"][tr_rej].astype(int)
groups = train["scenes"][tr_rej]

oof = np.full(len(tr_rej), np.nan)
cv = GroupKFold(n_splits=5)

for fold, (fit_idx, held_idx) in enumerate(
    cv.split(xtr, ytr, groups), 1
):
    m = model()
    m.fit(xtr[fit_idx], ytr[fit_idx])
    oof[held_idx] = m.predict_proba(xtr[held_idx])[:, 1]
    print("OOF FOLD", fold, "N=", len(held_idx))

assert np.isfinite(oof).all()

print("TRAIN OOF AUC:",
      round(roc_auc_score(ytr, oof), 6))
print("TRAIN OOF AP:",
      round(average_precision_score(ytr, oof), 6))

# Fit final predictor using Train only.
final_model = model()
final_model.fit(xtr, ytr)

scores_val = final_model.predict_proba(
    val["x"][va_rej]
)[:, 1]

print("VAL AUC (DIAGNOSTIC):",
      round(roc_auc_score(
          val["labels"][va_rej], scores_val
      ), 6))


# Read exact historical valid sample list.
with open(ROOT / "G3/cache_valid_samples.txt",
          newline="") as f:
    valid_keys = {
        (row["sample_idx"], row["scene_token"])
        for row in csv.DictReader(f, delimiter="\t")
    }


def baseline_counts(split_data):
    scenes = set(split_data["scenes"].tolist())
    total = np.zeros(3, dtype=np.int64)
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
            pv = d["Pv"]
            gt = d["GT"]
            valid = (
                d["gt_cell_valid_mask"].astype(bool)
                & d["future_valid_mask"].astype(bool)[:, None, None]
                & (gt != 255)
            )
            threshold = float(d["test_seg_thresh"].item())
            assert abs(threshold - 0.1) < 1e-8

            pred = pv > threshold
            target = gt == 1

            total[0] += np.count_nonzero(pred & target & valid)
            total[1] += np.count_nonzero(pred & ~target & valid)
            total[2] += np.count_nonzero(~pred & target & valid)

    # Apply all originally accepted V2 candidates at g=1.
    accepted = split_data["accept"]
    i1 = split_data["gate_index"][1.0]
    added_tp = int(split_data["ba"][accepted, i1].sum())
    added_fp = int(split_data["ha"][accepted, i1].sum())

    total += np.array(
        [added_tp, added_fp, -added_tp],
        dtype=np.int64
    )
    return total, n_frames


def iou(counts):
    tp, fp, fn = (int(v) for v in counts)
    return tp / (tp + fp + fn)


train_base, train_frames = baseline_counts(train)
val_base, val_frames = baseline_counts(val)

print("\n=== VERIFIED BASELINES ===")
print("TRAIN FRAMES:", train_frames)
print("TRAIN V2 IoU:", round(iou(train_base), 9))
print("VAL FRAMES:", val_frames)
print("VAL V2 IoU:", round(iou(val_base), 9))
print("VAL TP/FP/FN:", val_base.tolist())

assert train_frames == 361, "Train frame mismatch"
assert val_frames == 96, "Val frame mismatch"
assert abs(iou(val_base) - REF) < 1e-8, (
    "BASELINE MISMATCH: evaluation stopped"
)


def apply_policy(base, dataset, rejected_indices,
                 selected_indices, gate):
    idx = rejected_indices[selected_indices]
    j = dataset["gate_index"][gate]

    tp_add = int(dataset["ba"][idx, j].sum())
    fp_add = int(dataset["ha"][idx, j].sum())

    counts = base + np.array(
        [tp_add, fp_add, -tp_add],
        dtype=np.int64
    )
    return iou(counts), counts, tp_add, fp_add


def select_top(scores, fraction):
    count = int(np.floor(len(scores) * fraction))
    if count <= 0:
        return np.empty(0, dtype=int)
    # Deterministic: higher score first, then lower index.
    order = np.lexsort((np.arange(len(scores)), -scores))
    return order[:count]


# Select policy using Train OOF predictions and Train GT only.
options = []
for gate in ACTIONS:
    for fraction in FRACTIONS:
        chosen = select_top(oof, fraction)
        score, counts, tp, fp = apply_policy(
            train_base, train, tr_rej, chosen, gate
        )
        options.append(dict(
            gate=gate,
            fraction=fraction,
            train_iou=score,
            n=len(chosen),
            tp_added=tp,
            fp_added=fp
        ))

# Conservative tie-break: smaller selection, weaker gate.
best = max(
    options,
    key=lambda x: (
        x["train_iou"],
        -x["fraction"],
        -x["gate"]
    )
)

print("\n=== TRAIN-ONLY POLICY SELECTION ===")
print(json.dumps(best, indent=2))

# Freeze policy before Val evaluation.
chosen_val = select_top(scores_val, best["fraction"])

val_iou, val_counts, val_tp, val_fp = apply_policy(
    val_base, val, va_rej, chosen_val, best["gate"]
)

print("\n=== FINAL HELD-OUT SCENE VALIDATION ===")
print("V2 HARD:", f"{iou(val_base):.9f}")
print("HISTORICAL E1:", "0.363507762")
print("SELECTED GATE:", best["gate"])
print("SELECTED FRACTION:", best["fraction"])
print("READMITTED CANDIDATES:", len(chosen_val))
print("ADDED TP:", val_tp)
print("ADDED FP:", val_fp)
print("FINAL TP/FP/FN:", val_counts.tolist())
print("FINAL IoU100:", f"{val_iou:.9f}")
print("DELTA VS V2 (pp):",
      round((val_iou - iou(val_base)) * 100, 6))

report = dict(
    experiment="V2-4F-3 risk-constrained readmission",
    train_scene_oof_auc=float(roc_auc_score(ytr, oof)),
    train_scene_oof_ap=float(average_precision_score(ytr, oof)),
    train_v2_iou=iou(train_base),
    val_v2_iou=iou(val_base),
    train_selected_policy=best,
    val_iou100=val_iou,
    val_readmitted=len(chosen_val),
    val_added_tp=val_tp,
    val_added_fp=val_fp,
    val_counts=val_counts.tolist(),
    test_used=False
)

with open(OUT / "summary.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nREPORT:", OUT / "summary.json")
print("V2-4F-3 COMPLETE")
