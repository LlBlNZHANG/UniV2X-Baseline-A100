#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/UniV2X")

MAIN_SCRIPT = (
    ROOT
    / "G0-G1-G2/V2/V2-2_planning_task_consistency/"
      "V2-2C_counterfactual_planning.py"
)

spec = importlib.util.spec_from_file_location(
    "v22c",
    MAIN_SCRIPT,
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


# ============================================================
# Exact Val split
# ============================================================

(
    valid_sample_ids,
    train_scenes,
    val_scenes,
    test_scenes,
) = m.build_exact_val_split()

plan_files = sorted(
    m.PLAN_CACHE.glob("sample_*.npz")
)

val_files = []

for p in plan_files:

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

assert len(val_files) == 96


# ============================================================
# Exact feature extractor
# ============================================================

v1 = m.STCVOcc(
    checkpoint_path=str(m.V1_CKPT),
    threshold=0.70,
)
v1.eval()

sample_features = []
sample_counts = []

for p_plan in val_files:

    p_g0 = m.G0_CACHE / p_plan.name

    with np.load(
        p_g0,
        allow_pickle=False,
    ) as g0:

        pv_np = m.normalize_array(
            g0["Pv"], 3
        )
        pi_np = m.normalize_array(
            g0["Pi_aligned"], 3
        )
        ov_np = m.normalize_array(
            g0["Ov"], 3
        )
        oi_np = m.normalize_array(
            g0["Oi"], 3
        )
        warp_np = m.normalize_array(
            g0["warp_valid_mask"], 2
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

        warp = torch.from_numpy(
            warp_np
        ).bool()

        candidate_indices, features = (
            v1.extract_candidates_and_features(
                pv,
                pi,
                ov,
                oi,
                warp,
            )
        )

        assert features.shape[1] == 127

        sample_features.append(
            features.cpu()
        )

        sample_counts.append(
            len(candidate_indices)
        )

all_features = torch.cat(
    sample_features,
    dim=0,
)

assert all_features.shape == (
    18555,
    127,
)

print("=" * 100)
print("V2-2C V2 NUMERICAL EXECUTION AUDIT")
print("=" * 100)
print("Val samples       :", len(val_files))
print("Candidates        :", len(all_features))
print("CUDA available    :", torch.cuda.is_available())


# ============================================================
# Load exact frozen V2 model
# ============================================================

def build_model(device):

    model = m.V2UtilityPredictor(
        dropout=0.1
    )

    ckpt = torch.load(
        m.V2_CKPT,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# Prediction modes
# ============================================================

@torch.no_grad()
def predict_samplewise_cpu():

    model = build_model("cpu")

    out = []

    for feat in sample_features:

        if len(feat) == 0:
            continue

        mag, _ = model(feat)

        out.append(
            mag.cpu()
        )

    return torch.cat(out)


@torch.no_grad()
def predict_all_cpu():

    model = build_model("cpu")

    mag, _ = model(
        all_features
    )

    return mag.cpu()


@torch.no_grad()
def predict_batched(
    device,
    batch_size=2048,
):

    model = build_model(device)

    out = []

    for i in range(
        0,
        len(all_features),
        batch_size,
    ):

        x = all_features[
            i:i + batch_size
        ].to(device)

        mag, _ = model(x)

        out.append(
            mag.cpu()
        )

    return torch.cat(out)


pred = {}

pred["cpu_samplewise"] = (
    predict_samplewise_cpu()
)

pred["cpu_all"] = (
    predict_all_cpu()
)

pred["cpu_batch2048"] = (
    predict_batched(
        "cpu",
        2048,
    )
)

if torch.cuda.is_available():

    pred["cuda_batch2048"] = (
        predict_batched(
            "cuda",
            2048,
        )
    )


# ============================================================
# Audit counts / numerical differences
# ============================================================

print()
print("===== ACCEPT COUNTS =====")

for name, x in pred.items():

    n_accept = int(
        (x > 0.0).sum().item()
    )

    print(
        f"{name:20s}: "
        f"{n_accept:5d} / {len(x)} "
        f"rate={n_accept/len(x):.9f}"
    )


ref = pred["cpu_samplewise"]

print()
print("===== VS CURRENT CPU SAMPLEWISE =====")

for name, x in pred.items():

    abs_diff = (
        x - ref
    ).abs()

    decision_diff = (
        (x > 0.0)
        !=
        (ref > 0.0)
    )

    print(
        f"{name:20s}: "
        f"max_abs={abs_diff.max().item():.10e} "
        f"mean_abs={abs_diff.mean().item():.10e} "
        f"decision_diff={int(decision_diff.sum())}"
    )


# ============================================================
# Threshold-margin audit
# ============================================================

print()
print("===== MARGIN AROUND ZERO =====")

x = ref

for eps in [
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
]:

    print(
        f"|mag| < {eps:.0e}: "
        f"{int((x.abs() < eps).sum())}"
    )


idx = torch.argsort(
    x.abs()
)[:20]

print()
print("===== 20 CLOSEST TO ZERO =====")

for rank, i in enumerate(
    idx.tolist(),
    start=1,
):

    vals = [
        f"{name}={pred[name][i].item():+.10e}"
        for name in pred
    ]

    signs = [
        f"{name}:{int(pred[name][i] > 0)}"
        for name in pred
    ]

    print(
        f"{rank:02d} idx={i:05d} | "
        + " | ".join(vals)
        + " || "
        + " ".join(signs)
    )


# ============================================================
# Pairwise decision disagreement
# ============================================================

print()
print("===== PAIRWISE DECISION DIFFERENCE =====")

names = list(pred.keys())

for i in range(len(names)):

    for j in range(i + 1, len(names)):

        a = names[i]
        b = names[j]

        mismatch = (
            (pred[a] > 0)
            !=
            (pred[b] > 0)
        )

        ids = torch.where(
            mismatch
        )[0]

        print(
            f"{a} vs {b}: "
            f"{len(ids)} mismatches"
        )

        if len(ids) > 0:

            print(
                " indices:",
                ids[:20].tolist(),
            )


print()
print("Frozen historical V2-1C accept count: 6156")
print("=" * 100)
