import os
import numpy as np
import torch
import torch.nn as nn

ROOT = "/root/autodl-tmp/UniV2X"

V2B = os.path.join(
    ROOT,
    "G0-G1-G2/V2/V2-4_learnable_adaptive_fusion/"
    "V2-4B_train_learnable_fusion_dataset.npz"
)

V2C = os.path.join(
    ROOT,
    "G0-G1-G2/V2/V2-4_learnable_adaptive_fusion/"
    "V2-4C-2A_train_gate_utility_curves.npz"
)

V2_CKPT = os.path.join(
    ROOT,
    "G0-G1-G2/V2/V2-1_magnitude_learning/"
    "checkpoints/main_lambda_0p50/best.pth"
)

EPS = 1e-12
THR = 0.1
BATCH = 2048


class V2UtilityPredictor(nn.Module):
    """Exact architecture from train_v2_magnitude_utility.py."""

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
        h = self.input_proj(x)
        h = h + self.residual_block(h)
        h = self.down_256(h)
        h = self.down_128(h)

        magnitude = self.magnitude_head(h).squeeze(1)
        sign_logit = self.sign_head(h).squeeze(1)

        return magnitude, sign_logit


def candidate_keys(d):
    return [
        (
            str(s),
            int(h),
            int(r),
            int(c),
        )
        for s, h, r, c in zip(
            d["sample_idx"],
            d["horizon"],
            d["patch_r"],
            d["patch_c"],
        )
    ]


def compute_utility(
    pv,
    pi,
    ov,
    gt,
    warp,
    tp,
    fp,
    fn,
    gate,
):
    """
    Vectorized exact V2-4C-2A utility:
      Pf = Pv + g * max(Pi-Pv,0) * warp
      fused = Pf > 0.1
      ADD = (~Ov) & fused & warp
      U = (TP+BA)/(D+HA) - TP/D
    """
    pos_delta = np.maximum(pi - pv, 0.0)
    pos_delta = pos_delta * warp.astype(np.float32)

    pf = pv + gate * pos_delta
    fused = pf > THR

    add = (~ov) & fused & warp

    ba = np.sum(add & gt, axis=(1, 2)).astype(np.float64)
    ha = np.sum(add & (~gt), axis=(1, 2)).astype(np.float64)

    tp = tp.astype(np.float64)
    fp = fp.astype(np.float64)
    fn = fn.astype(np.float64)

    D = tp + fp + fn

    out = np.zeros_like(D, dtype=np.float64)
    valid = D > 0

    out[valid] = (
        (tp[valid] + ba[valid]) /
        (D[valid] + ha[valid])
        -
        tp[valid] / D[valid]
    )

    return out


def main():
    print("=" * 72)
    print("V2-4D Offline Exact Reproduction Audit")
    print("Training: NONE | Test: NO")
    print("=" * 72)

    b = np.load(V2B, allow_pickle=False)
    c = np.load(V2C, allow_pickle=False)

    n_b = len(b["sample_idx"])
    n_c = len(c["sample_idx"])

    print("\n[1] Candidate domain")
    print("V2-4B rows =", n_b)
    print("V2-4C rows =", n_c)

    keys_b = candidate_keys(b)
    keys_c = candidate_keys(c)

    exact_order = keys_b == keys_c
    set_b = set(keys_b)
    set_c = set(keys_c)

    set_equal = set_b == set_c

    print("row-order identical =", exact_order)
    print("candidate-set equal  =", set_equal)
    print("unique V2-4B =", len(set_b))
    print("unique V2-4C =", len(set_c))
    print("B-only =", len(set_b - set_c))
    print("C-only =", len(set_c - set_b))

    if not set_equal:
        print("\n[FAIL] Candidate domains differ.")
        return

    # If ordering differs, align C to B.
    if exact_order:
        c_order = np.arange(n_c)
    else:
        index_c = {k: i for i, k in enumerate(keys_c)}
        c_order = np.array([index_c[k] for k in keys_b], dtype=np.int64)

    print("\n[2] Canonical candidate definition")

    ov = b["Ov_patch"].astype(bool)
    oi = b["Oi_patch"].astype(bool)
    warp = b["warp_patch"].astype(bool)

    add_recomputed = (~ov) & oi & warp

    if "candidate_add_patch" in b.files:
        add_saved = b["candidate_add_patch"].astype(bool)
        add_patch_equal = np.array_equal(
            add_recomputed,
            add_saved
        )
        print("candidate_add_patch exact =", add_patch_equal)
    else:
        add_patch_equal = True
        print("candidate_add_patch = not stored")

    add_counts = add_recomputed.reshape(n_b, -1).sum(axis=1)

    print("ADD_R min =", int(add_counts.min()))
    print("ADD_R max =", int(add_counts.max()))
    print("ADD_R>0 rate =", float(np.mean(add_counts > 0)))

    canonical_ok = bool(np.all(add_counts > 0))

    print("\n[3] V2 predictor exact reproduction")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device =", device)

    model = V2UtilityPredictor().to(device)

    ckpt = torch.load(
        V2_CKPT,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    model.eval()

    feats = b["features_127"].astype(np.float32)

    mags = []
    signs = []

    with torch.no_grad():
        for st in range(0, len(feats), BATCH):
            x = torch.from_numpy(
                feats[st:st+BATCH]
            ).to(device)

            mag, sign_logit = model(x)

            mags.append(
                mag.detach().cpu().numpy()
            )
            signs.append(
                sign_logit.sigmoid()
                .detach().cpu().numpy()
            )

    mag_online = np.concatenate(mags)
    sign_online = np.concatenate(signs)

    mag_saved = b["v2_mag"].astype(np.float32)
    sign_saved = b["v2_sign_prob"].astype(np.float32)

    accept_online = mag_online > 0
    accept_saved = b["v2_accept"].astype(bool)

    mag_diff = np.abs(
        mag_online - mag_saved
    )
    sign_diff = np.abs(
        sign_online - sign_saved
    )

    accept_agree = np.mean(
        accept_online == accept_saved
    )

    print(
        "mag max abs diff =",
        float(mag_diff.max())
    )
    print(
        "mag mean abs diff =",
        float(mag_diff.mean())
    )
    print(
        "sign max abs diff =",
        float(sign_diff.max())
    )
    print(
        "sign mean abs diff =",
        float(sign_diff.mean())
    )

    print(
        "accept online =",
        int(accept_online.sum()),
        "/",
        len(accept_online),
        "=",
        float(accept_online.mean()),
    )

    print(
        "accept saved  =",
        int(accept_saved.sum()),
        "/",
        len(accept_saved),
        "=",
        float(accept_saved.mean()),
    )

    print(
        "accept agreement =",
        float(accept_agree)
    )

    print("\n[4] U(0.1) / U(1.0) exact reproduction")

    pv = b["Pv_patch"].astype(np.float32)
    pi = b["Pi_patch"].astype(np.float32)
    gt = b["GT_patch"].astype(bool)

    tp = b["ego_tp_100"]
    fp = b["ego_fp_100"]
    fn = b["ego_fn_100"]

    u01 = compute_utility(
        pv, pi, ov, gt, warp,
        tp, fp, fn,
        0.1,
    )

    u10 = compute_utility(
        pv, pi, ov, gt, warp,
        tp, fp, fn,
        1.0,
    )

    gate_grid = c["gate_grid"]

    idx01 = int(
        np.where(
            np.abs(gate_grid - 0.1) < 1e-5
        )[0][0]
    )

    idx10 = int(
        np.where(
            np.abs(gate_grid - 1.0) < 1e-5
        )[0][0]
    )

    curve = c["utility_curve"][c_order]

    u01_off = curve[:, idx01].astype(np.float64)
    u10_off = curve[:, idx10].astype(np.float64)

    u01_diff = np.abs(u01 - u01_off)
    u10_diff = np.abs(u10 - u10_off)

    y_online = u01 > (u10 + EPS)
    y_off = u01_off > (u10_off + EPS)

    label_agree = np.mean(
        y_online == y_off
    )

    tie_online = np.sum(
        np.abs(u01 - u10) <= EPS
    )

    print(
        "U01 max abs diff =",
        float(u01_diff.max())
    )
    print(
        "U01 mean abs diff =",
        float(u01_diff.mean())
    )
    print(
        "U10 max abs diff =",
        float(u10_diff.max())
    )
    print(
        "U10 mean abs diff =",
        float(u10_diff.mean())
    )

    print(
        "U01 min/mean/max =",
        float(u01.min()),
        float(u01.mean()),
        float(u01.max()),
    )

    print(
        "U10 min/mean/max =",
        float(u10.min()),
        float(u10.mean()),
        float(u10.max()),
    )

    du = u01 - u10

    print(
        "dU min/mean/max =",
        float(du.min()),
        float(du.mean()),
        float(du.max()),
    )

    print("tie count =", int(tie_online))

    print(
        "attenuate labels online =",
        int(y_online.sum()),
        "/",
        len(y_online),
        "=",
        float(y_online.mean()),
    )

    print(
        "attenuate labels offline =",
        int(y_off.sum()),
        "/",
        len(y_off),
        "=",
        float(y_off.mean()),
    )

    print(
        "label agreement =",
        float(label_agree)
    )

    print("\n[5] Final verdict")

    checks = {
        "row count equal":
            n_b == n_c,

        "candidate set equal":
            set_equal,

        "candidate ADD_R > 0":
            canonical_ok,

        "candidate_add_patch exact":
            add_patch_equal,

        "V2 accept agreement == 1":
            accept_agree == 1.0,

        "U01 max diff <= 1e-6":
            float(u01_diff.max()) <= 1e-6,

        "U10 max diff <= 1e-6":
            float(u10_diff.max()) <= 1e-6,

        "label agreement == 1":
            label_agree == 1.0,
    }

    all_ok = True

    for k, v in checks.items():
        print(
            f"[{'PASS' if v else 'FAIL'}] {k}"
        )
        all_ok &= bool(v)

    print()
    print(
        "VERDICT =",
        "PASS" if all_ok else "FAIL"
    )


if __name__ == "__main__":
    main()
