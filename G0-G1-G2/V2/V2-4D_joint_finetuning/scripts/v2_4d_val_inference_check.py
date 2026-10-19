"""
V2-4D Val Inference Check
One real validation sample, inference only.
No optimizer, no training, no Test split.
Random Innovation-3 weights: interface check only.
"""
import os
import sys
import importlib.util

ROOT = "/root/autodl-tmp/UniV2X"
BASE = os.path.join(ROOT, "G0-G1-G2/V2/V2-4D_joint_finetuning")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "projects"))
sys.path.insert(0, BASE)

import torch
import mmcv
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmcv.parallel import scatter

# Reuse audited classes. Importing does not execute audit main().
audit_path = os.path.join(
    BASE, "scripts/v2_4d_exact_label_audit.py"
)
spec = importlib.util.spec_from_file_location(
    "v2_4d_audit_reference", audit_path
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

MultiAgent = audit.MultiAgent
V2UtilityPredictor = audit.V2UtilityPredictor
STCVFeatureExtractor = audit.STCVFeatureExtractor
Innovation3LearnableFusion = audit.Innovation3LearnableFusion

def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda:0")

    print("\n=== V2-4D VAL INFERENCE CHECK ===")
    print("Training: NO")
    print("Test split: NO")
    print("Innovation-3: RANDOM INITIALIZATION")
    print("Purpose: INTERFACE CHECK ONLY")

    cfg = mmcv.Config.fromfile(
        os.path.join(
            ROOT,
            "projects/configs_e2e_univ2x/univ2x_coop_e2e.py"
        )
    )

    # Do not run historical STCV during V2-4D inference.
    cfg.model_ego_agent.occ_head.use_stcv_occ = False

    # Build exactly as in the audited pipeline.
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

    # Official Stage-2 initialization.
    ckpt = torch.load(audit.STAGE2_CKPT, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(
        state, strict=False
    )
    print("Stage-2 loaded")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    # Frozen Innovation-2.
    v2 = V2UtilityPredictor().to(device)
    v2_ckpt = torch.load(audit.V2_CKPT, map_location="cpu")
    v2.load_state_dict(v2_ckpt["model"], strict=True)

    # Random 385-D Innovation-3: no performance conclusions.
    inn3 = Innovation3LearnableFusion(
        bev_feat_dim=256,
        offline_feat_dim=129,
        dropout=0.1,
    ).to(device)

    model.attach_innovation3(inn3, v2)
    model.attach_stcv_extractor(STCVFeatureExtractor())
    model.eval()

    # Validation split ONLY.
    val_cfg = cfg.data.val.copy()
    val_cfg.pop('samples_per_gpu', None)
    val_cfg.pop("samples_per_gpu", None)
    val_cfg.test_mode = True

    dataset = build_dataset(val_cfg)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=42,
    )
    print("Val samples:", len(dataset))


    found = False
    max_samples = min(30, len(dataset))

    print("Searching up to", max_samples, "Val samples")

    for idx, data in enumerate(loader):
        if idx >= max_samples:
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

        dbg = getattr(model, "_v2_4d_debug", {})

        if dbg.get("mode") != "inference":
            raise RuntimeError(
                "V2-4D inference callback was NOT executed"
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

        occ = result[0]["occ"]
        seg = occ["seg_out"]

        assert tuple(seg.shape) == (1, 5, 1, 200, 200)
        assert torch.isfinite(seg.float()).all()

        print(
            f"[VAL {idx + 1:02d}/{max_samples}] "
            f"cand={n} accepted={a} "
            f"g0={g0} g01={g01} g1={g1}",
            flush=True,
        )

        if a > 0:
            found = True
            print("\\n=== V2-4D EFFECTIVE INFERENCE PASS ===")
            print("Val sample index:", idx)
            print("V2 accepted:", a)
            print("g=0:", g0)
            print("g=0.1:", g01)
            print("g=1:", g1)
            print("seg_out shape:", tuple(seg.shape))
            break

    if not found:
        print(
            "\\nINCONCLUSIVE: no V2 accepted candidates "
            f"in first {max_samples} Val samples."
        )

    print("Training: NO | Test used: NO")
    print("No checkpoint saved.")

if __name__ == "__main__":
    main()
