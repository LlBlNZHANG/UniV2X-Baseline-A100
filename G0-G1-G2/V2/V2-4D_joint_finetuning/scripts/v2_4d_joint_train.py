"""
V2-4D Joint Fine-tuning.

Train:
  - Innovation-3 (385-D)
  - OccHead dense_decoder
  - OccHead pred_prob

Frozen:
  - V2 utility predictor
  - Other UniV2X parameters

Online labels:
  - Strict exact utility labels only
  - Never proxy labels

Test split: NEVER USED
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path

import torch
import mmcv
from torch.optim import AdamW
from mmcv.parallel import scatter
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

ROOT = "/root/autodl-tmp/UniV2X"
BASE = os.path.join(ROOT, "G0-G1-G2/V2/V2-4D_joint_finetuning")

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=1)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            BASE, "checkpoints/V2-4D_joint_finetuning"
        ),
    )
    return p.parse_args()


def save_checkpoint(model, optimizer, step, output):
    Path(output).mkdir(parents=True, exist_ok=True)

    ego = model.model_ego_agent

    state = {
        "step": step,
        "innovation3": model.innovation3_module.state_dict(),
        "dense_decoder": ego.occ_head.dense_decoder.state_dict(),
        "pred_prob": ego.occ_head.pred_prob.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": {
            "stage2": audit.STAGE2_CKPT,
            "v2": audit.V2_CKPT,
            "innovation3_input_dim": 385,
            "strict_exact_label": True,
            "test_used": False,
            "loss": [
                "occ.loss_mask",
                "occ.loss_dice",
                "inn3.loss_att",
            ],
        },
    }

    target = Path(output) / f"step_{step:06d}.pth"
    temporary = target.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, target)

    print(f"[CHECKPOINT] {target}", flush=True)


def build_training_model(device):
    cfg = mmcv.Config.fromfile(
        os.path.join(
            ROOT,
            "projects/configs_e2e_univ2x/univ2x_coop_e2e.py",
        )
    )

    # V2-4D is separate from historical STCV inference.
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

    # Stage-2 initialization
    ckpt = torch.load(audit.STAGE2_CKPT, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    missing, unexpected = model.load_state_dict(
        state, strict=False
    )

    print(
        f"[STAGE2] missing={len(missing)} "
        f"unexpected={len(unexpected)}",
        flush=True,
    )

    if missing:
        raise RuntimeError(
            f"Stage-2 missing model keys: {missing[:20]}"
        )

    # Innovation-2
    v2 = V2UtilityPredictor().to(device)
    v2_ckpt = torch.load(audit.V2_CKPT, map_location="cpu")
    v2.load_state_dict(v2_ckpt["model"], strict=True)

    # Innovation-3
    inn3 = Innovation3LearnableFusion(
        bev_feat_dim=256,
        offline_feat_dim=129,
        dropout=0.1,
    ).to(device)

    extractor = STCVFeatureExtractor()

    model.attach_innovation3(inn3, v2)
    model.attach_stcv_extractor(extractor)

    # No proxy-label fallback.
    model.v2_4d_strict_exact_label = True

    # Freeze all, then unfreeze exactly three modules.
    for p in model.parameters():
        p.requires_grad_(False)

    train_modules = {
        "innovation3": model.innovation3_module,
        "dense_decoder": model.model_ego_agent.occ_head.dense_decoder,
        "pred_prob": model.model_ego_agent.occ_head.pred_prob,
    }

    for module in train_modules.values():
        for p in module.parameters():
            p.requires_grad_(True)

    # Freeze pretrained model's running statistics.
    model.eval()
    for module in train_modules.values():
        module.train()

    return cfg, model, train_modules


def main():
    args = parse_args()

    assert args.max_steps >= 1
    assert args.save_every >= 1
    assert torch.cuda.is_available(), "CUDA is required"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda:0")

    print("=" * 64)
    print("V2-4D JOINT FINE-TUNING")
    print("Strict exact label: ON")
    print("Test used: NO")
    print(f"Max steps: {args.max_steps}")
    print("=" * 64, flush=True)

    cfg, model, modules = build_training_model(device)

    ego = model.model_ego_agent

    groups = [
        {
            "params": list(modules["innovation3"].parameters()),
            "lr": 3e-4,
        },
        {
            "params": (
                list(modules["dense_decoder"].parameters())
                + list(modules["pred_prob"].parameters())
            ),
            "lr": 1e-5,
        },
    ]

    optimizer = AdamW(groups, weight_decay=1e-4)

    # Assert exact optimizer coverage.
    optimized_ids = {
        id(p)
        for group in optimizer.param_groups
        for p in group["params"]
    }

    trainable_ids = {
        id(p)
        for p in model.parameters()
        if p.requires_grad
    }

    assert optimized_ids == trainable_ids, (
        "Optimizer parameters do not match trainable parameters"
    )

    print("[PARAMETERS]")
    for name, module in modules.items():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {n:,}", flush=True)

    train_ds = build_dataset(cfg.data.train)
    train_dl = build_dataloader(
        train_ds,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=42,
    )

    print(f"[TRAIN] dataset={len(train_ds)}", flush=True)

    start_step = 0

    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device)

        modules["innovation3"].load_state_dict(
            resume["innovation3"], strict=True
        )
        modules["dense_decoder"].load_state_dict(
            resume["dense_decoder"], strict=True
        )
        modules["pred_prob"].load_state_dict(
            resume["pred_prob"], strict=True
        )

        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"])

        print(f"[RESUME] step={start_step}", flush=True)

    iterator = iter(train_dl)

    for step in range(start_step + 1, args.max_steps + 1):
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(train_dl)
            data = next(iterator)

        data_gpu = scatter(
            data, [torch.cuda.current_device()]
        )[0]

        optimizer.zero_grad(set_to_none=True)

        losses = model.forward_train(**data_gpu)

        required = ("occ.loss_mask", "occ.loss_dice")
        for key in required:
            if key not in losses:
                raise RuntimeError(f"Required loss missing: {key}")

        total = (
            losses["occ.loss_mask"]
            + losses["occ.loss_dice"]
        )

        att = losses.get("inn3.loss_att")

        if att is not None:
            total = total + att

        if not bool(torch.isfinite(total).all()):
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        dbg = getattr(model, "_v2_4d_debug", {})

        n_accept = int(dbg.get("v2_accept_count", 0))

        # V2-4D Valid-GT strict supervision contract.
        if n_accept > 0:
            if att is None:
                raise RuntimeError(
                    "Accepted candidates exist but attention loss is missing"
                )
            if dbg.get("label_source") != "valid_gt_exact":
                raise RuntimeError(
                    "Invalid Valid-GT label source during strict training: "
                    f"{dbg.get('label_source')}"
                )

            n_valid = int(dbg.get("label_valid_count", -1))
            n_positive = int(dbg.get("label_positive_count", -1))

            if not (0 <= n_positive <= n_valid <= n_accept):
                raise RuntimeError(
                    "Invalid Valid-GT supervision statistics: "
                    f"accepted={n_accept}, valid={n_valid}, "
                    f"positive={n_positive}"
                )

        # Record parameters before the update.
        before = {
            name: [
                p.detach().clone()
                for p in module.parameters()
                if p.requires_grad
            ]
            for name, module in modules.items()
        }

        total.backward()

        # Frozen parameters must not receive gradients.
        for name, p in model.named_parameters():
            if not p.requires_grad and p.grad is not None:
                raise RuntimeError(
                    f"Frozen parameter received gradient: {name}"
                )

        grad_norms = {}
        for name, module in modules.items():
            sq = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    sq += float(
                        p.grad.detach().float().pow(2).sum().item()
                    )
            grad_norms[name] = sq ** 0.5

        if not all(
            torch.isfinite(torch.tensor(v))
            for v in grad_norms.values()
        ):
            raise RuntimeError("Non-finite gradient norm")

        optimizer.step()

        # Compare all trainable parameters after optimizer.step().
        changed = {}
        for name, module in modules.items():
            after = [
                p.detach()
                for p in module.parameters()
                if p.requires_grad
            ]
            changed[name] = any(
                not torch.equal(a, b)
                for a, b in zip(before[name], after)
            )

        # Occupancy late modules must update.
        for name in ("dense_decoder", "pred_prob"):
            if not changed[name]:
                raise RuntimeError(
                    f"{name} did not update at step {step}"
                )

        # Valid-GT: only supervised accepted candidates generate BCE gradients.
        n_valid = int(dbg.get("label_valid_count", 0))
        if n_valid > n_accept:
            raise RuntimeError(
                "Valid-GT supervised count exceeds V2 accepted count"
            )
        if n_valid > 0 and not changed["innovation3"]:
            raise RuntimeError(
                "Innovation-3 did not update despite valid supervision"
            )

        print(
            f"[STEP {step:04d}] "
            f"loss={total.item():.6f} "
            f"mask={losses['occ.loss_mask'].item():.6f} "
            f"dice={losses['occ.loss_dice'].item():.6f} "
            f"att={att.item() if att is not None else 'skip'} "
            f"accepted={n_accept} "
            f"valid_labels={dbg.get('label_valid_count', 0)} "
            f"positive_labels={dbg.get('label_positive_count', 0)} "
            f"exact={dbg.get('label_source')} "
            f"grad={grad_norms} "
            f"updated={changed}",
            flush=True,
        )

        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(
                model, optimizer, step, args.output
            )

    print("=" * 64)
    print("TRAINING STEPS COMPLETED")
    print("Strict exact label: ON")
    print("Test used: NO")
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
