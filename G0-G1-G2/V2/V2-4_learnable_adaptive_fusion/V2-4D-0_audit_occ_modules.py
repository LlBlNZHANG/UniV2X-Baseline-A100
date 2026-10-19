#!/usr/bin/env python3

from pathlib import Path
import importlib

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model


ROOT = Path("/root/autodl-tmp/UniV2X")

CONFIG = ROOT / "projects/configs_e2e_univ2x/univ2x_coop_e2e.py"
CHECKPOINT = ROOT / "ckpts/univ2x_coop_e2e_stg2.pth"

KEYWORDS = (
    "occ",
    "seg",
    "fusion",
    "decoder",
    "predict",
)


def import_plugin(cfg):

    try:
        import projects.mmdet3d_plugin
        print("[Plugin] imported projects.mmdet3d_plugin")
    except Exception as e:
        print("[Plugin] direct import failed:", repr(e))

    if getattr(cfg, "plugin", False):

        plugin_dir = getattr(
            cfg,
            "plugin_dir",
            None,
        )

        if plugin_dir:

            module_path = (
                plugin_dir
                .replace("/", ".")
                .rstrip(".")
            )

            try:
                importlib.import_module(module_path)
                print("[Plugin] imported:", module_path)

            except Exception as e:
                print(
                    "[Plugin] config plugin import failed:",
                    module_path,
                    repr(e),
                )


def n_params(module):

    return sum(
        p.numel()
        for p in module.parameters()
    )


def direct_params(module):

    return sum(
        p.numel()
        for p in module.parameters(recurse=False)
    )


def matched(name):

    lower = name.lower()

    return any(
        k in lower
        for k in KEYWORDS
    )


def main():

    print("=" * 120)
    print("V2-4D-0 OCCUPANCY MODULE / PARAMETER AUDIT")
    print("=" * 120)

    print("Config     :", CONFIG)
    print("Checkpoint :", CHECKPOINT)
    print("Training   : NONE")
    print("Test used  : NO")
    print()

    assert CONFIG.exists(), CONFIG
    assert CHECKPOINT.exists(), CHECKPOINT

    cfg = Config.fromfile(
        str(CONFIG)
    )

    import_plugin(cfg)

    print()
    print("[1] Building model ...")

    model = build_model(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )

    print("[2] Loading official Stage-2 checkpoint ...")

    load_checkpoint(
        model,
        str(CHECKPOINT),
        map_location="cpu",
        strict=False,
    )

    print("[3] Model loaded.")
    print()

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Total model parameters:",
        f"{total:,}",
    )

    # ================================================================
    # A. top-level modules
    # ================================================================

    print()
    print("=" * 120)
    print("A. TOP-LEVEL MODULES")
    print("=" * 120)

    for name, module in model.named_children():

        print(
            f"{name:<45} "
            f"{module.__class__.__name__:<35} "
            f"params={n_params(module):>12,}"
        )

    # ================================================================
    # B. occupancy-related modules
    # ================================================================

    print()
    print("=" * 120)
    print("B. OCC / SEG / FUSION / DECODER / PREDICT MODULES")
    print("=" * 120)

    matched_modules = []

    for name, module in model.named_modules():

        if not name:
            continue

        if matched(name):

            matched_modules.append(
                (
                    name,
                    module.__class__.__name__,
                    n_params(module),
                    direct_params(module),
                )
            )

    for name, cls, total_p, direct_p in matched_modules:

        print(
            f"{name:<90} "
            f"{cls:<32} "
            f"total={total_p:>10,} "
            f"direct={direct_p:>10,}"
        )

    print()
    print(
        "Matched modules:",
        len(matched_modules),
    )

    # ================================================================
    # C. occupancy-related parameters
    # ================================================================

    print()
    print("=" * 120)
    print("C. OCCUPANCY-RELATED PARAMETER NAMES")
    print("=" * 120)

    matched_params = []

    for name, param in model.named_parameters():

        if matched(name):

            matched_params.append(
                (
                    name,
                    tuple(param.shape),
                    param.numel(),
                    param.requires_grad,
                )
            )

    for name, shape, count, grad in matched_params:

        print(
            f"{name:<105} "
            f"shape={str(shape):<25} "
            f"n={count:>10,} "
            f"grad={grad}"
        )

    print()
    print(
        "Matched parameter tensors:",
        len(matched_params),
    )

    print(
        "Matched parameter count  :",
        f"{sum(x[2] for x in matched_params):,}",
    )

    # ================================================================
    # D. leaf modules
    # ================================================================

    print()
    print("=" * 120)
    print("D. OCCUPANCY-RELATED LEAF MODULES WITH DIRECT PARAMETERS")
    print("=" * 120)

    leaf_modules = []

    for name, module in model.named_modules():

        if not name:
            continue

        if not matched(name):
            continue

        children = list(
            module.children()
        )

        dp = direct_params(module)

        if (
            len(children) == 0
            and dp > 0
        ):

            leaf_modules.append(
                (
                    name,
                    module.__class__.__name__,
                    dp,
                )
            )

    for name, cls, count in leaf_modules:

        print(
            f"{name:<105} "
            f"{cls:<35} "
            f"direct_params={count:>10,}"
        )

    print()
    print(
        "Leaf modules:",
        len(leaf_modules),
    )

    # ================================================================
    # E. last matched params
    # ================================================================

    print()
    print("=" * 120)
    print("E. LAST 80 MATCHED PARAMETER TENSORS")
    print("=" * 120)

    for name, shape, count, grad in matched_params[-80:]:

        print(
            f"{name:<105} "
            f"shape={str(shape):<25} "
            f"n={count:>10,}"
        )

    print()
    print("=" * 120)
    print("V2-4D-0 FINAL")
    print("=" * 120)
    print("Model load     : PASS")
    print("Parameter audit: PASS")
    print("Training       : NONE")
    print("Test used      : NO")
    print()
    print("[V2-4D-0 OCC MODULE AUDIT] COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
