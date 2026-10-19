"""
V2-4D-2A Batch Structure Audit
================================
打印真实 dataloader batch 的完整结构（DataContainer 详情），
然后测试 mmcv.parallel.scatter 后的字段类型。

Training : NONE
Test used : NO
不修改任何模型代码。
"""

import sys, os
sys.path.insert(0, '/root/autodl-tmp/UniV2X')
sys.path.insert(0, '/root/autodl-tmp/UniV2X/projects')

import torch
import mmcv

try:
    import mmdet3d_plugin  # noqa
except Exception as e:
    print(f"[WARN] plugin import: {e}")

from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv.parallel import DataContainer

# ── Build dataloader ──────────────────────────────────────────────────────────
cfg = mmcv.Config.fromfile('projects/configs_e2e_univ2x/univ2x_coop_e2e.py')
train_ds = build_dataset(cfg.data.train)
train_dl = build_dataloader(
    train_ds, samples_per_gpu=1, workers_per_gpu=0,
    num_gpus=1, dist=False, shuffle=False, seed=42
)
data = next(iter(train_dl))

# ── Print raw batch structure ─────────────────────────────────────────────────
print("=" * 100)
print("REAL TRAIN BATCH STRUCTURE (raw, before scatter)")
print("=" * 100)

for k, v in data.items():
    print(f"\n[{k}]")
    print("  type =", type(v).__name__)

    if isinstance(v, DataContainer):
        print("  DataContainer:")
        print("    stack         =", v.stack)
        print("    cpu_only      =", v.cpu_only)
        print("    padding_value =", v.padding_value)
        print("    data type     =", type(v.data).__name__)
        try:
            print("    data len      =", len(v.data))
        except Exception:
            pass
        try:
            x = v.data[0]
            print("    data[0] type  =", type(x).__name__)
            if torch.is_tensor(x):
                print("    data[0] shape =", tuple(x.shape))
            elif isinstance(x, (list, tuple)):
                print("    data[0] len   =", len(x))
                for i, item in enumerate(x[:3]):
                    shape = tuple(item.shape) if torch.is_tensor(item) else "N/A"
                    print(f"      [{i}] type={type(item).__name__}  shape={shape}")
        except Exception as e:
            print("    inspect error =", repr(e))

    elif torch.is_tensor(v):
        print("  shape =", tuple(v.shape), "  dtype =", v.dtype)

    elif isinstance(v, dict):
        print("  dict keys:", list(v.keys())[:10])
        for sk, sv in list(v.items())[:5]:
            print(f"    [{sk}] type={type(sv).__name__}", end="")
            if isinstance(sv, DataContainer):
                print(f"  (DC stack={sv.stack} cpu_only={sv.cpu_only})", end="")
                try:
                    print(f"  data type={type(sv.data).__name__}  len={len(sv.data)}", end="")
                    x = sv.data[0]
                    if torch.is_tensor(x):
                        print(f"  data[0].shape={tuple(x.shape)}", end="")
                    elif isinstance(x, (list, tuple)):
                        print(f"  data[0] len={len(x)}", end="")
                except Exception:
                    pass
            elif torch.is_tensor(sv):
                print(f"  shape={tuple(sv.shape)}", end="")
            print()

    elif isinstance(v, (list, tuple)):
        print("  len =", len(v))
        if len(v) > 0:
            print("  [0] type =", type(v[0]).__name__)

# ── Test scatter ──────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("AFTER scatter(data, [0]) — target GPU 0")
print("=" * 100)

from mmcv.parallel import scatter

device = torch.device("cuda:0")
data_gpu = scatter(data, [0])[0]

for k, v in data_gpu.items():
    print(f"\n[{k}]")
    print("  type =", type(v).__name__)
    if torch.is_tensor(v):
        print("  shape =", tuple(v.shape))
    elif isinstance(v, dict):
        print("  dict keys:", list(v.keys())[:10])
        for sk, sv in list(v.items())[:5]:
            print(f"    [{sk}] type={type(sv).__name__}", end="")
            if torch.is_tensor(sv):
                print(f"  shape={tuple(sv.shape)}", end="")
            elif isinstance(sv, (list, tuple)):
                print(f"  len={len(sv)}", end="")
                if len(sv) > 0:
                    x = sv[0]
                    if torch.is_tensor(x):
                        print(f"  [0].shape={tuple(x.shape)}", end="")
                    else:
                        print(f"  [0].type={type(x).__name__}", end="")
            print()
    elif isinstance(v, (list, tuple)):
        print("  len =", len(v))
        if len(v) > 0:
            x = v[0]
            print("  [0] type =", type(x).__name__)
            if torch.is_tensor(x):
                print("  [0] shape =", tuple(x.shape))

print("\nBATCH STRUCTURE AUDIT COMPLETE")
print("Training : NONE   Test used : NO")
