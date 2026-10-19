import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_label import compute_valid_binary_oracle_label_online


def test_ignore_cells_do_not_affect_oracle():
    pv = torch.zeros((1, 1, 4, 4))
    pi = torch.ones_like(pv)
    ov = torch.zeros_like(pv, dtype=torch.long)
    warp = torch.ones((1, 4, 4), dtype=torch.bool)

    gt = torch.full((1, 1, 4, 4), 255, dtype=torch.long)
    gt[0, 0, 0, 0] = 1

    h = torch.tensor([0])
    r = torch.tensor([0])
    c = torch.tensor([0])
    frame_valid = torch.tensor([[True]])

    labels, mask = compute_valid_binary_oracle_label_online(
        pv=pv,
        pi_aligned=pi,
        ov=ov,
        gt=gt,
        warp_valid=warp,
        horizons=h,
        patch_rs=r,
        patch_cs=c,
        frame_valid=frame_valid,
    )

    assert mask.tolist() == [True]
    assert labels.shape == (1,)


def test_invalid_horizon_excluded():
    pv = torch.zeros((1, 2, 4, 4))
    pi = torch.ones_like(pv)
    ov = torch.zeros_like(pv, dtype=torch.long)
    gt = torch.ones_like(ov)
    warp = torch.ones((1, 4, 4), dtype=torch.bool)

    labels, mask = compute_valid_binary_oracle_label_online(
        pv=pv,
        pi_aligned=pi,
        ov=ov,
        gt=gt,
        warp_valid=warp,
        horizons=torch.tensor([0, 1]),
        patch_rs=torch.tensor([0, 0]),
        patch_cs=torch.tensor([0, 0]),
        frame_valid=torch.tensor([[True, False]]),
    )

    assert mask.tolist() == [True, False]
    assert labels.shape == (2,)


def test_all_ignore_excluded():
    pv = torch.zeros((1, 1, 4, 4))
    pi = torch.ones_like(pv)
    ov = torch.zeros_like(pv, dtype=torch.long)
    gt = torch.full_like(ov, 255)
    warp = torch.ones((1, 4, 4), dtype=torch.bool)

    labels, mask = compute_valid_binary_oracle_label_online(
        pv=pv,
        pi_aligned=pi,
        ov=ov,
        gt=gt,
        warp_valid=warp,
        horizons=torch.tensor([0]),
        patch_rs=torch.tensor([0]),
        patch_cs=torch.tensor([0]),
        frame_valid=torch.tensor([[True]]),
    )

    assert not mask.any()
    assert labels.shape == (1,)


if __name__ == "__main__":
    for fn in (
        test_ignore_cells_do_not_affect_oracle,
        test_invalid_horizon_excluded,
        test_all_ignore_excluded,
    ):
        fn()
        print("PASS:", fn.__name__)
    print("ALL ORACLE TESTS PASS")
