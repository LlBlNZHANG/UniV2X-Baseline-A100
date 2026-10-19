
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from online_label import compute_valid_binary_oracle_label_online


def run_case(gt_value):
    # Evaluate the top-left 4x4 patch within an 8x8 frame.
    # A foreground cell outside the candidate ensures D > 0.
    pv = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    pi = torch.zeros_like(pv)
    pi[:, :, :4, :4] = 0.8
    ov = torch.zeros_like(pv, dtype=torch.long)
    gt = torch.zeros_like(ov)
    gt[:, :, :4, :4] = gt_value

    # Independent baseline true positive outside candidate patch.
    pv[0, 0, 7, 7] = 0.9
    ov[0, 0, 7, 7] = 1
    gt[0, 0, 7, 7] = 1

    warp = torch.ones((1, 8, 8), dtype=torch.bool)

    return compute_valid_binary_oracle_label_online(
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


def test_false_positive_should_attenuate():
    labels, mask = run_case(0)
    assert mask.tolist() == [True]
    assert labels.tolist() == [1.0], (
        "Full fusion creates false positives; attenuation should win"
    )


def test_true_positive_should_fuse():
    labels, mask = run_case(1)
    assert mask.tolist() == [True]
    assert labels.tolist() == [0.0], (
        "Full fusion recovers true positives; full fusion should win"
    )


if __name__ == "__main__":
    for test in (
        test_false_positive_should_attenuate,
        test_true_positive_should_fuse,
    ):
        test()
        print("PASS:", test.__name__)
    print("ALL NUMERIC ORACLE TESTS PASS")
