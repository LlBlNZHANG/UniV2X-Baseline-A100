import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_label import (
    compute_per_horizon_baseline_stats,
    compute_binary_oracle_label_online,
)


def test_ignore_255():
    gt = torch.tensor(
        [[[[0, 1, 255, 0],
           [1, 0, 255, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]]]],
        dtype=torch.long,
    )

    ov = torch.zeros_like(gt)

    tp, fp, fn = compute_per_horizon_baseline_stats(ov, gt)

    assert fn.item() == 2, (
        "FAIL: GT=255 incorrectly contributes to foreground FN"
    )


def test_invalid_horizon():
    # Future implementation must explicitly support this mask.
    from online_label import compute_valid_gt_mask

    gt = torch.zeros((1, 5, 200, 200), dtype=torch.long)
    frame_valid = torch.tensor(
        [[True, False, True, True, True]]
    )

    mask = compute_valid_gt_mask(gt, frame_valid)

    assert mask.shape == gt.shape
    assert not mask[:, 1].any()
    assert mask[:, 0].all()


def main():
    tests = [
        test_ignore_255,
        test_invalid_horizon,
    ]

    failures = []

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except Exception as exc:
            failures.append(test.__name__)
            print(f"FAIL: {test.__name__}: {exc}")

    if failures:
        raise SystemExit(
            f"EXPECTED RED: {len(failures)} tests failing"
        )

    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
