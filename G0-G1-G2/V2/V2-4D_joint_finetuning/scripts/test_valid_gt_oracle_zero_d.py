import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from online_label import compute_valid_binary_oracle_label_online

pv = torch.zeros((1, 1, 4, 4))
pi = torch.full_like(pv, 0.8)
ov = torch.zeros_like(pv, dtype=torch.long)
gt = torch.zeros_like(ov)
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

assert mask.tolist() == [False], "D=0 candidate must not be supervised"
assert labels.shape == (1,)
print("PASS: D=0 candidate excluded from supervision")
