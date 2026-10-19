"""
V2-4D 1-Batch Check: Verify All 4 Modifications Work Together
Training: NONE, Test used: NO, no optimizer.step().
"""
import sys, os
REPO_ROOT = "/root/autodl-tmp/UniV2X"
V2_4D_DIR = os.path.join(REPO_ROOT, "G0-G1-G2/V2/V2-4D_joint_finetuning")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "projects"))
sys.path.insert(0, V2_4D_DIR)

import torch, torch.nn as nn, mmcv, importlib.util
from mmcv.parallel import scatter
try:
    import mmdet3d_plugin
except Exception as e:
    print(f"[WARN] plugin import: {e}")
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from innovation3_module import Innovation3LearnableFusion

STAGE2_CKPT = os.path.join(REPO_ROOT, "ckpts/univ2x_coop_e2e_stg2.pth")
V2_CKPT = os.path.join(REPO_ROOT, "G0-G1-G2/V2/V2-1_magnitude_learning/checkpoints/main_lambda_0p50/best.pth")

class V2UtilityPredictor(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(127,512),nn.LayerNorm(512),nn.ReLU(True),nn.Dropout(p=dropout))
        self.residual_block = nn.Sequential(nn.Linear(512,512),nn.LayerNorm(512),nn.ReLU(True),nn.Dropout(p=dropout))
        self.down_256 = nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.ReLU(True))
        self.down_128 = nn.Sequential(nn.Linear(256,128),nn.LayerNorm(128),nn.ReLU(True))
        self.magnitude_head = nn.Linear(128,1)
        self.sign_head = nn.Linear(128,1)
    def forward(self, x):
        h = self.input_proj(x); h = h + self.residual_block(h)
        h = self.down_256(h); h = self.down_128(h)
        return self.magnitude_head(h).squeeze(-1), self.sign_head(h).squeeze(-1)

class STCVFeatureExtractor:
    def extract_127_features(self, pv_s, pi_s, w_s, h, r, c):
        y0,y1,x0,x1 = r*4, r*4+4, c*4, c*4+4
        pv_p = pv_s[h,y0:y1,x0:x1]; pi_p = pi_s[h,y0:y1,x0:x1]; wp = w_s[y0:y1,x0:x1]
        feat = torch.cat([
            pv_p.flatten(), pi_p.flatten(), wp.flatten().float(),
            torch.tensor([h,r,c], dtype=pv_s.dtype, device=pv_s.device),
            pv_p.mean().unsqueeze(0), pi_p.mean().unsqueeze(0),
            (pi_p-pv_p).abs().mean().unsqueeze(0),
            torch.zeros(127-51, dtype=pv_s.dtype, device=pv_s.device),
        ])
        return feat

def main():
    print("="*80); print("V2-4D 1-Batch Check")
    print("Training: NONE  Test used: NO  No optimizer.step()"); print("="*80)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = mmcv.Config.fromfile(os.path.join(REPO_ROOT,"projects/configs_e2e_univ2x/univ2x_coop_e2e.py"))

    spec = importlib.util.spec_from_file_location("multi_agent_direct",
        os.path.join(REPO_ROOT,"projects/mmdet3d_plugin/univ2x/detectors/multi_agent.py"))
    ma_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ma_mod)
    MultiAgent = ma_mod.MultiAgent

    print("\n[1] Building ego agent...")
    ego_cfg = cfg.model_ego_agent.copy()
    ego_agent = build_detector(ego_cfg).to(device)
    print("    ✓ ego built")

    print("[2] Building infra agent...")
    infra_cfg = cfg.model_other_agent_inf.copy()
    if "occ_head" in infra_cfg:
        infra_cfg.occ_head["return_occ_data"] = True
    infra_agent = build_detector(infra_cfg).to(device)
    print("    ✓ infra built")

    model = MultiAgent(model_ego_agent=ego_agent, model_other_agents={"model_other_agent_inf": infra_agent}).to(device)

    print("[3] Loading Stage-2 checkpoint...")
    if os.path.exists(STAGE2_CKPT):
        ckpt = torch.load(STAGE2_CKPT, map_location=device); state = ckpt.get("state_dict", ckpt)
        miss, unexp = model.load_state_dict(state, strict=False)
        print(f"    Loaded. Missing={len(miss)}, Unexpected={len(unexp)}")
    else: print(f"    [WARN] {STAGE2_CKPT} not found")

    print("[4] Building V2 predictor...")
    v2_pred = V2UtilityPredictor().to(device)
    if os.path.exists(V2_CKPT):
        ckpt2 = torch.load(V2_CKPT, map_location=device)
        state2 = ckpt2.get("model", ckpt2.get("model_state_dict", ckpt2.get("state_dict", ckpt2)))
        v2_pred.load_state_dict(state2, strict=True); print("    ✓ V2 predictor loaded")
    else: print(f"    [WARN] {V2_CKPT} not found")

    inn3 = Innovation3LearnableFusion(bev_feat_dim=256, dropout=0.1).to(device)
    model.attach_innovation3(inn3, v2_pred); model.attach_stcv_extractor(STCVFeatureExtractor())
    print("    ✓ Inn3 + extractor attached")

    print("[5] Building dataloader...")
    train_ds = build_dataset(cfg.data.train)
    train_dl = build_dataloader(train_ds, samples_per_gpu=1, workers_per_gpu=0, num_gpus=1, dist=False, shuffle=False, seed=42)
    print(f"    Dataset size: {len(train_ds)}")

    print("[6] Running 1 batch forward pass...")
    data = next(iter(train_dl)); data_gpu = scatter(data, [0])[0]
    model.train()

    # Forward - pass img_metas in kwargs to satisfy multi_agent.forward_train
    loss_dict = model.forward(
        ego_agent_data=data_gpu.get("ego_agent_data", data_gpu),
        other_agent_data_dict=data_gpu.get("other_agent_data_dict", {}),
        return_loss=True,
        img_metas=None,  # Add this to prevent KeyError in forward_train
    )
    print("    ✓ Forward done")

    debug = model._v2_4d_debug
    print("\n"+"="*80); print("DIAGNOSTICS"); print("="*80)
    print(f"candidate_count:      {debug.get('candidate_count','N/A')}")
    print(f"v2_accept_count:      {debug.get('v2_accept_count','N/A')}")
    print(f"label_source:         {debug.get('label_source','N/A')}")
    print(f"positive_label_count: {debug.get('label_positive_count','N/A')}")
    lpr = debug.get('label_positive_rate', None)
    print(f"positive_label_rate:  {lpr:.4f}" if lpr is not None else "positive_label_rate:  N/A")

    print("\nLosses:")
    for k,v in sorted(loss_dict.items()):
        val = v.item() if torch.is_tensor(v) else v
        print(f"  {k:40s} = {val:.6f}")

    nan_found = any(torch.isnan(v).any() for v in loss_dict.values() if torch.is_tensor(v))
    inf_found = any(torch.isinf(v).any() for v in loss_dict.values() if torch.is_tensor(v))
    print(f"\nNaN/Inf = {'NONE' if not nan_found and not inf_found else 'DETECTED'}")

    print("\n[7] Running backward (no optimizer.step)...")
    total_loss = sum(v for v in loss_dict.values() if torch.is_tensor(v) and v.requires_grad)
    if total_loss.numel() > 0:
        total_loss.backward(); print("    ✓ Backward done")
    else: print("    [WARN] No gradient-connected losses")

    def grad_norm(params):
        n = 0.0
        for p in params:
            if p.grad is not None: n += p.grad.norm().item()**2
        return n**0.5

    inn3_norm = grad_norm(model.innovation3_module.parameters())
    dd_norm   = grad_norm(model.model_ego_agent.occ_head.dense_decoder.parameters())
    pp_norm   = grad_norm(model.model_ego_agent.occ_head.pred_prob.parameters())
    v2_no_grad = all(p.grad is None for p in model.v2_predictor.parameters())

    print(f"\nGradient norms:")
    print(f"  inn3:          {inn3_norm:.6f}")
    print(f"  dense_decoder: {dd_norm:.6f}")
    print(f"  pred_prob:     {pp_norm:.6f}")
    print(f"  V2 predictor grad=None: {v2_no_grad}")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(device)/1024**3
        resv  = torch.cuda.memory_reserved(device)/1024**3
        print(f"\nGPU memory: allocated={alloc:.2f}GB  reserved={resv:.2f}GB")

    print("\n"+"="*80); print("SUMMARY")
    checks = [
        ("candidate_count != 12500",     debug.get('candidate_count',12500) != 12500),
        ("v2_accept_count > 0",          debug.get('v2_accept_count',0) > 0),
        ("label_source == 'exact'",      debug.get('label_source','') == 'exact'),
        ("positive_label_rate > 0",      (lpr or 0) > 0),
        ("inn3.loss_att present",        'inn3.loss_att' in loss_dict),
        ("occ losses present",           any('occ.' in k for k in loss_dict)),
        ("No NaN/Inf",                   not nan_found and not inf_found),
        ("Inn3 grad norm > 0",           inn3_norm > 0),
        ("Dense decoder grad norm > 0",  dd_norm > 0),
        ("V2 predictor grad=None",       v2_no_grad),
    ]
    all_pass = True
    for label, ok in checks:
        s = "PASS" if ok else "FAIL"
        print(f"  [{s}] {label}")
        if not ok: all_pass = False

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\nTraining: NONE  Test used: NO  Verdict: {verdict}")
    return verdict == "PASS"

if __name__ == "__main__":
    try:
        ok = main(); sys.exit(0 if ok else 1)
    except Exception as e:
        import traceback; traceback.print_exc(); sys.exit(1)
