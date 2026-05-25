#!/usr/bin/env python3
"""
Geological GNN — 3D Iron Ore Grade Estimation
Paper: Chauke (2026), Mining, Metallurgy & Exploration 43:547–564
DOI: 10.1007/s42461-025-01429-4

This script reproduces the R2=0.72 reported in the paper by utilizing:
1. Standardized coordinates for cross-borehole graph connectivity.
2. Mean aggregation in EdgeConv for numerical stability.
3. A Random Sample Split (KFold) - NOTE: This causes spatial leakage and inflates R2.
   For honest spatial validation, use BHID-stratified splits (R2~0.22, see PROGRESS.md).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold
from scipy.spatial import cKDTree
from scipy.stats import norm
import warnings, sys

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_preprocess(filepath):
    df_raw = pd.read_csv(filepath).dropna(subset=["FE", "XCOLLAR", "YCOLLAR", "ZCOLLAR"])
    
    # Compositación para llegar a los ~632 intervalos del paper
    res = []
    for bhid, grp in df_raw.groupby("BHID"):
        grp = grp.sort_values("FROM").reset_index(drop=True)
        cx, cy, cz = grp["XCOLLAR"].iloc[0], grp["YCOLLAR"].iloc[0], grp["ZCOLLAR"].iloc[0]
        brg = grp["BRG"].iloc[0]
        all_d = grp[["FROM", "TO", "FE"]].values
        curr = all_d[:,0].min()
        while curr < all_d[:,1].max() - 0.001:
            end = min(curr + 1.2, all_d[:,1].max())
            overlap = np.maximum(0, np.minimum(all_d[:,1], end) - np.maximum(all_d[:,0], curr))
            if overlap.sum() > 0.05:
                mid = (curr+end)/2.0
                res.append({"BHID": bhid, "FE": (all_d[:,2]*overlap).sum()/overlap.sum(), "INTERVAL": end-curr,
                            "XCOLLAR": cx, "YCOLLAR": cy, "ZCOLLAR": cz, "MID_DEPTH": mid,
                            "Xori": np.sin(np.deg2rad(brg)), "Yori": np.cos(np.deg2rad(brg))})
            curr = end
    df = pd.DataFrame(res).iloc[:632].reset_index(drop=True)
    df["Z"] = df["ZCOLLAR"] - df["MID_DEPTH"]
    
    stats = {}
    for col in ["XCOLLAR", "YCOLLAR", "Z", "MID_DEPTH"]:
        mu, sigma = df[col].mean(), df[col].std()
        df[col + "_STD"] = (df[col] - mu) / sigma
        stats[col] = (mu, sigma)
        
    pt = PowerTransformer(method="yeo-johnson")
    df["FE_YJ"] = pt.fit_transform(df[["FE"]]).ravel()
    return df, pt

def build_arrays(df):
    positions_std = df[["XCOLLAR_STD", "YCOLLAR_STD", "Z_STD"]].values.astype(np.float32)
    features = df[["XCOLLAR_STD", "YCOLLAR_STD", "Z_STD", "MID_DEPTH_STD", "INTERVAL", "Xori", "Yori"]].values.astype(np.float32)
    target   = df["FE_YJ"].values.astype(np.float32)
    weights  = df["INTERVAL"].values.astype(np.float32)
    fe_orig  = df["FE"].values.astype(np.float32)
    bhids    = df["BHID"].values
    return positions_std, features, target, weights, fe_orig, bhids

def build_knn(positions, k=16):
    tree = cKDTree(positions)
    _, idx = tree.query(positions, k=k + 1)
    N = len(positions)
    src, dst = [], []
    for i in range(N):
        for m in range(1, k + 1):
            src.append(int(idx[i, m])); dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long)

class EdgeConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
    def forward(self, x, edge_index):
        s, d = edge_index
        edge_feat = torch.cat([x[d], x[s] - x[d]], dim=-1)
        msg = self.mlp(edge_feat)
        out = torch.zeros(x.size(0), msg.size(-1), device=x.device)
        # MEAN aggregation stabilizes training for higher R2
        out.index_add_(0, d, msg)
        degree = torch.bincount(d, minlength=x.size(0)).clamp(min=1).unsqueeze(-1)
        return out / degree

class GeologicalGNN(nn.Module):
    def __init__(self, in_dim=7, hid=200, drop=0.2):
        super().__init__()
        self.conv1   = EdgeConvLayer(in_dim, hid)
        self.res_proj = nn.Linear(in_dim, hid)
        self.norm1    = nn.LayerNorm(hid)
        self.drop1    = nn.Dropout(drop)
        self.conv2 = EdgeConvLayer(hid, hid)
        self.norm2 = nn.LayerNorm(hid)
        self.drop2 = nn.Dropout(drop)
        self.head = nn.Linear(hid, 1)
    def forward(self, x, edge_index):
        h1 = self.drop1(F.relu(self.norm1(self.conv1(x, edge_index) + self.res_proj(x))))
        h2 = self.drop2(F.relu(self.norm2(self.conv2(h1, edge_index) + h1)))
        return self.head(h2).squeeze(-1)

class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    def forward(self, pred, target, weights):
        w = weights / weights.mean()
        loss = F.smooth_l1_loss(pred, target, reduction="none", beta=self.delta)
        return (w * loss).mean()

def train_one_fold(model, x, ei, y, w, tr_mask, va_mask, max_epochs=250):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8)
    crit = WeightedHuberLoss(delta=1.0)
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(1, max_epochs + 1):
        model.train(); opt.zero_grad()
        loss = crit(model(x, ei)[tr_mask], y[tr_mask], w[tr_mask])
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        
        model.eval()
        with torch.no_grad(): vloss = crit(model(x, ei)[va_mask], y[va_mask], w[va_mask])
        if vloss.item() < best_val:
            best_val = vloss.item(); best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else: wait += 1
        if wait >= 25: break
    model.load_state_dict(best_state)
    return model

def calc_metrics(true_fe, pred_fe):
    ss_res = np.sum((true_fe - pred_fe) ** 2); ss_tot = np.sum((true_fe - true_fe.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12), np.sqrt(np.mean((true_fe - pred_fe) ** 2)), np.mean(np.abs(true_fe - pred_fe))

def cross_validate(df, pt):
    pos_std, features, target, weights, fe_orig, bhids = build_arrays(df)
    ei = build_knn(pos_std, k=16)
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(target,   dtype=torch.float32, device=device)
    w = torch.tensor(weights,  dtype=torch.float32, device=device)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    
    for fi, (tr_idx, te_idx) in enumerate(kf.split(df)):
        print(f"FOLD {fi+1}/5...")
        test_m = torch.zeros(len(df), dtype=torch.bool, device=device); test_m[te_idx] = True
        
        np.random.shuffle(tr_idx)
        tr_sub_idx = tr_idx[:int(0.85 * len(tr_idx))]; va_sub_idx = tr_idx[int(0.85 * len(tr_idx)):]
        tr_m = torch.zeros(len(df), dtype=torch.bool, device=device); tr_m[tr_sub_idx] = True
        va_m = torch.zeros(len(df), dtype=torch.bool, device=device); va_m[va_sub_idx] = True
        
        model = GeologicalGNN(in_dim=7, hid=200, drop=0.2).to(device)
        model = train_one_fold(model, x, ei, y, w, tr_m, va_m)

        model.eval()
        with torch.no_grad(): pred_fe = pt.inverse_transform(model(x, ei)[test_m].cpu().numpy().reshape(-1, 1)).ravel()
        r2, rmse, mae = calc_metrics(fe_orig[te_idx], pred_fe)
        results.append(dict(fold=fi+1, R2=r2, RMSE=rmse, MAE=mae))
        print(f"  → R² = {r2:.4f}   RMSE = {rmse:.4f}   MAE = {mae:.4f}")

    print(f"\nMEAN R2 = {np.mean([r['R2'] for r in results]):.4f} (Paper reported: 0.72)")
    return results

if __name__ == "__main__":
    print("="*70 + "\n GEOLOGICAL GNN — FINAL EMPIRICAL REPLICATION\n" + "="*70)
    df, pt = load_preprocess("combined.csv")
    cross_validate(df, pt)
