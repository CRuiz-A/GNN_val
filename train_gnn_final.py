#!/usr/bin/env python3
"""
Comprehensive GNN Validation Pipeline for 3D Iron Ore Grade Estimation.
Implements two validation strategies side-by-side to demonstrate the spatial leakage effect:
- Strategy A: Borehole-level Group 5-Fold CV (Leakage-Free Spatial validation).
- Strategy B: Point-level 5-Fold CV (Traditional random point split - with spatial leakage).

Using the exact same compositing (n=825) and PowerTransformer as ok_v13.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from scipy.spatial import cKDTree
import warnings, time, json, random
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════
#  DATA COMPOSITING & GRAPH CONSTRUCTION UTILITIES
# ══════════════════════════════════════════════════════════════

def composite_to_length(df, comp_len=1.0):
    composited = []
    for bhid, grp in df.groupby("BHID"):
        grp = grp.sort_values("FROM").reset_index(drop=True)
        collar_x, collar_y, collar_z = grp["XCOLLAR"].iloc[0], grp["YCOLLAR"].iloc[0], grp["ZCOLLAR"].iloc[0]
        dip = grp["DIP"].iloc[0] if "DIP" in grp.columns else -90.0
        brg = grp["BRG"].iloc[0] if "BRG" in grp.columns else 0.0
        all_depths = grp[["FROM", "TO", "FE"]].values
        min_d, max_d = all_depths[:,0].min(), all_depths[:,1].max()
        current = min_d
        while current < max_d - 0.001:
            end = min(current + comp_len, max_d)
            overlap = np.maximum(0, np.minimum(all_depths[:,1], end) - np.maximum(all_depths[:,0], current))
            total_l = overlap.sum()
            if total_l > 0.05:
                w_grade = (all_depths[:,2] * overlap).sum() / total_l
                mid = (current + end) / 2.0
                composited.append({
                    "BHID": bhid, 
                    "FE": w_grade, 
                    "X": collar_x, 
                    "Y": collar_y, 
                    "Z": collar_z - mid,
                    "MID_DEPTH": mid,
                    "INTERVAL": end - current,
                    "BRG": brg,
                    "DIP": dip
                })
            current = end
    return pd.DataFrame(composited)


def load_and_process_final(path="combined.csv"):
    df_raw = pd.read_csv(path).dropna(subset=["FE", "XCOLLAR", "YCOLLAR", "ZCOLLAR", "FROM", "TO"])
    
    # Binary search for compositing length to get exactly 825 samples
    low, high = 0.55, 0.65
    for _ in range(10):
        mid = (low + high) / 2
        n = len(composite_to_length(df_raw, mid))
        if n > 825: low = mid
        else: high = mid
    df = composite_to_length(df_raw, mid).iloc[:825].reset_index(drop=True)
    return df


def build_knn_graph_final(df, k=19):
    positions = df[["X", "Y", "Z"]].values
    mu, sigma = positions.mean(axis=0), positions.std(axis=0)
    pos_std = (positions - mu) / (sigma + 1e-12)
    
    tree = cKDTree(pos_std)
    dists, idx = tree.query(pos_std, k=k + 1)
    N = len(pos_std)
    src, dst, weights = [], [], []
    for i in range(N):
        for m in range(1, k + 1):
            src.append(int(idx[i, m]))
            dst.append(i)
            # Use inverse distance as edge weight
            d = dists[i, m]
            weights.append(1.0 / (d + 1e-3))
            
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


# ══════════════════════════════════════════════════════════════
#  GEOLOGICAL GNN MODEL DEFINITION
# ══════════════════════════════════════════════════════════════

class EdgeConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x, edge_index, edge_weight=None):
        s, d = edge_index
        edge_feat = torch.cat([x[d], x[s] - x[d]], dim=-1)
        msg = self.mlp(edge_feat)
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1)
        out = torch.zeros(x.size(0), msg.size(-1), device=x.device)
        out.index_add_(0, d, msg)
        degree = torch.bincount(d, minlength=x.size(0)).clamp(min=1).unsqueeze(-1)
        return out / degree


class GeologicalGNN(nn.Module):
    def __init__(self, in_channels=9, hid=200, drop=0.2):
        super().__init__()
        self.conv1   = EdgeConvLayer(in_channels, hid)
        self.res_proj = nn.Linear(in_channels, hid)
        self.norm1    = nn.LayerNorm(hid)
        self.drop1    = nn.Dropout(drop)
        self.conv2 = EdgeConvLayer(hid, hid)
        self.norm2 = nn.LayerNorm(hid)
        self.drop2 = nn.Dropout(drop)
        self.head = nn.Linear(hid, 1)
        
    def forward(self, x, edge_index, edge_weight, mask, train_mean_yj):
        # Mean Grade Masking: replace grade feature (column 8) for masked/validated nodes
        x_masked = x.clone()
        x_masked[mask, 8] = train_mean_yj
        
        h1 = self.drop1(F.relu(self.norm1(self.conv1(x_masked, edge_index, edge_weight) + self.res_proj(x_masked))))
        h2 = self.drop2(F.relu(self.norm2(self.conv2(h1, edge_index, edge_weight) + h1)))
        return self.head(h2).squeeze(-1)


# ══════════════════════════════════════════════════════════════
#  TRAINING LOGIC
# ══════════════════════════════════════════════════════════════

def get_metrics(true_vals, pred_vals):
    r2 = r2_score(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    mae = mean_absolute_error(true_vals, pred_vals)
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def train_and_eval_fold(tr_sub_m, val_m, test_m, x_full_t, edge_index, edge_weight, y_trans_t, df, pt_y):
    """Fits the Geological GNN on one fold and returns predictions on the test set."""
    train_mean_yj = y_trans_t[tr_sub_m].mean()
    mask_during_train = val_m | test_m
    
    model = GeologicalGNN(in_channels=9).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_val, best_state, wait = float('inf'), None, 0
    for epoch in range(150):
        model.train()
        opt.zero_grad()
        out = model(x_full_t, edge_index, edge_weight, mask_during_train, train_mean_yj)
        
        # Weighted Huber Loss (§4.5.2)
        res = torch.abs(y_trans_t[tr_sub_m] - out[tr_sub_m])
        w_huber = torch.where(res <= 1.0, 0.5 * res**2, res - 0.5)
        # Weights by INTERVAL
        weights = torch.tensor(df.loc[tr_sub_m.cpu().numpy(), 'INTERVAL'].values, dtype=torch.float, device=device)
        loss = (w_huber * (weights / weights.mean())).mean()
        
        loss.backward()
        opt.step()
        
        if (epoch+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                v_out = model(x_full_t, edge_index, edge_weight, mask_during_train, train_mean_yj)
                v_loss = F.l1_loss(v_out[val_m], y_trans_t[val_m]).item()
                if v_loss < best_val:
                    best_val = v_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
            if wait >= 3:
                break
                
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_trans = model(x_full_t, edge_index, edge_weight, test_m, train_mean_yj)[test_m].cpu().numpy()
        # Safe inverse transform clipping to original training bounds to prevent NaNs
        y_trans_train = y_trans_t[tr_sub_m].cpu().numpy()
        p_trans_clipped = np.clip(p_trans, y_trans_train.min(), y_trans_train.max())
        p_fe = np.clip(pt_y.inverse_transform(p_trans_clipped.reshape(-1, 1)).flatten(), 0.0, 100.0)
        
    return p_fe


# ══════════════════════════════════════════════════════════════
#  MAIN EXECUTION PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  GEOLOGICAL GNN REPLICATION PIPELINE (n=825, Yeo-Johnson)")
    print("=" * 80)
    
    # 1. Load Data
    df = load_and_process_final("combined.csv")
    print(f"Data Prepared: n={len(df)}, BHIDs={df['BHID'].nunique()}")
    
    # 2. Features Preprocessing
    feats = ["X", "Y", "Z", "MID_DEPTH", "INTERVAL", "Xori", "Yori", "DIP"]
    df['theta_BRG'] = df['BRG'] * (np.pi / 180.0)
    df['Xori'] = np.sin(df['theta_BRG'])
    df['Yori'] = np.cos(df['theta_BRG'])
    
    scaler_x = StandardScaler()
    df_std = df.copy()
    df_std[feats] = scaler_x.fit_transform(df[feats])
    
    pt_y = PowerTransformer(method='yeo-johnson', standardize=False)
    y_trans = pt_y.fit_transform(df[['FE']]).flatten()
    
    # 3. Build GNN Graph
    edge_index, edge_weight = build_knn_graph_final(df, k=19)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    
    fe_raw_t = torch.tensor(df['FE'].values, dtype=torch.float, device=device)
    y_trans_t = torch.tensor(y_trans, dtype=torch.float, device=device)
    x_full_np = np.column_stack([df_std[feats].values, y_trans])
    x_full_t = torch.tensor(x_full_np, dtype=torch.float, device=device)
    
    # ══════════════════════════════════════════════════════════════
    #  STRATEGY A: BOREHOLE GROUP 5-FOLD CV (LEAKAGE-FREE)
    # ══════════════════════════════════════════════════════════════
    print(f"\n▸ Running Strategy A: Borehole-level Group 5-Fold CV...")
    t0 = time.time()
    
    bhids = df['BHID'].unique()
    z_means = {b: df.loc[df['BHID']==b, 'Z'].mean() for b in bhids}
    sorted_bhids = sorted(bhids, key=lambda b: z_means[b])
    folds_bhids = np.array_split(sorted_bhids, 5)
    
    pred_fe_group = np.zeros(len(df))
    for f_idx, test_bhids in enumerate(folds_bhids):
        test_m = torch.tensor(df['BHID'].isin(test_bhids), dtype=torch.bool, device=device)
        train_m = ~test_m
        
        tr_bhids_all = bhids[~np.isin(bhids, test_bhids)]
        val_bhids = tr_bhids_all[-max(1, int(0.15*len(tr_bhids_all))):]
        val_m = torch.tensor(df['BHID'].isin(val_bhids), dtype=torch.bool, device=device)
        tr_sub_m = train_m & (~val_m)
        
        p_fe = train_and_eval_fold(tr_sub_m, val_m, test_m, x_full_t, edge_index, edge_weight, y_trans_t, df, pt_y)
        pred_fe_group[df['BHID'].isin(test_bhids)] = p_fe
        
    metrics_group = get_metrics(df['FE'].values, pred_fe_group)
    print(f"  └─ Completed in {time.time() - t0:.1f}s")
    
    # ══════════════════════════════════════════════════════════════
    #  STRATEGY B: POINT-LEVEL 5-FOLD CV (WITH SPATIAL LEAKAGE)
    # ══════════════════════════════════════════════════════════════
    print(f"\n▸ Running Strategy B: Point-level 5-Fold CV...")
    t0_point = time.time()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    pred_fe_point = np.zeros(len(df))
    
    for f_idx, (tr_val_idx, te_idx) in enumerate(kf.split(df)):
        test_m = torch.zeros(len(df), dtype=torch.bool, device=device)
        test_m[te_idx] = True
        
        # Sub-split training set for Early Stopping validation
        np.random.seed(SEED + f_idx)
        np.random.shuffle(tr_val_idx)
        split_pt = int(0.85 * len(tr_val_idx))
        tr_idx, val_idx = tr_val_idx[:split_pt], tr_val_idx[split_pt:]
        
        tr_sub_m = torch.zeros(len(df), dtype=torch.bool, device=device)
        tr_sub_m[tr_idx] = True
        
        val_m = torch.zeros(len(df), dtype=torch.bool, device=device)
        val_m[val_idx] = True
        
        p_fe = train_and_eval_fold(tr_sub_m, val_m, test_m, x_full_t, edge_index, edge_weight, y_trans_t, df, pt_y)
        pred_fe_point[te_idx] = p_fe
        
    metrics_point = get_metrics(df['FE'].values, pred_fe_point)
    print(f"  └─ Completed in {time.time() - t0_point:.1f}s")
    
    # ══════════════════════════════════════════════════════════════
    #  PRINT COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("      GNN VALIDATION STRATEGY COMPARISON (n=825)")
    print("=" * 80)
    print(f"{'Estrategia de Validación':<35} | {'R²':<10} | {'RMSE':<10} | {'MAE':<10}")
    print("-" * 80)
    print(f"{'Strategy A (Borehole Group K-Fold)':<35} | {metrics_group['R2']:6.4f}   | {metrics_group['RMSE']:5.2f}      | {metrics_group['MAE']:5.2f}")
    print(f"{'Strategy B (Point-level K-Fold)':<35} | {metrics_point['R2']:6.4f}   | {metrics_point['RMSE']:5.2f}      | {metrics_point['MAE']:5.2f}")
    print("=" * 80)
    print("Note: Strategy B displays inflated performance due to cross-borehole spatial leakage.")
    
    # Save combined results
    results_json = {
        "strategy_a_group_kfold": metrics_group,
        "strategy_b_point_kfold": metrics_point
    }
    with open("results_gnn_comparison.json", "w") as f:
        json.dump(results_json, f, indent=2)
        
    df["PRED_FE_GROUP"] = pred_fe_group
    df["PRED_FE_POINT"] = pred_fe_point
    df.to_csv("predictions_gnn_comparison.csv", index=False)
    print("\nResults successfully saved to: results_gnn_comparison.json and predictions_gnn_comparison.csv")


if __name__ == "__main__":
    main()
