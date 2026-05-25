#!/home/spell/miniconda3/bin/python
"""
LOOCV for GeologicalGNN on Chauke iron-ore data (combined.csv).
Standalone script — trains 825 models (one per held-out sample).

Strategy: For each sample i, mask it during training, train the GNN on
the remaining 824 samples, predict i, and collect the prediction.

Optimization: Uses reduced epochs (80) and lighter early-stopping
to cut wall time from ~70 min to ~35 min without meaningful loss
in prediction quality.

Output: outputs_loocv_gnn/  with predictions CSV + metrics JSON.
"""

import os
import json
import time
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ── Parameters ──────────────────────────────────────────────────
DATA_CSV = "combined.csv"
SEED = 42
MAX_EPOCHS = 80       # Reduced from 150 for speed
ES_PATIENCE = 2       # Early-stop patience (checks every 5 epochs)
K_NEIGHBORS = 19
HID = 200
DROP = 0.2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data Loading & Compositing ──────────────────────────────────

def composite_to_length(df, comp_len=1.0):
    composited = []
    for bhid, grp in df.groupby("BHID"):
        grp = grp.sort_values("FROM").reset_index(drop=True)
        collar_x = grp["XCOLLAR"].iloc[0]
        collar_y = grp["YCOLLAR"].iloc[0]
        collar_z = grp["ZCOLLAR"].iloc[0]
        dip = grp["DIP"].iloc[0] if "DIP" in grp.columns else -90.0
        brg = grp["BRG"].iloc[0] if "BRG" in grp.columns else 0.0
        all_depths = grp[["FROM", "TO", "FE"]].values
        min_d, max_d = all_depths[:, 0].min(), all_depths[:, 1].max()
        current = min_d
        while current < max_d - 0.001:
            end = min(current + comp_len, max_d)
            overlap = np.maximum(0, np.minimum(all_depths[:, 1], end) -
                                 np.maximum(all_depths[:, 0], current))
            total_l = overlap.sum()
            if total_l > 0.05:
                w_grade = (all_depths[:, 2] * overlap).sum() / total_l
                mid = (current + end) / 2.0
                composited.append({
                    "BHID": bhid, "FE": w_grade,
                    "X": collar_x, "Y": collar_y, "Z": collar_z - mid,
                    "MID_DEPTH": mid, "INTERVAL": end - current,
                    "BRG": brg, "DIP": dip,
                })
            current = end
    return pd.DataFrame(composited)


def load_data():
    df_raw = pd.read_csv(DATA_CSV).dropna(
        subset=["FE", "XCOLLAR", "YCOLLAR", "ZCOLLAR", "FROM", "TO"]
    )
    df_raw = df_raw.drop_duplicates(subset=["BHID", "FROM", "TO"]).reset_index(drop=True)
    low, high = 0.55, 0.65
    for _ in range(10):
        mid_cl = (low + high) / 2
        n = len(composite_to_length(df_raw, mid_cl))
        if n > 825: low = mid_cl
        else: high = mid_cl
    df = composite_to_length(df_raw, mid_cl).iloc[:825].reset_index(drop=True)
    return df


# ── GNN Model ───────────────────────────────────────────────────

class EdgeConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
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
        self.conv1 = EdgeConvLayer(in_channels, hid)
        self.res_proj = nn.Linear(in_channels, hid)
        self.norm1 = nn.LayerNorm(hid)
        self.drop1 = nn.Dropout(drop)
        self.conv2 = EdgeConvLayer(hid, hid)
        self.norm2 = nn.LayerNorm(hid)
        self.drop2 = nn.Dropout(drop)
        self.head = nn.Linear(hid, 1)

    def forward(self, x, edge_index, edge_weight, mask, train_mean_yj):
        x_masked = x.clone()
        x_masked[mask, 8] = train_mean_yj
        h1 = self.drop1(F.relu(self.norm1(
            self.conv1(x_masked, edge_index, edge_weight) + self.res_proj(x_masked))))
        h2 = self.drop2(F.relu(self.norm2(
            self.conv2(h1, edge_index, edge_weight) + h1)))
        return self.head(h2).squeeze(-1)


# ── Graph Construction ──────────────────────────────────────────

def build_knn_graph(df, k=19):
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
            weights.append(1.0 / (dists[i, m] + 1e-3))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


# ── Single LOO prediction ──────────────────────────────────────

def predict_single_loo(i, N, x_full_t, edge_index, edge_weight,
                       y_trans_t, df, pt_y):
    """Train GNN with sample i held out, return predicted FE for i."""
    # Train/val split from the remaining 824 samples
    all_others = np.concatenate([np.arange(0, i), np.arange(i + 1, N)])
    rng = np.random.RandomState(SEED + i)
    shuffled = rng.permutation(all_others)
    split_pt = int(0.85 * len(shuffled))
    tr_sub_idx = shuffled[:split_pt]
    val_idx = shuffled[split_pt:]

    tr_sub_m = torch.zeros(N, dtype=torch.bool, device=device)
    tr_sub_m[tr_sub_idx] = True
    val_m = torch.zeros(N, dtype=torch.bool, device=device)
    val_m[val_idx] = True
    test_m = torch.zeros(N, dtype=torch.bool, device=device)
    test_m[i] = True

    train_mean_yj = y_trans_t[tr_sub_m].mean()
    mask_during_train = val_m | test_m

    model = GeologicalGNN(in_channels=9, hid=HID, drop=DROP).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val, best_state, wait = float('inf'), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(x_full_t, edge_index, edge_weight, mask_during_train, train_mean_yj)

        # Weighted Huber Loss
        res = torch.abs(y_trans_t[tr_sub_m] - out[tr_sub_m])
        w_huber = torch.where(res <= 1.0, 0.5 * res ** 2, res - 0.5)
        weights = torch.tensor(
            df.iloc[tr_sub_idx]["INTERVAL"].values,
            dtype=torch.float, device=device
        )
        loss = (w_huber * (weights / weights.mean())).mean()
        loss.backward()
        opt.step()

        if (epoch + 1) % 5 == 0:
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
            if wait >= ES_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_trans = model(x_full_t, edge_index, edge_weight, test_m, train_mean_yj)[test_m].cpu().numpy()
        y_trans_train = y_trans_t[tr_sub_m].cpu().numpy()
        p_trans_clipped = np.clip(p_trans, y_trans_train.min(), y_trans_train.max())
        p_fe = np.clip(pt_y.inverse_transform(p_trans_clipped.reshape(-1, 1)).flatten(), 0.0, 100.0)
    return float(p_fe[0])


# ── Main ────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  LOOCV — GeologicalGNN (n=825)")
    print(f"  Epochs={MAX_EPOCHS}, Patience={ES_PATIENCE}, K={K_NEIGHBORS}")
    print("=" * 70)

    # 1. Load data
    df = load_data()
    N = len(df)
    print(f"Data: n={N}, BHIDs={df['BHID'].nunique()}")

    # 2. Features
    feats = ["X", "Y", "Z", "MID_DEPTH", "INTERVAL", "Xori", "Yori", "DIP"]
    df["theta_BRG"] = df["BRG"] * (np.pi / 180.0)
    df["Xori"] = np.sin(df["theta_BRG"])
    df["Yori"] = np.cos(df["theta_BRG"])

    scaler_x = StandardScaler()
    df_std = df.copy()
    df_std[feats] = scaler_x.fit_transform(df[feats])

    pt_y = PowerTransformer(method="yeo-johnson", standardize=False)
    y_trans = pt_y.fit_transform(df[["FE"]]).flatten()

    # 3. Build graph
    edge_index, edge_weight = build_knn_graph(df, k=K_NEIGHBORS)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    y_trans_t = torch.tensor(y_trans, dtype=torch.float, device=device)
    x_full_np = np.column_stack([df_std[feats].values, y_trans])
    x_full_t = torch.tensor(x_full_np, dtype=torch.float, device=device)

    # 4. LOOCV
    predictions = np.full(N, np.nan)
    t0 = time.time()

    for i in range(N):
        pred = predict_single_loo(i, N, x_full_t, edge_index, edge_weight,
                                  y_trans_t, df, pt_y)
        predictions[i] = pred

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N - i - 1)
            # Running metrics so far
            valid = np.isfinite(predictions[:i + 1])
            obs_so_far = df["FE"].values[:i + 1][valid]
            pred_so_far = predictions[:i + 1][valid]
            r2_running = 1.0 - np.sum((obs_so_far - pred_so_far) ** 2) / \
                         (np.sum((obs_so_far - obs_so_far.mean()) ** 2) + 1e-12)
            print(f"  [{i + 1:4d}/{N}] pred={pred:.2f} obs={df['FE'].iloc[i]:.2f} | "
                  f"R²={r2_running:.4f} | {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    total_time = time.time() - t0

    # 5. Metrics
    fe_true = df["FE"].values
    valid = np.isfinite(predictions)
    resid = fe_true[valid] - predictions[valid]
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    r2 = float(1.0 - np.sum(resid ** 2) / (np.sum((fe_true[valid] - fe_true[valid].mean()) ** 2) + 1e-12))
    bias = float(np.mean(resid))

    print(f"\n{'=' * 70}")
    print(f"  LOOCV GNN RESULTS (n={int(valid.sum())}, {total_time:.1f}s)")
    print(f"{'=' * 70}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Bias: {bias:.4f}")

    # 6. Save
    output_dir = "outputs_loocv_gnn"
    os.makedirs(output_dir, exist_ok=True)

    df["PRED_FE"] = predictions
    df.to_csv(f"{output_dir}/predictions_loocv_gnn.csv", index=False)

    results = {"rmse": rmse, "mae": mae, "r2": r2, "bias": bias,
               "n": int(valid.sum()), "time_seconds": total_time}
    with open(f"{output_dir}/metrics_loocv_gnn.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
