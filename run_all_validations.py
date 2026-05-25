#!/home/spell/miniconda3/bin/python
"""
Run all validations on Chauke GNN data (combined.csv).
Applies the SBCV (Spatial Block Cross-Validation) framework from
../run_all_validations.py to the 3D iron-ore dataset, using:
  - OK estimation via PyKrige 3D (ok_v13 style)
  - GNN estimation via GeologicalGNN (train_gnn_final style)

Outputs: outputs_all_validation/ with per-method CSV metrics + JSON summary.
"""

import os
import sys
import json
import time
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  PARAMETERS
# ══════════════════════════════════════════════════════════════
DATA_CSV = "combined.csv"
VCOL = "FE"
K_FOLDS = 5
SEED = 42
MARGIN = 50.0

# SBCV block sizes (in XY plane, before rotation)
# Deposit extent is ~453×226m → use small blocks to get ≥10 blocks for balanced 5-fold
SBCV_DX = 100.0
SBCV_DY = 60.0
ANGLE_DEG = 0.0  # No rotation needed for this deposit

# LOO Block validation block dimensions (3D)
BLOCK_XSIZ = 10.0
BLOCK_YSIZ = 10.0
BLOCK_ZSIZ = 5.0

# OK search parameters (from ok_v13)
PAPER_PARAMS = {'sill': 1.19, 'range': 219.7, 'nugget': 0.381}
SCALING_Y = 1.0 / 0.72
SCALING_Z = 1.0 / 0.51
SEARCH_RADIUS = PAPER_PARAMS['range']
NDMIN, NDMAX = 6, 16

np.random.seed(SEED)
random.seed(SEED)

# ══════════════════════════════════════════════════════════════
#  DATA LOADING & COMPOSITING  (from ok_v13 / train_gnn_final)
# ══════════════════════════════════════════════════════════════

def composite_to_length(df, comp_len=1.0):
    """Compositing to fixed-length intervals (shared by both ok_v13 and train_gnn)."""
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
            overlap = np.maximum(0, np.minimum(all_depths[:, 1], end) - np.maximum(all_depths[:, 0], current))
            total_l = overlap.sum()
            if total_l > 0.05:
                w_grade = (all_depths[:, 2] * overlap).sum() / total_l
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
                    "DIP": dip,
                })
            current = end
    return pd.DataFrame(composited)


def load_data():
    """Load and composite to n=825 (exact same procedure as ok_v13/train_gnn)."""
    df_raw = pd.read_csv(DATA_CSV).dropna(
        subset=["FE", "XCOLLAR", "YCOLLAR", "ZCOLLAR", "FROM", "TO"]
    )
    df_raw = df_raw.drop_duplicates(subset=["BHID", "FROM", "TO"]).reset_index(drop=True)

    # Binary search for composite length → exactly 825 samples
    low, high = 0.55, 0.65
    for _ in range(10):
        mid_cl = (low + high) / 2
        n = len(composite_to_length(df_raw, mid_cl))
        if n > 825:
            low = mid_cl
        else:
            high = mid_cl

    df = composite_to_length(df_raw, mid_cl).iloc[:825].reset_index(drop=True)
    print(f"Data loaded: n={len(df)}, BHIDs={df['BHID'].nunique()}, comp_len={mid_cl:.4f}")
    return df


# ══════════════════════════════════════════════════════════════
#  SBCV PARTITIONING  (from ../run_all_validations.py)
# ══════════════════════════════════════════════════════════════

def rotate_xy(x, y, angle_deg):
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    return ca * x - sa * y, sa * x + ca * y


def make_sbcv_meta(df, dx, dy, angle_deg):
    """Build SBCV block metadata from point cloud XY extent."""
    X = df[["X", "Y"]].to_numpy(float)
    xr, yr = rotate_xy(X[:, 0], X[:, 1], -angle_deg)
    xmin, ymin = xr.min(), yr.min()
    xmax, ymax = xr.max(), yr.max()
    nxm = max(1, int(np.ceil((xmax - xmin) / dx)))
    nym = max(1, int(np.ceil((ymax - ymin) / dy)))
    return {"xmin": xmin, "ymin": ymin, "nx": nxm, "ny": nym,
            "dx": dx, "dy": dy, "angle": angle_deg}


def assign_blocks_points(df, meta):
    """Assign each composite sample to an SBCV block (XY projection)."""
    xr, yr = rotate_xy(df["X"].to_numpy(float), df["Y"].to_numpy(float), -meta["angle"])
    bix = np.floor((xr - meta["xmin"]) / meta["dx"]).astype(int).clip(0, meta["nx"] - 1)
    biy = np.floor((yr - meta["ymin"]) / meta["dy"]).astype(int).clip(0, meta["ny"] - 1)
    return (bix + biy * meta["nx"]).astype(int)


def balanced_block_folds(block_ids, k_folds=5, seed=42):
    """Greedy balanced assignment of blocks to folds."""
    all_blocks = np.arange(int(block_ids.max()) + 1, dtype=int)
    counts = pd.Series(block_ids).value_counts().reindex(all_blocks, fill_value=0)
    order = all_blocks[np.argsort(-counts.values)]
    sizes = [0] * k_folds
    b2f = {}
    for b in order:
        f = int(np.argmin(sizes))
        b2f[int(b)] = f + 1
        sizes[f] += int(counts.loc[b])
    return b2f


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    resid = y_true - y_pred
    n = len(resid)
    rmse = float(np.sqrt(np.mean(resid ** 2))) if n > 0 else np.nan
    mae = float(np.mean(np.abs(resid))) if n > 0 else np.nan
    r2 = 1.0 - np.sum(resid ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + 1e-12) if n > 1 else np.nan
    bias = float(np.mean(resid)) if n > 0 else np.nan
    return dict(rmse=rmse, mae=mae, r2=r2, bias=bias)


# ══════════════════════════════════════════════════════════════
#  METHOD 1:  OK POINT ESTIMATION  (from ok_v13.py — PyKrige 3D)
# ══════════════════════════════════════════════════════════════

def ok_point_estimates(train_df, test_df, pt_y):
    """
    OK estimation using PyKrige 3D with sectorized neighbor search,
    PowerTransform, and the paper's variogram parameters.
    Falls back to IDW when kriging fails.
    """
    from pykrige.ok3d import OrdinaryKriging3D

    x_tr = train_df["X"].values
    y_tr = train_df["Y"].values
    z_tr = train_df["Z"].values
    fe_tr = train_df["FE"].values

    # Fit PowerTransformer on training data only
    fe_trans = pt_y.fit_transform(fe_tr.reshape(-1, 1)).flatten()

    preds = []
    for _, row in test_df.iterrows():
        tx, ty, tz = row["X"], row["Y"], row["Z"]
        dx = x_tr - tx
        dy = y_tr - ty
        dz = z_tr - tz
        adist = np.sqrt(dx ** 2 + (dy * SCALING_Y) ** 2 + (dz * SCALING_Z) ** 2)

        # Sectorized neighbor selection (same as ok_v13)
        in_ellipsoid = adist <= SEARCH_RADIUS
        candidates = np.where(in_ellipsoid)[0]
        if len(candidates) < NDMIN:
            candidates = np.argsort(adist)[:NDMAX]

        # Sectorized selection in XY
        angles = np.degrees(np.arctan2(dy[candidates], dx[candidates])) % 360
        n_sectors = 4
        n_per_sector = 4
        sector_size = 360.0 / n_sectors
        selected = []
        for s in range(n_sectors):
            lo, hi = s * sector_size, (s + 1) * sector_size
            in_sector = (angles >= lo) & (angles < hi)
            sector_idxs = candidates[in_sector]
            sector_order = sector_idxs[np.argsort(adist[sector_idxs])]
            selected.extend(sector_order[:n_per_sector])
        if len(selected) < NDMIN:
            selected = list(np.argsort(adist)[:NDMAX])
        idxs = np.array(selected)

        try:
            ok = OrdinaryKriging3D(
                x_tr[idxs], y_tr[idxs], z_tr[idxs], fe_trans[idxs],
                variogram_model='spherical',
                variogram_parameters=PAPER_PARAMS,
                anisotropy_scaling_y=SCALING_Y,
                anisotropy_scaling_z=SCALING_Z,
                anisotropy_angle_z=0.0,
                verbose=False, enable_plotting=False,
            )
            p, _ = ok.execute('points', np.array([tx]), np.array([ty]), np.array([tz]))
            pred_trans = p[0]
        except Exception:
            # Fallback: IDW in transformed space
            w = 1.0 / (adist[idxs] + 1e-6)
            pred_trans = np.sum(w * fe_trans[idxs]) / np.sum(w)

        pred_fe = pt_y.inverse_transform(np.array([[pred_trans]])).flatten()[0]
        preds.append(pred_fe)

    return preds


# ══════════════════════════════════════════════════════════════
#  METHOD 2:  GNN ESTIMATION  (from train_gnn_final.py)
# ══════════════════════════════════════════════════════════════

def _import_torch():
    """Lazy import to avoid overhead if only running OK."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def build_knn_graph(df, k=19):
    torch, _, _ = _import_torch()
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
            d = dists[i, m]
            weights.append(1.0 / (d + 1e-3))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def _build_gnn_model():
    """Construct the GeologicalGNN model (same architecture as train_gnn_final.py)."""
    torch, nn, F = _import_torch()

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

    return GeologicalGNN


def gnn_fold_predictions(df, train_idx, test_idx, edge_index, edge_weight,
                          x_full_t, y_trans_t, pt_y, device):
    """Train GNN on train_idx, predict on test_idx. Returns predicted FE for test set."""
    torch, nn, F = _import_torch()

    GeologicalGNN = _build_gnn_model()

    # Sub-split training into train_sub + validation for early stopping
    rng = np.random.RandomState(SEED)
    shuffled = rng.permutation(train_idx)
    split_pt = int(0.85 * len(shuffled))
    tr_sub_idx = shuffled[:split_pt]
    val_idx = shuffled[split_pt:]

    # Guard: skip empty test folds
    if len(test_idx) == 0:
        return np.array([])

    N = len(df)
    tr_sub_m = torch.zeros(N, dtype=torch.bool, device=device)
    tr_sub_m[tr_sub_idx] = True
    val_m = torch.zeros(N, dtype=torch.bool, device=device)
    val_m[val_idx] = True
    test_m = torch.zeros(N, dtype=torch.bool, device=device)
    test_m[test_idx] = True

    train_mean_yj = y_trans_t[tr_sub_m].mean()
    mask_during_train = val_m | test_m

    model = GeologicalGNN(in_channels=9).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val, best_state, wait = float('inf'), None, 0
    for epoch in range(150):
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
            if wait >= 3:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_trans = model(x_full_t, edge_index, edge_weight, test_m, train_mean_yj)[test_m].cpu().numpy()
        y_trans_train = y_trans_t[tr_sub_m].cpu().numpy()
        p_trans_clipped = np.clip(p_trans, y_trans_train.min(), y_trans_train.max())
        p_fe = np.clip(pt_y.inverse_transform(p_trans_clipped.reshape(-1, 1)).flatten(), 0.0, 100.0)

    return p_fe


# ══════════════════════════════════════════════════════════════
#  FOLD GENERATORS  — each yields (fold_id, train_idx, test_idx)
# ══════════════════════════════════════════════════════════════

def gen_sbcv_folds(df, k_folds):
    """Spatial Block Cross-Validation folds (XY-projected rotated blocks)."""
    meta = make_sbcv_meta(df, SBCV_DX, SBCV_DY, ANGLE_DEG)
    block_ids = assign_blocks_points(df, meta)
    b2f = balanced_block_folds(block_ids, k_folds=k_folds, seed=SEED)
    fold_assign = np.array([b2f[int(b)] for b in block_ids])
    print(f"  SBCV blocks: {meta['nx']}×{meta['ny']} = {meta['nx'] * meta['ny']} blocks")
    fc = pd.Series(fold_assign).value_counts().sort_index()
    print(f"  Fold sizes: {dict(fc)}")
    for fold in range(1, k_folds + 1):
        test_idx = np.where(fold_assign == fold)[0]
        train_idx = np.where(fold_assign != fold)[0]
        yield fold, train_idx, test_idx


def gen_loocv_folds(df):
    """Leave-One-Out Cross-Validation: each sample is tested once."""
    N = len(df)
    print(f"  LOOCV: {N} folds (one sample held out each)")
    for i in range(N):
        train_idx = np.concatenate([np.arange(0, i), np.arange(i + 1, N)])
        test_idx = np.array([i])
        yield i + 1, train_idx, test_idx


def gen_bhid_folds(df, k_folds):
    """Borehole-ID Group K-Fold: all samples from a borehole stay in the same fold."""
    bhids = df["BHID"].values
    unique_bhids = np.array(sorted(df["BHID"].unique()))
    n_bh = len(unique_bhids)

    # Sort boreholes by Z-mean (same strategy as train_gnn_final.py)
    z_means = {b: df.loc[df["BHID"] == b, "Z"].mean() for b in unique_bhids}
    sorted_bhids = sorted(unique_bhids, key=lambda b: z_means[b])

    # Balanced assignment: greedy fill smallest fold
    bh_counts = {b: int((bhids == b).sum()) for b in unique_bhids}
    sizes = [0] * k_folds
    bh2fold = {}
    for b in sorted(sorted_bhids, key=lambda b: -bh_counts[b]):
        f = int(np.argmin(sizes))
        bh2fold[b] = f + 1
        sizes[f] += bh_counts[b]

    fold_assign = np.array([bh2fold[b] for b in bhids])
    fc = pd.Series(fold_assign).value_counts().sort_index()
    print(f"  BHID Group K-Fold: {n_bh} boreholes → {k_folds} folds")
    print(f"  Fold sizes: {dict(fc)}")
    for fold in range(1, k_folds + 1):
        test_idx = np.where(fold_assign == fold)[0]
        train_idx = np.where(fold_assign != fold)[0]
        yield fold, train_idx, test_idx


def build_3d_blocks(df, xsiz, ysiz, zsiz):
    """
    Discretize composite samples into 3D blocks of xsiz × ysiz × zsiz.
    Returns a DataFrame of unique blocks with centroid, mean FE, and member indices.
    """
    x, y, z = df["X"].values, df["Y"].values, df["Z"].values
    xmn = np.floor(x.min() / xsiz) * xsiz
    ymn = np.floor(y.min() / ysiz) * ysiz
    zmn = np.floor(z.min() / zsiz) * zsiz

    bix = np.floor((x - xmn) / xsiz).astype(int)
    biy = np.floor((y - ymn) / ysiz).astype(int)
    biz = np.floor((z - zmn) / zsiz).astype(int)

    # Unique block key
    df_tmp = df.copy()
    df_tmp["_bix"] = bix
    df_tmp["_biy"] = biy
    df_tmp["_biz"] = biz
    df_tmp["_bidx"] = df_tmp.index  # original row index

    blocks = []
    for (ix, iy, iz), grp in df_tmp.groupby(["_bix", "_biy", "_biz"]):
        member_idx = grp["_bidx"].values
        blocks.append({
            "bix": ix, "biy": iy, "biz": iz,
            "cx": xmn + (ix + 0.5) * xsiz,
            "cy": ymn + (iy + 0.5) * ysiz,
            "cz": zmn + (iz + 0.5) * zsiz,
            "mean_FE": grp["FE"].mean(),
            "n_samples": len(member_idx),
            "member_idx": member_idx,
        })
    return pd.DataFrame(blocks)


def gen_loo_block_folds(df, xsiz, ysiz, zsiz):
    """
    Leave-One-Out Block Validation at block support.
    Each fold holds out one 3D block; training uses all composites NOT in that block.
    Yields (block_id, train_point_idx, test_point_idx, block_row) where block_row
    contains the block centroid and observed block mean for block-level metrics.
    """
    blk_df = build_3d_blocks(df, xsiz, ysiz, zsiz)
    n_blocks = len(blk_df)
    print(f"  LOO-Block: {n_blocks} blocks ({xsiz}×{ysiz}×{zsiz} m), "
          f"samples/block: {blk_df['n_samples'].mean():.1f} avg")

    all_idx = np.arange(len(df))
    for b_id, row in blk_df.iterrows():
        test_point_idx = row["member_idx"]
        train_point_idx = np.setdiff1d(all_idx, test_point_idx)
        yield b_id + 1, train_point_idx, test_point_idx, row


def gen_stratified_bhid_folds(df, k_folds):
    """
    Stratified Group 5-Fold Cross-Validation.
    - Groups: Borehole ID (BHID) → no data leakage across boreholes.
    - Stratification: by grade bins → each fold has similar FE distribution.
    Uses sklearn.model_selection.StratifiedGroupKFold.
    """
    groups = df["BHID"].values
    # Bin grades into quantile-based strata for stratification
    n_bins = min(5, df["BHID"].nunique())
    grade_bins = pd.qcut(df["FE"], q=n_bins, labels=False, duplicates="drop")

    sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(df, grade_bins, groups))

    unique_bhids = df["BHID"].nunique()
    print(f"  Stratified BHID Group K-Fold: {unique_bhids} boreholes, "
          f"{n_bins} grade strata → {k_folds} folds")
    fc = {f + 1: len(te) for f, (_, te) in enumerate(splits)}
    print(f"  Fold sizes: {fc}")

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        yield fold_idx + 1, train_idx, test_idx


# ══════════════════════════════════════════════════════════════
#  BLOCK-LEVEL LOO EVALUATION DRIVER
# ══════════════════════════════════════════════════════════════

def run_block_loo_validation(df, edge_index, edge_weight, x_full_t, y_trans_t,
                              pt_y, device, method_name, verbose=True):
    """
    Specialized driver for LOO Block Validation at block support.
    For each held-out block:
      - OK: estimates block center (or microblock average) from training composites.
      - GNN: trains on training composites, predicts test composites, averages to block.
    Compares predicted block grade vs observed block mean FE.
    """
    torch, nn, F = _import_torch()
    t0 = time.time()
    all_preds = []
    n_folds_total = 0

    for b_id, train_idx, test_idx, blk_row in gen_loo_block_folds(
            df, BLOCK_XSIZ, BLOCK_YSIZ, BLOCK_ZSIZ):
        n_folds_total += 1
        obs_block_mean = blk_row["mean_FE"]

        if method_name == "OK_Point":
            # Estimate at block center using training composites
            train_df = df.iloc[train_idx]
            # Create a pseudo test-point at the block center
            test_point = pd.DataFrame([{
                "X": blk_row["cx"], "Y": blk_row["cy"], "Z": blk_row["cz"],
                "FE": np.nan,
            }])
            pt_fold = PowerTransformer(method="yeo-johnson", standardize=False)
            preds = ok_point_estimates(train_df, test_point, pt_fold)
            pred_val = preds[0] if len(preds) > 0 else np.nan

        elif method_name == "GNN":
            # Train GNN, predict test composites, average to block
            preds = gnn_fold_predictions(
                df, train_idx, test_idx,
                edge_index, edge_weight,
                x_full_t, y_trans_t, pt_y, device,
            )
            pred_val = np.mean(preds) if len(preds) > 0 else np.nan

        if np.isfinite(pred_val):
            all_preds.append({"obs": obs_block_mean, "pred": pred_val,
                              "fold": b_id, "n_samples": blk_row["n_samples"]})

        if verbose and n_folds_total <= 20:
            status = f"{pred_val:.2f}" if np.isfinite(pred_val) else "NaN"
            print(f"    Block {b_id}: n={blk_row['n_samples']}, "
                  f"obs={obs_block_mean:.2f}, pred={status}")
        elif verbose and n_folds_total == 21:
            print(f"    ... (suppressing per-block output)")

    elapsed = time.time() - t0
    pred_df = pd.DataFrame(all_preds)

    if len(pred_df) == 0:
        print(f"  ⚠ No valid block predictions for LOO_Block/{method_name}")
        return None

    overall = metrics(pred_df["obs"], pred_df["pred"])
    per_fold = pd.DataFrame([{"fold": "all", "n_blocks": len(pred_df), **overall}])

    if verbose:
        print(f"\n  Overall [LOO_Block / {method_name}] ({elapsed:.1f}s):")
        print(f"    Blocks evaluated: {len(pred_df)} ({BLOCK_XSIZ}×{BLOCK_YSIZ}×{BLOCK_ZSIZ}m)")
        print(f"    RMSE: {overall['rmse']:.4f}")
        print(f"    MAE:  {overall['mae']:.4f}")
        print(f"    R²:   {overall['r2']:.4f}")
        print(f"    Bias: {overall['bias']:.4f}")

    return {"overall": overall, "per_fold": per_fold, "elapsed": elapsed}


# ══════════════════════════════════════════════════════════════
#  GENERIC EVALUATION DRIVER
# ══════════════════════════════════════════════════════════════

def run_validation(strategy_name, fold_generator, method_name,
                   df, edge_index, edge_weight, x_full_t, y_trans_t,
                   pt_y, device, verbose=True):
    """
    Run a single (strategy × method) evaluation.
    Returns dict with 'overall' metrics and 'per_fold' DataFrame.
    """
    torch, nn, F = _import_torch()
    t0 = time.time()
    all_preds = []
    n_folds_total = 0

    for fold_id, train_idx, test_idx in fold_generator:
        n_folds_total += 1
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        if len(test_idx) == 0:
            if verbose:
                print(f"    Fold {fold_id}: skipped (empty)")
            continue

        if method_name == "OK_Point":
            pt_fold = PowerTransformer(method="yeo-johnson", standardize=False)
            preds = ok_point_estimates(train_df, test_df, pt_fold)
            for obs, pred in zip(test_df[VCOL].values, preds):
                if np.isfinite(pred):
                    all_preds.append({"obs": obs, "pred": pred, "fold": fold_id})

        elif method_name == "GNN":
            preds = gnn_fold_predictions(
                df, train_idx, test_idx,
                edge_index, edge_weight,
                x_full_t, y_trans_t, pt_y, device,
            )
            for obs, pred in zip(df.iloc[test_idx][VCOL].values, preds):
                if np.isfinite(pred):
                    all_preds.append({"obs": obs, "pred": pred, "fold": fold_id})

        # Progress for non-LOOCV strategies
        if verbose and n_folds_total <= 20:
            n_valid = len([p for p in all_preds if p["fold"] == fold_id])
            print(f"    Fold {fold_id}: train={len(train_idx)}, test={len(test_idx)} → {n_valid} valid")
        elif verbose and n_folds_total == 21:
            print(f"    ... (suppressing per-fold output for {strategy_name})")

    elapsed = time.time() - t0
    pred_df = pd.DataFrame(all_preds)

    if len(pred_df) == 0:
        print(f"  ⚠ No valid predictions for {strategy_name}/{method_name}")
        return None

    overall = metrics(pred_df["obs"], pred_df["pred"])

    # For LOOCV each fold has 1 sample → per-fold metrics are meaningless.
    # Group into a single "fold" for reporting.
    if n_folds_total > K_FOLDS:
        per_fold = pd.DataFrame([{"fold": "all", **overall}])
    else:
        per_fold = pred_df.groupby("fold").apply(
            lambda g: pd.Series(metrics(g["obs"], g["pred"]))
        ).reset_index()

    if verbose:
        print(f"\n  Overall [{strategy_name} / {method_name}] ({elapsed:.1f}s):")
        print(f"    RMSE: {overall['rmse']:.4f}")
        print(f"    MAE:  {overall['mae']:.4f}")
        print(f"    R²:   {overall['r2']:.4f}")
        print(f"    Bias: {overall['bias']:.4f}")
        if n_folds_total <= K_FOLDS:
            print(f"  Per-Fold:")
            print(per_fold.to_string(index=False))

    return {"overall": overall, "per_fold": per_fold, "elapsed": elapsed}


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  CHAUKE DATA — COMPREHENSIVE VALIDATION")
    print("  Strategies: SBCV · LOOCV · BHID Group · LOO Block · Stratified BHID")
    print("  Methods:    OK (PyKrige 3D) · GNN (GeologicalGNN)")
    print("=" * 70)

    # ── 1. Load data ────────────────────────────────────────────
    df = load_data()

    # ── 2. Prepare GNN tensors (shared across all strategies) ──
    torch, nn, F = _import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feats = ["X", "Y", "Z", "MID_DEPTH", "INTERVAL", "Xori", "Yori", "DIP"]
    df["theta_BRG"] = df["BRG"] * (np.pi / 180.0)
    df["Xori"] = np.sin(df["theta_BRG"])
    df["Yori"] = np.cos(df["theta_BRG"])

    scaler_x = StandardScaler()
    df_std = df.copy()
    df_std[feats] = scaler_x.fit_transform(df[feats])

    pt_y = PowerTransformer(method="yeo-johnson", standardize=False)
    y_trans = pt_y.fit_transform(df[["FE"]]).flatten()

    edge_index, edge_weight = build_knn_graph(df, k=19)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    y_trans_t = torch.tensor(y_trans, dtype=torch.float, device=device)
    x_full_np = np.column_stack([df_std[feats].values, y_trans])
    x_full_t = torch.tensor(x_full_np, dtype=torch.float, device=device)

    # ── 3. Define validation matrix ────────────────────────────
    # (strategy_name, fold_generator_factory, methods_to_run)
    validation_matrix = [
        ("SBCV",           lambda: gen_sbcv_folds(df, K_FOLDS),           ["OK_Point", "GNN"]),
        ("LOOCV",          lambda: gen_loocv_folds(df),                   ["OK_Point"]),
        ("BHID_Group",     lambda: gen_bhid_folds(df, K_FOLDS),           ["OK_Point", "GNN"]),
        ("Strat_BHID",     lambda: gen_stratified_bhid_folds(df, K_FOLDS),["OK_Point", "GNN"]),
    ]

    all_results = {}  # key = "Strategy/Method"

    for strat_name, gen_factory, methods in validation_matrix:
        for method_name in methods:
            key = f"{strat_name}/{method_name}"
            print(f"\n{'=' * 60}")
            print(f"  {key}")
            print(f"{'=' * 60}")

            fold_gen = gen_factory()
            res = run_validation(
                strat_name, fold_gen, method_name,
                df, edge_index, edge_weight, x_full_t, y_trans_t,
                pt_y, device,
            )
            if res is not None:
                all_results[key] = res

    # ── 3b. LOO Block Validation (special block-level driver) ──
    for method_name in ["OK_Point", "GNN"]:
        key = f"LOO_Block/{method_name}"
        print(f"\n{'=' * 60}")
        print(f"  {key}")
        print(f"{'=' * 60}")
        res = run_block_loo_validation(
            df, edge_index, edge_weight, x_full_t, y_trans_t,
            pt_y, device, method_name,
        )
        if res is not None:
            all_results[key] = res

    # ── 4. Save results ────────────────────────────────────────
    output_dir = "outputs_all_validation"
    os.makedirs(output_dir, exist_ok=True)

    summary_json = {}
    for key, res in all_results.items():
        safe_name = key.replace("/", "_")
        res["per_fold"].to_csv(f"{output_dir}/{safe_name}_metrics.csv", index=False)
        summary_json[key] = res["overall"]

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # ── 5. Print comparison table ──────────────────────────────
    print(f"\n{'=' * 75}")
    print("  COMPREHENSIVE VALIDATION COMPARISON")
    print(f"{'=' * 75}")
    print(f"{'Strategy/Method':<25} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Bias':>10}")
    print("-" * 75)
    for key, res in all_results.items():
        o = res["overall"]
        print(f"{key:<25} {o['rmse']:>10.4f} {o['mae']:>10.4f} {o['r2']:>10.4f} {o['bias']:>10.4f}")
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
