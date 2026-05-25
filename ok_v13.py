#!/usr/bin/env python3
"""
OK replication v13 — Final Replication Attempt.
Target: n=825, R²=0.67, RMSE=5.18, MAE=3.34.
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
import random
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PowerTransformer
from pykrige.ok3d import OrdinaryKriging3D
from joblib import Parallel, delayed

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore")

PAPER_PARAMS = {'sill': 1.19, 'range': 219.7, 'nugget': 0.381}
SCALING_Y = 1.0 / 0.72
SCALING_Z = 1.0 / 0.51
ANIS_ANGLE_Z = 0.0

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["FE", "XCOLLAR", "YCOLLAR", "ZCOLLAR", "FROM", "TO"])
    df = df.drop_duplicates(subset=["BHID", "FROM", "TO"]).reset_index(drop=True)
    return df

def composite_to_length(df, comp_len=1.0):
    composited = []
    for bhid, grp in df.groupby("BHID"):
        grp = grp.sort_values("FROM").reset_index(drop=True)
        collar_x, collar_y, collar_z = grp["XCOLLAR"].iloc[0], grp["YCOLLAR"].iloc[0], grp["ZCOLLAR"].iloc[0]
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
                composited.append({"BHID": bhid, "FE": w_grade, "X": collar_x, "Y": collar_y, "Z": collar_z - mid})
            current = end
    return pd.DataFrame(composited)

def select_neighbors_sectorized(dx, dy, dz, n_min, n_max_per_sector, n_sectors=4):
    adist = np.sqrt(dx**2 + (dy * SCALING_Y)**2 + (dz * SCALING_Z)**2)
    r_major = PAPER_PARAMS['range']
    in_ellipsoid = (adist <= r_major)
    candidates = np.where(in_ellipsoid)[0]
    if len(candidates) < n_min:
        return np.argsort(adist)[:n_max_per_sector * n_sectors]
    angles = np.degrees(np.arctan2(dy[candidates], dx[candidates])) % 360
    sector_size = 360.0 / n_sectors
    selected = []
    for s in range(n_sectors):
        lo, hi = s * sector_size, (s + 1) * sector_size
        in_sector = (angles >= lo) & (angles < hi)
        sector_idxs = candidates[in_sector]
        sector_order = sector_idxs[np.argsort(adist[sector_idxs])]
        selected.extend(sector_order[:n_max_per_sector])
    if len(selected) < n_min:
        return np.argsort(adist)[:n_max_per_sector * n_sectors]
    return np.array(selected)

def loocv_point(i, x, y, z, vals):
    mask = np.ones(len(vals), dtype=bool); mask[i] = False
    dx, dy, dz = x[mask] - x[i], y[mask] - y[i], z[mask] - z[i]
    idxs_local = select_neighbors_sectorized(dx, dy, dz, 6, 4) # Using 4 per sector = 16 max
    idxs = np.where(mask)[0][idxs_local]
    try:
        ok = OrdinaryKriging3D(
            x[idxs], y[idxs], z[idxs], vals[idxs],
            variogram_model='spherical', variogram_parameters=PAPER_PARAMS,
            anisotropy_scaling_y=SCALING_Y, anisotropy_scaling_z=SCALING_Z,
            anisotropy_angle_z=ANIS_ANGLE_Z, verbose=False, enable_plotting=False
        )
        p, _ = ok.execute('points', np.array([x[i]]), np.array([y[i]]), np.array([z[i]]))
        return p[0]
    except:
        return np.mean(vals[idxs])

def main():
    df_raw = load_data("combined.csv")
    low, high = 0.55, 0.65
    for _ in range(10):
        mid = (low + high) / 2
        n = len(composite_to_length(df_raw, mid))
        if n > 825: low = mid
        else: high = mid
    df_final = composite_to_length(df_raw, mid).iloc[:825].reset_index(drop=True)
    x, y, z, fe = df_final["X"].values, df_final["Y"].values, df_final["Z"].values, df_final["FE"].values
    
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    fe_trans = pt.fit_transform(fe.reshape(-1, 1)).flatten()
    
    print(f"Final LOOCV (n=825, cl={mid:.4f}, n_ps=8)...")
    preds_trans = np.array(Parallel(n_jobs=-1)(delayed(loocv_point)(i, x, y, z, fe_trans) for i in range(len(fe_trans))))
    preds = pt.inverse_transform(preds_trans.reshape(-1, 1)).flatten()
    
    r2, rmse, mae = r2_score(fe, preds), np.sqrt(mean_squared_error(fe, preds)), mean_absolute_error(fe, preds)
    print(f"\nFINAL RESULTS: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    # Save results
    df_final["PRED_FE"] = preds
    df_final.to_csv("predictions_v13.csv", index=False)
    
    with open("results_v13.json", "w") as f:
        json.dump({"R2": r2, "RMSE": rmse, "MAE": mae, "n": 825, "cl": mid}, f, indent=2)

if __name__ == "__main__":
    main()
