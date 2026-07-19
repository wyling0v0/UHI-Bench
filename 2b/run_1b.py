"""Task 2b — Air-T UHI sparse-station reconstruction.

Protocol:
  Munich (4203px) full grid = ground truth. Random spatial mask to keep fractions
  {10%, 25%, 50%} (simulate station sparsity). Reconstruct hidden pixels.
  Cologne (1148px) is the reference real-world station density (~27% of Munich).

Two-layer fairness (same as 1a):
  Layer1 (main): coords + UHI values only, no static.
  Layer2 (ablation): +static features.

Baselines (CPU, this script): IDW, OrdinaryKriging, UniversalKriging (coord-trend),
  RegressionKriging (static drift, L2), XGBoost.  GNN(IGNNK)/KCN queued (GPU).

Usage:
  python run_1b.py --city munich --years 2023 2024 2025
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import load_ta_field
from common.baselines import BASELINES, standardize_feats, _fit_variogram_gaussian
from common.masks import BIN_LABELS, random_mask_for_bin

NBINS = len(BIN_LABELS)   # 4 missing% bins, aligned with Task 2a's x-axis


def run_one(city, years, n_times, n_splits, max_pred, seed):
    rng = np.random.default_rng(seed)
    print(f"\n{'='*60}\n[1b] {city}  loading AirT {years} ...")
    t0 = time.time()
    fld = load_ta_field(city, years=years)
    print(f"    values {fld.values.shape}  feats {fld.feats.shape}  ({time.time()-t0:.0f}s)")
    feats_std = standardize_feats(fld.feats)
    xy = fld.xy_km.astype(np.float64)
    N = fld.values.shape[1]

    # city variogram (fit once, reuse)
    cov = np.isnan(fld.values).mean(1)
    clear_idx = np.where(cov < 0.02)[0]
    pool_co, pool_vo = [], []
    for ti in clear_idx[:30]:
        sc = fld.values[ti]; ok = np.isfinite(sc)
        pool_co.append(xy[ok]); pool_vo.append(sc[ok])
    pool_co = np.concatenate(pool_co); pool_vo = np.concatenate(pool_vo)
    if len(pool_co) > 8000:
        s = np.random.default_rng(0).choice(len(pool_co), 8000, replace=False)
        pool_co, pool_vo = pool_co[s], pool_vo[s]
    pv = (pool_vo - pool_vo.mean()) / (pool_vo.std() + 1e-9)
    vario = _fit_variogram_gaussian(pool_co, pv)
    print(f"    city variogram (pooled, ref): sill={vario[0]:.4f} range={vario[1]:.2f}km nugget={vario[2]:.4f}")
    # kriging fits variogram PER-SCENE inside _solve (pooled fit is reference only)

    # pick eval timestamps (with full/near-full valid grid)
    full_idx = np.where(cov < 0.01)[0]
    if len(full_idx) > n_times:
        full_idx = rng.choice(full_idx, n_times, replace=False)
    print(f"    eval timestamps: {len(full_idx)}  ×  splits: {n_splits}  ×  missing-bins{BIN_LABELS}")

    methods = {}
    for nm in ["IDW", "OrdinaryKriging", "UniversalKriging",
               "RandomForest", "XGBoost", "RegressionKriging",
               "RandomForest_static", "XGBoost_static"]:
        methods[nm] = {str(b): {"ae": []} for b in range(NBINS)}

    n_eval = 0
    for ti in full_idx:
        scene = fld.values[ti].astype(np.float64)
        if not np.isfinite(scene).all():
            scene = np.where(np.isfinite(scene), scene, np.nanmean(scene))
        for s in range(n_splits):
            for b in range(NBINS):
                hide_idx = np.where(random_mask_for_bin(N, b, rng))[0]
                obs_idx = np.setdiff1d(np.arange(N), hide_idx)
                if len(hide_idx) > max_pred:
                    hide_eval = rng.choice(hide_idx, max_pred, replace=False)
                else:
                    hide_eval = hide_idx
                co, vo = xy[obs_idx], scene[obs_idx]
                cp = xy[hide_eval]; yt = scene[hide_eval]
                fo, fp = feats_std[obs_idx], feats_std[hide_eval]

                methods["IDW"][str(b)]["ae"].append(np.abs(BASELINES["IDW"](co, vo, cp) - yt))
                methods["OrdinaryKriging"][str(b)]["ae"].append(np.abs(BASELINES["OrdinaryKriging"](co, vo, cp) - yt))
                # Universal Kriging = trend on coordinates
                uk = BASELINES["RegressionKriging"](co, vo, cp, feat_obs=co, feat_pred=cp)
                methods["UniversalKriging"][str(b)]["ae"].append(np.abs(uk - yt))
                methods["XGBoost"][str(b)]["ae"].append(np.abs(BASELINES["XGBoost"](co, vo, cp) - yt))
                methods["RandomForest"][str(b)]["ae"].append(np.abs(BASELINES["RandomForest"](co, vo, cp) - yt))
                rk = BASELINES["RegressionKriging"](co, vo, cp, feat_obs=fo, feat_pred=fp)
                methods["RegressionKriging"][str(b)]["ae"].append(np.abs(rk - yt))
                methods["XGBoost_static"][str(b)]["ae"].append(np.abs(
                    BASELINES["XGBoost"](co, vo, cp, feat_obs=fo, feat_pred=fp) - yt))
                methods["RandomForest_static"][str(b)]["ae"].append(np.abs(
                    BASELINES["RandomForest"](co, vo, cp, feat_obs=fo, feat_pred=fp) - yt))
                n_eval += 1
        if (np.where(full_idx == ti)[0][0] + 1) % 5 == 0:
            print(f"    timestamps done: {np.where(full_idx==ti)[0][0]+1}/{len(full_idx)}  (evals={n_eval})")

    out = {"city": city, "n_features": int(fld.feats.shape[1]), "feat_names": fld.feat_names,
           "bins": BIN_LABELS, "axis": "missing%", "n_times": len(full_idx),
           "n_splits": n_splits,
           "protocol": {"eval_years": list(map(int, years)),
                        "years": list(map(int, years)),
                        "mode": "scene-wise transductive",
                        "note": "Each eval scene uses only visible pixels at that timestamp."},
           "methods": {}}
    for mname, bins in methods.items():
        out["methods"][mname] = {}
        for b, d in bins.items():
            ae = np.concatenate(d["ae"]) if d["ae"] else np.array([])
            out["methods"][mname][str(b)] = {
                "MAE": float(ae.mean()) if ae.size else None,
                "RMSE": float(np.sqrt((ae ** 2).mean())) if ae.size else None,
                "N": int(ae.size),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="munich")
    ap.add_argument("--cologne_density_ref", action="store_true",
                   help="just print Cologne px count as reference")
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n_times", type=int, default=40)
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--max_pred", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    res = run_one(a.city, a.years, a.n_times, a.n_splits, a.max_pred, a.seed)
    (out_dir / f"1b_{a.city}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n--- {a.city}  MAE by missing% bin ---")
    print(f"{'method':>22}" + "".join(f"{b:>12}" for b in res["bins"]))
    for mname in res["methods"]:
        row = "".join(f"{(res['methods'][mname][str(k)]['MAE'] or float('nan')):>12.4f}" for k in range(NBINS))
        print(f"{mname:>22}{row}")
    print(f"\n[saved] {out_dir}/1b_{a.city}.json")


if __name__ == "__main__":
    main()
