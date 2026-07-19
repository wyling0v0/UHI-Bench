"""Task 2a — LST-UHI cloud-gap reconstruction.

Protocol:
  clear-scene GT + transferred REAL cloud mask, stratified by cloud coverage
  into 4 bins (0-25 / 25-50 / 50-75 / >75%).

Two-layer fairness:
  Layer1 (main): all baselines use minimal input (coords + UHI values), no static.
  Layer2 (ablation): +static features, report delta MAE.

Baselines (CPU, this script): IDW, OrdinaryKriging, RegressionKriging(L2 only),
  RandomForest, XGBoost.  GPU baselines (ConvLSTM/DGIN/TimesFM) queued separately.

Usage:
  python run_1a.py --cities cairo bucharest lagos --years 2023 2024 2025
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import load_lst_field
from common.masks import build_cloud_eval_pairs, BIN_LABELS
from common.baselines import BASELINES, standardize_feats, _fit_variogram_gaussian


def run_one(city, years, n_clear, masks_per_bin, max_pred, seed):
    rng = np.random.default_rng(seed)
    print(f"\n{'='*60}\n[1a] {city}  loading LST {years} ...")
    t0 = time.time()
    fld = load_lst_field(city, years=years)
    print(f"    values {fld.values.shape}  feats {fld.feats.shape}  ({time.time()-t0:.0f}s)")
    feats_std = standardize_feats(fld.feats)
    xy = fld.xy_km.astype(np.float64)

    # fit ONE city-level variogram (reuse across all pairs) on pooled clear scenes
    cov = np.isnan(fld.values).mean(1)
    clear_idx = np.where(cov < 0.02)[0]
    pool_co, pool_vo = [], []
    for ti in clear_idx[:30]:
        sc = fld.values[ti]
        ok = np.isfinite(sc)
        pool_co.append(xy[ok]); pool_vo.append(sc[ok])
    pool_co = np.concatenate(pool_co); pool_vo = np.concatenate(pool_vo)
    if len(pool_co) > 8000:
        s = np.random.default_rng(0).choice(len(pool_co), 8000, replace=False)
        pool_co, pool_vo = pool_co[s], pool_vo[s]
    # standardize before fitting variogram → sill≈1, matches _solve's internal standardization
    pv = (pool_vo - pool_vo.mean()) / (pool_vo.std() + 1e-9)
    vario = _fit_variogram_gaussian(pool_co, pv)
    print(f"    city variogram (pooled, for reference): sill={vario[0]:.3f} range={vario[1]:.2f}km nugget={vario[2]:.3f}")
    # NOTE: kriging fits the variogram PER-SCENE inside _solve (on standardized obs),
    # because pooling across timestamps inflates variance and corrupts the range.
    # The pooled vario above is only a sanity reference.

    pairs = build_cloud_eval_pairs(fld.values, fld.times, rng,
                                   n_clear=n_clear, masks_per_bin=masks_per_bin)
    print(f"    eval pairs: {len(pairs)}  (per bin: " +
          ", ".join(f"{BIN_LABELS[b]}={sum(1 for p in pairs if p['bin_idx']==b)}"
                    for b in range(4)) + ")")

    # which baselines per layer
    L1 = ["IDW", "OrdinaryKriging", "UniversalKriging", "RandomForest", "XGBoost"]
    L2 = ["IDW_static", "RandomForest_static", "XGBoost_static", "RegressionKriging"]
    methods = {m: {b: {"ae": [], "n": 0} for b in range(4)} for m in (L1 + L2)}

    for pi, pr in enumerate(pairs):
        scene = fld.values[pr["clear_t"]].astype(np.float64)     # full GT (clear)
        m = pr["mask"]                                            # bool[N] cloud pixels
        # observed = not-cloud AND finite in the clear scene (clear scene may have ~2% residual NaN)
        obs_idx = np.where(~m & np.isfinite(scene))[0]
        hide_idx = np.where(m & np.isfinite(scene))[0]
        if len(obs_idx) < 5 or len(hide_idx) < 1:
            continue
        # subsample target pixels for speed
        if len(hide_idx) > max_pred:
            hide_idx = rng.choice(hide_idx, max_pred, replace=False)
        co, vo = xy[obs_idx], scene[obs_idx]
        cp = xy[hide_idx]; yt = scene[hide_idx]
        fo, fp = feats_std[obs_idx], feats_std[hide_idx]
        b = pr["bin_idx"]

        # Layer 1 (no static)
        for name in ["IDW", "OrdinaryKriging"]:
            p = BASELINES[name](co, vo, cp)
            methods[name][b]["ae"].append(np.abs(p - yt))
        # Universal Kriging = Kriging with coordinate drift (trend on coords)
        uk = BASELINES["RegressionKriging"](co, vo, cp, feat_obs=co, feat_pred=cp)
        methods["UniversalKriging"][b]["ae"].append(np.abs(uk - yt))
        for name in ["RandomForest", "XGBoost"]:
            p = BASELINES[name](co, vo, cp)                      # L1: coords only
            methods[name][b]["ae"].append(np.abs(p - yt))
        # Layer 2 (+static)
        p = BASELINES["IDW"](co, vo, cp)                          # IDW ignores feat; same
        methods["IDW_static"][b]["ae"].append(np.abs(p - yt))
        for name in ["RandomForest", "XGBoost"]:
            p = BASELINES[name](co, vo, cp, feat_obs=fo, feat_pred=fp)
            methods[f"{name}_static"][b]["ae"].append(np.abs(p - yt))
        p = BASELINES["RegressionKriging"](co, vo, cp, feat_obs=fo, feat_pred=fp)
        methods["RegressionKriging"][b]["ae"].append(np.abs(p - yt))

        if (pi + 1) % 20 == 0:
            print(f"    [{pi+1}/{len(pairs)}] done")

    # aggregate
    out = {"city": city, "n_features": int(fld.feats.shape[1]),
           "feat_names": fld.feat_names, "n_pairs": len(pairs),
           "bins": BIN_LABELS,
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
    ap.add_argument("--cities", nargs="+", default=["cairo", "bucharest", "lagos"])
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--n_clear", type=int, default=80)
    ap.add_argument("--masks_per_bin", type=int, default=20)
    ap.add_argument("--max_pred", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for c in a.cities:
        res = run_one(c, a.years, a.n_clear, a.masks_per_bin, a.max_pred, a.seed)
        all_res[c] = res
        (out_dir / f"1a_{c}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        # quick console table
        print(f"\n--- {c}  MAE by cloud bin ---")
        print(f"{'method':>22}" + "".join(f"{BIN_LABELS[b]:>10}" for b in range(4)))
        for mname in res["methods"]:
            row = "".join(f"{res['methods'][mname][str(b)]['MAE'] or float('nan'):>10.3f}"
                          for b in range(4))
            print(f"{mname:>22}{row}")

    (out_dir / "1a_all.json").write_text(json.dumps(all_res, indent=2, ensure_ascii=False))
    print(f"\n[saved] {out_dir}/1a_all.json")


if __name__ == "__main__":
    main()
